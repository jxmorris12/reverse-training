from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load gpt2 model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# load wikitext data
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# W: lm head weights
original_W = model.lm_head.weight.detach().clone().to(device)  # Shape: [50257, 768]

# W': random weights
W_prime = torch.nn.Parameter(original_W.clone(), requires_grad=True).to(device)

# Hyperparameters
lambda_param = 100.0
learning_rate = 0.0001
num_iterations = 1000
batch_size = 32

# Optimizer
optimizer = optim.Adam([W_prime], lr=learning_rate)

def get_hidden_states(model, tokenizer, text_batch, device):
    inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
    return hidden_states


def run_gaussian_defense(model_state_dict: dict[str, torch.Tensor], weight: float = 0.01) -> dict[str, torch.Tensor]:
    for key, value in model_state_dict.items():
        model_state_dict[key] = torch.randn_like(value) * weight + value
    return model_state_dict


def run_lagrangian_defense(model, tokenizer, dataset, device, num_iterations, batch_size, learning_rate, lambda_param, patience):

    # data
    train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    val_dataset = dataset["validation"].shuffle(seed=42).select(range(100))

    model = model.to(device)

    # Training parameters
    best_val_kl = float('inf')
    patience = 5
    patience_counter = 0

    # Training loop
    for iteration in range(num_iterations):
        # Get batch of actual text
        idx = iteration % (len(train_dataset) // batch_size)
        text_batch = train_dataset[idx * batch_size: (idx + 1) * batch_size]["text"]
        
        # Get hidden states from GPT-2
        hidden_states = get_hidden_states(model, tokenizer, text_batch, device)
        x = hidden_states.view(-1, hidden_states.size(-1))  # Flatten sequence dimension
        
        # Scale hidden states
        x = F.normalize(x, dim=-1) * 10.0
        
        # Compute logits
        logits_original = torch.matmul(x, original_W.t())
        logits_prime = torch.matmul(x, W_prime.t())
        
        # Optional: Temperature scaling for softer probabilities
        temperature = 2.0
        logits_original = logits_original / temperature
        logits_prime = logits_prime / temperature
        
        # Compute KL divergence (switched order and using direct probs)
        kl_div = F.kl_div(
            F.log_softmax(logits_prime, dim=-1),
            F.softmax(logits_original, dim=-1),
            reduction='batchmean',
            log_target=False
        )
        
        # Compute L2 distance between W and W'
        weight_diff = torch.norm(W_prime - original_W)
        
        # Compute loss: maximize weight difference - lambda * KL divergence
        loss = -weight_diff + lambda_param * kl_div
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation every 50 iterations
        if iteration % 50 == 0:
            model.eval()
            val_kl_divs = []
            
            with torch.no_grad():
                for val_idx in range(0, len(val_dataset), batch_size):
                    val_batch = val_dataset[val_idx:val_idx + batch_size]["text"]
                    val_hidden_states = get_hidden_states(model, tokenizer, val_batch, device)
                    val_x = val_hidden_states.view(-1, val_hidden_states.size(-1))
                    val_x = F.normalize(val_x, dim=-1) * 10.0
                    
                    val_logits_original = torch.matmul(val_x, original_W.t()) / temperature
                    val_logits_prime = torch.matmul(val_x, W_prime.t()) / temperature
                    
                    val_kl = F.kl_div(
                        F.log_softmax(val_logits_prime, dim=-1),
                        F.softmax(val_logits_original, dim=-1),
                        reduction='batchmean',
                        log_target=False
                    )
                    val_kl_divs.append(val_kl.item())
            
            avg_val_kl = sum(val_kl_divs) / len(val_kl_divs)
            
            print(f"Iteration {iteration}")
            print(f"Training - Weight diff: {weight_diff.item():.4f}, KL div: {kl_div.item():.4f}")
            print(f"Validation - KL div: {avg_val_kl:.4f}")
            print(f"Loss: {loss.item():.4f}\n")
            
            # Early stopping based on validation KL
            if avg_val_kl < best_val_kl:
                best_val_kl = avg_val_kl
                best_W_prime = W_prime.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at iteration {iteration}")
                print(f"Best validation KL: {best_val_kl:.4f}")
                break

    # Use best weights found during validation
    obfuscated_W = best_W_prime