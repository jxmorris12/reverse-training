from typing import Iterable, Optional
import collections
import hashlib
import json
import numpy as np
import random
import time
import os
import pickle

import datasets
import torch
import torch.nn.functional as F
import transformers
import tqdm

from utils.batch import find_executable_batch_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def hash_model_params(model):
    """Generate a hash of model parameters."""
    # TODO: This is a hack to get around the fact that the model state dict is not serializable.
    # TODO: We should use a more robust method to hash the model parameters.
    # Extract state dict
    state_dict = model.state_dict()

    # Convert to JSON string
    json_str = json.dumps([n for n, _ in sorted(state_dict.items())], sort_keys=True)
    
    # Generate hash
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def limit_layers(model: transformers.PreTrainedModel, n_layers: int) -> transformers.PreTrainedModel:
    if hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'h'):
            # gpt2
            model.transformer.h = model.transformer.h[:n_layers]
        else:
            model.transformer.layer = model.transformer.layer[:n_layers]
    elif hasattr(model, 'encoder'):
        if hasattr(model.encoder, 'layers'):
            model.encoder.layers = model.encoder.layers[:n_layers]
        else:
            model.encoder.layer = model.encoder.layer[:n_layers]
    else:
        raise RuntimeError(f"unknown how to limit layers of model {type(model)}")
    return model


def state_dict_to_tensor(ordered_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.data.to(device).reshape(-1) for n, p in ordered_dict.items()], 0)


def get_token_embeddings_from_dataset(dataset_size: int, sequence_length: int, ds: datasets.Dataset, text_column_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """ initialize the synthetic data """
    student_net = get_model("gpt2")
    student_net.train()
    student_token_embeddings = student_net.get_input_embeddings().to(device)
    # TODO: Consider other initialization methods
    # def simple_soft_one_hot(x, num_classes, temperature=1.0):
    #     one_hot = F.one_hot(x, num_classes).float()
    #     return F.softmax(one_hot / temperature, dim=-1)
    
    hidden_size = student_token_embeddings.weight.shape[1]

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left" # important for correct truncation
    tokenizer.padding_side = "left" 
    tokens = tokenizer.batch_encode_plus(
        ds[text_column_name][0:dataset_size],
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=sequence_length-1, 
        return_attention_mask=False
    )
    token_embeddings_syn = student_token_embeddings(tokens["input_ids"].to(device))
    assert token_embeddings_syn.shape == (dataset_size, sequence_length - 1, hidden_size), f"invalid shape: {token_embeddings_syn.shape}, need {(dataset_size, sequence_length - 1, hidden_size)}"

    num_classes = 4
    CLASS_MAP = torch.tensor([352,  362,  657,  513], device=device)
    token_labels_syn = torch.randint(low=0, high=num_classes, size=[dataset_size], device=device)
    token_labels_syn = CLASS_MAP[token_labels_syn]

    return (token_embeddings_syn, token_labels_syn)


def get_token_embeddings_random_soft(student_net: transformers.AutoModel, dataset_size: int, sequence_length: int,) -> tuple[torch.Tensor, torch.Tensor]:
    """ initialize the synthetic data """
    student_token_embeddings = student_net.get_input_embeddings().to(device)
    # tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    def simple_soft_one_hot(x, num_classes, temperature=1.0):
        one_hot = F.one_hot(x, num_classes).float()
        return F.softmax(one_hot / temperature, dim=-1)
    
    rand_token_idxs = torch.randint(0, student_token_embeddings.num_embeddings, (dataset_size, sequence_length-1), device=device)
    rand_token_one_hots = simple_soft_one_hot(rand_token_idxs, student_token_embeddings.num_embeddings, temperature=0.1)
    token_embeddings_syn = rand_token_one_hots @ student_token_embeddings.weight
    
    hidden_size = student_token_embeddings.weight.shape[1]
    assert token_embeddings_syn.shape == (dataset_size, sequence_length - 1, hidden_size), f"invalid shape: {token_embeddings_syn.shape}, need {(dataset_size, sequence_length - 1, hidden_size)}"

    num_classes = 4
    CLASS_MAP = torch.tensor([352,  362,  657,  513], device=device) # for AG_News... tmp
    token_labels_syn = torch.randint(low=0, high=num_classes, size=[dataset_size], device=device)
    token_labels_syn = CLASS_MAP[token_labels_syn]

    return (token_embeddings_syn, token_labels_syn)


def get_model(model_path: str) -> dict[str, torch.Tensor]:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation="eager"
    )
    return limit_layers(model, 6)


@find_executable_batch_size
def _autolabel_dataset_uncached(
        dataset: datasets.Dataset,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        sequence_length: int,
        batch_size: int = 32,
    ) -> torch.Tensor:
    """Get model predictions for the sequence_length-th token for each example in the dataset.
    
    Args:
        dataset: Dataset to label
        model: Model to use for labeling
        tokenizer: Tokenizer for processing text
        sequence_length: Maximum sequence length for tokenization
        batch_size: Batch size for processing
        
    Returns:
        Tensor of shape (len(dataset),) containing the predicted token ids
    """
    model.eval()
    model.to(device)
    all_labels = []
    
    for batch_start in tqdm.trange(0, len(dataset), batch_size, desc="Autolabeling dataset", colour="RED"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset.select(range(batch_start, batch_end))
        
        # Tokenize batch
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
        )
        inputs = { k: v.to(device) for k, v in inputs.items() }
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, sequence_length-1, :] # Get logits for sequence_length-th token
            predictions = torch.argmax(logits, dim=-1)
            all_labels.append(predictions.cpu())
    
    return torch.cat(all_labels)


def autolabel_dataset(
        dataset: datasets.Dataset,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        sequence_length: int,
    ) -> torch.Tensor:
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    hash_kwargs = {
        "dataset_hash": dataset._fingerprint,
        "model_hash": hash_model_params(model),
        "tokenizer_hash": tokenizer.name_or_path,
        "sequence_length": sequence_length,
    }
    cache_key = _get_cache_key(**hash_kwargs)
    print(f"Autolabeling dataset with cache key: {cache_key}")
    cache_path = os.path.join(cache_dir, f"autolabel_dataset_{cache_key}.npz")

    if not os.path.exists(cache_path):
        labels = _autolabel_dataset_uncached(dataset, model, tokenizer, sequence_length)
        np.savez(cache_path, labels=labels.numpy())
    
    labels = np.load(cache_path)["labels"]
    return torch.from_numpy(labels)



def _get_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    
    # hack to get around gpt weight-tying:
    if torch.isclose(state_dict["transformer.wte.weight"], state_dict["lm_head.weight"]).all():
        del state_dict["lm_head.weight"]
    
    return { k: v.detach().clone() for k,v in state_dict.items() }


def get_model_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    return _get_state_dict(get_model(model_path))


def _get_cache_key(**kwargs) -> str:
    """Generate a cache key from arbitrary keyword arguments.
    
    The key is generated by sorting the kwargs by key and concatenating their string representations.
    Only serializable values are included in the key generation.
    """
    def is_serializable(value) -> bool:
        """Check if a value should be included in cache key generation."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return True
        if isinstance(value, (list, tuple)):
            return all(is_serializable(x) for x in value)
        if isinstance(value, dict):
            return all(isinstance(k, str) and is_serializable(v) for k, v in value.items())
        return False

    # Filter out non-serializable values and those that shouldn't affect caching
    cache_kwargs = {
        k: v for k, v in kwargs.items() 
        if is_serializable(v) and not isinstance(v, (torch.Tensor, datasets.Dataset, datasets.DatasetDict))
    }
    
    # Sort kwargs by key for consistent ordering
    sorted_items = sorted(cache_kwargs.items())
    
    # Convert each value to string, handling special cases
    def value_to_str(v):
        if isinstance(v, (list, tuple)):
            return '_'.join(str(x) for x in v)
        if isinstance(v, dict):
            return '_'.join(f"{k}:{value_to_str(v)}" for k, v in sorted(v.items()))
        return str(v)
    
    return '_'.join(f"{k}:{value_to_str(v)}" for k, v in sorted_items)


def _eval_expert_model(
    student_net: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    eval_ds: datasets.Dataset,
    text_column_name: str,
    label_column_name: Optional[str],
    num_expert_datapoints: int,
    sequence_length: int,
    num_evaluation_batches: int
) -> tuple[float, Optional[float], Optional[float]]:
    """Evaluate expert model on evaluation dataset.
    
    Args:
        student_net: Model to evaluate
        tokenizer: Tokenizer to use
        eval_ds: Evaluation dataset
        text_column_name: Name of text column in dataset
        label_column_name: Name of label column in dataset (if doing classification)
        num_expert_datapoints: Number of datapoints to evaluate on per batch
        sequence_length: Maximum sequence length for tokenization
        num_evaluation_batches: Number of evaluation batches to run
        
    Returns:
        Tuple of (eval_loss, eval_accuracy)
    """
    eval_metrics = []
    eval_accuracies = []
    
    for _ in range(num_evaluation_batches):
        batch_idxs = random.sample(range(len(eval_ds)), k=num_expert_datapoints)
        examples = eval_ds.select(batch_idxs)
        
        if label_column_name is not None:
            # Classification mode
            examples = [
                f"{text} {label}" 
                for text, label in zip(examples[text_column_name], examples[label_column_name])
            ]
        else:
            # Language modeling mode
            examples = examples[text_column_name]
        
        tokens = tokenizer(
            examples,
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=sequence_length, 
        ).to(device)
        
        if label_column_name is not None:
            # For classification, only predict the last token
            labels = tokens.input_ids[:, 1:].detach().clone() 
            labels[:, :-1] = -100
        else:
            # For language modeling, predict all next tokens
            labels = tokens.input_ids[:, 1:].detach().clone()
        
        outputs = student_net(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
        )
        logits = outputs.logits[:, :-1]
        
        eval_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            labels.reshape(-1),
            ignore_index=-100,
            reduction="mean"
        )
        eval_metrics.append(eval_loss.item())

        token_mask = (labels != -100)
        eval_accuracy = (logits.argmax(dim=-1)[token_mask] == labels[token_mask]).float().mean()
        eval_accuracies.append(eval_accuracy.item())

    eval_loss = sum(eval_metrics) / len(eval_metrics)
    eval_accuracy = sum(eval_accuracies) / len(eval_accuracies) if eval_accuracies else None
    
    return eval_loss, eval_accuracy


def _train_expert_model_uncached(
        num_experts: int, 
        num_steps_per_expert: int, 
        num_expert_datapoints: int, 
        expert_lr: float, 
        sequence_length: int,
        ds: datasets.DatasetDict, 
        text_column_name: str, 
        label_column_name: Optional[str] = None,
        ds_tokens: Optional[torch.Tensor] = None,
        ds_labels: Optional[torch.Tensor] = None,
        num_evaluation_batches: int = 16,
        early_stopping_patience: int = 3,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, dict[str, torch.Tensor]]:
    student_net = get_model("gpt2").to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left" # important for correct truncation
    tokenizer.padding_side = "left" 

    train_ds = ds["train"]
    eval_ds = ds["test"]

    # optim = torch.optim.Adam(student_net.parameters(), lr=expert_lr)
    optim = torch.optim.SGD(student_net.parameters(), lr=expert_lr)

    expert_state_dicts = [_get_state_dict(student_net)]
    step = 0
    pbar = tqdm_if_main_worker(total=num_experts * num_steps_per_expert, colour="CYAN")
    all_token_counts = torch.zeros([tokenizer.vocab_size], device=device)

    # training loop
    evaluation_metrics = collections.defaultdict(list)
    eval_loss = -1 
    eval_accuracy = -1
    
    # Early stopping variables
    best_eval_accuracy = float('-inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_experts):
        for _i in range(num_steps_per_expert):
            if ds_tokens is not None:
                # Handle pre-tokenized data case
                if len(ds_tokens) < num_expert_datapoints:
                    batch_idxs = torch.randint(0, len(ds_tokens), (num_expert_datapoints,)).tolist()
                else:
                    batch_idxs = random.sample(range(len(ds_tokens)), k=num_expert_datapoints)
                tokens = ds_tokens[batch_idxs].to(device)
                outputs = student_net(
                    input_ids=tokens,
                    attention_mask=(tokens != tokenizer.pad_token_id),
                )
                labels = ds_labels[batch_idxs] if ds_labels is not None else tokens[:, 1:]
                no_labels = torch.zeros_like(tokens[:, :-2], device=device, dtype=torch.long) - 100
                labels = torch.cat([no_labels, labels[:, None].to(device)], dim=1)
                all_token_counts += torch.bincount(tokens.flatten(), minlength=tokenizer.vocab_size)
            else:
                # Handle raw text data case
                batch_idxs = random.sample(range(len(train_ds)), k=num_expert_datapoints)
                examples = train_ds.select(batch_idxs)
                
                if label_column_name is not None:
                    # Classification mode - append label to text
                    examples = [
                        f"{text} {label}" 
                        for text, label in zip(examples[text_column_name], examples[label_column_name])
                    ]
                else:
                    # Language modeling mode - use text directly
                    examples = examples[text_column_name]
                
                tokens = tokenizer(
                    examples,
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=sequence_length, 
                ).to(device)
                
                labels = tokens.input_ids[:, 1:].detach().clone() 
                if label_column_name is not None:
                    # For classification, only predict the last token
                    labels[:, :-1] = -100

                outputs = student_net(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                )
                # Track token counts
                all_token_counts += torch.bincount(tokens.input_ids.flatten(), minlength=tokenizer.vocab_size)
            

            logits = outputs.logits[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                labels.reshape(-1),
                ignore_index=-100,
                reduction="mean"
            )
            pbar.set_description(f"Epoch {epoch} | Step {step+1} | Loss = {loss:.3f}")
            pbar.update(1)
            loss.backward()
            optim.step()
            optim.zero_grad()
            student_net.zero_grad()
            step += 1
            
        # evaluation loop every epoch
        eval_loss, eval_accuracy = _eval_expert_model(
            student_net=student_net,
            tokenizer=tokenizer,
            eval_ds=eval_ds,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
            num_expert_datapoints=num_expert_datapoints,
            sequence_length=sequence_length,
            num_evaluation_batches=num_evaluation_batches
        )
        
        evaluation_metrics[f"eval_step{step}_loss"].append(eval_loss)
        evaluation_metrics[f"eval_step{step}_accuracy"].append(eval_accuracy)

        # print(f"[Epoch {epoch} | Step {step}] | Eval loss: {eval_loss:.3f} | Eval accuracy: {eval_accuracy:.3f}")
        tqdm.tqdm.write(f"[Epoch {epoch} | Step {step}] | Eval loss: {eval_loss:.3f} | Eval accuracy: {eval_accuracy:.3f}")
        
        expert_state_dicts.append(_get_state_dict(student_net))
        
        # Check for early stopping
        if eval_accuracy is not None:
            if eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
                if epochs_without_improvement >= early_stopping_patience:
                    pbar.close()
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

    # Run one more evaluation
    eval_loss, eval_accuracy = _eval_expert_model(
        student_net=student_net,
        tokenizer=tokenizer,
        eval_ds=eval_ds,
        text_column_name=text_column_name,
        label_column_name=label_column_name,
        num_expert_datapoints=num_expert_datapoints,
        sequence_length=sequence_length,
        num_evaluation_batches=num_evaluation_batches
    )
    evaluation_metrics[f"eval_step{step}_loss"].append(eval_loss)
    evaluation_metrics[f"eval_step{step}_accuracy"].append(eval_accuracy)

    # filter None values
    for k in evaluation_metrics.keys(): evaluation_metrics[k] = [v for v in evaluation_metrics[k] if v is not None]
    evaluation_metrics = { k: torch.tensor(v).mean().item() for k, v in evaluation_metrics.items() }
    pbar.close()
    best_eval_loss = min({ v for k,v in evaluation_metrics.items() if "loss" in k } | { float("inf") })
    final_evaluation_metrics = { "best_eval_loss": best_eval_loss  }
    if label_column_name is not None:
        best_eval_accuracy = max({ v for k,v in evaluation_metrics.items() if "accuracy" in k } | { float("0") })
        print0(f"Best eval loss: {best_eval_loss:.3f} | Best eval accuracy: {best_eval_accuracy:.3f}")
        final_evaluation_metrics["best_eval_accuracy"] = best_eval_accuracy

    if all_token_counts.sum() == 0:
        print0("WARNING: no tokens were counted")
    return expert_state_dicts, all_token_counts, final_evaluation_metrics


def train_expert_model(
        num_experts: int, 
        num_steps_per_expert: int, 
        num_expert_datapoints: int, 
        expert_lr: float, 
        sequence_length: int,
        ds: datasets.DatasetDict, 
        text_column_name: str, 
        label_column_name: str,
        ds_tokens: Optional[torch.Tensor] = None,
        ds_labels: Optional[torch.Tensor] = None,
        num_evaluation_batches: int = 10,
        **kwargs
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, dict[str, torch.Tensor]]:
    """Train expert models with caching based on input parameters."""
    
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on all parameters
    cache_kwargs = {
        "num_experts": num_experts,
        "num_steps_per_expert": num_steps_per_expert,
        "num_expert_datapoints": num_expert_datapoints,
        "expert_lr": expert_lr,
        "sequence_length": sequence_length,
        "ds_name": ds.config_name if hasattr(ds, "config_name") else "custom",
        "text_column_name": text_column_name,
        "label_column_name": label_column_name,
        "num_evaluation_batches": num_evaluation_batches,
        "ds_fingerprint": "_".join(str(ds[k]._fingerprint) for k in sorted(ds.keys())),
        **kwargs  # Include any additional kwargs in cache key generation
    }
    cache_key = _get_cache_key(**cache_kwargs)
    cache_path = os.path.join(cache_dir, f"expert_model_{cache_key}.pkl")

    uncachable_args_provided = (ds_tokens is not None) or (ds_labels is not None)
    
    # Check if cached result exists
    if os.path.exists(cache_path) and not uncachable_args_provided:
        print0(f"Loading cached expert model results from {cache_path}")
        with open(cache_path, "rb") as f:
            expert_state_dicts, all_token_counts, final_evaluation_metrics = pickle.load(f)
        print0(f"Loaded cached expert model results from {cache_path} - "
               f"best eval loss: {final_evaluation_metrics['best_eval_loss']:.3f} /" 
               f"best eval accuracy: {final_evaluation_metrics['best_eval_accuracy']:.3f}")
        return expert_state_dicts, all_token_counts, final_evaluation_metrics
    
    # If not cached, run training and cache results
    num_datapoints = len(ds_tokens) if ds_tokens is not None else len(ds["train"])
    if not uncachable_args_provided:
        print0(f"Training expert model with {num_datapoints} datapoints and caching results to {cache_path}")
    else:
        print0(f"Training expert model with {num_datapoints} datapoints")
    
    results = _train_expert_model_uncached(
        num_experts=num_experts,
        num_steps_per_expert=num_steps_per_expert,
        num_expert_datapoints=num_expert_datapoints,
        expert_lr=expert_lr,
        sequence_length=sequence_length,
        ds=ds,
        text_column_name=text_column_name,
        label_column_name=label_column_name,
        ds_tokens=ds_tokens,
        ds_labels=ds_labels,
        num_evaluation_batches=num_evaluation_batches,
        **kwargs
    )
    
    # Cache results
    if not uncachable_args_provided:
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
    
    return results


@torch.no_grad
def project_x_to_embedding_space(
        X: torch.Tensor, 
        initial_student_net: transformers.PreTrainedModel,
        minibatch_size: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
    if minibatch_size is None:
        minibatch_size = X.shape[0]
    
    embeddings = initial_student_net.get_input_embeddings()
    embeddings = embeddings.to(device)
    Z = []
    Z_tokens = []

    for i in trange_if_main_worker(0, X.shape[0], minibatch_size, leave=False):
        X_batch = X[i:i+minibatch_size]
        Z_distances = torch.cdist(X_batch, embeddings.weight.to(device))
        Z_tokens_batch = Z_distances.argmin(dim=2)
        Z_batch = embeddings(Z_tokens_batch)
        Z.append(Z_batch)
        Z_tokens.append(Z_tokens_batch)
    
    Z = torch.cat(Z, dim=0)
    Z_tokens = torch.cat(Z_tokens, dim=0)
    return Z, Z_tokens


def load_dataset_from_name(dataset_name: str) -> tuple[datasets.Dataset, str, str]:
    if dataset_name == "ag_news":
        ds = datasets.load_dataset("fancyzhx/ag_news")
        text_column_name = "text"        
        label_column_name = "label"
    elif dataset_name.startswith("ag_news_") and dataset_name[8:].isdigit():
        num_samples = int(dataset_name[8:])
        ds = datasets.load_dataset("fancyzhx/ag_news")
        ds = ds.train_test_split(test_size=0.1, seed=42)
        ds["train"] = ds["train"].select(range(num_samples))
        text_column_name = "text"
        label_column_name = "label"
    elif dataset_name == "nq":
        ds = datasets.load_dataset("jxm/nq_corpus_dpr")["train"]
        ds = ds.train_test_split(test_size=0.1, seed=42)
        text_column_name = "text"
        label_column_name = None
    elif dataset_name.startswith("nq_") and dataset_name[3:].isdigit():
        # Handle nq_BBB where BBB is any integer
        num_samples = int(dataset_name[3:])
        ds = datasets.load_dataset("jxm/nq_corpus_dpr")["train"]
        ds = ds.train_test_split(test_size=0.1, seed=42)
        ds["train"] = ds["train"].select(range(num_samples))
        text_column_name = "text"
        label_column_name = None
    elif dataset_name == "msmarco":
        ds = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
        text_column_name = "text"
        label_column_name = None
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return ds, text_column_name, label_column_name

def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1


def get_rank() -> int:
    try:
        return int(torch.distributed.get_rank())
    except (RuntimeError, ValueError):
        return 0


def gather(t: torch.Tensor) -> torch.Tensor:
    # torch.distributed.nn.all_gather scales by world size since the reduce op is SUM
    # https://github.com/pytorch/pytorch/issues/58005
    # only should use torch.distributed.nn.all_gather if we implement a `local_loss`
    # like: https://github.com/mlfoundations/open_clip/issues/616
    world_size = get_world_size()
    if world_size == 1:
        return t

    if t.ndim == 0:
        t = t.unsqueeze(0)

    gathered = [torch.empty_like(t) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, t)
    gathered[get_rank()] = t

    return torch.cat(gathered, dim=0)


def print0(*args, **kwargs) -> None:
    if get_rank() == 0:
        print(*args, **kwargs)


def tqdm_if_main_worker(*args, **kwargs) -> Iterable:
    if get_rank() == 0:
        return tqdm.tqdm(*args, **kwargs)
    else:
        return tqdm.tqdm(*args, **kwargs, disable=True)


def trange_if_main_worker(*args, **kwargs) -> tqdm.tqdm:
    if get_rank() == 0:
        return tqdm.trange(*args, **kwargs)
    else:
        return tqdm.trange(*args, **kwargs, disable=True)
