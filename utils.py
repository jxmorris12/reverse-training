from typing import Optional
import collections
import random
import time

import datasets
import torch
import torch.nn.functional as F
import transformers
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    breakpoint() # TODO: Autogen CLASS_MAP!

    return (token_embeddings_syn, token_labels_syn)


def get_token_embeddings_random(dataset_size: int, sequence_length: int,) -> tuple[torch.Tensor, torch.Tensor]:
    """ initialize the synthetic data """
    student_net = get_model("gpt2")
    student_net.train()
    student_token_embeddings = student_net.get_input_embeddings().to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

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


def _get_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    
    # hack to get around gpt weight-tying:
    if torch.isclose(state_dict["transformer.wte.weight"], state_dict["lm_head.weight"]).all():
        del state_dict["lm_head.weight"]
    
    return { k: v.detach().clone() for k,v in state_dict.items() }


def get_model_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    return _get_state_dict(get_model(model_path))


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
        num_evaluation_batches: int = 10
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, dict[str, torch.Tensor]]:
    student_net = get_model("gpt2").to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left" # important for correct truncation
    tokenizer.padding_side = "left" 

    evaluation_steps: list[int] = [10, 100,]

    train_ds = ds["train"]
    eval_ds = ds["test"]

    # optim = torch.optim.Adam(student_net.parameters(), lr=expert_lr)
    optim = torch.optim.SGD(student_net.parameters(), lr=expert_lr)

    expert_state_dicts = [_get_state_dict(student_net)]
    step = 0
    pbar = tqdm.tqdm(total=num_experts * num_steps_per_expert, colour="CYAN")
    all_token_counts = torch.zeros([tokenizer.vocab_size], device=device)

    # training loop
    evaluation_metrics = collections.defaultdict(list)
    eval_loss = -1 
    eval_accuracy = -1
    for _i in range(num_experts):
        for _j in range(num_steps_per_expert):
            if ds_tokens is not None:
                if len(ds_tokens) < num_expert_datapoints:
                    batch_idxs = torch.randint(0, len(ds_tokens), (num_expert_datapoints,)).tolist()
                else:
                    batch_idxs = random.sample(range(len(ds_tokens)), k=num_expert_datapoints)
                tokens = ds_tokens[batch_idxs]
                labels = ds_labels[batch_idxs]
                outputs = student_net(
                    input_ids=tokens,
                    attention_mask=(tokens != tokenizer.pad_token_id),
                )
            else:
                batch_idxs = random.sample(range(len(train_ds)), k=num_expert_datapoints)
                examples = train_ds.select(batch_idxs)
                examples = [
                    f"{text} {label}" 
                    for text, label in zip(examples[text_column_name], examples[label_column_name])
                ]
                tokens = tokenizer(
                    examples,
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=sequence_length, 
                ).to(device)
                labels = tokens.input_ids[:, 1:].detach().clone() 
                labels[:, :-1] = -100
                outputs = student_net(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                )
                # use bincount to track token counts
                all_token_counts += torch.bincount(tokens.input_ids.flatten(), minlength=tokenizer.vocab_size)
            logits = outputs.logits[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(labels.numel(), -1), 
                labels.reshape(labels.numel(),),
                ignore_index=-100,
                reduction="mean"
            )
            pbar.set_description(f"Expert training step {step+1} | Loss = {loss:.3f}")
            pbar.update(1)
            loss.backward()
            optim.step()
            optim.zero_grad()
            student_net.zero_grad()
            step += 1
            if (step + 1) in evaluation_steps:
                 # evaluation loop
                for _ in range(num_evaluation_batches):
                    batch_idxs = random.sample(range(len(eval_ds)), k=num_expert_datapoints)
                    examples = eval_ds.select(batch_idxs)
                    examples = [
                        f"{text} {label}" 
                        for text, label in zip(examples[text_column_name], examples[label_column_name])
                    ]
                    tokens = tokenizer(
                        examples,
                        return_tensors="pt", 
                        padding="max_length", 
                        truncation=True, 
                        max_length=sequence_length, 
                    ).to(device)
                    outputs = student_net(
                        input_ids=tokens.input_ids,
                        attention_mask=tokens.attention_mask,
                    )
                    logits = outputs.logits[:, :-1]
                    labels = tokens.input_ids[:, 1:].detach().clone() 
                    labels[:, :-1] = -100
                    eval_loss = torch.nn.functional.cross_entropy(
                        logits.reshape(labels.numel(), -1), 
                        labels.reshape(labels.numel(),),
                        ignore_index=-100,
                        reduction="mean"
                    )
                    evaluation_metrics[f"eval_step{step}_loss"].append(eval_loss.item())

                    token_mask = (labels != -100)
                    eval_accuracy = (logits.argmax(dim=-1)[token_mask] == labels[token_mask]).float().mean()
                    evaluation_metrics[f"eval_step{step}_accuracy"].append(eval_accuracy.item())
        expert_state_dicts.append(_get_state_dict(student_net))
    
    evaluation_metrics = { k: torch.tensor(v).mean().item() for k, v in evaluation_metrics.items() }
    pbar.close()
    best_eval_loss = min({ v for k,v in evaluation_metrics.items() if "loss" in k } | { float("inf") })
    best_eval_accuracy = max({ v for k,v in evaluation_metrics.items() if "accuracy" in k } | { float("0") })
    print(f"Best eval loss: {best_eval_loss:.3f} | Best eval accuracy: {best_eval_accuracy:.3f}")

    return expert_state_dicts, all_token_counts, evaluation_metrics


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

    for i in tqdm.trange(0, X.shape[0], minibatch_size, leave=False):
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
    elif dataset_name == "nq":
        ds = datasets.load_dataset("jxm/nq_corpus_dpr")
        text_column_name = "text"
        label_column_name = None
    elif dataset_name == "msmarco":
        ds = datasets.load_dataset("Tevatron/msmarco-passage-corpus")
        text_column_name = "text"
        label_column_name = None
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return ds, text_column_name, label_column_name