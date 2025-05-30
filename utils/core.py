from typing import Iterable, Optional, Union
import collections
import gc
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


class ExpertModel:
    """This is a wrapper class for consistent train/eval loss and accuracy calculation.
    
    It also handles tokenization and label remapping.
    """
    student_net: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    all_labels: Optional[set[str]]
    max_sequence_length: int
    
    def __init__(self, base_model_name_or_path: str, max_sequence_length: int, all_labels: Optional[set[str]] = None):
        self.student_net = get_model(base_model_name_or_path).to(device)
        self.student_net.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left" # important for correct truncation
        self.tokenizer.padding_side = "left" 
        if all_labels is not None:
            self.all_labels = list(sorted(all_labels))
            self.all_labels_ids = [self.tokenizer.encode(f" {x}")[-1] for x in self.all_labels]
        else:
            self.all_labels = None
            self.all_labels_ids = None
        self.max_sequence_length = max_sequence_length
        self.verbose = False
        print(f"[ExpertModel] | {base_model_name_or_path} | all_labels: {self.all_labels} | all_labels_ids: {self.all_labels_ids}")
    
    @property
    def is_doing_classification(self) -> bool:
        return (self.all_labels is not None) and (len(self.all_labels) > 0)
    
    def prepare_examples(
            self, examples: list[str], label_column_name: Optional[str], text_column_name: str) -> list[str]:
        if label_column_name is not None:
            # Classification mode
            examples = [
                f"{text} {label}" 
                for text, label in zip(examples[text_column_name], examples[label_column_name])
            ]
        else:
            # Language modeling mode
            examples = examples[text_column_name]
        return examples

    def compute_loss_and_accuracy(
            self, 
            logits: torch.Tensor, 
            labels: torch.Tensor,
            vmap: bool = False,
        ) -> tuple[torch.Tensor, float, float]:
        if self.is_doing_classification:
            # For classification, only predict the last token
            if labels.ndim == 1:
                # unroll singleton for vmap
                labels = labels[None]

            labels = labels[:, -1]
            logits = logits[:, -1, :]

            # Mask logits by vocab
            token_is_masked = (
                ~(
                    torch.arange(logits.shape[1])[None, :] 
                    == 
                    torch.tensor(self.all_labels_ids)[:, None]
                ).any(dim=0)
            )
            logits[:, token_is_masked] = -float("inf")

            # Get predictions
            accuracy = (logits.argmax(-1) == labels).float().mean()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                labels.reshape(-1),
                reduction="mean"
            )
        else:
            if labels.ndim == 1:
                # unroll singleton for vmap
                labels = labels[None]

            labels = labels[:, -1]
            logits = logits[:, -1, :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                labels.reshape(-1),
                reduction="mean",
                ignore_index=-100,
            )
            accuracy = (logits.argmax(-1).float() == labels.float()).float().mean()
    

        # if not vmap:
        #     if torch.isinf(loss):
        #         print(f"[WARNING] | loss is inf")
        #         breakpoint()
    
        return logits, loss, accuracy

    def compute_outputs(
            self, 
            examples: list[str],
            output_hidden_states: bool = False,
        ) -> tuple[transformers.BatchEncoding, torch.Tensor]:

        tokenized_text = self.tokenizer(
            examples,
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_sequence_length, 
        ).to(device)
        input_ids = tokenized_text.input_ids
        attention_mask = tokenized_text.attention_mask

        # print a single input
        text = self.tokenizer.decode(input_ids[0])
        if self.verbose: print(f"[compute_outputs] | input_ids.shape: {input_ids.shape} | text: {text}")
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.student_net(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
            )
        
        return tokenized_text, outputs
    
    def get_logits_and_labels(
            self, 
            tokenized_text: transformers.BatchEncoding,
            outputs: dict[str, torch.Tensor], # transformers.CausalLMOutputWithPast,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = tokenized_text.input_ids[:, 1:].detach().clone() 
        logits = outputs.logits[:, :-1, :]
        return logits, labels
    
    def get_loss_and_accuracy(
            self, 
            examples: list[str], 
            label_column_name: Optional[str] = None,
        ) -> tuple[float, float]:

        tokenized_text, outputs = self.compute_outputs(examples)
        logits, labels = self.get_logits_and_labels(tokenized_text, outputs)

        return self.compute_loss_and_accuracy(logits, labels)
    

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


def get_token_embeddings_from_classification_dataset(dataset_size: int, sequence_length: int, classification_dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """ initialize the synthetic data from a ClassificationDataset """
    ds = classification_dataset.dataset
    text_column_name = classification_dataset.text_column_name
    
    student_net = get_model("gpt2")
    student_net.train()
    student_token_embeddings = student_net.get_input_embeddings().to(device)
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
    return model


@find_executable_batch_size
def _autolabel_dataset_uncached(
        expert: ExpertModel,
        dataset: datasets.Dataset,
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
    expert.student_net.eval()
    expert.student_net.to(device)

    all_labels = []
    true_labels = []

    label_map = dict(zip(
        expert.all_labels,
        expert.all_labels_ids,
    ))
    label_map_reverse = {v: k for k, v in label_map.items()}

    for batch_start in tqdm.trange(0, len(dataset), batch_size, desc="Autolabeling dataset", colour="RED"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset.select(range(batch_start, batch_end))
        examples = datasets.Dataset.from_dict(
            {
                "text": batch["text"],
                "label": ["A"] * len(batch),
            }
        )
        examples = expert.prepare_examples(
            examples=examples,
            label_column_name="label",
            text_column_name="text",
        )
        
        # Get model predictions
        with torch.no_grad():
            masked_logits, _, _ = expert.get_loss_and_accuracy(
                examples=examples,
            )
            pred_token_ids = masked_logits.argmax(-1).cpu()

        all_labels.append(torch.tensor(pred_token_ids))
        if "label" in batch: true_labels.append(torch.tensor(label_map[L] for L in batch["label"]))
    
    expert.student_net.cpu()
    expert.student_net.train()

    all_labels = torch.cat(all_labels)
    label_counts = all_labels.unique(return_counts=True)

    if len(true_labels) > 0:
        true_labels = torch.cat(true_labels)
        true_label_counts = true_labels.unique(return_counts=True)

        agreement = (all_labels == true_labels).float().mean()
        print(f"[autolabel_dataset] expert model | agreement: {agreement:.2f} | autolabeled counts: {label_counts} | true label counts: {true_label_counts}")
    else:
        print(f"[autolabel_dataset] expert model | autolabeled counts: {label_counts}")

    return [label_map_reverse[L] for L in all_labels.tolist()]


def autolabel_dataset(
        dataset: datasets.Dataset,
        expert: ExpertModel,
    ) -> torch.Tensor:
    cache_dir = os.path.join(os.path.dirname(__file__), os.pardir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    # TODO: Restore caching somehow.
    # hash_kwargs = {
    #     "dataset_hash": dataset._fingerprint,
    #     # "random_seed": random.randint(0, 1000000),
    #     # "model_hash": hash_model_params(expert.student_net),
    #     "tokenizer_hash": expert.tokenizer.name_or_path,
    #     "sequence_length": sequence_length,
    # }
    # cache_key = _get_cache_key(**hash_kwargs)
    # cache_key = hashlib.sha256(cache_key.encode()).hexdigest()
    # print(f"Autolabeling dataset with cache key: {cache_key}")
    # cache_path = os.path.join(cache_dir, f"autolabel_dataset_{cache_key}.npz")

    # if not os.path.exists(cache_path):
    labels = _autolabel_dataset_uncached(
        dataset=dataset, 
        expert=expert,
    )
    return labels


def _get_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    
    # hack to get around gpt weight-tying:
    try:
        # this literally only matters for gpt2
        if torch.isclose(state_dict["transformer.wte.weight"], state_dict["lm_head.weight"]).all():
            del state_dict["lm_head.weight"]
    except KeyError:
        pass
    
    return { k: v.detach().clone().cpu() for k,v in state_dict.items() }


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


@find_executable_batch_size
@torch.no_grad()
def _eval_expert_model(
    expert: ExpertModel,
    eval_ds: datasets.Dataset,
    text_column_name: str,
    label_column_name: Optional[str],
    batch_size: int = 32,
) -> tuple[float, Optional[float], Optional[float]]:
    """Evaluate expert model on evaluation dataset.
    
    Args:
        expert (ExpertModel): Expert model to evaluate
        eval_ds: Evaluation dataset
        text_column_name: Name of text column in dataset
        label_column_name: Name of label column in dataset (if doing classification)
        batch_size: Number of datapoints to evaluate on per batch
        
    Returns:
        Tuple of (eval_loss, eval_accuracy)
    """
    expert.student_net.eval()

    eval_metrics = []
    eval_accuracies = []

    all_idxs = list(range(len(eval_ds)))
    i = 0
    while i < len(all_idxs):
        batch_idxs = all_idxs[i:i+batch_size]
        i += batch_size
        
        examples = eval_ds.select(batch_idxs)
        examples = expert.prepare_examples(
            examples=examples,
            label_column_name=label_column_name,
            text_column_name=text_column_name,
        )
        _, loss, accuracy = expert.get_loss_and_accuracy(
            examples=examples,
            label_column_name=label_column_name,
        )
        eval_metrics.append(loss.item())
        eval_accuracies.append(accuracy.item())

    eval_loss = sum(eval_metrics) / len(eval_metrics)
    eval_accuracy = sum(eval_accuracies) / len(eval_accuracies) if eval_accuracies else None

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    expert.student_net.train()
    return eval_loss, eval_accuracy


def _train_expert_model_uncached(
        base_model_name_or_path: str,
        num_experts: int, 
        num_steps_per_expert: int, 
        expert_batch_size: int, 
        expert_lr: float, 
        sequence_length: int,
        ds: datasets.DatasetDict, 
        text_column_name: str, 
        label_column_name: Optional[str] = None,
        early_stopping_patience: int = 10,
        num_eval_datapoints: int = 2048,
    ) -> tuple[ExpertModel, list[dict[str, torch.Tensor]], torch.Tensor, dict[str, torch.Tensor]]:

    train_ds = ds["train"].shuffle(seed=42)
    eval_ds = ds["test"].shuffle(seed=42)
    eval_ds = eval_ds.select(range(min(num_eval_datapoints, len(eval_ds))))

    if label_column_name is not None:
        assert label_column_name in train_ds.column_names, f"label_column_name: {label_column_name} not in train_ds.column_names: {train_ds.column_names}"
        all_labels = list(sorted(set(train_ds[label_column_name]) | set(eval_ds[label_column_name])))
    else:
        all_labels = None
    expert = ExpertModel(base_model_name_or_path, max_sequence_length=sequence_length, all_labels=all_labels)
    optim = torch.optim.Adam(expert.student_net.parameters(), lr=expert_lr)
    # optim = torch.optim.SGD(student_net.parameters(), lr=expert_lr)

    expert_state_dicts = [_get_state_dict(expert.student_net)]
    step = 0
    pbar = tqdm_if_main_worker(total=num_experts * num_steps_per_expert, colour="CYAN")

    # training loop
    evaluation_metrics = collections.defaultdict(list)
    eval_loss = -1 
    eval_accuracy = -1

    # Early stopping variables
    best_eval_accuracy = float('-inf')
    epochs_without_improvement = 0

    # Print first datapoint
    print(f"[train_expert_model] First datapoint: {train_ds[0][text_column_name]}")
    if label_column_name is not None: print(f"[train_expert_model] First datapoint label: {train_ds[0][label_column_name]}")
    
    for epoch in range(num_experts):
        for _i in range(num_steps_per_expert):
            batch_idxs = random.sample(range(len(train_ds)), k=min(expert_batch_size, len(train_ds)))
            examples = train_ds.select(batch_idxs)
            examples = expert.prepare_examples(
                examples=examples,
                label_column_name=label_column_name,
                text_column_name=text_column_name,
            )
            _, loss, accuracy = expert.get_loss_and_accuracy(
                examples=examples,
                label_column_name=label_column_name,
            )
            pbar.set_description(f"Epoch {epoch} | Step {step+1} | Loss = {loss:.3f} | Accuracy = {accuracy:.3f}")
            pbar.update(1)
            loss.backward()
            optim.step()
            optim.zero_grad()
            expert.student_net.zero_grad()
            step += 1
            
        # evaluation loop every epoch
        eval_loss, eval_accuracy = _eval_expert_model(
            expert=expert,
            eval_ds=eval_ds,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
        )

        if (eval_accuracy == 0.0) or (eval_loss == float("inf")):
            print("Warning: 0.0 acc or inf loss!")
        
        evaluation_metrics[f"eval_step{step}_loss"].append(eval_loss)
        evaluation_metrics[f"eval_step{step}_accuracy"].append(eval_accuracy)

        tqdm.tqdm.write(f"[Epoch {epoch} | Step {step}] | Eval loss: {eval_loss:.3f} | Eval accuracy: {eval_accuracy:.3f}")
        expert_state_dicts.append(_get_state_dict(expert.student_net))
        
        # Check for early stopping
        if eval_accuracy is not None:
            if eval_accuracy > best_eval_accuracy:
                best_eval_accuracy = eval_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    pbar.close()
                    print(f"Early stopping triggered after {epoch+1} epochs (patience = {early_stopping_patience}, best eval accuracy = {best_eval_accuracy:.3f})")
                    break

    # Run one more evaluation
    eval_loss, eval_accuracy = _eval_expert_model(
        expert=expert,
        eval_ds=eval_ds,
        text_column_name=text_column_name,
        label_column_name=label_column_name,
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

    expert.student_net.cpu()
    return expert, expert_state_dicts, final_evaluation_metrics


def train_expert_model(
        base_model_name_or_path: str,
        num_experts: int, 
        num_steps_per_expert: int, 
        expert_batch_size: int, 
        expert_lr: float, 
        sequence_length: int,
        ds: datasets.DatasetDict, 
        text_column_name: str, 
        label_column_name: str,
        **kwargs
    ) -> tuple[ExpertModel, list[dict[str, torch.Tensor]], torch.Tensor, dict[str, torch.Tensor]]:
    """Train expert models with caching based on input parameters."""
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(os.path.dirname(__file__), os.pardir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on all parameters
    cache_kwargs = {
        "base_model_name_or_path": base_model_name_or_path,
        "num_experts": num_experts,
        "num_steps_per_expert": num_steps_per_expert,
        "expert_batch_size": expert_batch_size,
        "expert_lr": expert_lr,
        "sequence_length": sequence_length,
        "ds_name": ds.config_name if hasattr(ds, "config_name") else "custom",
        "text_column_name": text_column_name,
        "label_column_name": label_column_name,
        "ds_fingerprint": "_".join(str(ds[k]._fingerprint) for k in sorted(ds.keys())),
        **kwargs  # Include any additional kwargs in cache key generation
    }
    cache_key = _get_cache_key(**cache_kwargs)
    cache_key = hashlib.sha256(cache_key.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"expert_model_{cache_key}.pkl")

    # Check if cached result exists
    if os.path.exists(cache_path):
        print0(f"Loading cached expert model results from {cache_path}")

        #########################################################
        # Tmp: Save train set.
        # ds["train"].to_parquet("train_ds.parquet")
        # print(f"Saved train set to train_ds.parquet")
        #########################################################

        try:
            with open(cache_path, "rb") as f:
                expert, expert_state_dicts, final_evaluation_metrics = pickle.load(f)
            print0(f"Loaded cached expert model results from {cache_path} - "
                f"best eval loss: {final_evaluation_metrics['best_eval_loss']:.3f} / " 
                f"best eval accuracy: {final_evaluation_metrics['best_eval_accuracy']:.3f}")
            return expert, expert_state_dicts, final_evaluation_metrics
        except Exception as e:
            print0(f"Error loading cached expert model results from {cache_path}: {e}")
            pass
    
    # If not cached, run training and cache results
    num_datapoints = len(ds["train"])
    print0(f"Training expert model with {num_datapoints} datapoints / batch size {expert_batch_size} and caching results to {cache_path}")
    
    results = _train_expert_model_uncached(
        base_model_name_or_path=base_model_name_or_path,
        num_experts=num_experts,
        num_steps_per_expert=num_steps_per_expert,
        expert_batch_size=expert_batch_size,
        expert_lr=expert_lr,
        sequence_length=sequence_length,
        ds=ds,
        text_column_name=text_column_name,
        label_column_name=label_column_name,
        **kwargs
    )
    
    # Cache results
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
