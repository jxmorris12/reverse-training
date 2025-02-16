import copy
import gc
import logging

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed
import wandb

from utils import (
    INIT_CHARS, 
    find_executable_batch_size, 
    get_nonascii_toks, 
    get_model_params, 
    get_model_grad
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GCGConfig:
    num_steps: int = 2500
    optim_str_init: Union[str, List[str]] = ("x " * 16).strip()
    str_batch_size: int = 16
    search_width: int = 16
    batch_size: int = None
    topk: int = 128
    n_replace: int = 8
    buffer_size: int = 0
    allow_non_ascii: bool = False
    filter_ids: bool = False # TODO: make compatible with `str_batch_size` > 1
    seed: int = None
    verbosity: str = "INFO"
    use_wandb: bool = True

@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]

class AttackBuffer:
    def __init__(self, config: GCGConfig):
        self.buffer = [] # elements are (loss: float, optim_ids: Tensor)
        self.size = config.buffer_size
        self.str_batch_size = config.str_batch_size
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            wandb.init(
                project="jxm-rev",
                entity="jack-morris",
                reinit=True,
            )
            wandb.config.update(config)
        else:
            wandb.init(mode="disabled")

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
            return

        self.buffer[-1] = (loss, optim_ids)
        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]
    
    def log_buffer(self, tokenizer):
        message = "buffer:"
        min_loss = float("inf")
        min_loss_str = "" 
        for loss, ids in self.buffer:
            ids_unflattened = ids.reshape(self.str_batch_size, -1)
            optim_str = " | ".join(
                tokenizer.batch_decode(ids_unflattened)
            )
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
            if loss < min_loss:
                min_loss_str = optim_str
                min_loss = loss
        wandb.log({"buffer": message, "min_loss": min_loss, "min_loss_str": min_loss_str})
        logger.info(message)

def sample_ids_random(
    ids: Tensor, 
    search_width: int, 
    vocab_size: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized 
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace: int
            the number of token positions to update per sequence
        not_allowed_ids: Tensor, shape = (n_ids)
            the token ids that should not be used in optimization
    
    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    grad = torch.randn((n_optim_tokens, vocab_size), device=ids.device)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)
    return new_ids

def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids) 
            token ids 
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer
    
    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
           filtered_ids.append(ids[i]) 
    
    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )
    
    return torch.stack(filtered_ids)


class GCG:
    def __init__(
        self, 
        initial_model: transformers.PreTrainedModel,
        final_model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = initial_model
        self.final_model = final_model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = initial_model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=initial_model.device)

        if initial_model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {initial_model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if initial_model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")
    
    def run(
        self,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        target_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
        self.target_ids = target_ids.repeat(config.str_batch_size, 1).flatten()[None]
        optim_ids = self.target_ids.clone()
        
        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        
        for _ in tqdm(range(config.num_steps)):
            # Sample candidate token sequences based on the token gradient
            sampled_ids = sample_ids_random(
                ids=optim_ids.squeeze(0),
                search_width=config.search_width,
                vocab_size=self.tokenizer.vocab_size,
                topk=config.topk,
                n_replace=config.n_replace,
                not_allowed_ids=self.not_allowed_ids,
            )
            if config.filter_ids:
                sampled_ids = filter_ids(sampled_ids, tokenizer)

            new_search_width = sampled_ids.shape[0]

            # Compute loss on all candidate sequences 
            batch_size = new_search_width if config.batch_size is None else config.batch_size

            input_embeds = self.embedding_layer(sampled_ids)
            loss = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds)

            current_loss = loss.min().item()
            optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

            # Update the buffer based on the loss
            losses.append(current_loss)
            if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                buffer.add(current_loss, optim_ids)
            
            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)                

        min_loss_index = losses.index(min(losses)) 

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
        )

        return result
    
    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config=config)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            init_optim_ids = init_optim_ids.repeat(config.str_batch_size, 1).flatten()[None]

            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device, dtype=torch.float32)
                init_buffer_ids = [init_buffer_ids[torch.multinomial(init_buffer_ids, init_optim_ids.shape[1], replacement=True)].unsqueeze(0).long() for _ in range(config.buffer_size - 1)]
                init_buffer_ids = torch.cat([init_optim_ids] + init_buffer_ids, dim=0)
            else:
                init_buffer_ids = init_optim_ids
                
        true_buffer_size = max(1, config.buffer_size) 

        # Compute the loss on the initial buffer entries
        init_buffer_embeds = self.embedding_layer(init_buffer_ids)
        init_buffer_losses = find_executable_batch_size(self.compute_candidates_loss, true_buffer_size)(init_buffer_embeds)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        logger.info("Initialized attack buffer.")
        
        return buffer
    
    def compute_candidates_loss(
        self,
        search_batch_size: int, 
        input_embeds: Tensor, 
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        # ref:
        # https://github.com/GeorgeCazenavette/mtt-distillation/blob/main/distill.py
        # todo: think about how to do multi-step training / when to take grad steps.
        initial_params = get_model_params(self.model)
        final_params = get_model_params(self.final_model)

        estimated_grad = (initial_params - final_params)

        for i in range(0, input_embeds.shape[0], search_batch_size):
            input_embeds_batch = input_embeds[i:i+search_batch_size]
            current_batch_size = input_embeds_batch.shape[0]

            # reshape to add batch dimension
            input_embeds_batch = input_embeds_batch.reshape(-1, self.config.str_batch_size, input_embeds_batch.shape[-1])

            outputs = self.model(inputs_embeds=input_embeds_batch)

            shift_logits = outputs.logits.contiguous()
            shift_labels = self.target_ids.repeat(current_batch_size, 1)
            shift_labels = shift_labels.reshape(-1, self.config.str_batch_size)
            
            ce_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                reduction="mean"
            )
            ce_loss.backward()

            # 
            batch_grad = get_model_grad(self.model)
            loss = 1 - torch.nn.functional.cosine_similarity(estimated_grad, batch_grad, dim=-1)
            all_loss.append(loss)

            self.model.zero_grad()

            del outputs
            gc.collect()
            torch.cuda.empty_cache()

        return torch.tensor(all_loss, device=self.model.device)

# A wrapper around the GCG `run` method that provides a simple API
def run(
    initial_model: transformers.PreTrainedModel,
    final_model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using GCG. 

    Args:
        initial_model: The model to use for optimization.
        final_model: Parameters of the final model. Goal
            is to get close to this.
        tokenizer: The model's tokenizer.
        target: The target generation.
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()
    
    logger.setLevel(getattr(logging, config.verbosity))
    
    gcg = GCG(
        initial_model=initial_model, 
        final_model=final_model, 
        tokenizer=tokenizer, 
        config=config
    )
    result = gcg.run()
    return result
    