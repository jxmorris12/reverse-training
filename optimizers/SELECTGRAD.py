from .optimizer import SELECTOptimizer

import datasets
import numpy as np
import random
import torch
import tqdm
import wandb

from utils import device, project_x_to_embedding_space, state_dict_to_tensor

class SELECTGRAD(SELECTOptimizer):
    X: torch.Tensor
    Y: torch.Tensor
    syn_lr: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._score_dataset()

    def _score_dataset(self):
        minibatch_size = self.args.minibatch_size
        for i in range(0, len(self.dataset), minibatch_size):
            batch_documents = self.dataset.select(list(range(i, min(i + minibatch_size, len(self.dataset))))
            batch_documents = [d["text"] for d in random_documents]
            batch_documents_tokens = self.tokenizer(
                random_documents, 
                padding="max_length", 
                truncation=True, 
                max_length=self.args.sequence_length-1, 
                return_tensors="pt", 
                return_attention_mask=False
            )["input_ids"].to(device)
            # TODO: Compute & weight gradient here

    def step(self, it: int, buffer: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics = self.step_x(it, buffer)

        if it % 100 == 0:
            self._log_table(self.X_tokens, self.Y, step=it)

        return self.X_tokens, metrics
