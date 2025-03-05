from .GCG import GCGOptimizer

import numpy as np
import random
import torch
import tqdm
import wandb

from utils import device, project_x_to_embedding_space, state_dict_to_tensor

class GCGAOptimizer(GCGOptimizer):
    X: torch.Tensor
    Y: torch.Tensor
    syn_lr: torch.Tensor
    def step_x(self, it: int, buffer: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        start_epoch = np.random.randint(0, len(buffer) - self.args.expert_epochs)
        starting_params = buffer[start_epoch]
        target_params = buffer[start_epoch + self.args.expert_epochs]

        search_width = self.args.gcg_search_width
        tokens_to_swap = self.args.gcg_tokens_to_swap
        X_best = None
        X_best_loss = float("inf")
        all_losses = []
        X_tokens = self.X_tokens.clone()
        indices = torch.randperm(len(X_tokens))
        indices_chunks = list(torch.split(indices, self.args.minibatch_size))

        # Randomly mutate token(s) from a chunk
        for _ in range(self.args.syn_steps):
            chunk_idxs = indices_chunks[_ % len(indices_chunks)]
            random_token_mask_idxs = torch.randint(
                low=0,
                high=self.args.sequence_length - 1,
                size=(len(chunk_idxs), tokens_to_swap),
                device=device,
            )
            random_token_idxs = torch.randint(
                low=0,
                high=self.tokenizer.vocab_size,
                size=(len(chunk_idxs), tokens_to_swap),
                device=device,
            )
            X_tokens[chunk_idxs[:, None], random_token_mask_idxs] = random_token_idxs
        
        X_mask_on_count = torch.zeros(X_tokens.shape, dtype=torch.long, device=device)
        X_mask_on_sum = torch.zeros(X_tokens.shape, dtype=torch.double, device=device)
        X_mask_off_count = torch.zeros(X_tokens.shape, dtype=torch.long, device=device)
        X_mask_off_sum = torch.zeros(X_tokens.shape, dtype=torch.double, device=device)

        for i in tqdm.trange(search_width, colour="#bf40bf", desc="GCG", leave=False):
            indices_chunks = list(torch.split(indices, self.args.minibatch_size))
            X_mask = torch.bernoulli(0.5 * torch.ones_like(X_tokens, device=device)).bool()
            X_tokens_batch = torch.where(X_mask, X_tokens, self.X_tokens)
            with torch.no_grad():
                X = self.initial_student_net.get_input_embeddings()(X_tokens_batch)

            param_loss, ce_loss_avg = self.step_x_inner_loop(
                X=X, 
                Y=self.Y,
                starting_params=state_dict_to_tensor(starting_params), 
                target_params=state_dict_to_tensor(target_params),
                syn_lr=self.syn_lr,
                indices_chunks=indices_chunks,
            )
            
            X_mask_on_count[X_mask] += 1
            X_mask_on_sum[X_mask] += param_loss.detach()

            X_mask_off_count[~X_mask] += 1
            X_mask_off_sum[~X_mask] += param_loss.detach()
        
        X_mask_on_avg = X_mask_on_sum / X_mask_on_count
        X_mask_on_avg = torch.nan_to_num(X_mask_on_avg, nan=float("inf"))
        X_mask_off_avg = X_mask_off_sum / X_mask_off_count
        X_mask_off_avg = torch.nan_to_num(X_mask_off_avg, nan=float("inf"))
        
        # Take the token-swaps that improved the loss on average
        self.X_tokens = torch.where(X_mask_on_avg < X_mask_off_avg, X_tokens, self.X_tokens)

        metrics = {
            "param_loss": param_loss.detach().cpu(),
            "X_best_loss": min(X_mask_on_avg.min(), X_mask_off_avg.min()).detach().cpu(),
            "start_epoch": start_epoch,
            "ce_loss": ce_loss_avg,
            "synth_lr": self.syn_lr.detach().cpu(),
        }
        return metrics
