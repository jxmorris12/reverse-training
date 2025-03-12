from .GCG import GCGOptimizer

import random

import numpy as np
import torch
import tqdm

from utils import device, state_dict_to_tensor

class GCGAOptimizer(GCGOptimizer):
    X: torch.Tensor
    Y: torch.Tensor
    syn_lr: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_counter = 0
        self.random_switch_perc = 0.25 # TODO: Argparse for this.
        self.it_per_x = 40 # 200 # TODO: Argparse for this.
        self.X_tokens_full = self.X_tokens.clone()

    def step_x(self, it: int, buffer: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if it % self.it_per_x == 0:
            self.x_counter += 1
            self.X_tokens = torch.cat([self.X_tokens, self.X_tokens_full[-1].unsqueeze(0)], dim=0)
        start_epoch = np.random.randint(0, len(buffer) - self.args.expert_epochs)
        starting_params = buffer[start_epoch]
        target_params = buffer[start_epoch + self.args.expert_epochs]
        search_width = self.args.gcg_search_width
        tokens_to_swap = self.args.gcg_tokens_to_swap

        X_tokens = self.X_tokens[:self.x_counter].clone()

        # Randomly mutate token(s) from a chunk
        best_X_loss = float("inf")
        best_X_tokens = None

        for _ in tqdm.trange(search_width, colour="#bf40bf", desc="GCG", leave=False):
            X_tokens_batch = X_tokens.clone()
            # Randomly mutate `tokens_to_swap` tokens in the last document
            rand_token_indices = torch.randint(0, X_tokens.shape[1], (tokens_to_swap,))
            rand_tokens = torch.randint(0, self.tokenizer.vocab_size, (tokens_to_swap,)).to(device)
            if random.random() < self.random_switch_perc:
                swap_idx = torch.randint(0, X_tokens.shape[0], (1,))
            else:
                swap_idx = -1
            X_tokens_batch[swap_idx, rand_token_indices] = rand_tokens


            # indices_chunks = list(torch.split(indices, self.args.minibatch_size))
            with torch.no_grad():
                X = self.initial_student_net.get_input_embeddings()(X_tokens_batch)
            Y = self.Y[:self.x_counter]

            param_loss, ce_loss_avg = self.step_x_inner_loop(
                X=X, 
                Y=Y,
                starting_params=state_dict_to_tensor(starting_params), 
                target_params=state_dict_to_tensor(target_params),
                syn_lr=self.syn_lr,
                # TODO: Figure out why we can't pass indices_chunks
                # indices_chunks=indices_chunks,
            )
            if param_loss < best_X_loss:
                best_X_loss = param_loss
                best_X_tokens = X_tokens_batch
        
        # Take the token-swaps that improved the loss the most
        self.X_tokens = best_X_tokens
        metrics = {
            "param_loss": param_loss.detach().item(),
            "X_best_loss": best_X_loss,
            "start_epoch": start_epoch,
            "ce_loss": ce_loss_avg,
            "synth_lr": self.syn_lr.detach().item(),
        }
        return metrics
