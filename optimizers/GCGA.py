from .GCG import GCGOptimizer

import random

import numpy as np
import torch

from utils.core import gather, get_rank, get_world_size, device, state_dict_to_tensor, trange_if_main_worker

class GCGAOptimizer(GCGOptimizer):
    X: torch.Tensor
    Y: torch.Tensor
    syn_lr: torch.Tensor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_counter = 0
        self.random_switch_perc = 0.0 # TODO: Argparse for this.
        self.it_per_x = 500 # 200 # TODO: Argparse for this.
        self.X_tokens_full = self.X_tokens.clone()
        self.X_tokens = torch.tensor([], dtype=torch.int64).to(device)
        self.X_tokens.requires_grad_(False)
        self.X_tokens_full.requires_grad_(False)
        self.Y.requires_grad_(False)
        self._distributed_broadcast_everything()
        self.X_swap_indices = []


    def _distributed_broadcast_everything(self) -> None:
        if get_world_size() <= 1:
            return
        # broadcast from rank 0
        torch.distributed.broadcast(self.X_tokens_full, src=0)
        torch.distributed.broadcast(self.X_tokens, src=0)
        torch.distributed.broadcast(self.Y, src=0)

    def step_x(self, step: int, buffer: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if it % self.it_per_x == 0:
            self.x_counter += 1
            new_X_tokens = gather(self.X_tokens_full[self.x_counter * get_world_size() + get_rank()].unsqueeze(0))
            self.X_tokens = torch.cat([self.X_tokens, new_X_tokens], dim=0)
            self.X_swap_indices = list(range(self.X_tokens.shape[1]))

        start_epoch = np.random.randint(0, len(buffer) - self.args.expert_epochs)
        starting_params = buffer[start_epoch]
        target_params = buffer[start_epoch + self.args.expert_epochs]
        search_width = self.args.gcg_search_width
        tokens_to_swap = self.args.gcg_tokens_to_swap

        X_tokens = self.X_tokens.to(device)

        # Randomly mutate token(s) from a chunk
        best_X_loss = float("inf")
        best_X_tokens = None
        best_X_swap_idx = None
        best_X_swap_token_idxs = None
        all_param_losses = []

        for _ in trange_if_main_worker(search_width, colour="#bf40bf", desc="GCG", leave=False):
            X_tokens_batch = X_tokens.clone()
            # Randomly mutate `tokens_to_swap` tokens in the last document
            if random.random() < self.random_switch_perc:
                swap_idx = torch.randint(0, X_tokens.shape[0], (1,))
                rand_token_indices = torch.randint(0, X_tokens.shape[1], (tokens_to_swap,))
            else:
                swap_idx = int(-1 * get_rank())
                if len(self.X_swap_indices) < tokens_to_swap:
                    rand_token_indices = torch.randint(0, X_tokens.shape[1], (tokens_to_swap,))
                else:
                    rand_token_indices = random.sample(self.X_swap_indices, tokens_to_swap)
                    rand_token_indices = torch.tensor([rand_token_indices])

            rand_tokens = torch.randint(0, self.tokenizer.vocab_size, (tokens_to_swap,)).to(device)
            X_tokens_batch[swap_idx, rand_token_indices] = rand_tokens

            # indices_chunks = list(torch.split(indices, self.args.minibatch_size))
            with torch.no_grad():
                X = self.initial_student_net.get_input_embeddings()(X_tokens_batch)

            _, min_param_loss, ce_loss_avg = self.step_x_inner_loop(
                X=X, 
                Y=self.Y[:X.shape[0]],
                starting_params=state_dict_to_tensor(starting_params), 
                target_params=state_dict_to_tensor(target_params),
                syn_lr=self.syn_lr,
                # TODO: Figure out why we can't pass indices_chunks
                # indices_chunks=indices_chunks,
                keep_grads=False,
            )
            if min_param_loss < best_X_loss:
                best_X_loss = min_param_loss
                best_X_tokens = X_tokens_batch
                best_X_swap_idx = swap_idx
                best_X_swap_token_idxs = rand_token_indices.flatten().tolist()
            all_param_losses.append(min_param_loss)
        
        # Track the best token-swap
        for _j in best_X_swap_token_idxs: 
            try:
                self.X_swap_indices.remove(_j)
            except:
                continue
        
        # Take the token-swaps that improved the loss the most
        self.X_tokens = best_X_tokens.detach()

        # Broadcast for distributed setup
        if get_world_size() > 1:
            # TODO: Fix this to accomodate swap_idx
            # swap_idxs = gather(torch.tensor([best_X_swap_idx], device=device))
            # all_new_tokens = gather(self.X_tokens[-1 * get_rank()].unsqueeze(0))
            best_x_swap_idx_tensor = torch.tensor([best_X_swap_idx], device=device)
            concatenated_tensors = torch.cat([best_x_swap_idx_tensor, self.X_tokens[-1 * get_rank()]], dim=0)
            gathered_result = gather(concatenated_tensors[None])
            swap_idxs = gathered_result[:, 0]
            all_new_tokens = gathered_result[:, 1:]
            self.X_tokens[swap_idxs] = all_new_tokens
        
        # print("rank", get_rank(), "X_tokens", self.X_tokens.tolist())
        # print()
        all_param_losses = torch.tensor(all_param_losses).to(device)
        metrics = {
            "param_loss_mean": all_param_losses.mean().item(),
            "X_best_loss": best_X_loss,
            "start_epoch": start_epoch,
            "ce_loss": ce_loss_avg,
            "synth_lr": self.syn_lr.detach().item(),
            "x_counter": self.x_counter,
            "X_tokens_numel": self.X_tokens.numel(),
            "it_per_x": self.it_per_x,
        }
        return metrics
