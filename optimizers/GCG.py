from .optimizer import DiscreteOptimizer

import copy
import numpy as np
import random
import torch
import tqdm
import wandb

from utils import device, project_x_to_embedding_space, state_dict_to_tensor

class GCGOptimizer(DiscreteOptimizer):
    X: torch.Tensor
    Y: torch.Tensor
    syn_lr: torch.Tensor

    def __init__(self, args, X: torch.Tensor, Y: torch.Tensor, tokenizer, student_net, initial_student_net):
        super().__init__(args)
        self.tokenizer = tokenizer
        self.student_net = student_net
        self.initial_student_net = initial_student_net

        _, self.X_tokens = project_x_to_embedding_space(X, self.initial_student_net, self.args.minibatch_size)
        self.Y = Y

        syn_lr = torch.tensor(self.args.lr_teacher).to(device)
        self.syn_lr = syn_lr.detach().to(device).requires_grad_(True)

    def step_x(self, it: int, buffer: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        start_epoch = np.random.randint(0, len(buffer) - self.args.expert_epochs)
        starting_params = buffer[start_epoch]
        target_params = buffer[start_epoch + self.args.expert_epochs]

        search_width = self.args.gcg_search_width
        tokens_to_swap = self.args.gcg_tokens_to_swap
        X_best = None
        X_best_loss = float("inf")
        all_losses = []
        for i in tqdm.trange(search_width + 1, colour="#bf40bf", desc="GCG", leave=False):
            X_tokens = self.X_tokens.clone()
            indices = torch.randperm(len(X_tokens))
            indices_chunks = list(torch.split(indices, self.args.minibatch_size))

            if i > 0:
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
                        
            with torch.no_grad():
                X = self.initial_student_net.get_input_embeddings()(X_tokens)

            param_loss, ce_loss_avg = self.step_x_inner_loop(
                X=X, 
                Y=self.Y,
                starting_params=state_dict_to_tensor(starting_params), 
                target_params=state_dict_to_tensor(target_params),
                syn_lr=self.syn_lr,
                indices_chunks=indices_chunks,
            )
            
            all_losses.append(param_loss.detach().cpu())
            if (X_best is None) or param_loss < X_best_loss:
                X_best = X_tokens.detach()
                X_best_loss = param_loss.detach()
        
        # print(f"X_best_loss: {X_best_loss}")
        self.X_tokens = X_best

        metrics = {
            "param_loss": param_loss.detach().cpu(),
            "X_best_loss": X_best_loss.detach().cpu(),
            "start_epoch": start_epoch,
            "ce_loss": ce_loss_avg,
            "synth_lr": self.syn_lr.detach().cpu(),
            # "synth_lr_grad": self.syn_lr.grad.norm().detach().cpu().item(),
        }
        return metrics

    def step(self, it: int, buffer: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics = self.step_x(it, buffer)

        if it % 100 == 0:
            self._log_table(self.X_tokens, self.Y, step=it)

        return self.X_tokens, metrics
