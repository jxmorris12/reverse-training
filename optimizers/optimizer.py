from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F
import tqdm
import wandb


class DiscreteOptimizer(ABC):
    def __init__(self, args):
        self.args = args
    
    def step_x_inner_loop(
            self, 
            X: torch.Tensor, 
            Y: torch.Tensor, 
            starting_params: torch.Tensor, 
            target_params: torch.Tensor, 
            syn_lr: torch.Tensor,
            indices_chunks: Optional[list[torch.Tensor]] = None
        ):
        student_params = [starting_params.clone().detach().requires_grad_(True)]
        ce_losses = []
        for step in tqdm.trange(self.args.syn_steps, desc="Synthetic steps", leave=False):
            if (indices_chunks is None) or len(indices_chunks) == 0:
                indices = torch.randperm(len(X))
                indices_chunks = list(torch.split(indices, self.args.minibatch_size))

            these_indices = indices_chunks.pop()
            x = X[these_indices]
            y = Y[these_indices]
            
            output = self.student_net(inputs_embeds=x, flat_param=student_params[-1])

            # autoregressive classification loss on last token
            logits = output.logits[:, :-1]
            ce_loss = torch.nn.functional.cross_entropy(logits[:, -1], y, reduction="mean")
            ce_losses.append(ce_loss.detach().item())

            # exit on nan
            if torch.isnan(ce_loss):
                breakpoint()
                print("nan detected - stopping!")
                exit()

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)
        ce_loss_avg = sum(ce_losses) / len(ce_losses)

        with torch.no_grad():
            final_student_params = student_params[-1]
            param_loss = 1 - F.cosine_similarity(
                final_student_params - starting_params,
                target_params - starting_params,
                dim=0
            ).mean()
        
        # Clean up memory
        for _ in student_params:
            del _
        del student_params
        
        return param_loss, ce_loss_avg
    
    @abstractmethod
    def step(self) -> torch.Tensor:
        pass