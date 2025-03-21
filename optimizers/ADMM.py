from .optimizer import DiscreteOptimizer

import numpy as np
import torch

from utils.core import device, project_x_to_embedding_space, state_dict_to_tensor

class ADMMOptimizer(DiscreteOptimizer):
    X: torch.Tensor
    Y: torch.Tensor
    syn_lr: torch.Tensor
    optimizer_token_embeddings: torch.optim.Optimizer
    optimizer_lr: torch.optim.Optimizer
    Λ: torch.Tensor
    Z: torch.Tensor

    def __init__(self, args, X: torch.Tensor, Y: torch.Tensor, tokenizer, student_net, initial_student_net):
        super().__init__(args)
        self.tokenizer = tokenizer
        self.student_net = student_net
        self.initial_student_net = initial_student_net

        self.X = X
        self.Y = Y

        syn_lr = torch.tensor(self.args.lr_teacher).to(device)
        self.syn_lr = syn_lr.detach().to(device).requires_grad_(True)

        self.optimizer_token_embeddings = torch.optim.Adam([X], lr=self.args.lr_tokens)
        self.optimizer_lr = torch.optim.SGD([syn_lr], lr=self.args.lr_lr, momentum=0.5)
        self.optimizer_token_embeddings.zero_grad()
        self.Λ = torch.zeros_like(X, device=device).requires_grad_(False)
        self.Z, _ = project_x_to_embedding_space(
            self.X - self.Λ / self.ρ, 
            self.initial_student_net, 
            self.args.minibatch_size
        )   

    @property
    def ρ(self):
        return self.args.admm_penalty_term

    def step_x(self, step: int, buffer: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        X = self.X
        Y = self.Y
        Z = self.Z
        Λ = self.Λ
        syn_lr = self.syn_lr

        num_params = sum([np.prod(p.size()) for p in self.student_net.parameters()])

        start_epoch = np.random.randint(0, len(buffer) - self.args.expert_epochs)
        starting_params = buffer[start_epoch]
        target_params = buffer[start_epoch + self.args.expert_epochs]
        # 
        #  (1) Compute overall loss on X using trajectory matching
        # 
        param_loss_normalized, ce_loss_avg = self.step_x_inner_loop(
            X=X, 
            Y=Y,
            starting_params=state_dict_to_tensor(starting_params), 
            target_params=state_dict_to_tensor(target_params),
            syn_lr=syn_lr,
        )
        lagrangian_term = torch.sum(Λ * (X - Z), dim=2).mean()
        quadratic_penalty = (self.ρ / 2) * ((X - Z).norm(p=2, dim=2) ** 2).mean()
        aux_loss = lagrangian_term + quadratic_penalty

        (aux_loss + param_loss_normalized).backward()
        Z_dist = (X - Z).norm(p=2).mean().detach()

        metrics = {
            # "param_loss": param_loss.detach().cpu(),
            "param_loss_normalized": param_loss_normalized.detach().cpu(),
            # "param_dist": param_dist.detach().cpu(),
            # "param_loss_normalized_minus_one": (param_loss_normalized - 1).detach().cpu(),
            # "param_dist": param_dist.detach().cpu(),
            "aux_loss":  aux_loss.detach().cpu().item(),
            "aux_loss_lagrangian": lagrangian_term.detach().cpu().item(),
            "aux_loss_quadratic_penalty": quadratic_penalty.detach().cpu().item(),
            "start_epoch": start_epoch,
            "ce_loss": ce_loss_avg,
            "token_grad_norm": X.grad.norm().detach().cpu().item(),
            "synth_lr": syn_lr.detach().cpu(),
            "synth_lr_grad": syn_lr.grad.norm().detach().cpu().item(),
            "Z_dist": Z_dist.detach().cpu().item(),
        }
        return X, Y, metrics
    
    @torch.no_grad
    def step_z(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not (it % self.args.max_iterations_x == 0):
            return self.Z, self.Λ, {}
        
        X = self.X
        Y = self.Y
        Z = self.Z
        Λ = self.Λ
        # 
        #  (2) Compute new Z based on projecting X back to word embedding space
        #                       TODO: Use a language model for this, optionally.
        Z, Z_tokens = project_x_to_embedding_space(
            X - Λ / self.ρ, self.initial_student_net, self.args.minibatch_size)
        # 
        # Log Z
        # 
        self._log_table(Z_tokens, Y, step=step)
        # 
        #  (3) Update Λ for ADMM
        # 
        Λ = Λ + self.ρ * (X - Z)
        return Z, Λ, {}

    def step(self, step: int, buffer: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.X, self.Y, x_metrics = self.step_x(step, buffer)
        self.Z, self.Λ, z_metrics = self.step_z(step)

        self.optimizer_token_embeddings.step()
        self.optimizer_lr.step()
        self.optimizer_token_embeddings.zero_grad()
        self.optimizer_lr.zero_grad()

        return self.Z, {**x_metrics, **z_metrics}
        