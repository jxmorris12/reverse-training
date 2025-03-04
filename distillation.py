import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transformers
import wandb
import tqdm

from reparam_module import ReparamModule
from utils import device, get_model, get_token_embeddings_random, get_token_embeddings_from_dataset, state_dict_to_tensor, project_x_to_embedding_space, load_expert_trajectories


class DatasetDistiller:
    def __init__(self, args):
        # train model for a couple steps
        self.args = args
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        self.initial_student_net = get_model("gpt2")
        self.student_net = ReparamModule(get_model("gpt2")).to(device)
        self.ds, self.ds_text_column_name, self.ds_label_column_name = self._init_dataset()
    
    @property
    def ρ(self):
        return self.args.penalty_term
    
    def _init_dataset(self) -> tuple[datasets.Dataset, str, str]:
        if self.args.dataset == "ag_news":
            ds = datasets.load_dataset("fancyzhx/ag_news", split="train")
            text_column_name = "text"        
            label_column_name = "label"
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented")
        
        return ds, text_column_name, label_column_name

    
    def _init_synthetic_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.args.token_init == "random":
            X, Y = get_token_embeddings_random(
                dataset_size=self.args.dataset_size,
                sequence_length=self.args.sequence_length,
            )
        elif self.args.token_init == "dataset":
            X, Y = get_token_embeddings_from_dataset(
                dataset_size=self.args.dataset_size, 
                sequence_length=self.args.sequence_length,
                ds=self.ds, 
                text_column_name=self.ds_text_column_name
            )
        else:
            raise NotImplementedError(f"Token init {args.token_init} not implemented")
        X = X.detach().to(device).requires_grad_(True)
        return X, Y
    
    def step_x_inner_loop(self, X: torch.Tensor, Y: torch.Tensor, starting_params: torch.Tensor, target_params: torch.Tensor, syn_lr: torch.Tensor):
        student_params = [starting_params.clone().detach().requires_grad_(True)]
        ce_losses = []
        indices_chunks = []
        for step in tqdm.trange(self.args.syn_steps, desc="Synthetic steps", leave=False):
            if not indices_chunks:
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
        return student_params[-1], ce_loss_avg

    def step_x(self, it: int, buffer: list, syn_lr: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, Λ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        num_params = sum([np.prod(p.size()) for p in self.student_net.parameters()])

        start_epoch = np.random.randint(0, len(buffer) - self.args.expert_epochs)
        starting_params = buffer[start_epoch]
        target_params = buffer[start_epoch + self.args.expert_epochs]

        target_params = state_dict_to_tensor(target_params)
        starting_params = state_dict_to_tensor(starting_params)
        # 
        #  (1) Compute overall loss on X using trajectory matching
        # 
        final_student_params, ce_loss_avg = self.step_x_inner_loop(
            X=X, 
            Y=Y,
            starting_params=starting_params, 
            target_params=target_params,
            syn_lr=syn_lr,
        )
        param_loss = torch.nn.functional.mse_loss(final_student_params, target_params, reduction="sum") / num_params
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum") / num_params
        if torch.isclose(param_dist, torch.tensor(0.0)):
            print("zero distance detected – stopping!")
            exit()

        ρ = self.args.penalty_term     # [TODO]  Argparse this. Paper says: 
        param_loss = param_loss / param_dist

        lagrangian_term = torch.sum(Λ * (X - Z), dim=2).mean()
        quadratic_penalty = (ρ / 2) * ((X - Z).norm(p=2, dim=2) ** 2).mean()
        aux_loss = lagrangian_term + quadratic_penalty

        (aux_loss + param_loss).backward()
        Z_dist = (X - Z).norm(p=2).mean().detach()

        metrics = {
            "param_loss": param_loss.detach().cpu(),
            "param_loss_minus_one": (param_loss - 1).detach().cpu(),
            "param_dist": param_dist.detach().cpu(),
            "aux_loss":  aux_loss.detach().cpu(),
            "aux_loss_lagrangian": lagrangian_term.detach().cpu(),
            "aux_loss_quadratic_penalty": quadratic_penalty.detach().cpu(),
            "start_epoch": start_epoch,
            "ce_loss": ce_loss_avg,
            "token_grad_norm": X.grad.norm().detach().cpu(),
            "synth_lr": syn_lr.detach().cpu(),
            "synth_lr_grad": syn_lr.grad.norm().detach().cpu(),
            "Z_dist": Z_dist,
        }
        return X, Y, metrics
    
    @torch.no_grad
    def step_z(self, it: int, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, Λ: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if not (it % self.args.max_iterations_x == 0):
            return Z, Λ, {}
        # 
        #  (2) Compute new Z based on projecting X back to word embedding space
        #                       TODO: Use a language model for this, optionally.
        Z, Z_tokens = project_x_to_embedding_space(
            X, Λ, self.ρ, self.initial_student_net, self.args.minibatch_size)
        # 
        # Log Z
        # 
        tokens = self.tokenizer.batch_decode(Z_tokens.cpu(), add_special_tokens=False)
        labels = self.tokenizer.batch_decode(Y.cpu(), add_special_tokens=False)
        table_data = [(i, T, L) for i, (T, L) in enumerate(zip(tokens, labels))]
        tokens_table = wandb.Table(data=table_data, columns=["index", "text", "label"])
        wandb.log({ "Z": tokens_table }, step=it)
        # 
        #  (3) Update Λ for ADMM
        # 
        Λ = Λ + self.ρ * (X - Z)
        return Z, Λ, {}

    def run_distillation(self):
        X, Y = self._init_synthetic_data()
        syn_lr = torch.tensor(self.args.lr_teacher).to(device)
        syn_lr = syn_lr.detach().to(device).requires_grad_(True)

        optimizer_token_embeddings = torch.optim.Adam([X], lr=self.args.lr_tokens)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=self.args.lr_lr, momentum=0.5)
        optimizer_token_embeddings.zero_grad()
        Λ = torch.zeros_like(X, device=device).requires_grad_(False)
        Z, _ = project_x_to_embedding_space(X, Λ, self.ρ, self.initial_student_net, self.args.minibatch_size)

        # load/generate expert trajectories
        buffer = load_expert_trajectories(
            num_experts=self.args.num_experts,
            num_steps_per_expert=self.args.pretrain_iterations_x,
            num_expert_datapoints=self.args.minibatch_size,
            expert_lr=self.args.expert_lr,
            sequence_length=self.args.sequence_length,
            ds=self.ds,
            text_column_name=self.ds_text_column_name,
            label_column_name=self.ds_label_column_name,
        )

        # run optimization
        pbar = tqdm.trange(0, self.args.max_iterations+1, desc="iterations")
        for it in pbar:
            X, Y, x_metrics = self.step_x(it, buffer, syn_lr, X, Y, Z, Λ)
            Z, Λ, z_metrics = self.step_z(it, X, Y, Z, Λ)
            pbar.set_postfix(**x_metrics)
            wandb.log({**x_metrics, **z_metrics}, step=it)

            optimizer_token_embeddings.step()
            optimizer_lr.step()
            optimizer_token_embeddings.zero_grad()
            optimizer_lr.zero_grad()