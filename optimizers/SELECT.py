from .optimizer import DiscreteOptimizer

import collections
import copy
import random
from enum import Enum

import torch
import tqdm
from utils import (
    device, 
    load_dataset_from_name,
)

from projection_utils import (
    CudaProjector,
    ProjectionType,
    get_grads_final_layer
)

class SELECTOptimizer(DiscreteOptimizer):
    X: torch.Tensor
    Y: torch.Tensor
    syn_lr: torch.Tensor

    def __init__(self, args, X: torch.Tensor, Y: torch.Tensor, tokenizer, student_net, initial_student_net):
        super().__init__(args)
        self.tokenizer = tokenizer
        self.student_net = student_net
        self.initial_student_net = initial_student_net
        self.base_model = copy.deepcopy(initial_student_net)

        self.Y = Y

        syn_lr = torch.tensor(self.args.lr_teacher).to(device)
        self.syn_lr = syn_lr.detach().to(device).requires_grad_(True)

        # Load search dataset
        self.dataset, self.dataset_text_column_name, __ = (
            load_dataset_from_name(self.args.select_seed_dataset)
        )
        self.dataset = self.dataset["train"]
        print(f"SELECTOptimizer: dataset size: {len(self.dataset)}")
        
        # Setup projector
        param_count = sum(p.numel() for p in self.base_model.lm_head.parameters())
        self.projector = CudaProjector(
            grad_dim=param_count,
            proj_dim=args.select_projection_dim,
            seed=0,
            block_size=128,
            proj_type=ProjectionType.rademacher,
            device=device,
            max_batch_size=256,
        )
        
        self.best_idx_counter = collections.Counter()
        self.batch = []
        self.optimizer = torch.optim.SGD(
            self.base_model.lm_head.parameters(), 
            lr=self.args.select_lr_student
        )
    
    def _rank_dataset_by_influence(self, base_params: torch.Tensor, expert_model_params: torch.Tensor) -> list:
        """Rank dataset examples by influence on optimization direction"""
        assert self.args.select_max_batch_size < len(self.dataset), \
            f"select_max_batch_size: {self.args.select_max_batch_size} must be less than dataset size: {len(self.dataset)}"

        # Get all gradients
        # TODO: Control randomness to get same results each time
        self.base_model.to(device)
        grads = get_grads_final_layer(
            self.base_model, 
            self.dataset, 
            self.tokenizer, 
            self.projector, 
            sequence_length=self.args.sequence_length, 
            batch_size=self.args.select_grad_batch_size
        )
        
        # Sanity check dimensions
        projection_dim = self.args.select_projection_dim
        assert grads.shape == (len(self.dataset), projection_dim), \
            f"grads.shape: {grads.shape} != (len(self.dataset), projection_dim): {(len(self.dataset), projection_dim)}"
        
        # Greedily fill batch
        batch = []
        grads_norm = grads / (grads.norm(dim=1, p=2, keepdim=True) + 1e-10)
        best_sim = float("-inf")
        current_grad = torch.zeros_like(base_params)
        student_lr = self.args.select_lr_student
        
        batch_pbar = tqdm.trange(0, self.args.select_max_batch_size, disable=(self.args.select_max_batch_size < 32))
        while len(batch) < self.args.select_max_batch_size:
            # Project parameter difference
            base_params_diff_projected = self.projector.project(
                (base_params - current_grad * student_lr) - expert_model_params, 
                model_id=0
            )
        
            # Use cosine similarity to find best gradient
            base_params_diff_norm = base_params_diff_projected / (base_params_diff_projected.norm(dim=1, p=2, keepdim=True) + 1e-10)
            sims = grads_norm @ base_params_diff_norm.T
            
            # Exclude already selected examples
            for idx in batch:
                sims[idx] = float("-inf")

            # Check for NaNs
            if torch.isnan(sims).any():
                print("NaNs in sims")
                break

            # Get the best remaining gradient
            best_idx = torch.argmax(sims)

            # Add the selected gradient to our batch
            batch.append(best_idx.item())
            grads[best_idx] = torch.zeros_like(grads[best_idx])  # Zero out the selected gradient

            # Update best similarity for logging purposes
            best_sim = sims[best_idx].item()

            # Compute full-resolution gradient
            batch_grad = get_grads_final_layer(
                self.base_model, 
                self.dataset.select([best_idx]), 
                self.tokenizer, 
                self.projector, 
                sequence_length=self.args.sequence_length, 
                batch_size=1, 
                do_projection=False
            )
            current_grad += batch_grad.flatten()
            batch_pbar.update(1)
            batch_pbar.set_description(f"Best sim: {best_sim:.3f} | Best idx: {best_idx.item()} | Batch size: {len(batch)}")
        

        print(f"Picked new batch of size {len(batch)}: {sorted(batch)}")
        
        return batch, best_sim

    def step_x(self, it: int, buffer: list) -> dict[str, torch.Tensor]:
        """Perform one step of SELECT optimization"""
        # Get current model parameters
        base_params = torch.cat([p.flatten().detach().requires_grad_(False) for p in self.base_model.lm_head.parameters()]).double().to(device)
        
        # Get expert model parameters from buffer
        expert_state_dict = buffer[-1]
        expert_model_params = torch.cat([v.flatten() for k, v in expert_state_dict.items() if "wte" in k]).double().to(device)
        
        # Update batch if needed
        if it % self.args.select_steps_per_grad == 0:
            self.batch, best_sim = self._rank_dataset_by_influence(base_params, expert_model_params)
        

        # Calculate current MSE
        current_mse = (base_params - expert_model_params).double().pow(2).sum().item()
        
        # Select random minibatch from batch
        minibatch = random.sample(self.batch, min(self.args.select_minibatch_size, len(self.batch)))

        print("** TAKING STEP **")
        # Take step on batch
        inputs = self.tokenizer(
            self.dataset.select(minibatch)["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.sequence_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].detach().clone()
        outputs = self.base_model(**inputs)
        loss = outputs.loss
        loss.backward()

        # Calculate gradient direction and step size
        base_params_diff = base_params - expert_model_params
        with torch.no_grad():
            grad = torch.cat([p.grad.flatten().detach() for p in self.base_model.lm_head.parameters()], dim=0)
            grad_direction = grad / (grad.norm(dim=0, p=2, keepdim=True) + 1e-10)
            grad_step_size = (grad_direction.double() @ base_params_diff).float()

        # Create optimizer and take step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.base_model.zero_grad()
        
        # Print loss
        print(f"loss: {loss.item():.3f}")

        # Report metrics
        metrics = {
            "param_mse": current_mse,
            "ce_loss": loss.detach().item(),
            "grad_step_size": grad_step_size.item() if not torch.isnan(grad_step_size) else 0.0,
            "synth_lr": self.syn_lr.detach().item(),
        }
        
        # Free memory
        torch.cuda.empty_cache()
        
        #   add minibatch to self.best_idx_counter
        self.best_idx_counter.update(minibatch)
        
        # Print top-10 most selected indices
        top_10_selected = self.best_idx_counter.most_common(10)
        print(f"Top-10 most-selected indices: {top_10_selected}")
        
        return metrics

    def step(self, it: int, buffer: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics = self.step_x(it, buffer)
        X_text = self.dataset.select(self.batch)["text"]
        X_tokens = self.tokenizer(X_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.args.sequence_length)["input_ids"]
        return X_tokens, metrics
