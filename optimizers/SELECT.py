from .optimizer import DiscreteOptimizer

import collections
import copy
import random

import torch
import tqdm
from utils import (
    autolabel_dataset,
    device, 
    get_model,
    load_dataset_from_name,
)

from utils.projection import (
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
        self.base_model = get_model("gpt2")

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
            self.base_model.parameters(), 
            lr=self.args.select_lr_student
        )
        self.steps_since_last_improvement = 0
        self.best_mse = float("inf")
        self.tokenization_cache = {}
        self.Y = None
        self.dataset_labels = None
        # TODO: Make this dynamic
        self.dataset_label_map = {
            "0": "World",
            "1": "Sports",
            "2": "Business",
            "3": "Sci/Tech",
        }
    
    def _rank_dataset_by_influence(
            self, 
            full_base_params, full_expert_model_params,
            last_layer_base_params, last_layer_expert_model_params
        ) -> list:
        """Rank dataset examples by influence on optimization direction"""
        assert self.args.select_full_dataset_size <= len(self.dataset), \
            f"select_full_dataset_size: {self.args.select_full_dataset_size} must be less than dataset size: {len(self.dataset)}"

        # Get all gradients
        # TODO: Control randomness to get same results each time
        self.base_model.to(device)

        base_model = copy.deepcopy(self.base_model)
        grads = []
        for i in range(self.args.select_num_pseudoexperts):
            step_frac = i / self.args.select_num_pseudoexperts
            pseudoexpert_params = (
                step_frac * full_base_params + (1 - step_frac) * full_expert_model_params
            )
            # Set params
            counter = 0
            for i, p in enumerate(base_model.parameters()):
                p.data = pseudoexpert_params[counter:counter+p.numel()].reshape(p.data.shape).to(p.data.dtype)
                counter += p.numel()

            batch_grads = get_grads_final_layer(
                base_model, 
                self.dataset, 
                self.dataset_labels,
                self.tokenizer, 
                self.projector,
                sequence_length=self.args.sequence_length, 
            )
            grads.append(batch_grads)
        
        grads = torch.stack(grads, dim=0).double().sum(dim=0).float()
        
        # Sanity check dimensions
        projection_dim = self.args.select_projection_dim
        assert grads.shape == (len(self.dataset), projection_dim), \
            f"grads.shape: {grads.shape} != (len(self.dataset), projection_dim): {(len(self.dataset), projection_dim)}"
        
        # Greedily fill batch
        batch = []
        grads_norm = grads / (grads.norm(dim=1, p=2, keepdim=True) + 1e-10)
        grads_norm = grads_norm.to(device)
        best_sim = float("-inf")
        current_grad = torch.zeros_like(last_layer_base_params)
        student_lr = self.args.select_lr_student
        
        batch_pbar = tqdm.trange(0, self.args.select_full_dataset_size, disable=(self.args.select_full_dataset_size < 32))
        while len(batch) < self.args.select_full_dataset_size:
            # Project parameter difference
            base_params_diff_projected = self.projector.project(
                (last_layer_base_params - current_grad * student_lr) - last_layer_expert_model_params, 
                model_id=0
            )
        
            # Use cosine similarity to find best gradient
            base_params_diff_norm = base_params_diff_projected / (base_params_diff_projected.norm(dim=1, p=2, keepdim=True) + 1e-10)
            base_params_diff_norm = base_params_diff_norm.to(device)
            sims = grads_norm @ base_params_diff_norm.T
            
            # Exclude already selected examples
            batch_idxs = torch.tensor(batch, device=device, dtype=torch.long)
            sims[batch_idxs] = torch.ones_like(sims[batch_idxs]) * float("-inf")

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
                self.dataset_labels[None, best_idx],
                self.tokenizer, 
                self.projector, 
                sequence_length=self.args.sequence_length, 
                do_projection=False
            )
            current_grad += batch_grad.flatten()
            batch_pbar.update(1)
            batch_pbar.set_description(f"Best sim: {best_sim:.3f} | Best idx: {best_idx.item()} | Batch size: {len(batch)}")
        

        print(f"Picked new batch of size {len(batch)}: {sorted(batch)}")
        
        return batch

    def step_with_grad(self, it: int, buffer: list) -> dict[str, torch.Tensor]:
        """Perform one step of SELECT optimization"""
        # Get current model parameters
        full_base_params = torch.cat([p.flatten().cpu().detach().requires_grad_(False) for p in self.base_model.parameters()]).double().to(device)
        last_layer_base_params = torch.cat([p.flatten().cpu().detach().requires_grad_(False) for p in self.base_model.lm_head.parameters()]).double().to(device)
        
        # Get expert model parameters from buffer
        expert_state_dict = buffer[-1]
        full_expert_model_params = torch.cat([v.flatten() for k, v in expert_state_dict.items()]).double().to(device)
        last_layer_expert_model_params = torch.cat([v.flatten() for k, v in expert_state_dict.items() if "wte" in k]).double().to(device)
        
        # Update batch if needed
        should_update_batch = (self.args.select_steps_per_grad > 0) and (it % self.args.select_steps_per_grad == 0)
        if (not len(self.batch)) or should_update_batch:
            self.batch = self._rank_dataset_by_influence(
                full_base_params, full_expert_model_params,
                last_layer_base_params, last_layer_expert_model_params
            )
        
        # Calculate current MSE
        base_params_diff = full_base_params - full_expert_model_params
        current_mse = (base_params_diff).double().pow(2).sum().item()
        
        # Select random minibatch from batch
        minibatch = random.sample(self.batch, min(self.args.minibatch_size, len(self.batch)))
        assert len(minibatch) > 0, f"Minibatch is empty"

        # Take step on batch
        inputs = self.tokenizer(
            self.dataset.select(minibatch)["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.args.sequence_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if self.args.select_do_classification:
            no_labels = (
                torch.zeros(len(minibatch), self.args.sequence_length - 1, device=device, dtype=torch.long) 
                - 
                100
            )
            last_token_labels = self.dataset_labels[minibatch][:, None]
            inputs["labels"] = torch.cat([no_labels, last_token_labels], dim=1)
        else:
            inputs["labels"] = inputs["input_ids"].detach().clone()
        outputs = self.base_model(**inputs)
        loss = outputs.loss
        loss.backward()

        base_model_grad_norm = torch.cat([p.grad.flatten().detach().norm(dim=0, p=2, keepdim=True) for p in self.base_model.parameters()], dim=0).norm(dim=0, p=2, keepdim=True)
        base_model_grad_norm = base_model_grad_norm.detach().cpu().item()

        # Calculate gradient direction and step size
        with torch.no_grad():
            grad = torch.cat([p.grad.flatten().detach() for p in self.base_model.parameters()], dim=0)
            grad_direction = grad / (grad.norm(dim=0, p=2, keepdim=True) + 1e-10)
            grad_step_size = (grad_direction.double() @ base_params_diff).float()

        # Create optimizer and take step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.base_model.zero_grad()
        
        # Report metrics
        metrics = {
            "param_mse": current_mse,
            "ce_loss": loss.detach().item(),
            "grad_step_size": grad_step_size.item() if not torch.isnan(grad_step_size) else 0.0,
            "synth_lr": self.syn_lr.detach().item(),
            "base_model_grad_norm": base_model_grad_norm,
        }
        
        torch.cuda.empty_cache()
        
        self.best_idx_counter.update(minibatch)

        top_10_selected = self.best_idx_counter.most_common(10)
        print(f"Top-10 most-selected indices: {top_10_selected}")

        if current_mse < self.best_mse:
            self.steps_since_last_improvement = 0
            self.best_mse = current_mse
        else:
            self.steps_since_last_improvement += 1

        # if self.steps_since_last_improvement > (2 * self.args.select_steps_per_grad):
        #     print(f"Stopping distillation early since no improvement for {2 * self.args.select_steps_per_grad} steps")
        #     self.should_stop = True

        return metrics

    def step(self, it: int, buffer: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.args.select_do_classification and self.dataset_labels is None:
            # Get labels from expert model
            expert_state_dict = buffer[-1]
            expert_state_dict["lm_head.weight"] = expert_state_dict["transformer.wte.weight"].to(device)
            expert_model = copy.deepcopy(self.base_model)
            expert_model.load_state_dict(expert_state_dict)
            self.dataset_labels = autolabel_dataset(
                dataset=self.dataset, 
                model=expert_model, 
                tokenizer=self.tokenizer, 
                sequence_length=self.args.sequence_length,
            )
            expert_state_dict.pop("lm_head.weight")
            
        metrics = self.step_with_grad(it, buffer)
        X_tokens = torch.stack([
            self._tokenize_dataset_cached(i)
            for i in self.batch
        ])
        self.Y = self.dataset_labels[self.batch]
        return X_tokens, metrics

    def _tokenize_dataset_cached(self, i: int) -> torch.Tensor:
        """Tokenize dataset entry with caching.
        
        Args:
            i: Index of dataset entry to tokenize
            
        Returns:
            Tokenized text as tensor
        """
        if i not in self.tokenization_cache:
            text = self.dataset[i][self.dataset_text_column_name]
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length", 
                truncation=True,
                max_length=self.args.sequence_length
            )["input_ids"].squeeze(0)
            self.tokenization_cache[i] = tokens
        return self.tokenization_cache[i]
