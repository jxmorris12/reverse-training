from .optimizer import DiscreteOptimizer

import collections
import copy
import random
import math
import torch
import tqdm
from utils import (
    autolabel_dataset,
    device, 
    get_model,
    ClassificationDataset,
)
from utils.projection import (
    CudaProjector,
    ProjectionType,
    get_grads_final_layer
)
from utils.vector_search import (
    BatchedExactVectorDatabase,
)

def _create_pseudoexperts(model: torch.nn.Module, full_base_params: torch.Tensor, full_expert_model_params: torch.Tensor, num_pseudoexperts: int) -> list[tuple[torch.nn.Module, int]]:
    """Create pseudoexperts by interpolating between base and expert models.
    
    Args:
        model: The model to update with pseudoexpert parameters
        full_base_params: Flattened parameters of the base model
        full_expert_model_params: Flattened parameters of the expert model
        num_pseudoexperts: Number of pseudoexperts to create
        
    Returns:
        List of tuples containing (model, idx) for each pseudoexpert
    """
    pseudoexperts = []
    
    # Validate parameter counts
    model_param_count = sum(p.numel() for p in model.parameters())
    if full_base_params.numel() != model_param_count or full_expert_model_params.numel() != model_param_count:
        raise ValueError(f"Parameter count mismatch: model has {model_param_count} parameters, "
                         f"but got {full_base_params.numel()} base params and {full_expert_model_params.numel()} expert params")
    
    for i in range(num_pseudoexperts):
        step_frac = i / num_pseudoexperts
        print(f"[create_pseudoexperts] Pseudoexpert {i}/{num_pseudoexperts} -> step_frac = {step_frac}")
        
        # Interpolate linearly from base model to expert model
        pseudoexpert_params = (
            (1 - step_frac) * full_base_params + step_frac * full_expert_model_params
        )
        
        # Create a new model copy and set its parameters
        counter = 0
        new_model = copy.deepcopy(model)
        for p in new_model.parameters():
            p.data = pseudoexpert_params[counter:counter+p.numel()].reshape(p.data.shape).to(p.data.dtype)
            counter += p.numel()
        pseudoexperts.append((new_model, i))
        
    return pseudoexperts

class SELECTOptimizer(DiscreteOptimizer):
    X: torch.Tensor
    Y: torch.Tensor
    syn_lr: torch.Tensor

    def __init__(
            self, 
            args, 
            X: torch.Tensor, 
            Y: torch.Tensor, 
            tokenizer, 
            student_net, 
            initial_student_net,
            true_classification_dataset: ClassificationDataset):
        super().__init__(args)
        self.tokenizer = tokenizer
        self.student_net = student_net
        self.initial_student_net = initial_student_net
        self.base_model = get_model("gpt2")

        self.Y = Y

        syn_lr = torch.tensor(self.args.lr_teacher).to(device)
        self.syn_lr = syn_lr.detach().to(device).requires_grad_(True)

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
        self.optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.args.select_lr_student)
        self.steps_since_last_improvement = 0
        self.best_mse = float("inf")
        self.tokenization_cache = {}
        self.Y = None

        # Load search dataset
        self.true_classification_dataset = true_classification_dataset
        self.seed_dataset = ClassificationDataset.from_dataset_name(self.args.select_seed_dataset, seed=self.args.seed)
        self.seed_dataset_train_split = self.seed_dataset.dataset["train"]
        print(f"SELECTOptimizer: dataset size: {len(self.seed_dataset_train_split)}")
        self.dataset_autolabels = None
    
    
    def _rank_dataset_by_influence(
            self, 
            full_base_params, full_expert_model_params,
            last_layer_base_params, last_layer_expert_model_params
        ) -> list:
        """Rank dataset examples by influence on optimization direction"""
        assert self.args.select_full_dataset_size <= len(self.seed_dataset_train_split), \
            f"select_full_dataset_size: {self.args.select_full_dataset_size} must be less than dataset size: {len(self.seed_dataset_train_split)}"
    

        # If we're using random batch fill strategy, we can already return the random batch
        if self.args.select_batch_fill_strategy == "random":
            return random.sample(range(len(self.seed_dataset_train_split)), k=self.args.select_full_dataset_size)

        # Get all gradients
        base_model = copy.deepcopy(self.base_model)
        grads = []
        
        # Use the new helper function to create pseudoexperts
        pseudoexperts = _create_pseudoexperts(
            base_model, 
            full_base_params, 
            full_expert_model_params, 
            num_pseudoexperts=self.args.select_num_pseudoexperts
        )
        for model, i in pseudoexperts:
            batch_grads = get_grads_final_layer(
                model, 
                self.seed_dataset_train_split, 
                self.dataset_autolabels,
                self.tokenizer, 
                self.projector,
                sequence_length=self.args.sequence_length, 
                use_cache=True,
                model_cache_key=self.base_model.config._name_or_path + f"_{i}_{self.args.select_num_pseudoexperts}",
            )
            grads.append(batch_grads)
        
        grads = torch.stack(grads, dim=0).double().sum(dim=0).float()
        grads_db = BatchedExactVectorDatabase(grads)
        
        # Sanity check dimensions
        projection_dim = self.args.select_projection_dim
        assert grads.shape == (len(self.seed_dataset_train_split), projection_dim), \
            f"grads.shape: {grads.shape} != (len(self.seed_dataset_train_split), projection_dim): {(len(self.seed_dataset_train_split), projection_dim)}"
        
        # Fill batch using greedy algorithm
        if self.args.select_batch_fill_strategy == "greedy":
            batch = self._fill_batch_greedy(
                grads_db=grads_db,
                last_layer_base_params=last_layer_base_params,
                last_layer_expert_model_params=last_layer_expert_model_params,
                pseudoexperts=pseudoexperts,
            )
        elif self.args.select_batch_fill_strategy == "topk":
            batch = self._fill_batch_topk(
                grads_db=grads_db,
                last_layer_base_params=last_layer_base_params,
                last_layer_expert_model_params=last_layer_expert_model_params,
            )
        elif self.args.select_batch_fill_strategy == "topk_balanced":
            batch = self._fill_batch_topk(
                grads_db=grads_db,
                last_layer_base_params=last_layer_base_params,
                last_layer_expert_model_params=last_layer_expert_model_params,
                balanced=True,
            )
        elif self.args.select_batch_fill_strategy == "bottomk":
            batch = self._fill_batch_topk(
                grads_db=grads_db,
                last_layer_base_params=last_layer_base_params,
                last_layer_expert_model_params=last_layer_expert_model_params,
                reverse=True,
            )
        elif self.args.select_batch_fill_strategy == "random":
            batch = random.choices(range(len(self.seed_dataset_train_split)), k=self.args.select_full_dataset_size)
        else:
            raise ValueError(f"Invalid batch fill strategy: {self.args.select_batch_fill_strategy}")
        
        print(f"Picked new batch of size {len(batch)}: {sorted(batch)[:5]}...{sorted(batch)[-5:]}")
        
        return batch

    def _fill_batch_topk(
            self, 
            grads_db: BatchedExactVectorDatabase,
            last_layer_base_params: torch.Tensor,
            last_layer_expert_model_params: torch.Tensor,
            reverse: bool = False,
            balanced: bool = False,
        ) -> list:
        """Fill a batch with the top k examples that have the most influence on optimization direction."""
        base_params_diff_projected = self.projector.project(
            (last_layer_base_params - last_layer_expert_model_params), 
            model_id=0
        )
        
        # Use cosine similarity to find best gradient
        base_params_diff_norm = base_params_diff_projected / (base_params_diff_projected.norm(dim=1, p=2, keepdim=True) + 1e-10)
        base_params_diff_norm = base_params_diff_norm.to(device)
        
        # Get the best remaining gradient
        if balanced:
            # Optionally we balance by label to avoid over-representing any one label
            default_label_map = self.true_classification_dataset.label_map
            tokenized_labels = [self.tokenizer.encode(f" {x}")[0] for x in default_label_map]
            best_idxs = []
            label_counts = { label: 0 for label in tokenized_labels }
            selected_data_by_label = { label: [] for label in tokenized_labels }
            label_is_removed = { label: False for label in tokenized_labels }
            pbar = tqdm.trange(self.args.select_full_dataset_size, desc="Filling and balancing batch")
            max_per_label = math.ceil(self.args.select_full_dataset_size / len(tokenized_labels))
            last_best_sim_length = 0
            while len(best_idxs) < self.args.select_full_dataset_size:
                num_to_fill = self.args.select_full_dataset_size - len(best_idxs)
                tqdm.tqdm.write(f"Searching for {num_to_fill} examples (len(best_idxs): {len(best_idxs)} / label_counts: {label_counts} / label_is_removed: {label_is_removed})")
                _, best_idxs_full = grads_db.search(base_params_diff_norm, num_to_fill)
                for best_idx in best_idxs_full.cpu().tolist():
                    label = self.dataset_autolabels[best_idx].item()
                    if label_counts[label] < max_per_label:
                        best_idxs.append(best_idx)
                        label_counts[label] += 1
                        selected_data_by_label[label].append(best_idx)

                # remove vectors from search from all datapoints w/ labels that are over max_per_label
                for label in tokenized_labels:
                    if (label_counts[label] >= max_per_label) and (not label_is_removed[label]):
                        label_is_removed[label] = True
                        all_datapoints_within_label = (self.dataset_autolabels == label).nonzero().flatten() 
                        grads_db.remove_vectors(all_datapoints_within_label)
                        tqdm.tqdm.write(f"Filled quota for label {label}; Removed {len(all_datapoints_within_label)} datapoints")
                
                grads_db.remove_vectors(torch.tensor(best_idxs))
                num_added = len(best_idxs) - last_best_sim_length
                last_best_sim_length = len(best_idxs)
                pbar.update(num_added)
            best_idxs = torch.tensor(best_idxs)
        else:
            best_sims, best_idxs = grads_db.search(base_params_diff_norm, self.args.select_full_dataset_size)
        if reverse:
            best_sims = best_sims.flip(dims=[0])
            best_idxs = best_idxs.flip(dims=[0])
        return best_idxs.tolist()


    def _fill_batch_greedy(
            self, 
            grads_db: BatchedExactVectorDatabase,
            last_layer_base_params: torch.Tensor,
            last_layer_expert_model_params: torch.Tensor,
            pseudoexperts: list[tuple[torch.nn.Module, int]],
            grad_recompute_steps: int = 32,
        ) -> list:
        """Greedily fill a batch with examples that have the most influence on optimization direction.
        
        Args:
            grads_db: Database of projected gradients for all examples
            last_layer_base_params: Parameters of the last layer of the base model
            last_layer_expert_model_params: Parameters of the last layer of the expert model
            
        Returns:
            List of indices of selected examples
        """
        batch = []
        best_sim = float("-inf")
        # Project parameter difference
        base_params_diff_projected = self.projector.project(
            (last_layer_base_params - last_layer_expert_model_params), 
            model_id=0
        )

        og_grads_db_vectors = grads_db.vectors.clone()
        full_resolution_batch_gradient = torch.zeros_like(last_layer_base_params)
        batch_pbar = tqdm.trange(0, self.args.select_full_dataset_size, disable=(self.args.select_full_dataset_size < 32))
        overall_best_sim = float("-inf")
        while len(batch) < self.args.select_full_dataset_size:
            # Use cosine similarity to find best gradient
            best_sim, best_idx = grads_db.search(base_params_diff_projected, 1)
            best_idx = best_idx.item()
            best_sim = best_sim.item()
            overall_best_sim = max(overall_best_sim, best_sim)

            # Add the selected gradient to our batch
            batch.append(best_idx)
            grads_db.remove_vectors(best_idx)

            # Compute full-resolution gradient
            grads_db.vectors += og_grads_db_vectors[None, best_idx]

            if len(batch) % grad_recompute_steps == 0:
                last_n_batch = batch[-grad_recompute_steps:]
                full_grad = None
                for model, _ in pseudoexperts:
                    batch_grad = get_grads_final_layer(
                        model, 
                        self.seed_dataset_train_split.select(last_n_batch), 
                        self.dataset_autolabels[last_n_batch],
                        self.tokenizer, 
                        self.projector, 
                        sequence_length=self.args.sequence_length,
                        do_projection=False,
                        use_cache=False,
                    )
                    if full_grad is None:
                        full_grad = batch_grad.sum(dim=0)
                    else:
                        full_grad += batch_grad.sum(dim=0)
                
                full_resolution_batch_gradient += full_grad.to(device)
                full_resolution_batch_gradient_proj = self.projector.project(full_resolution_batch_gradient, model_id=0)
                grads_db.vectors = og_grads_db_vectors.clone() + full_resolution_batch_gradient_proj

            batch_pbar.update(1)
            batch_pbar.set_description(f"Best sim: {best_sim:.3f} | Best idx: {best_idx} | Batch size: {len(batch)} | Current sim: {best_sim:.3f} | Overall best sim: {overall_best_sim:.3f}")

        label_distribution = self.dataset_autolabels[batch].unique(return_counts=True)
        print(f"[SELECTOptimizer._fill_batch_greedy] Overall best sim: {overall_best_sim:.3f} | Label distribution: {label_distribution}")
        return batch

    def step_with_grad(self, step: int, buffer: list) -> dict[str, torch.Tensor]:
        """Perform one step of SELECT optimization"""
        # Get current model parameters
        full_base_params = torch.cat([p.flatten().cpu().detach().requires_grad_(False) for p in self.base_model.parameters()]).double().to(device)
        last_layer_base_params = torch.cat([p.flatten().cpu().detach().requires_grad_(False) for p in self.base_model.lm_head.parameters()]).double().to(device)
        
        # Get expert model parameters from buffer
        expert_state_dict = buffer[-1]
        full_expert_model_params = torch.cat([v.flatten() for k, v in expert_state_dict.items()]).double().to(device)
        last_layer_expert_model_params = torch.cat([v.flatten() for k, v in expert_state_dict.items() if "wte" in k]).double().to(device)
        
        # Update batch if needed
        should_update_batch = (self.args.select_steps_per_grad > 0) and (step % self.args.select_steps_per_grad == 0)
        if (not len(self.batch)) or should_update_batch:
            self.base_model.to(device)
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
            self.seed_dataset_train_split.select(minibatch)["text"],
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
            last_token_labels = self.dataset_autolabels[minibatch][:, None].to(device)
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
        # top_10_selected = self.best_idx_counter.most_common(10)

        if current_mse < self.best_mse:
            self.steps_since_last_improvement = 0
            self.best_mse = current_mse
        else:
            self.steps_since_last_improvement += 1

        return metrics

    def step(self, step: int, buffer: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.args.select_do_classification and self.dataset_autolabels is None:
            # Get labels from expert model
            expert_state_dict = buffer[-1]
            expert_state_dict["lm_head.weight"] = expert_state_dict["transformer.wte.weight"].to(device)
            expert_model = copy.deepcopy(self.base_model)
            expert_model.load_state_dict(expert_state_dict)
            if self.args.select_label_strategy == "auto":
                self.dataset_autolabels = autolabel_dataset(
                    dataset=self.seed_dataset_train_split,
                    model=expert_model, 
                    tokenizer=self.tokenizer, 
                    sequence_length=self.args.sequence_length,
                    label_map=self.true_classification_dataset.label_map,
                )
            elif self.args.select_label_strategy == "random":
                self.dataset_autolabels = torch.randint(
                    0, 
                    len(self.true_classification_dataset.label_map), 
                    (len(self.seed_dataset_train_split),),
                    device=device,
                )
            else:
                raise ValueError(f"Invalid label strategy: {self.args.select_label_strategy}")
            expert_state_dict.pop("lm_head.weight")
            
        metrics = self.step_with_grad(step, buffer)
        X_tokens = torch.stack([
            self._tokenize_dataset_cached(i)
            for i in self.batch
        ])
        self.Y = self.dataset_autolabels[self.batch]
        return X_tokens, metrics

    def _tokenize_dataset_cached(self, i: int) -> torch.Tensor:
        """Tokenize dataset entry with caching.
        
        Args:
            i: Index of dataset entry to tokenize
            
        Returns:
            Tokenized text as tensor
        """
        if i not in self.tokenization_cache:
            text = self.seed_dataset_train_split[i][
                self.seed_dataset.text_column_name
            ]
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length", 
                truncation=True,
                max_length=self.args.sequence_length
            )["input_ids"].squeeze(0)
            self.tokenization_cache[i] = tokens
        return self.tokenization_cache[i]
