from .optimizer import DiscreteOptimizer
import datasets
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
from utils.core import ExpertModel
from utils.projection import (
    CudaProjector,
    ProjectionType,
    get_grads_final_layer,
    get_grads_full_model,
)
from utils.vector_search import (
    BatchedExactVectorDatabase,
)


class SELECTOptimizer(DiscreteOptimizer):
    X: torch.Tensor
    Y: torch.Tensor
    syn_lr: torch.Tensor
    expert_model: ExpertModel

    def __init__(
            self, 
            args, 
            expert_model: ExpertModel,
            X: torch.Tensor, 
            Y: torch.Tensor, 
            tokenizer, 
            student_net, 
            initial_student_net,
            true_classification_dataset: ClassificationDataset
        ):
        super().__init__(args)
        self.tokenizer = tokenizer
        self.base_model = get_model(args.base_model_name_or_path)
        self.expert_model = expert_model
        self.Y = Y

        # Setup projector
        param_count = sum(p.numel() for p in self.base_model.lm_head.parameters())
        self.projector = CudaProjector(
            grad_dim=param_count,
            proj_dim=args.select_projection_dim,
            seed=0,
            block_size=256,
            proj_type=ProjectionType.rademacher,
            device=device,
            max_batch_size=256,
        )
        
        self.best_idx_counter = collections.Counter()
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

    def _run_create_pseudoexperts(
        self,
        model: torch.nn.Module, 
        expert_model: torch.nn.Module,
        full_base_params: torch.Tensor, 
        full_expert_model_params: torch.Tensor, 
        num_pseudoexperts: int,
        dataset: datasets.Dataset,
    ) -> list[tuple[torch.nn.Module, int]]:
        """Create pseudoexperts by interpolating from base to expert models.
        
        Args:
            model: The model to update with pseudoexpert parameters
            full_base_params: Flattened parameters of the base model
            full_expert_model_params: Flattened parameters of the expert model
            num_pseudoexperts: Number of pseudoexperts to create
            
        Returns:
            List of tuples containing (model, idx) for each pseudoexpert
        """
        if self.args.select_num_pseudoexperts <= 1:
            return [(model, 0)]

        pseudoexperts = []

        label_mask = (
            expert_model.all_labels_ids[:, None].to(device) == torch.arange(self.tokenizer.vocab_size, device=device)[None, :]
        ).any(dim=0)
        num_steps_per_pseudoexpert = 10
        # optim = torch.optim.SGD(model.parameters(), lr=1e-4)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        for i in range(num_pseudoexperts):
            for j in range(num_steps_per_pseudoexpert):
                # Classify batch
                batch = dataset.select(random.sample(range(len(dataset)), k=self.args.expert_batch_size))
                inputs = self.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.args.sequence_length-1,
                ).to(device)
                base_model_outputs = model(**inputs)
                with torch.no_grad():
                    expert_model_outputs = expert_model(**inputs)
                
                # Compute KL divergence between base and expert model outputs
                base_logits = base_model_outputs.logits[:, -1, :]
                expert_logits = expert_model_outputs.logits[:, -1, :]

                base_logits = base_logits[:, label_mask]
                expert_logits = expert_logits[:, label_mask]

                base_logprobs = base_logits.log_softmax(dim=-1)
                expert_logprobs = expert_logits.log_softmax(dim=-1)
                kl_div = torch.nn.functional.kl_div(
                    base_logprobs, 
                    expert_logprobs, 
                    log_target=True,
                    reduction="batchmean",
                )
                # Compute MSE between parameters
                mse = torch.tensor(0.0, device=device)
                for base_param, expert_param in zip(model.parameters(), expert_model.parameters()):
                    mse += torch.nn.functional.mse_loss(base_param, expert_param.detach(), reduction="sum")

                # loss = kl_div + mse
                loss = (mse + kl_div)
                loss.backward()
                optim.step()
                optim.zero_grad()
                
            print(f"Step {i*num_steps_per_pseudoexpert} | Pseudoexpert {i} | kl_div: {kl_div} | mse: {mse}")
            # add pseudoexpert to list
            pseudoexpert_model = copy.deepcopy(model).cpu()
            pseudoexperts.append((pseudoexpert_model, i))
            
        
        print(f"[SELECTOptimizer._run_create_pseudoexperts] Created {num_pseudoexperts} pseudoexperts | Final loss: {loss} | kl_div: {kl_div} | mse: {mse}")
        return pseudoexperts
        
    def _rank_dataset_by_influence(
            self,
            expert_state_dict,
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

        if self.args.select_do_warmup:
            self._run_warmup(base_model, last_layer_expert_model_params)
        
        # Create pseudoexperts by trajectory matching with random data
        expert_model = copy.deepcopy(self.base_model)
        expert_model.load_state_dict(self._restore_model_state_dict(expert_state_dict))

        expert_model_num_points = self.args.expert_batch_size * self.args.expert_epochs
        random_idxs = random.sample(range(len(self.seed_dataset_train_split)), k=expert_model_num_points)

        pseudoexperts = self._run_create_pseudoexperts(
            base_model, 
            expert_model=expert_model,
            full_base_params=full_base_params, 
            full_expert_model_params=full_expert_model_params, 
            num_pseudoexperts=self.args.select_num_pseudoexperts,
            dataset=self.seed_dataset_train_split.select(random_idxs),
        )

        get_grads = get_grads_full_model if self.args.select_grads_full_model else get_grads_final_layer
        for model, i in pseudoexperts:
            # TODO: Add other expert hparams to model_cache_key – lr, steps, batch size.
            model_cache_key = (
                self.base_model.config._name_or_path
                + self.args.dataset
                + f"_{self.args.expert_batch_size}"
                + f"_{self.args.expert_epochs}"
                + f"_{self.args.select_num_pseudoexperts}"
                + f"_{self.args.select_lr_student}"
                + f"_{self.args.select_steps_per_grad}"
                + f"_{self.args.select_do_warmup}"
                + f"_{self.args.select_do_classification}"
            )
            if self.args.select_do_warmup:
                model_cache_key += f"_warmup"
            print(f"[SELECTOptimizer._rank_dataset_by_influence] Getting grads for model {i} | model_cache_key: {model_cache_key}")

            if self.args.select_use_expert_grads:
                batch_grads = get_grads(
                    expert=self.expert_model,
                    dataset=self.seed_dataset_train_split, 
                    labels=self.dataset_autolabels,
                    projector=self.projector,
                    use_cache=False,
                    do_projection=True,
                    model_cache_key=model_cache_key + "_expert_grads",
                )
            else:
                batch_grads = get_grads(
                    expert=self.expert_model, 
                    dataset=self.seed_dataset_train_split, 
                    labels=self.dataset_autolabels,
                    projector=self.projector,
                    use_cache=False,
                    do_projection=True,
                    model_cache_key=model_cache_key,
                )
            grads.append(batch_grads)
        
        grads = torch.stack(grads, dim=0)
        grads_db = BatchedExactVectorDatabase(grads)
        
        # Fill batch using greedy algorithm
        base_params = full_base_params if self.args.select_grads_full_model else last_layer_base_params
        expert_params = full_expert_model_params if self.args.select_grads_full_model else last_layer_expert_model_params

        if self.args.select_batch_fill_strategy == "greedy":
            batch = self._fill_batch_greedy(
                grads_db=grads_db,
                base_params=base_params,
                expert_params=expert_params,
                pseudoexperts=pseudoexperts,
            )
        elif self.args.select_batch_fill_strategy == "greedy_batched":
            batch = self._fill_batch_greedy(
                grads_db=grads_db,
                base_params=base_params,
                expert_params=expert_params,
                pseudoexperts=pseudoexperts,
                batched=True,
            )
        elif self.args.select_batch_fill_strategy == "topk":
            batch = self._fill_batch_topk(
                grads_db=grads_db,
                base_params=base_params,
                expert_params=expert_params,
            )
        elif self.args.select_batch_fill_strategy == "topk_balanced":
            batch = self._fill_batch_topk(
                grads_db=grads_db,
                base_params=base_params,
                expert_params=expert_params,
                balanced=True,
            )
        elif self.args.select_batch_fill_strategy == "bottomk":
            batch = self._fill_batch_topk(
                grads_db=grads_db,
                base_params=base_params,
                expert_params=expert_params,
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
            base_params: torch.Tensor,
            expert_params: torch.Tensor,
            reverse: bool = False,
            balanced: bool = False,
        ) -> list:
        """Fill a batch with the top k examples that have the most influence on optimization direction."""
        base_params_diff_projected = self.projector.project(
            (base_params - expert_params), 
            model_id=0
        )
        
        # Use cosine similarity to find best gradient
        base_params_diff_norm = base_params_diff_projected / (base_params_diff_projected.norm(dim=1, p=2, keepdim=True) + 1e-10)
        base_params_diff_norm = base_params_diff_norm.to(device)
        
        # Get the best remaining gradient
        if balanced:
            # Optionally we balance by label to avoid over-representing any one label
            tokenized_labels = set(self.dataset_autolabels.flatten().tolist())
            label_counts = { label: 0 for label in tokenized_labels }
            selected_data_by_label = { label: [] for label in tokenized_labels }
            label_is_removed = { label: False for label in tokenized_labels }
            pbar = tqdm.trange(self.args.select_full_dataset_size, desc="Filling and balancing batch")
            max_per_label = math.ceil(self.args.select_full_dataset_size / len(tokenized_labels))
            last_best_sim_length = 0
            best_idxs = []
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
            base_params: torch.Tensor,
            expert_params: torch.Tensor,
            pseudoexperts: list[tuple[torch.nn.Module, int]],
            grad_recompute_steps: int = 32,
            batched: bool = False,
        ) -> list:
        """Greedily fill a batch with examples that have the most influence on optimization direction.
        
        Args:
            grads_db: Database of projected gradients for all examples
            base_params: Parameters of the last layer of the base model
            expert_params: Parameters of the last layer of the expert model
            
        Returns:
            List of indices of selected examples
        """
        batch = []
        best_sim = float("-inf")
        assert base_params.shape == expert_params.shape, f"base_params.shape: {base_params.shape} != expert_params.shape: {expert_params.shape}"

        # TODO: Make the following work w/ just last layer grads.
        # Project parameter differences
        # pseudoexperts_projected_diffs = []
        # last_params = torch.cat([p.flatten().detach() for p in self.base_model.parameters()])
        # for model, i in pseudoexperts:
        #     model_params = torch.cat([p.flatten().detach() for p in model.parameters()])
        #     model_params_diff = model_params - last_params
        #     model_params_diff_projected = self.projector.project(
        #         model_params_diff, 
        #         model_id=0
        #     )
        #     pseudoexperts_projected_diffs.append(model_params_diff_projected)
        #     last_params = model_params
        # pseudoexperts_projected_diffs = torch.stack(pseudoexperts_projected_diffs, dim=0)

        params_diff_projected = self.projector.project(base_params - expert_params, model_id=0)
        # Split grads_db per-label
        og_grads_db_vectors = grads_db.vectors.cpu().clone()
        batch_pbar = tqdm.trange(0, self.args.select_full_dataset_size, disable=(self.args.select_full_dataset_size < 32))
        overall_best_sim = float("-inf")
        while len(batch) < self.args.select_full_dataset_size:
            # Use cosine similarity to find best gradient
            best_sim, best_idx = grads_db.search(params_diff_projected, 1)
            best_idx = best_idx.item()
            best_sim = best_sim.item()
            overall_best_sim = max(overall_best_sim, best_sim)

            assert best_idx not in batch, f"Best idx {best_idx} is already in batch"

            # Add the selected gradient to our batch
            grads_db.remove_vectors(best_idx)
            batch.append(best_idx)
            # Compute full-resolution gradient
            # TODO: get actually best_idx (don't just set to 0)

            this_grads_db_vector = og_grads_db_vectors[:, best_idx].mean(dim=0, keepdim=True)
            grads_db.vectors += this_grads_db_vector

            if batched:
                if len(batch) % self.args.expert_batch_size == 0:
                    batch_grads_list = []
                    # reset vectors
                    grads_db.vectors = og_grads_db_vectors.clone()

            batch_pbar.update(1)
            batch_pbar.set_description(f"Best sim: {best_sim:.3f} | Best idx: {best_idx} | Batch size: {len(batch)} | Current sim: {best_sim:.3f} | Overall best sim: {overall_best_sim:.3f}")

        label_distribution = collections.Counter([self.dataset_autolabels[j] for j in batch])
        print(f"[SELECTOptimizer._fill_batch_greedy] Overall best sim: {overall_best_sim:.3f} | Label distribution: {label_distribution}")
        return batch
    
    def _restore_model_state_dict(self, expert_state_dict: dict) -> dict:
        """Prepare expert model state dict for use in SELECT."""
        is_gpt2 = "gpt2" in self.args.base_model_name_or_path
        if is_gpt2:
            expert_state_dict["lm_head.weight"] = expert_state_dict["transformer.wte.weight"]
        return expert_state_dict
    
    def _flatten_model_params(self, model_state_dict: dict) -> torch.Tensor:
        """Flatten model parameters."""
        is_gpt2 = "gpt2" in self.args.base_model_name_or_path
        forbidden_keys = {}
        if is_gpt2 and "transformer.wte.weight"  in model_state_dict:
            forbidden_keys = {"transformer.wte.weight"}
        return torch.cat([v.flatten() for k, v in model_state_dict.items() if k not in forbidden_keys]).cpu().double()
    
    def _get_last_layer_params(self, model_state_dict: dict) -> torch.Tensor:
        """Get last layer parameters."""
        is_gpt2 = "gpt2" in self.args.base_model_name_or_path
        if is_gpt2:
            return torch.cat([v.flatten() for k, v in model_state_dict.items() if "wte" in k]).cpu().double()
        else:
            return torch.cat([v.flatten() for k, v in model_state_dict.items() if "lm_head" in k]).cpu().double()

    def _get_batch_idxs(self, step: int, buffer: list) -> list[int]:
        """Perform one step of SELECT optimization"""
        # Get current model parameters
        full_base_params = self._flatten_model_params(self.base_model.state_dict())
        last_layer_base_params = self._get_last_layer_params(self.base_model.state_dict())
        
        # Get expert model parameters from buffer
        expert_state_dict = buffer[-1]
        full_expert_model_params = self._flatten_model_params(expert_state_dict)
        last_layer_expert_model_params = self._get_last_layer_params(expert_state_dict)
        
        # Update batch if needed
        return self._rank_dataset_by_influence(
            expert_state_dict=expert_state_dict,
            full_base_params=full_base_params, 
            full_expert_model_params=full_expert_model_params,
            last_layer_base_params=last_layer_base_params, 
            last_layer_expert_model_params=last_layer_expert_model_params
        )

    def step(self, step: int, buffer: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.args.select_do_classification and self.dataset_autolabels is None:
            # Get labels from expert model
            if self.args.select_label_strategy == "auto":
                self.dataset_autolabels = autolabel_dataset(
                    expert=self.expert_model,
                    dataset=self.seed_dataset_train_split,
                )
            elif self.args.select_label_strategy == "random":
                tokenized_labels = torch.tensor(self.expert_model.all_labels_ids)
                dataset_autolabels_idxs = torch.randint(
                    0, 
                    len(tokenized_labels), 
                    (len(self.seed_dataset_train_split),),
                    device="cpu",
                )
                self.dataset_autolabels = tokenized_labels[dataset_autolabels_idxs.cpu()]
            else:
                raise ValueError(f"Invalid label strategy: {self.args.select_label_strategy}")
        
        batch = self._get_batch_idxs(step, buffer)
        X = self.seed_dataset_train_split.select(batch)["text"]
        X_tokens = torch.stack([
            self._tokenize_dataset_cached(i) for i in batch
        ])
        
        Y = [self.dataset_autolabels[i] for i in batch]
        Y_set = set(Y)
        assert Y_set <= set(self.expert_model.all_labels), f"Y: {Y_set} is not a subset of expert_model.all_labels: {set(self.expert_model.all_labels)}"

        return X, X_tokens.cpu(), Y, {}

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
