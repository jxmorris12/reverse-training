import hashlib
import os
from enum import Enum

import datasets
import numpy as np
import torch
import tqdm


from utils.core import device, hash_model_params, _get_cache_key
from utils.batch import find_executable_batch_size
from utils.core import ExpertModel

def vectorize(grads: dict, device=None) -> torch.Tensor:
    """Convert a dictionary of gradients to a flat tensor."""
    return torch.cat([p.flatten() for p in grads.values()], dim=0).to(device)

class ProjectionType(str, Enum):
    rademacher = "rademacher"
    normal = "normal"


class CudaProjector:
    """A performant implementation of the projection for CUDA."""
    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device,
        max_batch_size: int = 256,
        block_size: int = 512,
    ) -> None:
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device
        self.max_batch_size = max_batch_size
        self.block_size = block_size

        self._dimension_lock = None
        
        if isinstance(device, str):
            device = torch.device(device)

        if device.type != "cuda":
            err = "CudaProjector only works on a CUDA device"
            raise ValueError(err)

        self.num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count

        try:
            import fast_jl
            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(
                torch.zeros(8, 1_000, device="cuda"), 512, 0, self.num_sms
            )
        except ImportError:
            err = "You should make sure to install the CUDA projector for traker (called fast_jl)."
            raise ModuleNotFoundError(err)

    def project(
        self,
        x: torch.Tensor,
        model_id: int,
    ) -> torch.Tensor:
        # Convert dict to tensor if needed
        if isinstance(x, dict):
            x = vectorize(x, device=self.device)

        x = x.to(self.device)
        
        # Optionally add a batch dimension
        if x.ndim == 1:
            x = x[None]

        batch_size = x.shape[0]
        dim = x.shape[1]

        if self._dimension_lock is None:
            print(f"[CudaProjector] Setting dimension lock to {dim}")
            self._dimension_lock = dim
        elif self._dimension_lock != dim:
            raise ValueError(f"Dimension of input to projector changed from {self._dimension_lock} to {dim}.")

        # Downcast from double to float
        if x.dtype == torch.float64:
            x = x.to(torch.float32)

        effective_batch_size = 32
        if batch_size <= 8:
            effective_batch_size = 8
        elif batch_size <= 16:
            effective_batch_size = 16

        effective_batch_size = min(self.max_batch_size, effective_batch_size)

        function_name = f"project_{self.proj_type.value}_{effective_batch_size}"
        import fast_jl

        fn = getattr(fast_jl, function_name)

        try:
            result = fn(
                x, self.proj_dim, self.seed + int(1e4) * model_id, self.num_sms
            )
        except RuntimeError as e:
            if "CUDA error: too many resources requested for launch" in str(e):
                # provide a more helpful error message
                raise RuntimeError(
                    (
                        "The batch size of the CudaProjector is too large for your GPU. "
                        "Reduce it by using the proj_max_batch_size argument."
                    )
                )
            else:
                print("Error for dtype: ", x.dtype, "and shape: ", x.shape)
                raise e

        return result

    def free_memory(self):
        """A no-op method."""
        pass

    def deterministic_hash(self) -> int:
        """
        Return a hash based on the essential properties of this CudaProjector.
        
        Returns:
            int: A hash value uniquely identifying this projector configuration.
        """
        # Use a tuple of the key properties that uniquely define a projector
        key_tuple = (
            self.grad_dim,
            self.proj_dim,
            self.seed,
            self.proj_type,
        )
        
        # Return the hash of this tuple
        return int(hashlib.md5(str(key_tuple).encode()).hexdigest(), 16)
        

@find_executable_batch_size
def _get_grads_final_layer_uncached(
        expert: ExpertModel, 
        dataset, 
        labels,
        projector, 
        do_projection: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
    """
    Computes gradients for each example in the dataset with respect to the model parameters.
    
    Args:
        expert: The expert model to compute gradients for
        dataset: The dataset containing examples
        labels: Optional labels for classification tasks
        projector: Projector for projecting gradients to a lower-dimensional space
        sequence_length: Maximum sequence length for tokenization
        do_projection: Whether to project the gradients
        batch_size: Number of examples to process at once
        
    Returns:
        A tensor of shape (len(dataset), projection_dim) containing the gradients
    """
    expert.student_net.eval()
    expert.student_net.to(device)
    
    do_classification = (labels is not None)
    all_grads = []
    all_losses = []
    pbar = tqdm.trange(0, len(dataset), batch_size, disable=(len(dataset) // batch_size < 10))
    
    # Get parameters for the final layer only
    params = {k: v.detach() for k, v in expert.student_net.lm_head.named_parameters()}
    buffers = {k: v.detach() for k, v in expert.student_net.lm_head.named_buffers()}
    
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset.select(range(batch_start, batch_end))
        
        if do_classification:
            batch_labels = [labels[i] for i in range(batch_start, batch_end)]
            batch = datasets.Dataset.from_dict(
                {
                    "text": batch["text"],
                    "label": batch_labels,
                }
            )
            examples = expert.prepare_examples(
                examples=batch,
                label_column_name="label",
                text_column_name="text"
            )
        else:
            batch = datasets.Dataset.from_dict(
                {
                    "text": batch["text"],
                }
            )
            examples = expert.prepare_examples(
                examples=batch,
                label_column_name=None,
                text_column_name="text"
            )
        
        # Get last hidden state
        with torch.no_grad():
            tokenized_text, outputs = expert.compute_outputs(
                examples=examples,
                output_hidden_states=True,
            )
            _, label_ids = expert.get_logits_and_labels(tokenized_text, outputs)
            last_hidden_state = outputs.hidden_states[-1]

        @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        def compute_loss(params, buffers, last_hidden_states, labels):
            logits = torch.func.functional_call(
                module=expert.student_net.lm_head,
                parameter_and_buffer_dicts=(params, buffers),
                args=(last_hidden_states,),
            )
            _, loss, _ = expert.compute_loss_and_accuracy(logits[None], labels[None], vmap=True)
            return loss
        
        ft_compute_grad_and_loss = torch.func.grad_and_value(compute_loss)
        ft_compute_sample_grad_and_loss = torch.func.vmap(ft_compute_grad_and_loss, in_dims=(None, None, 0, 0))

        # Compute per-sample gradients and losses
        ft_per_sample_grads, losses_batch = ft_compute_sample_grad_and_loss(params, buffers, last_hidden_state, label_ids)

        # Process gradients as before
        grads_batch = torch.cat([ft_per_sample_grads[n].reshape(batch_end - batch_start, -1) for n, _ in expert.student_net.lm_head.named_parameters()], dim=1)

        if do_projection:
            projected_grads = projector.project(grads_batch, model_id=0)
            all_grads.extend(projected_grads.cpu())
            all_losses.extend(losses_batch.cpu())
        else:
            all_grads.extend(grads_batch.cpu())
            all_losses.extend(losses_batch.cpu())
    
    expert.student_net.cpu()
    expert.student_net.train()
    
    return torch.stack(all_grads).to(torch.float16), torch.stack(all_losses).to(torch.float16)


def get_grads_final_layer(
        expert: ExpertModel, 
        dataset, 
        labels,
        projector, 
        do_projection: bool = True,
        use_cache: bool = False,
        model_cache_key: str = "",
    ) -> torch.Tensor:
    """Get model predictions for the sequence_length-th token for each example in the dataset.
    
    Args:
        dataset: Dataset to label
        model: Model to use for labeling
        tokenizer: Tokenizer for processing text
        sequence_length: Maximum sequence length for tokenization
        batch_size: Batch size for processing
        
    Returns:
        Tensor of shape (len(dataset),) containing the predicted token ids
    """
    use_cache = False
    if not use_cache:
        return _get_grads_final_layer_uncached(
            expert=expert, 
            dataset=dataset, 
            labels=labels, 
            projector=projector, 
            do_projection=do_projection,
        )
    hash_kwargs = {
        # TODO: hash labels...
        "student_net_hash": hash_model_params(expert.student_net),
        "dataset_hash": dataset._fingerprint,
        "tokenizer_hash": expert.tokenizer.name_or_path,
        "projector_hash": projector.deterministic_hash(),
        "do_projection": do_projection,
        "model_cache_key": model_cache_key,
    }
    cache_dir = os.path.join(os.path.dirname(__file__), os.pardir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    full_cache_key = _get_cache_key(**hash_kwargs)
    cache_key = hashlib.sha256(full_cache_key.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"get_grads_final_layer_{cache_key}.npz")

    try:
        grads = np.load(cache_path)["grads"]
    except:
        print(f"Getting grads for final layer with cache key: {full_cache_key} => {cache_key}")
        grads, losses = _get_grads_final_layer_uncached(
            expert=expert, 
            dataset=dataset,
            labels=labels, 
            projector=projector, 
            do_projection=do_projection,
        )
        np.savez(cache_path, grads=grads.numpy(), losses=losses.numpy())

    grads = np.load(cache_path)["grads"]
    losses = np.load(cache_path)["losses"]
    return torch.from_numpy(grads), torch.from_numpy(losses)


@find_executable_batch_size
def _get_grads_full_model_uncached(
        expert: ExpertModel, 
        dataset, 
        labels,
        projector, 
        sequence_length: int, 
        do_projection: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
    """
    Computes gradients for each example in the dataset with respect to the model parameters.
    
    Args:
        student_net: The model to compute gradients for
        dataset: The dataset containing examples
        tokenizer: Tokenizer for processing text examples
        projector: Projector for projecting gradients to a lower-dimensional space
        sequence_length: Maximum sequence length for tokenization
        batch_size: Number of examples to process at once
        do_projection: Whether to project the gradients
        
    Returns:
        A tensor of shape (len(dataset), projection_dim) containing the gradients
    """
    all_grads = []
    all_losses = []
    pbar = tqdm.trange(0, len(dataset), batch_size, disable=(len(dataset) // batch_size < 10))

    # print first datapoint
    print(f"[get_grads_full_model] First datapoint: {dataset[0]['text']}")
    print(f"[get_grads_full_model] First datapoint label: {labels[0]}")

    # helpful page: pytorch.org/tutorials/intermediate/per_sample_grads.html
    params = {k: v.detach() for k, v in expert.student_net.named_parameters()}
    buffers = {k: v.detach() for k, v in expert.student_net.named_buffers()}
    
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset.select(range(batch_start, batch_end))
        batch_labels = [labels[i] for i in range(batch_start, batch_end)]

        batch = datasets.Dataset.from_dict(
            {
                "text": batch["text"],
                "label": batch_labels,
            }
        )
    
        inputs = expert.prepare_examples(
            examples=batch,
            label_column_name="label",
            text_column_name="text"
        )
        inputs = { k: v.to(device) for k, v in inputs.items() }
        
        @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        def compute_loss(params, buffers, input_ids, attention_mask, labels):
            net_output = torch.func.functional_call(
                module=expert.student_net,
                parameter_and_buffer_dicts=(params, buffers),
                kwargs={"input_ids": input_ids[None], "attention_mask": attention_mask[None]},
            )[0]
            
            logits = net_output[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                labels.reshape(-1).to(device),
                ignore_index=-100,
                reduction="mean"
            )
            return loss
    
        ft_compute_grad_and_loss = torch.func.grad_and_value(compute_loss)
        ft_compute_sample_grad_and_loss = torch.func.vmap(ft_compute_grad_and_loss, in_dims=(None, None, 0, 0, 0))

        # Compute per-sample gradients and losses
        ft_per_sample_grads, losses_batch = ft_compute_sample_grad_and_loss(params, buffers, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])

        # Process gradients as before
        grads_batch = torch.cat([ft_per_sample_grads[n].reshape(batch_end - batch_start, -1) for n, _ in expert.student_net.named_parameters()], dim=1)
        grads_batch = grads_batch.double() / grads_batch.norm(dim=1, keepdim=True)
        all_losses.extend(losses_batch.double())

        if do_projection:
            projected_grads = projector.project(grads_batch, model_id=0)
            all_grads.extend(projected_grads.cpu())
        else:
            all_grads.extend(grads_batch.cpu())
    return torch.stack(all_grads), torch.stack(all_losses)


def get_grads_full_model(
        expert: ExpertModel, 
        dataset, 
        labels,
        projector, 
        do_projection: bool = True,
        use_cache: bool = True,
        model_cache_key: str = "",
    ) -> torch.Tensor:
    """Get model predictions for the sequence_length-th token for each example in the dataset.
    
    Args:
        dataset: Dataset to label
        model: Model to use for labeling
        tokenizer: Tokenizer for processing text
        sequence_length: Maximum sequence length for tokenization
        batch_size: Batch size for processing
        
    Returns:
        Tensor of shape (len(dataset),) containing the predicted token ids
    """
    if not use_cache:
        return _get_grads_full_model_uncached(
            expert=expert, 
            dataset=dataset, 
            labels=labels, 
            projector=projector, 
            do_projection=do_projection,
        )
    hash_kwargs = {
        "student_net_hash": hash_model_params(expert.student_net),
        "dataset_hash": dataset._fingerprint,
        "tokenizer_hash": expert.tokenizer.name_or_path,
        "projector_hash": projector.deterministic_hash(),
        "do_projection": do_projection,
        "model_cache_key": model_cache_key,
    }
    cache_dir = os.path.join(os.path.dirname(__file__), os.pardir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    full_cache_key = _get_cache_key(**hash_kwargs)
    cache_key = hashlib.sha256(full_cache_key.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"get_grads_full_model_{cache_key}.npz")

    try:
        grads = np.load(cache_path)["grads"]
    except:
        print(f"Getting grads for full model with cache key: {full_cache_key} => {cache_key}")
        grads = _get_grads_full_model_uncached(
            expert=expert, 
            dataset=dataset,
            labels=labels, 
            projector=projector, 
            do_projection=do_projection,
        )
        np.savez(cache_path, grads=grads.numpy())

    grads = np.load(cache_path)["grads"]
    return torch.from_numpy(grads)