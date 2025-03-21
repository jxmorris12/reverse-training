import torch
from enum import Enum
import tqdm

from utils.core import device

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
        block_size: int = 128,
    ) -> None:
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        
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
        grads: torch.Tensor,
        model_id: int,
    ) -> torch.Tensor:
        # Convert dict to tensor if needed
        if isinstance(grads, dict):
            grads = vectorize(grads, device=self.device)
        
        # Optionally add a batch dimension
        if grads.ndim == 1:
            grads = grads[None]

        batch_size = grads.shape[0]

        # Downcast from double to float
        if grads.dtype == torch.float64:
            grads = grads.to(torch.float32)

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
                grads, self.proj_dim, self.seed + int(1e4) * model_id, self.num_sms
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
                print("Error for dtype: ", grads.dtype, "and shape: ", grads.shape)
                raise e

        return result

    def free_memory(self):
        """A no-op method."""
        pass


def get_grads_final_layer(
        student_net, 
        dataset, 
        labels,
        tokenizer, 
        projector, 
        sequence_length: int, 
        batch_size: int, 
        do_projection: bool = True
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
    do_classification = (labels is not None)
    all_grads = []
    pbar = tqdm.trange(0, len(dataset), batch_size, disable=(batch_size < 32))

    # helpful page: pytorch.org/tutorials/intermediate/per_sample_grads.html
    params = {k: v.detach() for k, v in student_net.lm_head.named_parameters()}
    buffers = {k: v.detach() for k, v in student_net.lm_head.named_buffers()}
    
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset.select(range(batch_start, batch_end))
        
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
        )
        inputs = { k: v.to(device) for k, v in inputs.items() }

        if do_classification:
            inputs["labels"] = labels[batch_start:batch_end]
        else:
            inputs["labels"] = inputs["input_ids"].detach().clone()

        # get last hidden state for inputs
        with torch.no_grad():
            outputs = student_net(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]
        
        def compute_loss(params, buffers, last_hidden_states, labels):
            logits = torch.func.functional_call(
                module=student_net.lm_head,
                parameter_and_buffer_dicts=(params, buffers),
                args=(last_hidden_states,),
            )
            if do_classification:
                logits = logits[-1, :]
            else:
                logits = logits[:-1, :]
                logits.reshape(-1, logits.size(-1)), 
                labels = labels[1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                labels.reshape(-1),
                ignore_index=-100,
                reduction="mean"
            )
            return loss

        # Create vectorized gradient function
        ft_compute_grad = torch.func.grad(compute_loss)
        ft_compute_sample_grad = torch.func.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

        # Compute per-sample gradients
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, last_hidden_state, inputs["labels"])
        grads_batch = torch.cat([ft_per_sample_grads[n].reshape(batch_end - batch_start, -1) for n, _ in student_net.lm_head.named_parameters()], dim=1)

        if do_projection:
            projected_grads = projector.project(grads_batch, model_id=0)
            all_grads.extend(projected_grads)
        else:
            all_grads.extend(grads_batch)
        
    return torch.stack(all_grads) 