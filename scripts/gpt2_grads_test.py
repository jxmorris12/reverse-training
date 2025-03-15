from abc import ABC, abstractmethod
from typing import Dict, Sequence, Union, Optional
from dataclasses import dataclass

import collections
import copy
import random

import datasets
import torch
import transformers
import tqdm
from enum import Enum

from torch.utils.data import Dataset

# from utils import train_expert_model

def limit_layers(model: transformers.PreTrainedModel, n_layers: int) -> transformers.PreTrainedModel:
    if hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'h'):
            model.transformer.h = model.transformer.h[:n_layers]
        else:
            model.transformer.layer = model.transformer.layer[:n_layers]
    elif hasattr(model, 'encoder'):
        if hasattr(model.encoder, 'layers'):
            model.encoder.layers = model.encoder.layers[:n_layers]
        else:
            model.encoder.layer = model.encoder.layer[:n_layers]
    else:
        raise RuntimeError(f"unknown how to limit layers of model {type(model)}")
    return model


class ProjectionType(str, Enum):
    rademacher = "rademacher"
    normal = "normal"


class AbstractProjector(ABC):
    """Implementations of the Projector class must implement the
    :meth:`AbstractProjector.project` method, which takes in model gradients and
    returns
    """

    @abstractmethod
    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: Union[str, ProjectionType],
        device: Union[str, torch.device],
    ) -> None:
        """Initializes hyperparameters for the projection.

        Args:
            grad_dim (int):
                number of parameters in the model (dimension of the gradient
                vectors)
            proj_dim (int):
                dimension after the projection
            seed (int):
                random seed for the generation of the sketching (projection)
                matrix
            proj_type (Union[str, ProjectionType]):
                the random projection (JL transform) guearantees that distances
                will be approximately preserved for a variety of torchoices of the
                random matrix (see e.g. https://arxiv.org/abs/1411.2404). Here,
                we provide an implementation for matrices with iid Gaussian
                entries and iid Rademacher entries.
            device (Union[str, torch.device]):
                CUDA device to use

        """
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.proj_type = proj_type
        self.device = device

    @abstractmethod
    def project(self, grads: torch.Tensor, model_id: int) -> torch.Tensor:
        """Performs the random projection. Model ID is included
        so that we generate different projection matrices for every
        model ID.

        Args:
            grads (Tensor): a batch of gradients to be projected
            model_id (int): a unique ID for a torcheckpoint

        Returns:
            torch.Tensor: the projected gradients
        """

    def free_memory(self):
        """Frees up memory used by the projector."""


class CudaProjector(AbstractProjector):
    """
    A performant implementation of the projection for CUDA with compute
    capability >= 7.0.
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        proj_type: ProjectionType,
        device,
        max_batch_size: int,
        *args,
        **kwargs,
    ) -> None:
        """

        Args:
            grad_dim (int):
                Number of parameters
            proj_dim (int):
                Dimension we project *to* during the projection step
            seed (int):
                Random seed
            proj_type (ProjectionType):
                Type of randomness to use for projection matrix (rademacher or normal)
            device:
                CUDA device
            max_batch_size (int):
                Explicitly constraints the batch size the CudaProjector is going
                to use for projection. Set this if you get a 'The batch size of
                the CudaProjector is too large for your GPU' error. Must be
                either 8, 16, or 32.

        Raises:
            ValueError:
                When attempting to use this on a non-CUDA device
            ModuleNotFoundError:
                When fast_jl is not installed

        """
        super().__init__(grad_dim, proj_dim, seed, proj_type, device)
        self.max_batch_size = max_batch_size

        if isinstance(device, str):
            device = torch.device(device)

        if device.type != "cuda":
            err = "CudaProjector only works on a CUDA device; Either switch to a CUDA device, or use the BasicProjector"
            raise ValueError(err)

        self.num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count

        try:
            import fast_jl

            # test run to catch at init time if projection goes through
            fast_jl.project_rademacher_8(
                torch.zeros(8, 1_000, device="cuda"), 512, 0, self.num_sms
            )
        except ImportError:
            err = "You should make sure to install the CUDA projector for traker (called fast_jl).\
                  See the installation FAQs for more details."
            raise ModuleNotFoundError(err)

    def project(
        self,
        grads: Union[dict, torch.Tensor],
        model_id: int,
    ) -> torch.Tensor:
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
                        "Reduce it by using the proj_max_batch_size argument of the TRAKer.\nOriginal error:"
                    )
                )
            else:
                print("Error for dtype: ", grads.dtype, "and shape: ", grads.shape)
                raise e

        return result

    def free_memory(self):
        """A no-op method."""
        pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_model(model_path: str) -> dict[str, torch.Tensor]:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation="eager"
    )
    return limit_layers(model, 6)

def _get_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    
    # hack to get around gpt weight-tying:
    if torch.isclose(state_dict["transformer.wte.weight"], state_dict["lm_head.weight"]).all():
        del state_dict["lm_head.weight"]
    
    return { k: v.detach().clone() for k,v in state_dict.items() }


def get_model_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    return _get_state_dict(get_model(model_path))

def train_expert_model(
        num_experts: int, 
        num_steps_per_expert: int, 
        num_expert_datapoints: int, 
        expert_lr: float, 
        sequence_length: int,
        train_ds: datasets.Dataset, 
        tokenizer: transformers.AutoTokenizer,
        text_column_name: str,
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor]:
    student_net = get_model("gpt2").to(device)

    # optim = torch.optim.Adam(student_net.parameters(), lr=expert_lr)
    optim = torch.optim.SGD(student_net.parameters(), lr=expert_lr)

    expert_state_dicts = [_get_state_dict(student_net)]
    step = 0
    pbar = tqdm.tqdm(total=num_experts * num_steps_per_expert, colour="CYAN")
    all_token_counts = torch.zeros([tokenizer.vocab_size], device=device)

    # training loop
    for _i in range(num_experts):
        for _j in range(num_steps_per_expert):
            batch_idxs = random.sample(
                range(len(train_ds)), 
                k=num_expert_datapoints
            )
            examples = train_ds.select(batch_idxs)[text_column_name]
            tokens = tokenizer(
                examples,
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=sequence_length, 
            ).to(device)
            outputs = student_net(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )
            # use bincount to track token counts
            all_token_counts += torch.bincount(tokens.input_ids.flatten(), minlength=tokenizer.vocab_size)
            
            labels = tokens.input_ids[:, 1:].detach().clone() 
            logits = outputs.logits[:, :-1, :]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(labels.numel(), -1), 
                labels.reshape(labels.numel(),),
                ignore_index=-100,
                reduction="mean"
            )
            pbar.set_description(f"Expert training step {step+1} | Loss = {loss:.3f}")
            pbar.update(1)
            loss.backward()
            optim.step()
            optim.zero_grad()
            student_net.zero_grad()
            step += 1
        expert_state_dicts.append(_get_state_dict(student_net))
    
    pbar.close()

    return expert_state_dicts, all_token_counts


def get_grads(base_model, dataset, tokenizer, projector, sequence_length: int) -> torch.Tensor:
    all_grads = []
    pbar = tqdm.trange(0, len(dataset), 1)
    for i in pbar:
        batch = dataset[i]
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
        )
        inputs = { k: v.to(device) for k, v in inputs.items() }
        inputs["labels"] = inputs["input_ids"].detach().clone()
        outputs = base_model(**inputs)
        outputs.loss.backward()
        grad_flattened = torch.cat([
            p.grad.flatten() for p in base_model.parameters()]
        )

        # Project the gradients
        projected_grad = projector.project(grad_flattened.unsqueeze(0), model_id=0)
        all_grads.append(projected_grad.squeeze(0))
        base_model.zero_grad()
    return torch.stack(all_grads)


def main():
    base_model = get_model("gpt2")
    base_model.to(device)

    train_dataset = datasets.load_dataset("tatsu-lab/alpaca")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "gpt2",
        model_max_length=256,
        padding_side="right",
        truncation_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    def make_text(ex: dict[str, str]) -> dict[str, str]:
        ex["text"] = ex["instruction"] + "\n" + ex["output"]
        return ex
    train_dataset = train_dataset.map(make_text)

    # train_dataset = train_dataset.select(range(4_096))
    train_dataset["train"] = train_dataset["train"].select(range(64))
    expert_state_dicts, _ = train_expert_model(
        num_experts=1,
        num_steps_per_expert=100,
        num_expert_datapoints=32,
        expert_lr=1e-2, 
        sequence_length=32,
        train_ds=train_dataset["train"],
        tokenizer=tokenizer,
        text_column_name="text",
    )
    expert_model_params = expert_state_dicts[-1]
    initial_expert_model_params = torch.cat([p.flatten() for p in expert_model_params.values()]).double().to(device).requires_grad_(False)

    base_params = torch.cat([p.flatten() for p in base_model.parameters()]).double().to(device)
    number_of_params = len(base_params)

    initial_mse = (base_params[None] - initial_expert_model_params[None]).double().pow(2).mean().item()
    print(f"Initial parameter MSE: {initial_mse:.8f}")

    projection_dim = 1024
    projector = CudaProjector(
        grad_dim=number_of_params,
        proj_dim=projection_dim,
        seed=0,
        proj_type=ProjectionType.rademacher,
        device=device,
        max_batch_size=32
    )

    # train + filter
    student_lr = 1e-2
    optimizer = torch.optim.SGD(base_model.parameters(), lr=student_lr)
    batch_size = 8
    j = 0
    # num_steps = min(1000, len(train_dataset)) // batch_size
    num_steps = 10
    pbar = tqdm.trange(0, num_steps)

    for _ in pbar:
        # get all grads
        grads = get_grads(base_model, train_dataset["train"], tokenizer, projector, sequence_length=32)
        assert grads.shape == (len(train_dataset["train"]), projection_dim)

        # project params
        base_params = torch.cat([p.flatten() for p in base_model.parameters()]).double().to(device)
        base_params = projector.project(base_params, model_id=0)
        expert_model_params = projector.project(initial_expert_model_params, model_id=0)

        projected_mse = (base_params - expert_model_params).double().pow(2).mean().item()
        print(f"Projected MSE: {projected_mse:.8f}")

        # greedily fill batch
        batch = []
        batch_grads_sum = torch.zeros_like(grads[0])
        while len(batch) < batch_size:
            # minimize |\sum_{i} batch_i  + x - grads_i|^2
            # sims = torch.nn.functional.cosine_similarity(
            #     batch_grads_sum[None] + grads + base_params,
            #     expert_model_params,
            #     dim=1
            # )
            # use MSE
            new_base_params = base_params + grads * student_lr
            sims = (new_base_params - expert_model_params).double().pow(2).mean(dim=1)
            # check for nans
            if torch.isnan(sims).any():
                print("NaNs in sims")
                breakpoint()
                break
            # filter out already selected grad
            batch_idxs = torch.tensor(batch, dtype=torch.int64)
            sims[batch_idxs] = float("inf")
            best_idx = torch.argmin(sims)
            batch.append(best_idx)
            batch_grads_sum += grads[best_idx]
            grads[best_idx] = torch.zeros_like(grads[best_idx])
            print(f"Best sim: {sims[best_idx].item():.3f} | Batch size: {len(batch)} | Best idx: {best_idx}")

        # take step on batch
        inputs = tokenizer(
            train_dataset["train"].select(batch)["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        )
        inputs = { k: v.to(device) for k, v in inputs.items() }
        inputs["labels"] = inputs["input_ids"].detach().clone()
        outputs = base_model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"loss: {loss.item():.3f}")
        print(f"loss: {loss.item():.3f}")
        j += 1
    
    final_mse = (base_params[None] - expert_model_params[None]).double().pow(2).mean().item()
    print(f"Final parameter MSE: {final_mse:.3f}")
    
if __name__ == "__main__":
    main()