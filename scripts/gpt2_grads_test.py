from abc import ABC, abstractmethod
from typing import Dict, Sequence, Union, Optional
from dataclasses import dataclass

import collections
import random

import datasets
import torch
import transformers
import tqdm
import wandb
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
    # disable dropout
    dropout_modules = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m in dropout_modules:
        m.p = 0.0

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


def get_grads(base_model, dataset, tokenizer, projector, sequence_length: int, batch_size: int, do_projection: bool = True) -> torch.Tensor:
    dataset = dataset.select(range(5, 7))
    
    all_grads = []
    all_grads_true = []
    pbar = tqdm.trange(0, len(dataset), batch_size)

    # helpful page: pytorch.org/tutorials/intermediate/per_sample_grads.html
    params = {k: v.detach() for k, v in base_model.lm_head.named_parameters()}
    buffers = {k: v.detach() for k, v in base_model.lm_head.named_buffers()}
    
    for batch_start in pbar:
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset.select(range(batch_start, batch_end))
        print(batch)
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=sequence_length,
        )
        inputs = { k: v.to(device) for k, v in inputs.items() }
        inputs["labels"] = inputs["input_ids"].detach().clone()

        # measure grads one-by-one
        batch_grads_true_list = []
        for i in range(len(batch)):
            batch_inputs = { k: v[i:i+1].to(device) for k, v in inputs.items() }
            loss = base_model(**batch_inputs).loss
            loss.backward()
            batch_grads_true = torch.cat([p.grad.flatten().detach() for p in base_model.lm_head.parameters()], dim=0)
            batch_grads_true_list.append(batch_grads_true)
            batch_grads_true_projected = projector.project(batch_grads_true, model_id=0)
            all_grads_true.extend(batch_grads_true_projected)
            base_model.zero_grad()
            print("loss 1: ", loss)
        batch_grads_true_list = torch.stack(batch_grads_true_list)

        # get last hidden state for inputs
        with torch.no_grad():
            outputs = base_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]
        
        # def compute_loss(params, buffers, last_hidden_states, labels):
        #     base_model.lm_head.requires_grad_(False)
        #     logits = torch.func.functional_call(
        #         module=base_model.lm_head,
        #         parameter_and_buffer_dicts=(params, buffers),
        #         args=(last_hidden_states,),
        #     )
        #     logits = logits[:-1, :]
        #     labels = labels[1:]
        #     loss = torch.nn.functional.cross_entropy(
        #         logits, 
        #         labels,
        #         ignore_index=-100,
        #         reduction="mean"
        #     )
        #     return loss

        # # Create vectorized gradient function
        # ft_compute_grad = torch.func.grad(compute_loss)
        # ft_compute_sample_grad = torch.func.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        # Compute per-sample gradients
        # ft_per_sample_grads = ft_compute_sample_grad(params, buffers, last_hidden_state, inputs["labels"])

        # Compute per-sample gradients
        cs = torch.nn.functional.cosine_similarity
        with torch.no_grad():
            logits = base_model.lm_head(last_hidden_state)
            logits = logits[:, :-1, :]
            labels = inputs["labels"][:, 1:]
            loss_mask = (inputs["attention_mask"][:, :-1] & (inputs["labels"] != -100)[:, :-1])

            # Calculate probs directly
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Initialize the one-hot targets with zeros
            batch_size, seq_len, vocab_size = logits.shape
            labels_one_hot = torch.zeros_like(probs)

            # Only set one-hot values for valid positions
            valid_indices = loss_mask.nonzero(as_tuple=True)
            valid_labels = labels[valid_indices]
            labels_one_hot[valid_indices[0], valid_indices[1], valid_labels] = 1.0

            # Gradient calculation
            grad_logits = probs - labels_one_hot
            grad_logits = grad_logits * loss_mask.unsqueeze(-1)

            # compute gradient
            # TODO: Find out why my gradient only has a cosine sim of 0.98 with the true gradient
            grad_W = (
                torch.einsum(
                    "bsd, bsv -> bdv", 
                    grad_logits, 
                    last_hidden_state[:, :-1, :] # * loss_mask[:, :, None]
                ) # / loss_mask.sum(dim=1)[:, None, None]
            )
            grad_W = grad_W.reshape(len(logits), -1)
            # TODO: Diagnose.
        breakpoint()

        grads_batch = torch.cat([ft_per_sample_grads[n].reshape(batch_size, -1) for n, _ in base_model.lm_head.named_parameters()], dim=1)

        if do_projection:
            projected_grads = projector.project(grads_batch, model_id=0)
            all_grads.extend(projected_grads)
        else:
            all_grads.extend(grads_batch)
        
    # return torch.stack(all_grads_true)
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

    # def make_text(ex: dict[str, str]) -> dict[str, str]:
    #     breakpoint()
    #     ex["text"] = ex["instruction"] + "\n" + ex["response"] + "\n" + ex["output"]
    #     return ex
    # train_dataset = train_dataset.map(make_text)

    # train_dataset = train_dataset.select(range(4_096))
    train_dataset["train"] = train_dataset["train"].select(range(64))
    train_dataset = train_dataset.map(lambda ex: { "text": ex["instruction"] + "\n" + ex["output"] })
    expert_state_dicts, _ = train_expert_model(
        num_experts=1,
        num_steps_per_expert=10,
        num_expert_datapoints=64,
        expert_lr=1e-3, 
        sequence_length=32,
        train_ds=train_dataset["train"],
        tokenizer=tokenizer,
        text_column_name="text",
    )
    expert_model_params = expert_state_dicts[-1]
    expert_model_params = torch.cat([p.flatten() for k, p in expert_model_params.items() if "wte" in k]).double().to(device).requires_grad_(False)

    base_params = torch.cat([p.flatten() for k, p in base_model.lm_head.named_parameters()]).double().to(device)
    number_of_params = len(base_params)

    initial_mse = (base_params - expert_model_params).double().pow(2).sum().item()
    print(f"Initial parameter MSE: {initial_mse:.8f}")

    projection_dim = 2048
    projector = CudaProjector(
        grad_dim=number_of_params,
        proj_dim=projection_dim,
        seed=0,
        block_size=128,
        proj_type=ProjectionType.rademacher,
        device=device,
        max_batch_size=256,
    )

    # create a larger dataset for search
    # other_dataset = datasets.load_dataset("fancyzhx/ag_news")
    other_dataset = datasets.load_dataset("jxm/nq_corpus_dpr")
    other_dataset["train"] = other_dataset["train"].select(range(256))
    # other_dataset["train"] = other_dataset["train"].select(range(1_280))
    search_dataset = datasets.concatenate_datasets([train_dataset["train"], other_dataset["train"]])
    # search_dataset = other_dataset["train"]


    #########################################################
    ############### Diagnostic Code 
    #########################################################
    all_grads = {}
    param_diff_proj = projector.project(base_params - expert_model_params, model_id=0)
    param_diff_proj = param_diff_proj / (param_diff_proj.norm(dim=1, p=2, keepdim=True) + 1e-10)
    for i in range(10 + 1):
        params_diff_step = (
            (base_params * (1 - i / 10))
            + 
            (expert_model_params * (i / 10))
        )
        # set base model params
        counter = 0
        for k, v in base_model.lm_head.named_parameters():
            v.data = params_diff_step[counter:counter+v.numel()].reshape(v.data.shape).to(v.data.dtype)
            counter += v.numel()
        all_grads[i] = get_grads(base_model, search_dataset, tokenizer, projector, sequence_length=32, batch_size=128, do_projection=True).cpu()
        all_grads[i] = all_grads[i] / (all_grads[i].norm(dim=1, p=2, keepdim=True) + 1e-10)
    #########################################################

    # num_steps = min(1000, len(train_dataset)) // batch_size
    num_steps = 2000
    grad_batch_size = 32

    pbar = tqdm.trange(0, num_steps)


    wandb.init(
        sync_tensorboard=False,
        project="dataset-distillation-2",
        # config=args,
        # mode="disabled" if get_rank() > 0 else os.getenv("WANDB_MODE", "online"),
    )
    best_idx_counter = collections.Counter()
    student_lr = 1e-3
    optimizer = torch.optim.SGD(base_model.parameters(), lr=student_lr)
    minibatch_size = 16
    for j in pbar:
        minibatch = random.sample(range(len(search_dataset)), minibatch_size)

        print("** TAKING STEP **")
        # take step on batch
        inputs = tokenizer(
            search_dataset.select(minibatch)["text"],
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

        # project grad from base_model to expert_model_params
        base_params_diff = base_params - expert_model_params
        with torch.no_grad():
            grad = torch.cat([p.grad.flatten().detach() for p in base_model.parameters()], dim=0)
            grad_direction = grad / (grad.norm(dim=0, p=2, keepdim=True) + 1e-10)
            grad_step_size = (grad_direction.double() @ base_params_diff).float()

        # [basic SGD update]
        optimizer.step()
        optimizer.zero_grad()
        base_model.zero_grad()
        pbar.set_description(f"loss: {loss.item():.3f}")
        print(f"loss: {loss.item():.3f}")

        # [projected update]
        # if grad_step_size.item() <= 0.0:
        #     print(f"Step size is negative or zero (got {grad_step_size.item()})")
        #     break

        #     # update params
        #     base_params = base_params - (grad_step_size * grad_direction)
        #     counter = 0
        #     for _, p in base_model.named_parameters():
        #         p.data = base_params[counter:counter+p.numel()].reshape(p.data.shape)
        #         counter += p.numel()
        
        # pbar.set_description(f"loss: {loss.item():.3f} | step size: {grad_step_size.item():.3f} | projected mse: {projected_mse:.3f}")

        if grad_step_size.item() <= 0.0:
            print(f"Warning: Step size is negative or zero (got {grad_step_size.item()})")

        wandb.log({
            "loss": loss.item(),
            "current_mse": current_mse,
            # "projected_mse": projected_mse,
            "grad_step_size": grad_step_size.item(),
            "best_sim": best_sim,
            "batch_size": len(batch),
            "num_steps": j,
            "grad_batch_size": grad_batch_size,
            "projection_dim": projection_dim,
            "sims_max": sims.max().item(),
        })

        # free memory
        torch.cuda.empty_cache()

        # Print top-10 most-selected indices
        top_10_selected = best_idx_counter.most_common(10)
        print(f"Top-10 most-selected indices: {top_10_selected}")
        
    
    final_mse = (base_params[None] - expert_model_params[None]).double().pow(2).mean().item()
    print(f"Final parameter MSE: {final_mse:.3f}")
    
if __name__ == "__main__":
    main()
