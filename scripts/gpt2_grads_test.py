from typing import Dict, Sequence
from dataclasses import dataclass
import copy
import datasets
import logging
import torch
import transformers
import tqdm

from torch.utils.data import Dataset


# class AbstractProjector(ABC):
#     """Implementations of the Projector class must implement the
#     :meth:`AbstractProjector.project` method, which takes in model gradients and
#     returns
#     """

#     @abstractmethod
#     def __init__(
#         self,
#         grad_dim: int,
#         proj_dim: int,
#         seed: int,
#         proj_type: Union[str, ProjectionType],
#         device: Union[str, torch.device],
#     ) -> None:
#         """Initializes hyperparameters for the projection.

#         Args:
#             grad_dim (int):
#                 number of parameters in the model (dimension of the gradient
#                 vectors)
#             proj_dim (int):
#                 dimension after the projection
#             seed (int):
#                 random seed for the generation of the sketching (projection)
#                 matrix
#             proj_type (Union[str, ProjectionType]):
#                 the random projection (JL transform) guearantees that distances
#                 will be approximately preserved for a variety of choices of the
#                 random matrix (see e.g. https://arxiv.org/abs/1411.2404). Here,
#                 we provide an implementation for matrices with iid Gaussian
#                 entries and iid Rademacher entries.
#             device (Union[str, torch.device]):
#                 CUDA device to use

#         """
#         self.grad_dim = grad_dim
#         self.proj_dim = proj_dim
#         self.seed = seed
#         self.proj_type = proj_type
#         self.device = device

#     @abstractmethod
#     def project(self, grads: Tensor, model_id: int) -> Tensor:
#         """Performs the random projection. Model ID is included
#         so that we generate different projection matrices for every
#         model ID.

#         Args:
#             grads (Tensor): a batch of gradients to be projected
#             model_id (int): a unique ID for a checkpoint

#         Returns:
#             Tensor: the projected gradients
#         """

#     def free_memory(self):
#         """Frees up memory used by the projector."""


# class CudaProjector(AbstractProjector):
#     """
#     A performant implementation of the projection for CUDA with compute
#     capability >= 7.0.
#     """

#     def __init__(
#         self,
#         grad_dim: int,
#         proj_dim: int,
#         seed: int,
#         proj_type: ProjectionType,
#         device,
#         max_batch_size: int,
#         *args,
#         **kwargs,
#     ) -> None:
#         """

#         Args:
#             grad_dim (int):
#                 Number of parameters
#             proj_dim (int):
#                 Dimension we project *to* during the projection step
#             seed (int):
#                 Random seed
#             proj_type (ProjectionType):
#                 Type of randomness to use for projection matrix (rademacher or normal)
#             device:
#                 CUDA device
#             max_batch_size (int):
#                 Explicitly constraints the batch size the CudaProjector is going
#                 to use for projection. Set this if you get a 'The batch size of
#                 the CudaProjector is too large for your GPU' error. Must be
#                 either 8, 16, or 32.

#         Raises:
#             ValueError:
#                 When attempting to use this on a non-CUDA device
#             ModuleNotFoundError:
#                 When fast_jl is not installed

#         """
#         super().__init__(grad_dim, proj_dim, seed, proj_type, device)
#         self.max_batch_size = max_batch_size

#         if isinstance(device, str):
#             device = ch.device(device)

#         if device.type != "cuda":
#             err = "CudaProjector only works on a CUDA device; Either switch to a CUDA device, or use the BasicProjector"
#             raise ValueError(err)

#         self.num_sms = ch.cuda.get_device_properties(device.index).multi_processor_count

#         try:
#             import fast_jl

#             # test run to catch at init time if projection goes through
#             fast_jl.project_rademacher_8(
#                 ch.zeros(8, 1_000, device="cuda"), 512, 0, self.num_sms
#             )
#         except ImportError:
#             err = "You should make sure to install the CUDA projector for traker (called fast_jl).\
#                   See the installation FAQs for more details."
#             raise ModuleNotFoundError(err)

#     def project(
#         self,
#         grads: Union[dict, Tensor],
#         model_id: int,
#     ) -> Tensor:
#         if isinstance(grads, dict):
#             grads = vectorize(grads, device=self.device)

#         batch_size = grads.shape[0]

#         effective_batch_size = 32
#         if batch_size <= 8:
#             effective_batch_size = 8
#         elif batch_size <= 16:
#             effective_batch_size = 16

#         effective_batch_size = min(self.max_batch_size, effective_batch_size)

#         function_name = f"project_{self.proj_type.value}_{effective_batch_size}"
#         import fast_jl

#         fn = getattr(fast_jl, function_name)

#         try:
#             result = fn(
#                 grads, self.proj_dim, self.seed + int(1e4) * model_id, self.num_sms
#             )
#         except RuntimeError as e:
#             if "CUDA error: too many resources requested for launch" in str(e):
#                 # provide a more helpful error message
#                 raise RuntimeError(
#                     (
#                         "The batch size of the CudaProjector is too large for your GPU. "
#                         "Reduce it by using the proj_max_batch_size argument of the TRAKer.\nOriginal error:"
#                     )
#                 )
#             else:
#                 raise e

#         return result

#     def free_memory(self):
#         """A no-op method."""
        # pass


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, list_data_dict: datasets.Dataset, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

device = "cuda"

def get_sims(base_model, dataset, data_collator, param_diff) -> torch.Tensor:
    all_sims = []
    pbar = tqdm.trange(0, len(dataset), 1)
    for i in pbar:
        batch = dataset[i]
        inputs = data_collator([batch])
        inputs = { k: v.to(device) for k, v in inputs.items() }
        outputs = base_model(**inputs)
        outputs.loss.backward()
        grad_flattened = torch.cat([
            p.grad.flatten() for p in base_model.parameters()]
        )

        # TODO: down-project and store.
        sim = torch.nn.functional.cosine_similarity(grad_flattened, param_diff, dim=0)
        all_sims.append(sim.item())
        pbar.set_description(f"similarity: {sim.item():.3f}")

        base_model.zero_grad()
    return torch.tensor(all_sims)

def main():
    base_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    other_model = transformers.AutoModelForCausalLM.from_pretrained("vicgalle/gpt2-alpaca")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "gpt2",
        model_max_length=256,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=other_model,
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=base_model,
    )
    

    # average params
    for (n1, p1), (n2, p2) in zip(base_model.named_parameters(), other_model.named_parameters()):
        p1.data = (p1.data + p2.data) / 2
    
    base_model_param_names = [(n, p.shape) for n, p in base_model.named_parameters()]
    other_model_param_names = [(n, p.shape) for n, p in other_model.named_parameters()]

    base_params = torch.cat([p.flatten() for p in base_model.parameters()]).double().to(device)
    other_params = torch.cat([p.flatten() for p in other_model.parameters()]).double().to(device)
    param_diff = base_params - other_params

    train_dataset = datasets.load_dataset("tatsu-lab/alpaca")["train"]
    # dataset = datasets.load_dataset("yahma/alpaca-cleaned")["train"]
    # data = datasets.load_dataset("wentingzhao/one-million-instructions")["train"]
    train_dataset = train_dataset.select(range(4_096))

    dataset = SupervisedDataset(train_dataset, tokenizer)


    # project_interval = 16  # project every 16 batches
    # projection_dim = 1024

    # proj = CudaProjector(grad_dim=number_of_params,
    #                     proj_dim=dim,
    #                     seed=0,
    #                     proj_type=ProjectionType.rademacher,
    #                     device=device,
    #                     dtype=dtype,
    #                     block_size=block_size,
    #                     max_batch_size=projector_batch_size)
    # def _project(current_full_grads, projected_grads):
    #     current_full_grads = torch.stack(current_full_grads).to(torch.float16)
    #     for i, projector in enumerate(projectors):
    #         current_projected_grads = projector.project(
    #             current_full_grads, model_id=model_id)
    #         projected_grads[proj_dim[i]].append(current_projected_grads.cpu())


    # get all grads
    base_model.to(device)

    # initial filtering
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    all_sims = get_sims(base_model, dataset, data_collator, param_diff)
    top_k = 1000
    top_k_indices = torch.argsort(all_sims, descending=True)[:top_k]

    # train + filter
    train_dataset = train_dataset.select(top_k_indices)
    filtered_train_dataset = train_dataset
    dataset = SupervisedDataset(filtered_train_dataset, tokenizer)
    batch_size = 8
    steps_before_reranking = 10
    steps_total = 1000
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-5)
    j = 0
    pbar = tqdm.trange(0, 1000)
    for _ in pbar:
        batch = [dataset[j] for j in range(j * batch_size, (j + 1) * batch_size)]
        inputs = data_collator(batch)
        inputs = { k: v.to(device) for k, v in inputs.items() }
        outputs = base_model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # print(f"loss: {loss.item()}")
        pbar.set_description(f"loss: {loss.item():.3f}")
        j += 1
        if (j > 0) and (j % steps_before_reranking == 0):
            all_sims = get_sims(base_model, dataset, data_collator, param_diff)
            top_indices = torch.argsort(all_sims, descending=True)
            filtered_train_dataset = train_dataset.select(top_indices)
            dataset = SupervisedDataset(filtered_train_dataset, tokenizer)
            j = 0
            breakpoint()


if __name__ == "__main__":
    main()