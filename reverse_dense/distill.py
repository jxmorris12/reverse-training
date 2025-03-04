import argparse
import copy
import glob
import os
import re

import datasets
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import transformers
import wandb

from utils import get_network, evaluate_synset, get_time, limit_layers
from reparam_module import ReparamModule


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MODEL_NUM_LAYERS = 8
sequence_length = 32
text_column_name = "text"        
label_column_name = "label"

num_experts = 20
num_steps_per_expert = 50
num_expert_datapoints = 256
expert_lr = 1e-4


def state_dict_to_tensor(ordered_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.data.to(device).reshape(-1) for n, p in ordered_dict.items()], 0)


def get_token_embeddings_from_dataset(ds: datasets.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """ initialize the synthetic data """
    student_net = get_model("gpt2")
    student_net.train()
    student_token_embeddings = student_net.get_input_embeddings().to(device)
    # TODO: Consider other initialization methods
    # def simple_soft_one_hot(x, num_classes, temperature=1.0):
    #     one_hot = F.one_hot(x, num_classes).float()
    #     return F.softmax(one_hot / temperature, dim=-1)
    
    hidden_size = student_token_embeddings.weight.shape[1]

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left" # important for correct truncation
    tokenizer.padding_side = "left" 
    tokens = tokenizer.batch_encode_plus(
        ds["text"][0:args.dataset_size],
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=sequence_length-1, 
        return_attention_mask=False
    )
    token_embeddings_syn = student_token_embeddings(tokens["input_ids"].to(device))
    assert token_embeddings_syn.shape == (args.dataset_size, sequence_length - 1, hidden_size), f"invalid shape: {token_embeddings_syn.shape}, need {(args.dataset_size, sequence_length - 1, hidden_size)}"

    num_classes = 4
    CLASS_MAP = torch.tensor([352,  362,  657,  513], device=device) # for AG_News... tmp
    token_labels_syn = torch.randint(low=0, high=num_classes, size=[args.dataset_size], device=device)
    token_labels_syn = CLASS_MAP[token_labels_syn]

    return (token_embeddings_syn, token_labels_syn)

def get_token_embeddings_random() -> tuple[torch.Tensor, torch.Tensor]:
    """ initialize the synthetic data """
    student_net = get_model("gpt2")
    student_net.train()
    student_token_embeddings = student_net.get_input_embeddings().to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    def simple_soft_one_hot(x, num_classes, temperature=1.0):
        one_hot = F.one_hot(x, num_classes).float()
        return F.softmax(one_hot / temperature, dim=-1)
    
    rand_token_idxs = torch.randint(0, student_token_embeddings.num_embeddings, (args.dataset_size, sequence_length-1), device=device)
    rand_token_one_hots = simple_soft_one_hot(rand_token_idxs, student_token_embeddings.num_embeddings, temperature=0.1)
    token_embeddings_syn = rand_token_one_hots @ student_token_embeddings.weight
    
    hidden_size = student_token_embeddings.weight.shape[1]
    assert token_embeddings_syn.shape == (args.dataset_size, sequence_length - 1, hidden_size), f"invalid shape: {token_embeddings_syn.shape}, need {(args.dataset_size, sequence_length - 1, hidden_size)}"

    num_classes = 4
    CLASS_MAP = torch.tensor([352,  362,  657,  513], device=device) # for AG_News... tmp
    token_labels_syn = torch.randint(low=0, high=num_classes, size=[args.dataset_size], device=device)
    token_labels_syn = CLASS_MAP[token_labels_syn]

    return (token_embeddings_syn, token_labels_syn)


def get_model(model_path: str) -> dict[str, torch.Tensor]:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation="eager"
    )
    print("Warning: Limiting layers to 4")
    return limit_layers(model, MODEL_NUM_LAYERS) # TODO: Override from config?


def _get_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    
    # hack to get around gpt weight-tying:
    if torch.isclose(state_dict["transformer.wte.weight"], state_dict["lm_head.weight"]).all():
        del state_dict["lm_head.weight"]
    
    return { k: v.detach().clone() for k,v in state_dict.items() }


def get_model_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    return _get_state_dict(get_model(model_path))


def load_expert_trajectories(ds: datasets.Dataset):
    student_net = get_model("gpt2").to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left" # important for correct truncation
    tokenizer.padding_side = "left" 

    optim = torch.optim.Adam(student_net.parameters(), lr=expert_lr)

    expert_paths = [_get_state_dict(student_net)]
    step = 0
    pbar = tqdm.tqdm(total=num_experts * num_steps_per_expert, colour="CYAN")
    for _i in range(num_experts):
        for _j in range(num_steps_per_expert):
            batch_idxs = random.sample(range(len(ds)), k=num_expert_datapoints)
            examples = ds.select(batch_idxs)
            examples = [
                f"{text} {label}" 
                for text, label in zip(examples[text_column_name], examples[label_column_name])
            ]
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
            logits = outputs.logits[:, :-1]
            labels = tokens.input_ids[:, 1:].detach().clone() 
            labels[:, :-1] = -100
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
        expert_paths.append(_get_state_dict(student_net))

    return expert_paths


def project_x_to_embedding_space(X, Λ, ρ, initial_student_net) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        embeddings = initial_student_net.get_input_embeddings()
        embeddings = embeddings.to(device)
        X_proj = X + Λ / ρ
        Z_distances = torch.cdist(X_proj, embeddings.weight.to(device))
        Z_tokens = Z_distances.argmin(dim=2)
        Z = embeddings(Z_tokens)
    return Z, Z_tokens


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    assert args.expert_epochs < args.max_experts, "Expert epochs must be less than max experts"
    data_save = []

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    wandb.init(
        sync_tensorboard=False,
        project="dataset-distillation",
        config=args,
        # mode="disabled",
    )

    args = type('', (), {})()
    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    if args.batch_syn is None:
        args.batch_syn = args.dataset_size # Full-batch GD
    print('Hyperparameters: \n', dict(vars(args)))

    ds = datasets.load_dataset("fancyzhx/ag_news", split="train")

    if args.token_init == "random":
        X, Y = get_token_embeddings_random()
    elif args.token_init == "dataset":
        X, Y = get_token_embeddings_from_dataset(ds=ds)
    else:
        raise NotImplementedError(f"Token init {args.token_init} not implemented")
    X = X.detach().to(device).requires_grad_(True)

    # train model for a couple steps
    student_net = get_model("gpt2")
    student_net = ReparamModule(student_net).to(device)

    syn_lr = torch.tensor(args.lr_teacher).to(device)
    syn_lr = syn_lr.detach().to(device).requires_grad_(True)

    # optimizer_token_embeddings = torch.optim.AdamW([token_embeddings_syn], lr=args.lr_tokens)
    # optimizer_token_embeddings = torch.optim.SGD([X], lr=args.lr_tokens, momentum=0.5)
    optimizer_token_embeddings = torch.optim.Adam([X], lr=args.lr_tokens)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_token_embeddings.zero_grad()

    print('%s [time] training begins' % get_time())

    initial_student_net = get_model("gpt2")
    # Λ = torch.rand_like(X, device=device).requires_grad_(False)
    Λ = torch.zeros_like(X, device=device).requires_grad_(False)
    ρ = args.penalty_term     # [TODO]  Argparse this. Paper says: 
                                #                   > ρ is chosen from
                                #                   > the set {0.001, 0.05, 0.01, . . . , 10}

    Z, _ = project_x_to_embedding_space(X, Λ, ρ, initial_student_net)

    # load/generate expert trajectories
    buffer = load_expert_trajectories(ds=ds)

    pbar = tqdm.trange(0, args.max_iterations+1, desc="iterations")
    for it in pbar:
        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        start_epoch = np.random.randint(0, len(buffer) - args.expert_epochs)
        starting_params = buffer[start_epoch]
        target_params = buffer[start_epoch+args.expert_epochs]
        student_params = [torch.cat([p.data.to(device).reshape(-1) for n, p in starting_params.items()], 0).requires_grad_(True)]

        target_params = state_dict_to_tensor(target_params)
        starting_params = state_dict_to_tensor(starting_params)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []
        ce_losses = []

        for step in tqdm.trange(args.syn_steps, desc="Synthetic steps", leave=False):
            if not indices_chunks:
                indices = torch.randperm(len(X))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()
            x = X[these_indices]
            y = Y[these_indices]
            
            output = student_net(inputs_embeds=x, flat_param=student_params[-1])

            # autoregressive classification loss on last token
            logits = output.logits[:, :-1]
            ce_loss = torch.nn.functional.cross_entropy(logits[:, -1], y, reduction="mean")
            ce_losses.append(ce_loss.detach().item())

            # exit on nan
            if torch.isnan(ce_loss):
                print("nan detected - stopping!")
                exit()

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)

        # 
        #  (1) Compute overall loss on X using trajectory matching
        # 
        param_loss = torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum") / num_params
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum") / num_params
        if torch.isclose(param_dist, torch.tensor(0.0)):
            print("zero distance detected – stopping!")
            exit()

        param_loss_list.append(param_loss * num_params)
        param_dist_list.append(param_dist * num_params)
        param_loss = param_loss / param_dist

        optimizer_token_embeddings.zero_grad()
        optimizer_lr.zero_grad() 

        lagrangian_term = torch.sum(Λ * (X - Z), dim=2).mean()
        quadratic_penalty = (ρ / 2) * ((X - Z).norm(p=2, dim=2) ** 2).mean()
        aux_loss = lagrangian_term + quadratic_penalty

        if (it >= args.pretrain_iterations_x):
            (aux_loss + param_loss).backward()
        else:
            param_loss.backward()
            Z = X.detach().clone()

        optimizer_token_embeddings.step()
        optimizer_lr.step()

        if (it >= args.pretrain_iterations_x) and (it % args.max_iterations_x == 0):
            with torch.no_grad():
                # 
                #  (2) Compute new Z based on projecting X back to word embedding space
                #                       TODO: Use a language model for this, optionally.
                Z, Z_tokens = project_x_to_embedding_space(X, Λ, ρ, initial_student_net)
                # 
                # Log Z
                # 
                tokens = tokenizer.batch_decode(Z_tokens.cpu(), add_special_tokens=False)
                labels = tokenizer.batch_decode(Y.cpu(), add_special_tokens=False)
                table_data = [(i, T, L) for i, (T, L) in enumerate(zip(tokens, labels))]
                tokens_table = wandb.Table(data=table_data, columns=["index", "text", "label"])
                wandb.log({ "Z": tokens_table }, step=it)
                # 
                #  (3) Update Λ for ADMM
                # 
                Λ = Λ + ρ * (X - Z)

                # TODO: Should we reset Adam here?
                optimizer_token_embeddings = torch.optim.Adam([X], lr=args.lr_tokens)
                optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)

        Z_dist = (X - Z).norm(p=2).mean().detach()
        ce_loss_avg = sum(ce_losses) / len(ce_losses)
        wandb.log(
            {
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
                "Z_dist": Z_dist.detach().cpu(),
            },
            step=it
        )
        # print("aux_loss", aux_loss, "param_loss", param_loss)
        # print("%s step = %04d, loss = %.8f, oneoffloss = %.8f, ce_loss = %.8f" % (get_time(), it, param_loss.item(), (param_loss - 1).item(), ce_loss_avg))
        pbar.set_postfix(
            aux_loss=f"{aux_loss.item():.3f}",
            param_loss=f"{param_loss.item():.3f}",
            ce_loss=f"{ce_loss_avg:.3f}",
            param_dist=f"{param_dist.item():.3f}",
            Z_dist=f"{Z_dist.item():.3f}",
            lr=f"{syn_lr.item():.3f}",
        )
        for _ in student_params:
            del _
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", "--ds", type=int, default=1000, help="size of distilled dataset")
    parser.add_argument("--batch_syn", "--batch_size_synthetic", type=int, default=None, help='should only use this if you run out of VRAM')
    
    parser.add_argument("--penalty_term", "--rho", type=float, default=0.1, help="ADMM penalty term (ρ)")

    parser.add_argument('--max_iterations', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--max_iterations_x', type=int, default=40, help='how many gradient steps per X update')
    parser.add_argument('--pretrain_iterations_x', type=int, default=100, help='how many gradient steps in initial X training phase')

    parser.add_argument('--lr_tokens', type=float, default=0.001, help='learning rate for updating synthetic tokens')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--expert_epochs', type=int, default=1, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=8, help='how many steps to take on synthetic data')
    parser.add_argument('--max_experts', type=int, default=32, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--token_init", type=str, default="random", help="how to initialize the synthetic tokens", choices=["random", "dataset"])

    args = parser.parse_args()

    main(args)
