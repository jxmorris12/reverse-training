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


sequence_length = 32
text_column_name = "text"        
label_column_name = "label"

num_experts = 3
num_steps_per_expert = 3
num_expert_datapoints = 8
expert_lr = 1e-3

torch.autograd.set_detect_anomaly(True)


def state_dict_to_tensor(ordered_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.data.to(args.device).reshape(-1) for n, p in ordered_dict.items()], 0)

def get_token_embeddings_syn(ds: datasets.Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """ initialize the synthetic data """
    student_net = get_model("gpt2")
    student_net.train()
    student_token_embeddings = student_net.get_input_embeddings().to(args.device)
    # TODO: Consider other initialization methods
    def simple_soft_one_hot(x, num_classes, temperature=1.0):
        one_hot = F.one_hot(x, num_classes).float()
        return F.softmax(one_hot / temperature, dim=-1)
    
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
    token_embeddings_syn = student_token_embeddings(tokens["input_ids"].to(args.device))
    # rand_token_idxs = torch.randint(0, student_token_embeddings.num_embeddings, (args.dataset_size, sequence_length - 1), device=args.device)
    # rand_token_one_hots = simple_soft_one_hot(rand_token_idxs, student_token_embeddings.num_embeddings, temperature=0.1)
    # token_embeddings_syn = rand_token_one_hots @ student_token_embeddings.weight
    # token_embeddings_syn = token_embeddings_syn[None, None].repeat(args.dataset_size, sequence_length, 1)
    # token_embeddings_syn = torch.randn(
    #     size=(args.dataset_size, sequence_length, hidden_size
    #     ), dtype=torch.float32
    # )
    assert token_embeddings_syn.shape == (args.dataset_size, sequence_length - 1, hidden_size), f"invalid shape: {token_embeddings_syn.shape}, need {(args.dataset_size, sequence_length - 1, hidden_size)}"

    num_classes = 4
    CLASS_MAP = torch.tensor([352,  362,  657,  513], device=args.device) # for AG_News... tmp
    token_labels_syn = torch.randint(low=0, high=num_classes, size=[args.dataset_size], device=args.device)
    token_labels_syn = CLASS_MAP[token_labels_syn]
    return (token_embeddings_syn, token_labels_syn)


def get_model(model_path: str) -> dict[str, torch.Tensor]:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, 
        attn_implementation="eager"
    )
    print("Warning: Limiting layers to 4")
    return limit_layers(model, 4) # TODO: Override from config?

def _get_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    
    # hack to get around gpt weight-tying:
    if torch.isclose(state_dict["transformer.wte.weight"], state_dict["lm_head.weight"]).all():
        del state_dict["lm_head.weight"]
    
    return { k: v.detach().clone() for k,v in state_dict.items() }


def get_model_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    return _get_state_dict(get_model(model_path))


def load_expert_paths(ds: datasets.Dataset):
    student_net = get_model("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left" # important for correct truncation
    tokenizer.padding_side = "left" 

    optim = torch.optim.Adam(student_net.parameters(), lr=expert_lr)

    expert_paths = [_get_state_dict(student_net)]
    step = 0
    for _i in range(num_experts):
        for _j in range(num_steps_per_expert):
            step += 1
            batch_idxs = random.sample(range(len(ds)), k=num_expert_datapoints)
            examples = ds.select(batch_idxs)
            examples = [
                f"{text} {label}" 
                for text, label in zip(examples[text_column_name], examples[label_column_name])
            ]
            tokens = tokenizer.batch_encode_plus(
                examples,
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=sequence_length, 
            )
            outputs = student_net(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )
            labels = tokens.input_ids.detach().clone()
            labels[:, :-1] = -100
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.reshape(labels.numel(), -1), 
                labels.reshape(labels.numel(),),
                ignore_index=-100
            )
            print(f"Step {step+1} | Loss = {loss:.3f}")
            loss.backward()
            optim.step()
            optim.zero_grad()
            student_net.zero_grad()
        expert_paths.append(_get_state_dict(student_net))

    # # random.shuffle(expert_revisions)
    # if args.max_experts is not None:
    #     expert_start_epoch = random.randint(0, len(expert_paths) - args.max_experts)
    #     expert_paths = expert_paths[expert_start_epoch:expert_start_epoch + args.max_experts]
    return expert_paths


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    assert args.expert_epochs < args.max_experts, "Expert epochs must be less than max experts"

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_save = []

    wandb.init(
        sync_tensorboard=False,
        project="dataset-distillation",
        config=args,
        mode="disabled",
    )

    args = type('', (), {})()
    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    if args.batch_syn is None:
        args.batch_syn = args.dataset_size # Full-batch GD
    print('Hyperparameters: \n', dict(vars(args)))

    # student_net = get_model(EXPERT_PATHS[0]).to(args.device)
    ds = datasets.load_dataset("fancyzhx/ag_news", split="train")
    X, Y = get_token_embeddings_syn(ds=ds)
    X = X.detach().to(args.device).requires_grad_(True)

    # train model for a couple steps
    student_net = get_model("gpt2")
    student_net = ReparamModule(student_net)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)

    # optimizer_token_embeddings = torch.optim.AdamW([token_embeddings_syn], lr=args.lr_img)
    optimizer_token_embeddings = torch.optim.SGD([X], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5) # momentum=0.5
    optimizer_token_embeddings.zero_grad()

    print('%s [time] training begins' % get_time())

    Z = X
    Λ = torch.rand_like(Z, device=Z.device)
    ρ = 1.0 # TODO: Argparse. Paper says: 
            #                   > ρ is chosen from
            #                   > the set {0.001, 0.05, 0.01, . . . , 10}

    # load expert trajectories
    buffer = load_expert_paths(ds=ds)
    for it in tqdm.trange(0, args.max_iterations+1, desc="iterations"):
        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        start_epoch = np.random.randint(0, len(buffer) - args.expert_epochs)
        starting_params = buffer[start_epoch]
        target_params = buffer[start_epoch+args.expert_epochs]
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for n, p in starting_params.items()], 0).requires_grad_(True)]

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

        aux_loss = (ρ / 2) * (X - Z - Λ / ρ).norm(p=2)
        (aux_loss + param_loss).backward()

        optimizer_token_embeddings.step()
        optimizer_lr.step()

        # 
        #  (2) Compute new Z based on projecting X back to word embedding space
        # 
        # TODO: Use a language model for this
        with torch.no_grad():
            embeddings = student_net.module.get_input_embeddings()
            embeddings_norm = embeddings.weight / embeddings.weight.norm(p=2, dim=1, keepdim=True)
            X_norm = X / X.norm(p=2, dim=2, keepdim=True)
            Z_tokens = (X_norm @ embeddings_norm.T).argmax(dim=-1)
            Z = embeddings(Z_tokens)

        # 
        #  (3) Update Λ for ADMM
        # 
        with torch.no_grad():
            Λ = Λ + ρ * (X - Z)


        ce_loss_avg = sum(ce_losses) / len(ce_losses)
        wandb.log(
            {
                "param_loss": param_loss.detach().cpu(),
                "param_loss_minus_one": (param_loss - 1).detach().cpu(),
                "param_dist": param_dist.detach().cpu(),
                "aux_loss":  aux_loss.detach().cpu(),
                "start_epoch": start_epoch,
                "ce_loss": ce_loss_avg,
                "token_grad_norm": X.grad.norm().detach().cpu(),
                "synth_lr": syn_lr.detach().cpu()
            }
        )

        for _ in student_params:
            del _

        if it % 10 == 0:
            print('%s iter = %04d, loss = %.8f, oneofloss = %.8f, ce_loss = %.8f' % (get_time(), it, param_loss.item(), (param_loss - 1).item(), ce_loss_avg))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')

    parser.add_argument("--dataset_size", "--ds", type=int, default=1000, help="size of distilled dataset")
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')

    parser.add_argument('--max_iterations', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--expert_epochs', type=int, default=1, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_experts', type=int, default=32, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    main(args)
