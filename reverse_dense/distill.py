import argparse
import copy
import glob
import os

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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# PYTHIA_REVISIONS = [f"step{step}" for step in range(0, 128_000, 1000)]
EXPERT_PATHS = glob.glob("/home/jxm3/research/reverse-training/train/saves/checkpoint-*")

def get_model_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    model = limit_layers(model, 4) # TODO: Override from config?
    print("Warning: Limiting layers to 4")
    return model.state_dict()

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    assert args.expert_epochs < args.max_experts, "Expert epochs must be less than max experts"

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    hidden_size = 512
    sequence_length = 32
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

    args.distributed = torch.cuda.device_count() > 1
    print('Hyper-parameters: \n', args.__dict__)

    student_net = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="eager",
        # attn_imports="flex_attention",
    ).to(args.device)  # get a random model
    student_net = ReparamModule(student_net)

    if args.distributed:
        student_net = torch.nn.DataParallel(student_net)

    student_net.train()
    """ initialize the synthetic data """
    student_token_embeddings = student_net.module.get_input_embeddings().to(args.device)
    # TODO: Consider other initialization methods
    def simple_soft_one_hot(x, num_classes, temperature=1.0):
        one_hot = F.one_hot(x, num_classes).float()
        return F.softmax(one_hot / temperature, dim=-1)

    rand_token_idxs = torch.randint(0, student_token_embeddings.num_embeddings, (args.dataset_size, sequence_length), device=args.device)
    rand_token_one_hots = simple_soft_one_hot(rand_token_idxs, student_token_embeddings.num_embeddings, temperature=0.1)
    token_embeddings_syn = rand_token_one_hots @ student_token_embeddings.weight
    # token_embeddings_syn = token_embeddings_syn[None, None].repeat(args.dataset_size, sequence_length, 1)
    # token_embeddings_syn = torch.randn(
    #     size=(args.dataset_size, sequence_length, hidden_size
    #     ), dtype=torch.float32
    # )
    assert token_embeddings_syn.shape == (args.dataset_size, sequence_length, hidden_size), f"invalid shape: {token_embeddings_syn.shape}"

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    print('initialize synthetic data from random noise')


    #   training
    token_embeddings_syn = token_embeddings_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    # optimizer_token_embeddings = torch.optim.AdamW([token_embeddings_syn], lr=args.lr_img)
    optimizer_token_embeddings = torch.optim.SGD([token_embeddings_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5) # momentum=0.5
    optimizer_token_embeddings.zero_grad()

    print('%s [time] training begins'%get_time())

    # load expert trajectories
    expert_paths = EXPERT_PATHS
    # random.shuffle(expert_revisions)
    if args.max_experts is not None:
        expert_start_epoch = random.randint(0, len(expert_paths) - args.max_experts)
        expert_paths = expert_paths[expert_start_epoch:expert_start_epoch + args.max_experts]
    
    buffer = [get_model_state_dict(model_path) for revision in expert_paths]
    for it in tqdm.trange(0, args.Iteration+1, desc="Iterations"):
        save_this_it = False

        wandb.log({"Progress": it, "Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        start_epoch = np.random.randint(0, len(buffer) - args.expert_epochs)
        starting_params = buffer[start_epoch]
        target_params = buffer[start_epoch+args.expert_epochs]
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for n, p in starting_params.items()], 0).requires_grad_(True)]

        target_params = torch.cat([p.data.to(args.device).reshape(-1) for n, p in target_params.items()], 0)
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for n, p in starting_params.items()], 0)


        syn_token_embeddings = token_embeddings_syn

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []
        ce_losses = []

        for step in tqdm.trange(args.syn_steps, desc="Synthetic Steps", leave=False):
            if not indices_chunks:
                indices = torch.randperm(len(syn_token_embeddings))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()
            x = syn_token_embeddings[these_indices]

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            output = student_net(inputs_embeds=x, flat_param=forward_params)

            # autoregressive mimicry loss            
            outputs_softmax = F.softmax(output.logits[:, :-1], dim=-1)
            outputs_softmax = outputs_softmax.contiguous().view(-1, outputs_softmax.shape[-1])

            with student_net.unflattened_param(forward_params):
                token_embeddings = student_net.module.get_input_embeddings().to(args.device)
            x_inputs = x[:, 1:, :].contiguous().view(-1, hidden_size)
            x_inputs_logits = x_inputs @ token_embeddings.weight.T
            #  TODO: Consider Gumbel
            inputs_softmax = F.softmax(x_inputs_logits * 80, dim=-1) # TODO: Think about temp
            
            assert outputs_softmax.shape == inputs_softmax.shape, f"invalid shape: {outputs_softmax.shape} != {inputs_softmax.shape}"
            # ce_loss = torch.nn.functional.mse_loss(mimiced_embeddings, x_inputs, reduction="sum") / sequence_length
            ce_loss = torch.nn.functional.cross_entropy(outputs_softmax, inputs_softmax, reduction="mean")
            ce_losses.append(ce_loss.detach().item())

            # exit on nan
            if torch.isnan(ce_loss):
                print("nan detected - stopping!")
                exit()

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            student_params.append(student_params[-1] - syn_lr * grad)


        # param_loss = torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        # param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        # param_loss /= num_params
        param_dist /= num_params
        # param_loss /= param_dist
        param_loss = 1 - torch.nn.functional.cosine_similarity(student_params[-1] - starting_params, target_params - starting_params, dim=0)
        # param_loss = 1 - torch.nn.functional.cosine_similarity(student_params[-1] - target_params, starting_param - target_params, dim=0)

        optimizer_token_embeddings.zero_grad()
        optimizer_lr.zero_grad()
    
        # https://github.com/pytorch/pytorch/issues/116350
        param_loss.backward()

        optimizer_token_embeddings.step()
        optimizer_lr.step()

        ce_loss_avg = sum(ce_losses) / len(ce_losses)
        wandb.log(
            {
                "param_loss": param_loss.detach().cpu(),
                # "param_loss_minus_one": (param_loss - 1).detach().cpu(),
                "param_dist": param_dist.detach().cpu(),
                "start_epoch": start_epoch,
                "ce_loss": ce_loss_avg,
                "token_grad_norm": token_embeddings_syn.grad.norm().detach().cpu(),
            }
        )

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.8f, oneofloss = %.8f, ce_loss = %.8f' % (get_time(), it, param_loss.item(), (param_loss - 1).item(), ce_loss_avg))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')

    parser.add_argument("--dataset_size", "--ds", type=int, default=1000, help="size of distilled dataset")
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')

    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m', help='model name')

    parser.add_argument('--expert_epochs', type=int, default=1, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_experts', type=int, default=16, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    main(args)