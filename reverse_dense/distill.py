import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from utils import get_network, get_eval_pool, evaluate_synset, get_time
import wandb
import copy
import random
from reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


PYTHIA_REVISIONS = [f"step{step}" for step in range(0, 128_000, 1000)]

def get_model_state_dict(model_name: str, revision: str) -> dict[str, torch.Tensor]:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, 
        revision=revision
    )
    return model.state_dict()

def main(args):
    random.seed(args.seed)
    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    assert args.expert_epochs < args.max_experts, "Expert epochs must be less than max experts"

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    hidden_size = 512
    sequence_length = 32
    
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

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
    print('Evaluation model pool: ', model_eval_pool)

    """ initialize the synthetic data """
    token_embeddings_syn = torch.randn(
        size=(args.dataset_size, sequence_length, hidden_size
        ), dtype=torch.float32
    )

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    print('initialize synthetic data from random noise')


    """ training """
    token_embeddings_syn = token_embeddings_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([token_embeddings_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    # load expert trajectories
    expert_revisions = PYTHIA_REVISIONS
    # random.shuffle(expert_revisions)
    if args.max_experts is not None:
        expert_start_epoch = random.randint(0, len(expert_revisions) - args.max_experts)
        expert_revisions = expert_revisions[expert_start_epoch:expert_start_epoch + args.max_experts]
    
    buffer = [get_model_state_dict(args.model_name, revision) for revision in expert_revisions]

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    student_net = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="eager",
        # attn_imports="flex_attention",
    ).to(args.device)  # get a random model
    student_net = ReparamModule(student_net)

    if args.distributed:
        student_net = torch.nn.DataParallel(student_net)

    student_net.train()

    for it in range(0, args.Iteration+1):
        save_this_it = False

        wandb.log({"Progress": it, "Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        start_epoch = np.random.randint(0, len(buffer) - args.expert_epochs)
        starting_params = buffer[start_epoch]
        print("start_epoch", start_epoch, "len(buffer)", len(buffer))

        target_params = buffer[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for n, p in target_params.items()], 0)
        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for n, p in starting_params.items()], 0).requires_grad_(True)]
        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for n, p in starting_params.items()], 0)

        syn_images = token_embeddings_syn

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):
            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()
            x = syn_images[these_indices]

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            output = student_net(inputs_embeds=x, flat_param=forward_params)

            # autoregressive mimicry loss            
            outputs_softmax = F.log_softmax(output.logits[:, :-1], dim=-1)
            with student_net.unflattened_param(forward_params):
                word_embeddings = student_net.module.get_input_embeddings().to(args.device)

            mimiced_embeddings = torch.einsum("bsv,vd->bsd", outputs_softmax, word_embeddings.weight)
            mimiced_embeddings = mimiced_embeddings.view(-1, hidden_size)

            x_inputs = x[:, 1:, :].contiguous().view(-1, hidden_size)
            assert x_inputs.shape == mimiced_embeddings.shape, f"invalid shape: {x_inputs.shape} != {mimiced_embeddings.shape}"
            mse_loss = torch.nn.functional.mse_loss(mimiced_embeddings, x_inputs, reduction="mean")

            grad = torch.autograd.grad(mse_loss, student_params[-1], create_graph=True)[0]

            student_params.append(student_params[-1] - syn_lr * grad)


        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
    
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            # https://github.com/pytorch/pytorch/issues/116350
            grand_loss.backward()

        optimizer_img.step()
        optimizer_lr.step()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--res', type=int, default=128, help='resolution for imagenet')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')

    parser.add_argument("--dataset_size", "--ds", type=int, default=1000, help="size of distilled dataset")

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-70m', help='model name')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_experts', type=int, default=16, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    main(args)