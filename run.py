import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import wandb

from distillation import DatasetDistiller
from utils import get_rank, get_time


def main(args):
    if torch.cuda.device_count() > 1:
        dist.init_process_group('nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
    
    random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    torch.cuda.manual_seed_all(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())

    wandb.init(
        sync_tensorboard=False,
        project="dataset-distillation",
        config=args,
        # mode="disabled" if get_rank() > 0 else os.getenv("WANDB_MODE", "online"),
    )

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    if args.minibatch_size is None:
        args.minibatch_size = args.dataset_size # Full-batch GD

    print(f"[rank {get_rank()}] Distillation begins:", get_time())
    distiller = DatasetDistiller(args=args)
    distiller.run_distillation()
        
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ag_news", help="dataset to distill (optional)")
    parser.add_argument("--dataset_size", "--ds", type=int, default=100, help="size of distilled dataset")
    parser.add_argument("--minibatch_size", "--batch_size_synthetic", type=int, default=None, help='minibatch size for synthetic data (optional)')
    parser.add_argument("--eval_every", type=int, default=200, help="how many steps between evaluations")

    parser.add_argument('--max_iterations', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--max_iterations_x', type=int, default=40, help='how many gradient steps per X update')
    parser.add_argument('--pretrain_iterations_x', type=int, default=100, help='how many gradient steps in initial X training phase')

    parser.add_argument('--lr_tokens', type=float, default=0.001, help='learning rate for updating synthetic tokens')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.001, help='initialization for synthetic learning rate')
    parser.add_argument('--syn_steps', type=int, default=8, help='how many steps to take on synthetic data')
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--token_init", type=str, default="random", help="how to initialize the synthetic tokens", choices=["random", "random_soft", "dataset"])

    parser.add_argument('--expert_epochs', type=int, default=1, help='how many expert epochs the target params are')
    parser.add_argument("--sequence_length", default=32, type=int, help="sequence length for the model")
    parser.add_argument("--num_experts", default=20, type=int, help="number of experts to read")
    parser.add_argument("--num_steps_per_expert", default=50, type=int, help="number of steps per expert")
    parser.add_argument("--num_expert_datapoints", default=256, type=int, help="number of datapoints per expert")
    parser.add_argument("--expert_lr", default=1e-4, type=float, help="learning rate for expert")

    parser.add_argument("--discrete_optimizer", "--opt", default="ADMM", type=str, help="discrete optimizer to use", choices=["ADMM", "GCG", "GCGA", "SELECT"])
    parser.add_argument("--admm_penalty_term", "--admm_rho", type=float, default=0.1, help="ADMM penalty term (ρ)")
    parser.add_argument("--gcg_search_width", type=int, default=8, help="how many random samples to take in GCG")
    parser.add_argument("--gcg_tokens_to_swap", type=int, default=1, help="how many tokens to swap in GCG")
    parser.add_argument("--gcg_documents_to_swap", type=int, default=None, help="how many documents to swap in GCG")

    parser.add_argument("--select_seed_dataset", type=str, default="nq", help="dataset to use for SELECT")
    parser.add_argument("--select_projection_dim", type=int, default=8192, help="projection dimension for SELECT")
    parser.add_argument("--select_steps_per_grad", type=int, default=256, help="how many steps between gradient rerankings in SELECT")
    parser.add_argument("--select_full_dataset_size", type=int, default=2048, help="how many examples to select per gradient reranking in SELECT")
    parser.add_argument("--select_lr_student", type=float, default=0.001, help="learning rate for SELECT")  
    parser.add_argument("--select_num_pseudoexperts", type=int, default=1, help="number of pseudoexperts to use in SELECT")
    parser.add_argument("--select_do_classification", type=bool, default=True, help="whether to do classification in SELECT")
    parser.add_argument("--select_batch_fill_strategy", type=str, default="greedy", help="strategy to fill the batch in SELECT", choices=["topk", "greedy"])
    parser.add_argument("--num_eval_epochs", type=int, default=50, help="number of evaluation epochs")


    parser.add_argument("--exp_name", type=str, required=True, help="experiment name [user-provided str]")

    args = parser.parse_args()

    main(args)
