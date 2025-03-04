import argparse
import random
import numpy as np
import torch
import transformers
import wandb

from distillation import DatasetDistiller
from utils import device, get_time, limit_layers


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
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

    print("[time] Distillation begins:", get_time())
    distiller = DatasetDistiller(args=args)
    distiller.run_distillation()
        
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ag_news", help="dataset to distill (optional)")
    parser.add_argument("--dataset_size", "--ds", type=int, default=100, help="size of distilled dataset")
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
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--token_init", type=str, default="random", help="how to initialize the synthetic tokens", choices=["random", "dataset"])

    parser.add_argument("--sequence_length", default=32, type=int, help="sequence length for the model")
    parser.add_argument("--num_experts", default=20, type=int, help="number of experts to read")
    parser.add_argument("--num_steps_per_expert", default=50, type=int, help="number of steps per expert")
    parser.add_argument("--num_expert_datapoints", default=256, type=int, help="number of datapoints per expert")
    parser.add_argument("--expert_lr", default=1e-4, type=float, help="learning rate for expert")


    args = parser.parse_args()

    main(args)
