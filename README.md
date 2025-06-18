# Approximating Language Model Training Data from Weights

This repository contains all the code for our paper "Approximating Language Model Training from Weights".  It also contains implementations of discrete optimizers such as ADMM and GCG and helper functions for finetuning language models with various datasets and objectives, computing per-example gradients, and projecting gradients using JL based on code from [trak](https://github.com/MadryLab/trak).

<img src="https://github.com/user-attachments/assets/4322cadd-9561-4203-adbd-a96e3d2af26a" width="500" />



## Example commands

### SELECT – LLAMA (SFT)

```bash
python run.py --max_iterations 1 --num_experts 1 --num_steps_per_expert 938 --expert_batch_size 16 --dataset msmarco_10000 --opt SELECT --expert_lr 2e-5 --select_seed_dataset nq_100000 --select_num_pseudoexperts 1 --select_steps_per_grad -1 --select_full_dataset_size 500 --exp_name msmarco_10000-nq_100000-500-greedy-45-4096-llama3-3b-1 --seed 45 --select_batch_fill_strategy greedy --select_projection_dim 4096 --select_label_strategy auto --results_dir results/main-exp-llama3b --model llama3-3b --select_use_expert_grads 0
```

### SELECT – GPT2 (Classification)

```bash
python run.py --max_iterations 1 --num_experts 1 --num_steps_per_expert 938 --expert_batch_size 32 --dataset dbpedia_10000 --opt SELECT --expert_lr 2e-5 --select_seed_dataset nq_10000 --select_num_pseudoexperts 1 --select_steps_per_grad -1 --select_full_dataset_size 10000 --exp_name dbpedia_10000-nq_10000-10000-greedy-45-4096-gpt2-0 --seed 45 --select_batch_fill_strategy greedy --select_projection_dim 4096 --select_label_strategy auto --results_dir results/main-exp-gpt2 --model gpt2 --select_use_expert_grads 0
```

### Optimizer ablation

```bash
python run.py --max_iterations 1 --num_experts 1 --num_steps_per_expert 938 --expert_batch_size 32 --dataset msmarco_10000 --opt SELECT --expert_lr 2e-5 --select_seed_dataset nq_10000 --select_num_pseudoexperts 1 --select_steps_per_grad -1 --select_full_dataset_size 1000 --exp_name msmarco_10000-nq_10000-1000-greedy-43-4096-gpt2-1 --seed 43 --select_batch_fill_strategy greedy --select_projection_dim 4096 --select_label_strategy auto --results_dir results/optimizer-ablation --model gpt2 --select_use_expert_grads 0 --optimizer sgd --optimizer_test sgd
```



At the start of the project, this code was forked from https://github.com/GeorgeCazenavette/mtt-distillation/tree/main.
