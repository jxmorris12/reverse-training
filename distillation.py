import datasets
import torch
import transformers
import wandb
import tqdm

from optimizers import ADMMOptimizer, GCGOptimizer, GCGAOptimizer, SELECTOptimizer
from reparam_module import ReparamModule
from utils import device, get_model, get_token_embeddings_random, get_token_embeddings_from_dataset, load_expert_trajectories


class DatasetDistiller:
    def __init__(self, args):
        # train model for a couple steps
        self.args = args
        tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.initial_student_net = get_model("gpt2")
        self.student_net = ReparamModule(get_model("gpt2")).to(device)
        self.ds, self.ds_text_column_name, self.ds_label_column_name = self._init_dataset()

    def _init_synthetic_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.args.token_init == "random":
            X, Y = [], []
            init_minibatch_size = 512
            for _ in tqdm.trange(0, self.args.dataset_size, init_minibatch_size, desc="Initializing"):
                x, y = get_token_embeddings_random(
                    dataset_size=min(init_minibatch_size, self.args.dataset_size),
                    sequence_length=self.args.sequence_length,
                )
                X.append(x)
                Y.append(y)
            X = torch.cat(X, dim=0)
            Y = torch.cat(Y, dim=0)
        elif self.args.token_init == "dataset":
            X, Y = get_token_embeddings_from_dataset(
                dataset_size=self.args.dataset_size, 
                sequence_length=self.args.sequence_length,
                ds=self.ds, 
                text_column_name=self.ds_text_column_name
            )
        else:
            raise NotImplementedError(f"Token init {args.token_init} not implemented")
        X = X.detach().to(device).requires_grad_(True)
        return X, Y
    
    def _init_dataset(self) -> tuple[datasets.Dataset, str, str]:
        if self.args.dataset == "ag_news":
            ds = datasets.load_dataset("fancyzhx/ag_news", split="train")
            text_column_name = "text"        
            label_column_name = "label"
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented")
        
        return ds, text_column_name, label_column_name
    
    def _init_discrete_optimizer(self):
        X, Y = self._init_synthetic_data()
        if self.args.discrete_optimizer == "ADMM":
            optimizer = ADMMOptimizer(
                args=self.args,
                X=X, 
                Y=Y,
                tokenizer=self.tokenizer,
                student_net=self.student_net,
                initial_student_net=self.initial_student_net,
            )
        elif self.args.discrete_optimizer == "GCG":
            optimizer = GCGOptimizer(
                args=self.args,
                X=X, 
                Y=Y,
                tokenizer=self.tokenizer,
                student_net=self.student_net,
                initial_student_net=self.initial_student_net,
            )
        elif self.args.discrete_optimizer == "GCGA":
            optimizer = GCGAOptimizer(
                args=self.args,
                X=X, 
                Y=Y,
                tokenizer=self.tokenizer,
                student_net=self.student_net,
                initial_student_net=self.initial_student_net,
            )
        elif self.args.discrete_optimizer == "SELECT":
            optimizer = SELECTOptimizer(
                args=self.args,
                X=X, 
                Y=Y,
                tokenizer=self.tokenizer,
                student_net=self.student_net,
                initial_student_net=self.initial_student_net,
            )
        else:
            raise NotImplementedError(f"Optimizer {args.discrete_optimizer} not implemented")
        return optimizer

    def run_distillation(self):
        # load/generate expert trajectories
        expert_buffer = load_expert_trajectories(
            num_experts=self.args.num_experts,
            num_steps_per_expert=self.args.num_steps_per_expert,
            num_expert_datapoints=self.args.minibatch_size,
            expert_lr=self.args.expert_lr,
            sequence_length=self.args.sequence_length,
            ds=self.ds,
            text_column_name=self.ds_text_column_name,
            label_column_name=self.ds_label_column_name,
        )

        # initialize parameters & optimizers
        discrete_optimizer = self._init_discrete_optimizer()

        # run optimization
        pbar = tqdm.trange(0, self.args.max_iterations+1, desc="iterations")
        for it in pbar:
            Z, metrics = discrete_optimizer.step(it, expert_buffer)
            pbar.set_postfix(**metrics)
            wandb.log(metrics, step=it)