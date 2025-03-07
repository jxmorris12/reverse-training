import datasets
import torch
import transformers
import wandb
import tqdm

from optimizers import ADMMOptimizer, GCGOptimizer, GCGAOptimizer, SELECTOptimizer
from reparam_module import ReparamModule
from utils import device, get_model, get_token_embeddings_random, get_token_embeddings_from_dataset, train_expert_model


LABEL_MAP = {
    "0": "World",
    "1": "Sports",
    "2": "Business",
    "3": "Sci/Tech",
}


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
        self.dataset_token_counts = None

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
            ds = datasets.load_dataset("fancyzhx/ag_news")
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
    
    def _log_table(self, tokens: torch.Tensor, labels: torch.Tensor, step: int) -> None:
        tokens = self.tokenizer.batch_decode(tokens.cpu(), add_special_tokens=False)
        labels = self.tokenizer.batch_decode(labels.cpu(), add_special_tokens=False)
        labels = list(map(lambda x: LABEL_MAP[x.strip()], labels))
        table_data = [(i, T, L) for i, (T, L) in enumerate(zip(tokens, labels))]
        tokens_table = wandb.Table(data=table_data, columns=["index", "text", "label"])
        wandb.log({ "Z": tokens_table }, step=step)

    def _evaluate_and_log(self, tokens: torch.Tensor, labels: torch.Tensor, step: int) -> None:
        self._log_table(tokens, labels, step=step)
        # TODO: Recheck the below logic!
        # compare tokens to dataset_token_counts
        dataset_token_counts = self.dataset_token_counts.bool()
        token_counts = tokens.flatten().bincount(minlength=self.tokenizer.vocab_size).bool()
        
        # compute precision & recall
        precision = (dataset_token_counts & token_counts).sum() / token_counts.sum()
        recall = (dataset_token_counts & token_counts).sum() / dataset_token_counts.sum()
        f1 = 2 * (precision * recall) / (precision + recall)
        dataset_metrics = {
            "dataset_token_precision": precision.detach().cpu().item(),
            "dataset_token_recall": recall.detach().cpu().item(),
            "dataset_token_f1": f1.detach().cpu().item(),
        }

        # run full evaluation
        _, __, evaluation_metrics = train_expert_model(
            num_experts=self.args.num_experts,
            num_steps_per_expert=self.args.num_steps_per_expert,
            num_expert_datapoints=self.args.minibatch_size,
            expert_lr=self.args.expert_lr,
            sequence_length=self.args.sequence_length,
            ds=self.ds,
            text_column_name=self.ds_text_column_name,
            label_column_name=self.ds_label_column_name,
            ds_tokens=tokens,
            ds_labels=labels,
            num_evaluation_batches=10, # TODO: Argparse!
        )

        # log
        wandb.log({ **dataset_metrics, **evaluation_metrics }, step=step)

    def run_distillation(self):
        # load/generate expert trajectories
        expert_buffer, dataset_token_counts, _ = train_expert_model(
            num_experts=self.args.num_experts,
            num_steps_per_expert=self.args.num_steps_per_expert,
            num_expert_datapoints=self.args.minibatch_size,
            expert_lr=self.args.expert_lr,
            sequence_length=self.args.sequence_length,
            ds=self.ds,
            text_column_name=self.ds_text_column_name,
            label_column_name=self.ds_label_column_name,
        )
        self.dataset_token_counts = dataset_token_counts

        # initialize parameters & optimizers
        discrete_optimizer = self._init_discrete_optimizer()

        # run optimization
        pbar = tqdm.trange(0, self.args.max_iterations+1, desc="iterations")
        for it in pbar:
            Z, metrics = discrete_optimizer.step(it, expert_buffer)
            pbar.set_postfix(**metrics)
            wandb.log(metrics, step=it)

            if it % 100 == 0:
                self._evaluate_and_log(Z, discrete_optimizer.Y, step=it)