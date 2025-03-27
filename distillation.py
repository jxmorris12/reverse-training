import gc
import os
import pickle
import time

import torch
import transformers
import wandb


from optimizers import ADMMOptimizer, GCGOptimizer, GCGAOptimizer, SELECTOptimizer
from reparam_module import ReparamModule
from utils import (
    device, 
    get_model, 
    get_world_size,
    get_token_embeddings_random_soft,
    get_token_embeddings_from_classification_dataset, 
    train_expert_model,
    trange_if_main_worker,
    ClassificationDataset,
)
from utils.dataset_evaluation_metrics import evaluate_dataset_similarity


class DatasetDistiller:
    def __init__(self, args):
        # train model for a couple steps
        self.args = args
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.initial_student_net = get_model(args.base_model_name_or_path)
        self.student_net = ReparamModule(get_model(args.base_model_name_or_path)).to(device)
        self.classification_dataset = self._load_dataset()
        self.dataset_token_counts = None

    def _init_synthetic_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.args.token_init == "random":
            student_token_embeddings = self.initial_student_net.get_input_embeddings().to(device)
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.base_model_name_or_path)
            X_tokens = torch.randint(
                low=0,
                high=tokenizer.vocab_size,
                size=[self.args.dataset_size, self.args.sequence_length - 1],
                device=device,
            )
            X = student_token_embeddings(X_tokens)

            num_classes = 4
            CLASS_MAP = torch.tensor([352,  362,  657,  513], device=device) # for AG_News... tmp
            token_labels_syn = torch.randint(low=0, high=num_classes, size=[self.args.dataset_size], device=device)
            Y = CLASS_MAP[token_labels_syn]

        elif self.args.token_init == "random_soft":
            X, Y = [], []
            init_minibatch_size = 512
            for _ in trange_if_main_worker(0, self.args.dataset_size, init_minibatch_size, desc="Initializing"):
                x, y = get_token_embeddings_random_soft(
                    student_net=self.initial_student_net,
                    dataset_size=min(init_minibatch_size, self.args.dataset_size),
                    sequence_length=self.args.sequence_length,
                )
                X.append(x)
                Y.append(y)
            X = torch.cat(X, dim=0)
            Y = torch.cat(Y, dim=0)
        elif self.args.token_init == "dataset":
            X, Y = get_token_embeddings_from_classification_dataset(
                dataset_size=self.args.dataset_size, 
                sequence_length=self.args.sequence_length,
                classification_dataset=self.classification_dataset
            )
        else:
            raise NotImplementedError(f"Token init {self.args.token_init} not implemented")
        X = X.detach().to(device).requires_grad_(True)
        return X, Y
    
    def _load_dataset(self) -> ClassificationDataset:
        return ClassificationDataset.from_dataset_name(self.args.dataset, seed=self.args.seed)
    
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
                true_classification_dataset=self.classification_dataset,
            )
        else:
            raise NotImplementedError(f"Optimizer {self.args.discrete_optimizer} not implemented")
        
        # Use label map from the classification dataset
        optimizer.dataset_label_map = self.classification_dataset.label_map
        return optimizer
    
    def _log_table(self, tokens: torch.Tensor, labels: torch.Tensor, step: int) -> None:
        tokens = self.tokenizer.batch_decode(tokens.cpu(), add_special_tokens=False)
        if labels is not None:
            labels = self.tokenizer.batch_decode(labels.cpu(), add_special_tokens=False)
            labels = list(map(lambda x: self.classification_dataset.label_map.get(x.strip(), "[?]"), labels))
        else:
            labels = [""] * len(tokens)
        table_data = [(i, T, L) for i, (T, L) in enumerate(zip(tokens, labels))]
        tokens_table = wandb.Table(data=table_data, columns=["index", "text", "label"])
        wandb.log({ "Z": tokens_table }, step=step)

    def _evaluate_and_log(self, Z_text: torch.Tensor, Z_tokens: torch.Tensor, Y: torch.Tensor, step: int) -> dict[str, float]:
        self._log_table(Z_tokens, Y, step=step)
        tokens = Z_tokens.cpu()
        labels = Y.cpu()

        # TODO: Recheck the below logic!
        # compare tokens to dataset_token_counts
        dataset_token_counts = self.dataset_token_counts.bool()
        token_counts = tokens.flatten().bincount(minlength=self.tokenizer.vocab_size).bool()
        
        # compute precision & recall
        precision = (dataset_token_counts & token_counts).sum() / token_counts.sum()
        recall = (dataset_token_counts & token_counts).sum() / dataset_token_counts.sum()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        dataset_metrics = {
            "dataset_token_precision": precision.detach().cpu().item(),
            "dataset_token_recall": recall.detach().cpu().item(),
            "dataset_token_f1": f1.detach().cpu().item(),
        }

        # run full evaluation
        _, __, evaluation_metrics = train_expert_model(
            base_model_name_or_path=self.args.base_model_name_or_path,
            num_experts=self.args.num_eval_epochs,
            num_steps_per_expert=max(1, len(tokens) // self.args.expert_batch_size),
            expert_batch_size=self.args.expert_batch_size,
            expert_lr=self.args.expert_lr,
            sequence_length=self.args.sequence_length,
            ds=self.classification_dataset.dataset,
            text_column_name=self.classification_dataset.text_column_name,
            label_column_name=self.classification_dataset.label_column_name,
            ds_tokens=tokens,
            ds_labels=labels,
        )

        # log
        metrics = { **dataset_metrics, **evaluation_metrics }
        wandb.log(metrics, step=step)
        return metrics
    
    def _distributed_broadcast_everything(self, expert_buffer: list[dict[str, torch.Tensor]], dataset_token_counts: torch.Tensor) -> None:
        if get_world_size() <= 1:
            return
        # broadcast expert_buffer from rank 0 to all other ranks
        # for i in range(len(expert_buffer)):
        #     expert_buffer[i] = torch.broadcast_coalesced(expert_buffer[i], devices=[0])[0]
        torch.distributed.broadcast_object_list(expert_buffer, src=0)
        # broadcast dataset_token_counts from rank 0 to all other ranks
        torch.distributed.broadcast(dataset_token_counts, src=0)
    
    def _run_defense(self, expert_buffer: list[dict[str, torch.Tensor]]) -> tuple[list[dict[str, torch.Tensor]], dict[str, float]]:
        if self.args.defense == None:
            return expert_buffer, {}
        else:
            raise NotImplementedError(f"Defense {self.args.defense} not implemented")

    def _run_distillation(self) -> tuple:
        # initialize parameters & optimizers
        discrete_optimizer = self._init_discrete_optimizer()

        # load/generate expert trajectories
        expert_buffer, dataset_token_counts, expert_evaluation_metrics = train_expert_model(
            base_model_name_or_path=self.args.base_model_name_or_path,
            num_experts=self.args.num_experts,
            num_steps_per_expert=self.args.num_steps_per_expert,
            expert_batch_size=self.args.expert_batch_size,
            expert_lr=self.args.expert_lr,
            sequence_length=self.args.sequence_length,
            ds=self.classification_dataset.dataset,
            text_column_name=self.classification_dataset.text_column_name,
            label_column_name=self.classification_dataset.label_column_name,
        )
        new_expert_buffer, new_expert_evaluation_metrics = self._run_defense(expert_buffer)
        self.dataset_token_counts = dataset_token_counts.cpu()
        
        # handle distributed
        self._distributed_broadcast_everything(expert_buffer, dataset_token_counts)

        # run optimization
        pbar = trange_if_main_worker(0, self.args.max_iterations+1, desc="iterations")
        all_evaluation_metrics = []
        total_time_in_evaluation = 0
        for step in pbar:
            Z_text, Z_tokens, Y, metrics = discrete_optimizer.step(step, expert_buffer)
            pbar.set_postfix(**metrics)
            wandb.log(metrics, step=step)

            if (step + 1) % self.args.eval_every == 0:
                eval_start_time = time.time()
                evaluation_metrics = self._evaluate_and_log(Z_text, Z_tokens, Y, step=step)
                evaluation_metrics["step"] = step
                all_evaluation_metrics.append(evaluation_metrics)
                eval_end_time = time.time()
                total_time_in_evaluation += eval_end_time - eval_start_time

            # clear cache
            gc.collect()
            torch.cuda.empty_cache()

            if discrete_optimizer.should_stop:
                break

        print("Stopping distillation...")
        eval_start_time = time.time()
        final_evaluation_metrics = self._evaluate_and_log(Z_text, Z_tokens, Y, step=step)
        wandb.finish()
        eval_end_time = time.time()
        total_time_in_evaluation += eval_end_time - eval_start_time

        output = { 
            "args": dict(vars(self.args)),
            "expert_evaluation_metrics": expert_evaluation_metrics,
            "evaluation_metrics": final_evaluation_metrics,
            "total_time_in_evaluation": total_time_in_evaluation,
        }

        # clear cache oncemore
        gc.collect()
        torch.cuda.empty_cache()

        return (Z_text, Z_tokens.cpu(), Y.cpu(), output)

    def run_distillation(self):
        import time
        start_time = time.time()
        output_dataset, tokens, labels, output_metrics = self._run_distillation()
        end_time = time.time()
        print(f"Distillation total time: {end_time - start_time} seconds")

        # Compute dataset-level metrics
        input_dataset = self.classification_dataset.dataset["train"][self.classification_dataset.text_column_name]
        print(f"[run_distillation] Computing dataset-level metrics with {len(input_dataset)} examples and {len(output_dataset)} output examples")
        dataset_evaluation_metrics = evaluate_dataset_similarity(input_dataset, output_dataset, max_tokens=self.args.sequence_length)
        print(dataset_evaluation_metrics)

        data = {
            "data": {
                "tokens": tokens,
                "labels": labels,
            },
            "time_elapsed": (end_time - start_time),
            **dataset_evaluation_metrics,
            **output_metrics,
        }
        output_dir = os.path.join(os.path.dirname(__file__), self.args.results_dir)
        os.makedirs(output_dir, exist_ok=True)
        pkl_path = os.path.join(output_dir, f"{self.args.exp_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        return data