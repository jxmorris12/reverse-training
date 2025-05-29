import gc
import os
import pickle
import time

import datasets
import torch
import transformers
import wandb


from optimizers import ADMMOptimizer, GCGOptimizer, GCGAOptimizer, SELECTOptimizer
from reparam_module import ReparamModule
from utils import (
    device, 
    get_model, 
    get_world_size,
    # get_token_embeddings_random_soft,
    # get_token_embeddings_from_classification_dataset, 
    trange_if_main_worker,
    ClassificationDataset,
)
from utils.core import _train_expert_model_uncached, train_expert_model, ExpertModel

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
    
    def _load_dataset(self) -> ClassificationDataset:
        return ClassificationDataset.from_dataset_name(self.args.dataset, seed=self.args.seed)
    
    def _init_discrete_optimizer(self, expert_model: ExpertModel):
        X, Y = [None, None]
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
                expert_model=expert_model,
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

    def _evaluate_and_log(
            self, 
            expert_model: ExpertModel, 
            Z_text: torch.Tensor, 
            Z_tokens: torch.Tensor, 
            Y: torch.Tensor, 
            step: int
        ) -> dict[str, float]:
        self._log_table(Z_tokens, Y, step=step)
        Y = Y.cpu().tolist()
        
        # compute token-level precision & recall
        # precision = (dataset_token_counts & token_counts).sum() / token_counts.sum()
        # recall = (dataset_token_counts & token_counts).sum() / dataset_token_counts.sum()
        # f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        # dataset_metrics = {
        #     "dataset_token_precision": precision.detach().cpu().item(),
        #     "dataset_token_recall": recall.detach().cpu().item(),
        #     "dataset_token_f1": f1.detach().cpu().item(),
        # }
        train_ds = datasets.Dataset.from_dict(
            {
                self.classification_dataset.text_column_name: Z_text,
                self.classification_dataset.label_column_name: Y,
            }
        )
        test_ds = self.classification_dataset.dataset["test"]
        ds = datasets.DatasetDict({
            "train": train_ds,
            "test": test_ds,
        })

        assert set(Y) <= set(test_ds[self.classification_dataset.label_column_name])

        # run full evaluation
        expert, __, evaluation_metrics = _train_expert_model_uncached(
            base_model_name_or_path=self.args.base_model_name_or_path,
            num_experts=self.args.num_eval_epochs,
            num_steps_per_expert=max(1, len(train_ds) // self.args.expert_batch_size),
            expert_batch_size=self.args.expert_batch_size,
            expert_lr=self.args.expert_lr,
            sequence_length=self.args.sequence_length,
            ds=ds,
            text_column_name=self.classification_dataset.text_column_name,
            label_column_name=self.classification_dataset.label_column_name,
        )

        # log
        metrics = { **evaluation_metrics }
        wandb.log(metrics, step=step)
        return metrics
    
    def _distributed_broadcast_everything(self, expert_buffer: list[dict[str, torch.Tensor]]) -> None:
        if get_world_size() <= 1:
            return
        # broadcast expert_buffer from rank 0 to all other ranks
        torch.distributed.broadcast_object_list(expert_buffer, src=0)
    
    def _run_defense(self, expert_buffer: list[dict[str, torch.Tensor]]) -> tuple[list[dict[str, torch.Tensor]], dict[str, float]]:
        if self.args.defense == None:
            return expert_buffer, {}
        else:
            raise NotImplementedError(f"Defense {self.args.defense} not implemented")

    def _run_distillation(self) -> tuple:
        # load/generate expert trajectories
        expert_model, expert_buffer, expert_evaluation_metrics = train_expert_model(
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

        # initialize parameters & optimizers
        discrete_optimizer = self._init_discrete_optimizer(expert_model=expert_model)

        # new_expert_buffer, new_expert_evaluation_metrics = self._run_defense(expert_buffer)

        # handle distributed
        self._distributed_broadcast_everything(expert_buffer)

        print(f"[run_distillation - start] Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # run optimization
        pbar = trange_if_main_worker(0, self.args.max_iterations+1, desc="iterations")
        all_evaluation_metrics = []
        total_time_in_evaluation = 0
        for step in pbar:
            Z_text, Z_tokens, Y, metrics = discrete_optimizer.step(step, expert_buffer)
            pbar.set_postfix(**metrics)
            wandb.log(metrics, step=step)

            # clear cache
            gc.collect()
            torch.cuda.empty_cache()

            if (step + 1) % self.args.eval_every == 0:
                eval_start_time = time.time()
                evaluation_metrics = self._evaluate_and_log(
                    expert_model=expert_model, 
                    Z_text=Z_text, 
                    Z_tokens=Z_tokens, 
                    Y=Y, 
                    step=step
                )
                evaluation_metrics["step"] = step
                all_evaluation_metrics.append(evaluation_metrics)
                eval_end_time = time.time()
                total_time_in_evaluation += eval_end_time - eval_start_time
                gc.collect()
                torch.cuda.empty_cache()

            if discrete_optimizer.should_stop:
                break

        print(f"[run_distillation - end] Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print("Stopping distillation...")
        eval_start_time = time.time()
        final_evaluation_metrics = self._evaluate_and_log(
            expert_model=expert_model, 
            Z_text=Z_text, 
            Z_tokens=Z_tokens, 
            Y=Y, 
            step=step
        )
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