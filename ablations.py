import json
import numpy as np
import pandas as pd
import random

from utils.dataset_evaluation_metrics import evaluate_dataset_similarity


def perform_ablation_study(ref_texts: list[str], rec_texts: list[str], replacement_percents: list[int], num_trials: int = 5, seed: int = 42) -> dict:
    """
    For each replacement percentage, randomly swap that percentage of entries in the recovered dataset with
    the corresponding entries from the reference dataset. For each replacement level, run multiple trials
    and average the evaluation metrics.
    
    Parameters:
      - ref_texts (list of str): The ground truth texts.
      - rec_texts (list of str): The recovered texts.
      - replacement_percents (list[int]): List of percentages (0 to 100) indicating the fraction of recovered examples to replace.
      - num_trials (int): Number of random trials to average over.
      - seed (int): Seed for reproducibility.
      
    Returns:
      - ablation_results (dict): Dictionary mapping replacement percentage to averaged metrics.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    n = len(rec_texts)
    ablation_results = {}
    
    for percent in replacement_percents:
        metrics_list = []
        num_replace = int(round((percent / 100.0) * n))
        for trial in range(num_trials):
            # Create a copy of the recovered texts
            modified_rec = rec_texts.copy()
            # Randomly select indices to replace
            indices = np.random.choice(n, size=num_replace, replace=False)
            # Replace recovered examples with the ground truth ones at the selected indices
            for idx in indices:
                modified_rec[idx] = ref_texts[idx]
            # Evaluate the similarity between the reference and modified recovered dataset
            metrics = evaluate_dataset_similarity(ref_texts, modified_rec)
            metrics_list.append(metrics)

        # Average over the trials for each metric.
        avg_full_ot = np.mean([m["full_ot_distance"] for m in metrics_list])
        avg_sinkhorn = np.mean([m["sinkhorn_distance"] for m in metrics_list])
        avg_optimal_matching_relaxed_wmd = np.mean([m["optimal_matching_relaxed_wmd"]["average_relaxed_wmd"] for m in metrics_list])
        avg_optimal_matching_relaxed_wmd_min = np.mean([m["optimal_matching_relaxed_wmd"]["min_relaxed_wmd"] for m in metrics_list])
        avg_optimal_matching_relaxed_wmd_max = np.mean([m["optimal_matching_relaxed_wmd"]["max_relaxed_wmd"] for m in metrics_list])

        avg_jaccard_examples = np.mean([m["jaccard_overlap_examples"] for m in metrics_list])
        avg_jaccard_vocab = np.mean([m["jaccard_overlap_vocabulary"] for m in metrics_list])

        avg_lev_avg = np.mean([m["levenshtein_stats"]["average_distance"] for m in metrics_list])
        avg_lev_min = np.mean([m["levenshtein_stats"]["min_distance"] for m in metrics_list])
        avg_lev_max = np.mean([m["levenshtein_stats"]["max_distance"] for m in metrics_list])
        
        avg_containment_similarity_examples = np.mean([m["containment_similarity_examples"] for m in metrics_list])
        avg_containment_similarity_vocab = np.mean([m["containment_similarity_vocabulary"] for m in metrics_list])
        
        ablation_results[f"{percent}%"] = {
            "average_full_ot_distance": float(avg_full_ot),
            "average_sinkhorn_distance": float(avg_sinkhorn),
            "average_optimal_matching_relaxed_wmd": float(avg_optimal_matching_relaxed_wmd),
            "average_jaccard_overlap_examples": float(avg_jaccard_examples),
            "average_jaccard_overlap_vocabulary": float(avg_jaccard_vocab),
            "average_containment_similarity_examples": float(avg_containment_similarity_examples),
            "average_containment_similarity_vocabulary": float(avg_containment_similarity_vocab),
            "average_levenshtein_distance": float(avg_lev_avg),
            "min_levenshtein_distance": float(avg_lev_min),
            "max_levenshtein_distance": float(avg_lev_max)
        }
    return ablation_results

def load_datasets(reference_parquet: str, recovered_json: str, ) -> tuple[list[str], list[str]]:
    """
    Load the ground truth dataset from a parquet file and the recovered dataset from a JSON file.
    Both files are expected to contain a "text" field.
    
    Parameters:
      - reference_parquet (str): Path to the ground truth parquet file.
      - recovered_json (str): Path to the recovered dataset JSON file.
      
    Returns:
      - (reference_texts, recovered_texts): Tuple of lists of strings.
    """
    # Load the reference dataset (e.g., train_ds.parquet)
    ref_df = pd.read_parquet(reference_parquet)
    # Assume the column "text" exists
    reference_texts = ref_df["text"].tolist()
    
    # Load the recovered dataset (e.g., ex_1000.json)
    rec_df = pd.read_json(recovered_json, orient="split")

    # Assume the column "text" exists
    recovered_texts = rec_df["text"].tolist()
    
    return reference_texts, recovered_texts

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study on dataset recovery")
    parser.add_argument("--reference", required=True, help="Path to reference dataset (parquet)")
    parser.add_argument("--recovered", required=True, help="Path to recovered dataset (json)")
    parser.add_argument("--output", required=True, help="Path to output file")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per percentage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--percentages", type=int, nargs="+", help="Replacement percentages")
    
    args = parser.parse_args()
  
    
    # File paths (adjust if needed)
    reference_file = args.reference
    recovered_file = args.recovered
    
    # Load datasets and extract the "text" field
    reference_texts, recovered_texts = load_datasets(reference_file, recovered_file)
    
    # For ablation study, we assume both lists are of equal length.
    # If not, you might need to sample or match lengths.

    # Found the "problem" #
    # min_len = min(len(reference_texts), len(recovered_texts))
    # reference_texts = reference_texts[:min_len]
    # recovered_texts = recovered_texts[:min_len]
    
    # Define the replacement percentages: 0%, 10%, ..., 100%
    replacement_percents = list(range(0, 101, 10))
    
    # Perform ablation study
    ablation_results = perform_ablation_study(reference_texts, recovered_texts, replacement_percents, num_trials=1, seed=42)
    
    # Save the ablation results to a JSON file
    output_filename = args.output
    with open(output_filename, "w") as f:
        json.dump(ablation_results, f, indent=4)
    
    print(f"Ablation study results saved to {output_filename}")
