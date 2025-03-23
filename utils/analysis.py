import glob
import pandas as pd
import pickle
import os

def load_results(path: str) -> dict:
    files = glob.glob(os.path.join(path, "*.pkl"))
    results = []
    for file in files:
        with open(file, "rb") as f:
            results.append(pickle.load(f))
    return results

def load_results_as_df(path: str) -> pd.DataFrame:
    results = load_results(path)
    n_results = len(results)
    results = [ex for ex in results if ("evaluation_metrics" in ex and len(ex["evaluation_metrics"]) > 0)]
    print(f"Filtered {n_results - len(results)}/{n_results} results with no evaluation metrics")
    print(results[0].keys())
    parsed_results = [
        {
            **ex["args"],
            **ex["expert_evaluation_metrics"],
            **ex["evaluation_metrics"],
            "dataset_sinkhorn_distance": ex["sinkhorn_distance"],
            "dataset_full_ot_distance": ex["full_ot_distance"],
            "dataset_jaccard_overlap_examples": ex["jaccard_overlap_examples"],
            "dataset_jaccard_overlap_vocabulary": ex["jaccard_overlap_vocabulary"],
        } for ex in results
    ]
    return pd.DataFrame(parsed_results)