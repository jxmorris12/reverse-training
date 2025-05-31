import glob
import pandas as pd
import pickle
import os

def load_results(path: str) -> dict:
    pkl_glob = os.path.join(path, "*.pkl")
    files = glob.glob(pkl_glob)
    print(f"Found {len(files)} files in {pkl_glob}")
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
            **{ f"expert_evaluation_{k}": v for k, v in ex["expert_evaluation_metrics"].items() },
            **{ f"evaluation_{k}": v for k, v in ex["evaluation_metrics"].items() },
            "dataset_sinkhorn_distance": ex.get("sinkhorn_distance", None),
            "dataset_full_ot_distance": ex.get("full_ot_distance", None),
            "dataset_jaccard_overlap_examples": ex.get("jaccard_overlap_examples", None),
            "dataset_jaccard_overlap_vocabulary": ex.get("jaccard_overlap_vocabulary", None),
            "dataset_containment_similarity_examples": ex.get("containment_similarity_examples", None),
            "dataset_containment_similarity_vocabulary": ex.get("containment_similarity_vocabulary", None),
            # "dataset_levenshtein_stats": ex["levenshtein_stats"],
            # "dataset_discrete_ot_distance_levenshtein": ex["discrete_ot_distance_levenshtein"],
            # "dataset_discrete_ot_distance_jaccard": ex["discrete_ot_distance_jaccard"],
            **{
                f"dataset_levenshtein_stats_{k}": v
                for k, v in ex.get("levenshtein_stats", {}).items()
            }
        } for ex in results
    ]
    return pd.DataFrame(parsed_results)