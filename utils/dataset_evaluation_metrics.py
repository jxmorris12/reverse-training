import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import ot  # Optimal Transport library
import Levenshtein # Token Edit
import math
import tqdm
from scipy.optimize import linear_sum_assignment

from utils.batch import find_executable_batch_size

# Load Sentence T5 model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "sentence-transformers/sentence-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


##############################
# Lexical-Based Metrics
##############################

def jaccard_similarity(set1, set2):
    """
    Compute the Jaccard similarity between two sets.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(intersection) / len(union)

def jaccard_overlap_examples(dataset1, dataset2):
    """
    Compute Jaccard similarity on the set of entire example strings.
    This measures exact overlap of examples.
    """
    set1 = set(dataset1)
    set2 = set(dataset2)
    return jaccard_similarity(set1, set2)

def jaccard_overlap_vocabulary(dataset1, dataset2, tokenizer=tokenizer):
    """
    Compute Jaccard similarity based on the union of tokens from each dataset.
    Uses the T5 tokenizer to tokenize each example.
    """
    tokens1 = set()
    for text in dataset1:
        tokens1.update(tokenizer.tokenize(text))
    tokens2 = set()
    for text in dataset2:
        tokens2.update(tokenizer.tokenize(text))
    return jaccard_similarity(tokens1, tokens2)


def compute_levenshtein_distance(s1, s2):
    """
    Levenshtein distance between two strings using the python-Levenshtein library.
    """
    return Levenshtein.distance(s1, s2)


def dataset_levenshtein_closest_pair_statistics(dataset_A, dataset_B):
    """
    For each example in dataset_A, compute the Levenshtein distance to every example in dataset_B.
    For each entry in dataset_A, identify the closest matching example in dataset_B (i.e. with the smallest distance).
    Returns a dictionary with summary statistics over these minimal distances and a list of the closest pairs.

    Output dictionary keys:
      - "average_distance": average minimal distance
      - "min_distance": minimum minimal distance
      - "max_distance": maximum minimal distance
      - "closest_pairs": a list of tuples (index_A, index_B, distance)
    """
    if not dataset_A or not dataset_B:
        raise ValueError("Both datasets must contain at least one example.")

    min_distances = []
    closest_pairs = []  # each element: (index_in_A, best_index_in_B, distance)

    for i, text_A in enumerate(dataset_A):
        best_distance = math.inf
        best_j = None
        for j, text_B in enumerate(dataset_B):
            distance = compute_levenshtein_distance(text_A, text_B)
            if distance < best_distance:
                best_distance = distance
                best_j = j
        min_distances.append(best_distance)
        closest_pairs.append((i, best_j, best_distance))

    stats = {
        "average_distance": sum(min_distances) / len(min_distances),
        "min_distance": min(min_distances),
        "max_distance": max(min_distances),
        # "closest_pairs": closest_pairs
    }
    return stats



##############################
# Embedding-Based OT Metrics
##############################
def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling over token embeddings, accounting for the attention mask.
    Returns a tensor of shape (batch_size, hidden_size).
    """
    token_embeddings = model_output.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


@find_executable_batch_size
def get_sentence_embeddings(texts, tokenizer, model, device, batch_size=128):
    """
    Compute sentence embeddings for a list of texts using the provided Sentence T5 model.
    Uses mean pooling over the token embeddings and then normalizes the embeddings.
    Returns a NumPy array of shape (num_texts, hidden_size).
    """
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
    all_embeddings = []
    pbar = tqdm.trange(0, len(texts), batch_size, desc="Computing sentence embeddings", colour="GREEN")
    for i in pbar:
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
        encoded_input = { key: val.to(device) for key, val in encoded_input.items() }
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model_output = model.encoder(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def dataset_level_full_ot_with_embeddings(emb_A, emb_B):
    """
    Compute the full optimal transport (Wasserstein) distance between two datasets using pre-computed embeddings.
    
    Uniform weights are assigned to each example. The cost matrix is computed using the Euclidean distance
    between embeddings, and the full Wasserstein distance is computed via ot.emd2.
    
    Returns a scalar distance value. Lower values indicate higher similarity.
    """
    m = emb_A.shape[0]
    n = emb_B.shape[0]
    wA = np.ones(m) / m
    wB = np.ones(n) / n
    cost_matrix = ot.dist(emb_A, emb_B, metric='sqeuclidean')
    distance = ot.emd2(wA, wB, cost_matrix)
    return distance


def dataset_level_sinkhorn_ot_with_embeddings(emb_A, emb_B, reg=0.1):
    """
    Compute the Sinkhorn optimal transport distance between two datasets using pre-computed embeddings.
    
    Uniform weights are assigned to each example. The cost matrix (Euclidean distances between embeddings) 
    is computed and the Sinkhorn distance is obtained using ot.sinkhorn2.
    
    The regularization parameter 'reg' controls the trade-off between accuracy and computation speed.
    Returns a scalar distance value. Lower values indicate higher similarity.
    """
    m = emb_A.shape[0]
    n = emb_B.shape[0]
    wA = np.ones(m) / m
    wB = np.ones(n) / n
    cost_matrix = ot.dist(emb_A, emb_B, metric='sqeuclidean')
    distance = ot.sinkhorn2(wA, wB, cost_matrix, reg)
    return distance


def example_level_relaxed_wmd_with_embeddings(emb_A, emb_B):
    """
    Compute a relaxed Word Mover's Distance (RWMD) between two documents using pre-computed embeddings.
    
    For each embedding in doc1, find the minimal Euclidean distance to any embedding in doc2,
    and vice versa. The relaxed WMD is defined as the maximum of these two average distances.
    """
    if emb_A.size == 0 or emb_B.size == 0:
        return float('inf')

    # For each embedding in doc1, compute minimal Euclidean distance to any embedding in doc2.
    distances_1_to_2 = [np.min(np.linalg.norm(emb_B - e, axis=1)) for e in emb_A]
    avg_1_to_2 = np.mean(distances_1_to_2)

    # For each embedding in doc2, compute minimal Euclidean distance to any embedding in doc1.
    distances_2_to_1 = [np.min(np.linalg.norm(emb_A - e, axis=1)) for e in emb_B]
    avg_2_to_1 = np.mean(distances_2_to_1)

    return max(avg_1_to_2, avg_2_to_1)


def optimal_matching_relaxed_wmd_with_embeddings(emb_A, emb_B):
    """
    Compute a cost matrix between every document in dataset_A and dataset_B using relaxed WMD
    with pre-computed embeddings. Then, use the Hungarian algorithm to obtain an optimal one-to-one matching.
    
    Returns summary statistics and the matched pairs.
    """
    m = len(emb_A)
    n = len(emb_B)
    cost_matrix = np.zeros((m, n))
    print(cost_matrix.shape)

    for i in tqdm.trange(m, desc="Computing relaxed WMD cost matrix", colour="MAGENTA"):
        for j in range(n):
            # Create single-example embeddings for the relaxed WMD calculation
            single_emb_A = emb_A[i:i+1]
            single_emb_B = emb_B[j:j+1]
            cost_matrix[i, j] = example_level_relaxed_wmd_with_embeddings(single_emb_A, single_emb_B)

    row_idx, col_idx = linear_sum_assignment(cost_matrix) # Hungarian algorithm for 1-1 matching
    matched_costs = cost_matrix[row_idx, col_idx]

    stats = {
        "average_relaxed_wmd": float(np.mean(matched_costs)),
        "min_relaxed_wmd": float(np.min(matched_costs)),
        "max_relaxed_wmd": float(np.max(matched_costs)),
        "matched_pairs": list(zip(row_idx.tolist(), col_idx.tolist(), matched_costs.tolist()))
    }
    return stats


# Keep these functions for backward compatibility
def dataset_level_full_ot(dataset_A, dataset_B, tokenizer=tokenizer, model=model, device=device):
    """
    Compute the full optimal transport (Wasserstein) distance between two datasets.
    
    Each dataset is first converted into a distribution of sentence embeddings (using the Sentence T5 model).
    Uniform weights are assigned to each example. The cost matrix is computed using the Euclidean distance
    between embeddings, and the full Wasserstein distance is computed via ot.emd2.
    
    Returns a scalar distance value. Lower values indicate higher similarity.
    """
    emb_A = get_sentence_embeddings(dataset_A, tokenizer, model, device)
    emb_B = get_sentence_embeddings(dataset_B, tokenizer, model, device)
    return dataset_level_full_ot_with_embeddings(emb_A, emb_B)


def dataset_level_sinkhorn_ot(dataset_A, dataset_B, reg=0.1, tokenizer=tokenizer, model=model, device=device):
    """
    Compute the Sinkhorn optimal transport distance between two datasets using entropic regularization.
    
    Each dataset is converted into a distribution of sentence embeddings (via the Sentence T5 model)
    with uniform weights. The cost matrix (Euclidean distances between embeddings) is computed and the
    Sinkhorn distance is obtained using ot.sinkhorn2.
    
    The regularization parameter 'reg' controls the trade-off between accuracy and computation speed.
    Returns a scalar distance value. Lower values indicate higher similarity.
    """
    emb_A = get_sentence_embeddings(dataset_A, tokenizer, model, device)
    emb_B = get_sentence_embeddings(dataset_B, tokenizer, model, device)
    return dataset_level_sinkhorn_ot_with_embeddings(emb_A, emb_B, reg)


def example_level_relaxed_wmd(dataset_A, dataset_B, tokenizer=tokenizer, model=model, device=device):
    """
    Compute a relaxed Word Mover's Distance (RWMD) between two documents.
    For each token embedding in doc1, find the minimal Euclidean distance to any token in doc2,
    and vice versa. The relaxed WMD is defined as the maximum of these two average distances.
    """
    emb_A = get_sentence_embeddings(dataset_A, tokenizer, model, device)
    emb_B = get_sentence_embeddings(dataset_B, tokenizer, model, device)
    return example_level_relaxed_wmd_with_embeddings(emb_A, emb_B)


def optimal_matching_relaxed_wmd(dataset_A, dataset_B, tokenizer=tokenizer, model=model, device=device):
    """
    Compute a cost matrix between every document in dataset_A and dataset_B using relaxed WMD.
    Then, use the Hungarian algorithm to obtain an optimal one-to-one matching.
    Returns summary statistics and the matched pairs.
    """
    emb_A = get_sentence_embeddings(dataset_A, tokenizer, model, device)
    emb_B = get_sentence_embeddings(dataset_B, tokenizer, model, device)
    return optimal_matching_relaxed_wmd_with_embeddings(emb_A, emb_B)


def evaluate_dataset_similarity(reference_dataset: list[str], recovered_dataset: list[str]) -> dict[str, float]:
    """
    Evaluate two datasets (reference and recovered) using dataset-level OT metrics,
    and save the results to a JSON file.
    
    Parameters:
      - reference_dataset (list of str): The ground truth dataset.
      - recovered_dataset (list of str): The recovered dataset.
      
    Returns:
      - results (dict): A dictionary containing the computed metrics.
    """
    # Compute embeddings once for both datasets
    emb_A = get_sentence_embeddings(reference_dataset, tokenizer, model, device)
    emb_B = get_sentence_embeddings(recovered_dataset, tokenizer, model, device)
    
    # Compute embedding-based metrics using the pre-computed embeddings
    full_ot_distance = dataset_level_full_ot_with_embeddings(emb_A, emb_B)
    sinkhorn_distance = dataset_level_sinkhorn_ot_with_embeddings(emb_A, emb_B, reg=0.1)
    # optimal_matching_relaxed_wmd_stats = optimal_matching_relaxed_wmd_with_embeddings(emb_A, emb_B)
    
    # Lexical-Based Metrics (these don't use embeddings)
    jaccard_overlap_examples_score = jaccard_overlap_examples(reference_dataset, recovered_dataset)
    jaccard_overlap_vocabulary_score = jaccard_overlap_vocabulary(reference_dataset, recovered_dataset)
    levenshtein_stats = dataset_levenshtein_closest_pair_statistics(reference_dataset, recovered_dataset)
    
    results = {
        "full_ot_distance": full_ot_distance,
        "sinkhorn_distance": sinkhorn_distance,
        # "optimal_matching_relaxed_wmd": optimal_matching_relaxed_wmd_stats,
        "jaccard_overlap_examples": jaccard_overlap_examples_score,
        "jaccard_overlap_vocabulary": jaccard_overlap_vocabulary_score,
        "levenshtein_stats": levenshtein_stats
    }
    return results


def dummy_test():
    """
    A dummy test to check the evaluation metrics on two datasets.
    """
    dataset_A = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test sentence.",
        "Data science and machine learning are evolving fields.",
        "Transformers have revolutionized NLP."
    ]
    
    dataset_B = [
        "Transformers have changed the landscape of NLP.",
        "Hello world, this is an example test sentence.",
        "Data science and AI are rapidly evolving.",
        "The quick brown fox leaps over the lazy dog."
    ]
    
    results = evaluate_dataset_similarity(dataset_A, dataset_B)
    print("Evaluation Metrics:")
    print(results)

if __name__ == "__main__":
    # Run dummy test
    dummy_test()
    
