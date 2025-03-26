import abc
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VectorDatabase(abc.ABC):
    def __init__(self, vectors: torch.Tensor):
        self.vectors = vectors
    
    @abc.abstractmethod
    def remove_vectors(self, idxs: torch.Tensor):
        pass

    @abc.abstractmethod
    def search(self, query_vector: torch.Tensor, k: int) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def reset_removed_vectors(self):
        pass


class ExactVectorDatabase(VectorDatabase):
    def search(self, query_vector: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        sims = torch.nn.functional.cosine_similarity(self.vectors, query_vector, dim=1)
        sims[self.ignore_mask] = -float("inf")
        idxs = torch.argsort(sims, descending=True)[:k]
        return sims[idxs], idxs
    
    def remove_vectors(self, idxs: torch.Tensor):
        # Zero out vectors at the given indices
        self.ignore_mask[idxs] = True
    
    def reset_removed_vectors(self):
        self.ignore_mask = torch.zeros(self.vectors.shape[0], dtype=bool)


class BatchedExactVectorDatabase(VectorDatabase):
    # supports a batch dimension, auto-takes a max
    # vectors.shape = (num_databases, database_size, num_vectors)
    def __init__(self, vectors: torch.Tensor, batch_size: int = 100_000):
        super().__init__(vectors.to(torch.float64))
        self.batch_size = batch_size
        self.ignore_mask = torch.zeros(vectors.shape[1], dtype=bool)
        self.vectors = self.vectors.to(torch.float64)
        
    def search(self, query_vector: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Make sure query vector is on GPU
        query_vector = query_vector.to(torch.float64)
        query_vector = query_vector.to(device)
            
        # Setup tensors to store results
        all_sims = []
        num_vectors = self.vectors.shape[1]

        query_vector_norm = query_vector / query_vector.norm(dim=-1, keepdim=True)
        
        # Process in batches
        for i in range(0, num_vectors, self.batch_size):
            # Get current batch
            end_idx = min(i + self.batch_size, num_vectors)
            batch_vectors = self.vectors[:, i:end_idx].to(device).double()
            batch_vectors_norm = batch_vectors / batch_vectors.norm(dim=-1, keepdim=True)
            batch_sims = batch_vectors_norm @ query_vector_norm.T
            batch_sims = batch_sims.mean(dim=0)
            all_sims.append(batch_sims.flatten())

        # Combine all batches
        combined_sims = torch.cat(all_sims)
        combined_sims[self.ignore_mask] = -float("inf")
        top_k_sims, top_k_indices = torch.topk(combined_sims, k, largest=True)

        return top_k_sims, top_k_indices
    
    def remove_vectors(self, idxs: torch.Tensor):
        # Zero out vectors at the given indices
        self.ignore_mask[idxs] = True
    
    def reset_removed_vectors(self):
        self.ignore_mask = torch.zeros(self.vectors.shape[0], dtype=bool)