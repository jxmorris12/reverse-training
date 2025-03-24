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
        super().__init__(vectors.to(torch.float16))
        self.batch_size = batch_size
        self.ignore_mask = torch.zeros(vectors.shape[1], dtype=bool)
        self.vectors = self.vectors.to(device)
        
    def search(self, query_vector: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Make sure query vector is on GPU
        query_vector = query_vector.to(torch.float16)
        query_vector = query_vector.to(device)
            
        # Setup tensors to store results
        all_sims = []
        num_vectors = self.vectors.shape[1]

        query_vector_norm = query_vector / query_vector.norm(dim=0, keepdim=True)
        
        # Process in batches
        for i in range(0, num_vectors, self.batch_size):
            # Get current batch
            end_idx = min(i + self.batch_size, num_vectors)
            batch_vectors = self.vectors[:, i:end_idx].to(device)
            batch_vectors_norm = batch_vectors / batch_vectors.norm(dim=0, keepdim=True)
            batch_sims = batch_vectors_norm @ query_vector_norm.T
            batch_sims = batch_sims.max(dim=0).values
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


class FaissVectorDatabase(VectorDatabase):
    def __init__(self, vectors: torch.Tensor, batch_size: int = 32000):
        import faiss
        print(f"Initializing FaissVectorDatabase with {vectors.shape[0]} vectors of dimension {vectors.shape[1]}")
        quantizer = faiss.IndexFlatL2(vectors.shape[1])
        index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], 128)

        # Train the index (necessary for IVF)
        # For large datasets, you can train on a smaller subset
        train_size = min(100_000, vectors.shape[0])
        index.train(vectors[:train_size])
        index.add(vectors)

        res = faiss.StandardGpuResources()  # GPU resources
        # self.index = faiss.index_cpu_to_gpu(res, 0, index)  # 0 is the GPU id
        self.index = index
        self.removed_ids = set()
    
    def remove_vectors(self, idxs: torch.Tensor):
        if not isinstance(idxs, torch.Tensor):
            idxs = torch.tensor(idxs)
        self.removed_ids.update(idxs.flatten().cpu().numpy())
    
    def search(self, query_vector: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: Investigate why FAISS doesn't support GPU tensors
        sims, ids = self.index.search(query_vector.cpu(), k + len(self.removed_ids))
        
        # Remove removed ids
        id_list = ids.flatten().tolist()
        sim_list = [sim for sim, id in zip(sims.flatten().tolist(), id_list) if id not in self.removed_ids]
        id_list = [id for id in id_list if id not in self.removed_ids]

        id_list = id_list[:k]
        sim_list = sim_list[:k]

        assert len(id_list) == len(sim_list) == k, \
            f"len(id_list): {len(id_list)} != len(sim_list): {len(sim_list)} != k: {k}"

        return torch.tensor(sim_list), torch.tensor(id_list)


# def get_vector_database(vectors: torch.Tensor, use_batched: bool = False, batch_size: int = 32000) -> VectorDatabase:
#     try:
#         import faiss
#         cls_name = FaissVectorDatabase
#     except ImportError:
#         if use_batched:
#             cls_name = BatchedExactVectorDatabase
#             return cls_name(vectors, batch_size=batch_size)
#         else:
#             cls_name = ExactVectorDatabase
#     return cls_name(vectors)
