import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
from typing import Dict, List, Optional

class EmbeddingDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        embedding_config: Dict[str, int],
    ):
        self.embedding_config = embedding_config
        self.names = list(embedding_config.keys())
        self.dims = [embedding_config[name] for name in self.names]
        self.cum_dims = torch.cumsum(torch.tensor([0] + self.dims), dim=0)
        self.total_float_dim = self.cum_dims[-1].item()

        # Preload all data
        self.indices = []
        self.embeddings = []  # list of tensors or one big tensor
        with open(file_path, 'r') as f:
            lines = f.readlines()

        total_fields = 1 + self.total_float_dim
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != total_fields:
                raise ValueError(f"Line {i} has {len(parts)} fields, expected {total_fields}")
            
            idx = int(parts[0])
            floats = list(map(float, parts[1:]))
            self.indices.append(idx)
            self.embeddings.append(torch.tensor(floats, dtype=torch.float32))
        self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.embeddings = torch.stack(self.embeddings)  # Shape: [N, total_float_dim]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_index = self.indices[idx]
        full_tensor = self.embeddings[idx]

        embeddings = {'index': sample_index.unsqueeze(0)}
        for i, name in enumerate(self.names):
            start = self.cum_dims[i].item()
            end = self.cum_dims[i + 1].item()
            embeddings[name] = full_tensor[start:end]

        return embeddings


def get_dataloaders(
    data_file: str,
    embedding_config: Optional[Dict[str, int]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    is_shuffle: bool = True,
):
    """
    Create a distributed dataloader for arbitrary embedding configurations.

    Args:
        data_file (str): Path to the data file (train/val/test).
        embedding_config (dict, optional): e.g., {"text": 512, "image": 1024, "cf": 128}
        batch_size (int): Per-GPU batch size.
        num_workers (int): Number of subprocesses for data loading.
        is_shuffle (bool): Whether to shuffle (via sampler).

    Returns:
        dataloader (DataLoader), sampler (DistributedSampler)
    """
    if embedding_config is None:
        embedding_config = {"text": 512, "image": 1024, "cf": 32}

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dataset = EmbeddingDataset(data_file, embedding_config)

    # Training mode: use DistributedSampler
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=is_shuffle,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # must be False when sampler is used
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return dataloader, sampler
