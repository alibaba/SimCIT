# -*- coding: utf-8 -*-
import json
import os
import torch
import argparse
import linecache
import multiprocessing as mp
import torch.distributed as dist
from models.ssid_rq import SpatialSemanticIdentifier
from torch.utils.data import DataLoader, Subset
from typing import Dict, Optional


# ================================
# EmbeddingDataset with 'index'
# ================================

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, embedding_config: Dict[str, int]):
        self.file_path = file_path
        self.embedding_config = embedding_config
        self.names = list(embedding_config.keys())
        self.dims = [embedding_config[name] for name in self.names]
        self.cum_dims = torch.cumsum(torch.tensor([0] + self.dims), dim=0)
        self.total_float_dim = self.cum_dims[-1].item()
        self.total_fields = 1 + self.total_float_dim

        with open(file_path, 'r') as f:
            self.num_lines = sum(1 for _ in f)

    def __len__(self) -> int:
        return self.num_lines

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        line = linecache.getline(self.file_path, idx + 1).strip()
        if not line:
            raise IndexError(f"Line {idx} is empty or out of range")
        parts = line.split()
        if len(parts) != self.total_fields:
            raise ValueError(f"Line {idx} has {len(parts)} fields, expected {self.total_fields}")
        
        index = int(parts[0])
        floats = list(map(float, parts[1:]))
        tensor = torch.tensor(floats, dtype=torch.float32)
        
        sample = {'index': torch.tensor(index, dtype=torch.long)}
        for i, name in enumerate(self.names):
            start = self.cum_dims[i].item()
            end = self.cum_dims[i + 1].item()
            sample[name] = tensor[start:end]
        return sample



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--state_dict_save_path", required=True, type=str)
    parser.add_argument("--use_columns", default="text_emb,image_emb,cf_emb", type=str)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--config_path", default='configs/content_3m.json', type=str)

    args = parser.parse_args()
    model_config = json.load(open(args.config_path, 'r'))
    args.model_config = model_config
    # Detect distributed setting
    rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = rank >= 0

    if is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        args.device = torch.device(rank % torch.cuda.device_count())
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1

    # Load model
    checkpoint = torch.load(args.state_dict_save_path, map_location=args.device)
    model_config = checkpoint['config']
    model = SpatialSemanticIdentifier(model_config=model_config, device=args.device)
    state_dict = checkpoint['model']
    # Handle DDP saved models
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Build dataset
    embedding_config = args.model_config["encoder"]["input_dim"]
    dataset = EmbeddingDataset(args.input_file, embedding_config)

    # Shard data across ranks (only by line number, not by index value)
    total = len(dataset)
    if world_size > 1:
        per_rank = total // world_size
        start = rank * per_rank
        end = start + per_rank if rank != world_size - 1 else total
        local_indices = list(range(start, end))
        dataset = Subset(dataset, local_indices)
        output_path = f"{args.output_file}_rank{rank}.txt"
    else:
        output_path = args.output_file

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    with open(output_path, 'w') as writer:
        with torch.no_grad():
            for batch in dataloader:
                batch_input = {k: v.to(args.device) for k, v in batch.items() if k != 'index'}
                _, sids = model.get_codes(batch_input)
                sids = sids.cpu().numpy()
                indices = batch['index'].cpu().numpy()
                for idx, sid in zip(indices, sids):
                    writer.write(f"{idx} {json.dumps(sid.tolist())}\n")

    print(f"[Rank {rank}] Done. Output saved to {output_path}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()