# -*- coding: utf-8 -*-
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import argparse
from modules.scheduler import WarmUpAndCosineDecayScheduler
import torch.distributed as dist
from trainers.batch_trainer import trainer as batch_trainer
from models.ssid_rq import SpatialSemanticIdentifier
from dataset.sid_dataset import get_dataloaders


def main(args):
    dist.init_process_group(backend=args.backend, init_method='env://')
    dist.barrier()
    args.world_size = dist.get_world_size()
    args.rank = dist.get_rank()
    device_id = args.rank % torch.cuda.device_count()
    args.device = torch.device(device_id)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    embedding_config = args.model_config["encoder"]["input_dim"]

    train_loader, train_sampler = get_dataloaders(
        data_file=args.train_file,
        embedding_config=embedding_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_shuffle=True
    )
    # During validation
    val_loader, _ = get_dataloaders(
        data_file=args.val_file,
        embedding_config=embedding_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_shuffle=False
    )

    model = SpatialSemanticIdentifier(model_config=args.model_config, device=args.device)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    optimizer = torch.optim.AdamW([{"params": model.parameters()}], lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = WarmUpAndCosineDecayScheduler(optimizer, start_lr=1e-6, base_lr=args.learning_rate, final_lr=1e-6,
                                              epoch_num=args.num_epochs, warmup_epoch_num=100)
    trainer = batch_trainer(model, optimizer, scheduler, args)
    trainer.train(train_loader, val_loader,train_sampler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--early_stop", default=10, type=int)
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument("--save_per_epochs", default=1000, type=int)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--lr_scheduler_steps", default=20, type=int)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--train_from_pretrained", action="store_true")
    parser.add_argument("--state_dict_save_path", default='./output_dir', type=str)
    parser.add_argument("--config_path", default='configs/content_3m.json', type=str)
    parser.add_argument("--use_columns", default='text_emb,image_emb,cf_emb', type=str)
    parser.add_argument("--train_file", default='./data/train.txt', type=str)
    parser.add_argument("--val_file", default='./data/val.txt', type=str)

    args = parser.parse_args()
    model_config = json.load(open(args.config_path, 'r'))
    args.model_config = model_config
    main(args)