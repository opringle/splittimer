import random
import torch
import numpy as np
import os
import argparse
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from collections import defaultdict

from tqdm import tqdm
from trainer import get_trainer_class
from utils import count_parameters, get_default_device_name, setup_logging, setup_seed


def worker_init_fn(worker_id):
    """Initialize random seed for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    setup_seed(worker_seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train a position classifier on preprocessed video clip data.")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--device', type=str, default=get_default_device_name(),
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--artifacts_dir', type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--trainer_type', type=str, required=True,
                        help='Type of model to train')

    args, _ = parser.parse_known_args()
    TrainerClass = get_trainer_class(args.trainer_type)
    TrainerClass.add_args(parser)
    args = parser.parse_args()

    setup_logging(args.log_level)
    setup_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path(
        f'artifacts/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=artifacts_dir)
    logging.info(f"View logs at `tensorboard --logdir={artifacts_dir.parent}`")
    checkpoint_dir = artifacts_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    data_dir = Path(args.data_dir)
    train_dir, val_dir = data_dir / 'train', data_dir / 'val'
    train_files = list(train_dir.glob('*.npz'))
    val_files = list(val_dir.glob('*.npz'))

    train_loader = TrainerClass.get_dataloader(
        train_files, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    val_loader = TrainerClass.get_dataloader(
        val_files, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

    trainer = TrainerClass.from_args(args, train_loader)

    start_epoch = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        trainer, epoch = TrainerClass.load(args.resume_from, args.device)
        start_epoch = epoch + 1
        logging.info(f"Resumed training from epoch {start_epoch}")

    logging.info(
        f"Total trainable parameters: {count_parameters(trainer.model)}")

    for epoch in range(start_epoch, args.num_epochs):
        train_metrics = trainer.fit(train_loader)
        log_metrics(f"Epoch {epoch} train metrics, ", train_metrics)
        for metric_name, metric_value in train_metrics.items():
            writer.add_scalar(f"{metric_name}/Train", metric_value, epoch)

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.num_epochs - 1:
            val_metrics = trainer.evaluate(val_loader)
            log_metrics(f"Epoch {epoch} val metrics, ", val_metrics)
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f"{metric_name}/Val", metric_value, epoch)

        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.num_epochs - 1:
            trainer.save(checkpoint_dir, epoch)
            logging.info(f'Saved checkpoint {epoch}')

    writer.close()


def log_metrics(prefix: str, metrics: dict) -> str:
    log_str = prefix
    for metric_name, metric_value in metrics.items():
        log_str += f'{metric_name} {metric_value:.2f}'
    logging.info(log_str)


if __name__ == "__main__":
    main()
