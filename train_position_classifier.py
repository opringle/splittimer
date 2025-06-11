import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import random  # Added for random.seed
from sklearn.metrics import classification_report
import logging
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import InteractionType, PositionClassifier, SequencePositionClassifier, count_parameters, get_default_device_name, setup_logging

class NPZDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        clip1 = data['clip1s'].astype(np.float32)
        clip2 = data['clip2s'].astype(np.float32)
        label = data['labels'].astype(np.float32)
        return torch.from_numpy(clip1), torch.from_numpy(clip2), torch.from_numpy(label)

def worker_init_fn(worker_id):
    """Initialize random seed for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    parser = argparse.ArgumentParser(description="Train a position classifier on preprocessed video clip data.")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--device', type=str, default=get_default_device_name(), help='Device to use (cuda or cpu)')
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--compress_sizes', type=str, default=None)
    parser.add_argument("--interaction_type", type=str, default="dot", choices=["dot", "mlp"])
    parser.add_argument('--post_lstm_sizes', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--artifacts_dir', type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')  # Added seed argument
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {args.seed} for reproducibility")

    # Warn about potential non-determinism with num_workers > 0
    if torch.utils.data.get_worker_info() is not None:
        logging.warning("DataLoader num_workers > 0 may introduce non-determinism despite seeding. Consider setting num_workers=0 for full reproducibility.")

    compress_sizes = [int(x) for x in args.compress_sizes.split(',')] if args.compress_sizes else []
    post_lstm_sizes = [int(x) for x in args.post_lstm_sizes.split(',')] if args.post_lstm_sizes else []

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path(f'artifacts/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=artifacts_dir)
    logging.info(f"View logs at `tensorboard --logdir={artifacts_dir.parent}`")
    checkpoint_dir = artifacts_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    data_dir = Path(args.data_dir)
    train_dir, val_dir = data_dir / 'train', data_dir / 'val'
    train_files = list(train_dir.glob('*.npz'))
    val_files = list(val_dir.glob('*.npz'))

    train_dataset = NPZDataset(train_files)
    val_dataset = NPZDataset(val_files) if val_files else None
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn) if val_dataset else None

    # Determine input shape from one batch
    clip1, clip2, _ = next(iter(train_loader))
    _, B, F, input_size = clip1.shape
    logging.info(f"Each sample's clips have shape = {clip1.shape}")

    model = PositionClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        interaction_type=InteractionType(args.interaction_type),
        bidirectional=args.bidirectional,
        compress_sizes=compress_sizes,
        post_lstm_sizes=post_lstm_sizes,
        dropout=args.dropout
    ).to(args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=args.device)
        model_type = checkpoint['model_type']
        if model_type == 'position':
            model = PositionClassifier(**checkpoint['model_config']).to(args.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resumed {model_type} model from epoch {start_epoch}")

    logging.info(f"Total trainable parameters: {count_parameters(model)}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for clip1, clip2, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            clip1 = clip1.to(args.device).squeeze()
            clip2 = clip2.to(args.device).squeeze()
            labels = labels.to(args.device).squeeze()
            optimizer.zero_grad()
            outputs = model(clip1, clip2)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
        train_loss = total_loss / total_samples
        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
        writer.add_scalar('Loss/train', train_loss, epoch)

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.num_epochs - 1:
            if val_loader:
                model.eval()
                val_loss = 0.0
                total_samples = 0
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for clip1, clip2, labels in val_loader:
                        clip1 = clip1.to(args.device).squeeze()
                        clip2 = clip2.to(args.device).squeeze()
                        labels = labels.to(args.device).squeeze()
                        outputs = model(clip1, clip2)
                        loss = criterion(outputs.squeeze(), labels)
                        val_loss += loss.item() * labels.size(0)
                        total_samples += labels.size(0)
                        preds = torch.sigmoid(outputs).squeeze() > 0.5
                        all_preds.extend(preds.cpu().numpy().astype(int))
                        all_labels.extend(labels.cpu().numpy().astype(int))

                val_loss = val_loss / total_samples if total_samples > 0 else float('inf')
                
                if all_labels:
                    report_str = classification_report(all_labels, all_preds, labels=[0, 1], target_names=['Class 0', 'Class 1'], zero_division=0)
                    report_dict = classification_report(all_labels, all_preds, labels=[0, 1], target_names=['Class 0', 'Class 1'], zero_division=0, output_dict=True)

                    logging.info(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')
                    logging.info(f'Classification Report:\n{report_str}')

                    writer.add_scalar('Loss/val', val_loss, epoch)
                    writer.add_scalar('Accuracy/val', report_dict['accuracy'], epoch)
                    for class_name in ['Class 0', 'Class 1']:
                        writer.add_scalar(f'Precision/val/{class_name}', report_dict[class_name]['precision'], epoch)
                        writer.add_scalar(f'Recall/val/{class_name}', report_dict[class_name]['recall'], epoch)
                        writer.add_scalar(f'F1-score/val/{class_name}', report_dict[class_name]['f1-score'], epoch)
                    writer.add_scalar('Macro Avg/Precision/val', report_dict['macro avg']['precision'], epoch)
                    writer.add_scalar('Macro Avg/Recall/val', report_dict['macro avg']['recall'], epoch)
                    writer.add_scalar('Macro Avg/F1-score/val', report_dict['macro avg']['f1-score'], epoch)
                    writer.add_scalar('Weighted Avg/Precision/val', report_dict['weighted avg']['precision'], epoch)
                    writer.add_scalar('Weighted Avg/Recall/val', report_dict['weighted avg']['recall'], epoch)
                    writer.add_scalar('Weighted Avg/F1-score/val', report_dict['weighted avg']['f1-score'], epoch)
                else:
                    logging.warning(f'Epoch {epoch+1}: No predictions or labels collected for validation')
            else:
                logging.info(f'Epoch {epoch+1}: Skipping validation as no validation files are present')

        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            model.save(checkpoint_path, epoch, optimizer)
            logging.info(f'Saved checkpoint to {checkpoint_path}')

    writer.close()

if __name__ == "__main__":
    main()