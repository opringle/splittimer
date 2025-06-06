import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report
import logging
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import PositionClassifier, count_parameters, setup_logging

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

def main():
    parser = argparse.ArgumentParser(description="Train a position classifier on preprocessed video clip data.")
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_epochs', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--compress_sizes', type=str, default='1024,512')
    parser.add_argument('--post_lstm_sizes', type=str, default='256,128')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--artifacts_dir', type=str, default=None)
    args = parser.parse_args()

    setup_logging()

    compress_sizes = [int(x) for x in args.compress_sizes.split(',')] if args.compress_sizes else []
    post_lstm_sizes = [int(x) for x in args.post_lstm_sizes.split(',')] if args.post_lstm_sizes else []

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path(f'artifacts/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=artifacts_dir)
    checkpoint_dir = artifacts_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    data_dir = Path(args.data_dir)
    train_dir, val_dir = data_dir / 'train', data_dir / 'val'
    train_files = list(train_dir.glob('*.npz'))
    val_files = list(val_dir.glob('*.npz'))

    train_dataset = NPZDataset(train_files)
    val_dataset = NPZDataset(val_files) if val_files else None
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4) if val_dataset else None

    model = PositionClassifier(
        input_size=2049,
        hidden_size=args.hidden_size,
        bidirectional=args.bidirectional,
        compress_sizes=compress_sizes,
        post_lstm_sizes=post_lstm_sizes,
        dropout=args.dropout
    ).to(args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        model, start_epoch, optimizer_state = PositionClassifier.load(args.resume_from, args.device)
        optimizer.load_state_dict(optimizer_state)
        start_epoch += 1
        logging.info(f"Resuming from epoch {start_epoch}")

    logging.info(f"Total trainable parameters: {count_parameters(model)}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for clip1, clip2, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            clip1 = clip1.to(args.device).flatten(start_dim=0, end_dim=1)
            clip2 = clip2.to(args.device).flatten(start_dim=0, end_dim=1)
            labels = labels.to(args.device).flatten(start_dim=0, end_dim=1)
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
                        clip1 = clip1.to(args.device).flatten(start_dim=0, end_dim=1)
                        clip2 = clip2.to(args.device).flatten(start_dim=0, end_dim=1)
                        labels = labels.to(args.device).flatten(start_dim=0, end_dim=1)
                        outputs = model(clip1, clip2)
                        loss = criterion(outputs.squeeze(), labels)
                        val_loss += loss.item() * labels.size(0)
                        total_samples += labels.size(0)
                        preds = torch.sigmoid(outputs).squeeze() > 0.5
                        all_preds.extend(preds.cpu().numpy().astype(int))
                        all_labels.extend(labels.cpu().numpy().astype(int))

                val_loss = val_loss / total_samples if total_samples > 0 else float('inf')
                
                # Compute classification report string and dictionary
                if all_labels:  # Only generate report if there are labels
                    report_str = classification_report(all_labels, all_preds, labels=[0, 1], target_names=['Class 0', 'Class 1'], zero_division=0)
                    report_dict = classification_report(all_labels, all_preds, labels=[0, 1], target_names=['Class 0', 'Class 1'], zero_division=0, output_dict=True)

                    # Log to console
                    logging.info(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')
                    logging.info(f'Classification Report:\n{report_str}')

                    # Log to TensorBoard
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