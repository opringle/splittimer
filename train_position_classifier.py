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
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class NPZDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        clip1 = data['clip1s'].astype(np.float32)  # Shape: (B, F, 2049)
        clip2 = data['clip2s'].astype(np.float32)  # Shape: (B, F, 2049)
        label = data['labels'].astype(np.float32)  # Shape: (B,)
        return torch.from_numpy(clip1), torch.from_numpy(clip2), torch.from_numpy(label)

class PositionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, compress_sizes=[], post_lstm_sizes=[]):
        super(PositionClassifier, self).__init__()
        self.bidirectional = bidirectional
        
        # Build compression layers
        if compress_sizes:
            layers = []
            in_size = input_size
            for size in compress_sizes:
                layers.append(nn.Linear(in_size, size))
                layers.append(nn.ReLU())
                in_size = size
            self.compression = nn.Sequential(*layers)
            lstm_input_size = compress_sizes[-1]
        else:
            self.compression = nn.Identity()
            lstm_input_size = input_size
        
        # LSTM layer
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        
        # Determine the size of the concatenated hidden states
        lstm_output_size = 2 * hidden_size if bidirectional else hidden_size
        concat_size = 2 * lstm_output_size
        
        # Build post-LSTM layers
        if post_lstm_sizes:
            layers = []
            in_size = concat_size
            for size in post_lstm_sizes:
                layers.append(nn.Linear(in_size, size))
                layers.append(nn.ReLU())
                in_size = size
            self.post_lstm = nn.Sequential(*layers)
            final_input_size = post_lstm_sizes[-1]
        else:
            self.post_lstm = nn.Identity()
            final_input_size = concat_size
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(final_input_size, 1)

    def forward(self, clip1, clip2):
        # Apply compression to each clip
        compressed_clip1 = self.compression(clip1)  # (B, F, lstm_input_size)
        compressed_clip2 = self.compression(clip2)  # (B, F, lstm_input_size)
        
        # Pass through LSTM
        if self.bidirectional:
            _, (h_n, _) = self.lstm(compressed_clip1)
            h1 = torch.cat((h_n[-2], h_n[-1]), dim=1)  # Last forward and backward hidden states
            _, (h_n, _) = self.lstm(compressed_clip2)
            h2 = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            _, (h1, _) = self.lstm(compressed_clip1)
            h1 = h1[-1]  # Last hidden state
            _, (h2, _) = self.lstm(compressed_clip2)
            h2 = h2[-1]
        
        # Concatenate the hidden states from both clips
        combined = torch.cat((h1, h2), dim=1)
        
        # Pass through post-LSTM layers
        post_lstm_output = self.post_lstm(combined)
        
        # Final classification layer
        output = self.fc(post_lstm_output)
        return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description="Train a position classifier on preprocessed video clip data.")
    parser.add_argument('data_dir', type=str, help='Directory containing .npz files with preprocessed data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    # hyperparameters
    parser.add_argument('--num_epochs', type=int, default=1500, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional LSTM')
    parser.add_argument('--compress_sizes', type=str, default='1024,512', help='Comma-separated list of sizes for compression layers, e.g., "1024,512"')
    parser.add_argument('--post_lstm_sizes', type=str, default='256,128', help='Comma-separated list of sizes for post-LSTM layers, e.g., "256,128"')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of LSTM')

    # validation
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of data for validation')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate the model on the validation set every N epochs')
    
    # artifacts
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Save model checkpoint every N epochs')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--artifacts_dir', type=str, default=None, help='Directory for TensorBoard logs and checkpoints. If not specified, a new one will be created.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Parse compress_sizes and post_lstm_sizes
    compress_sizes = [int(x) for x in args.compress_sizes.split(',')] if args.compress_sizes else []
    post_lstm_sizes = [int(x) for x in args.post_lstm_sizes.split(',')] if args.post_lstm_sizes else []

    # Set up artifacts directory
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        artifacts_dir = Path(f'artifacts/experiment_{timestamp}')
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Artifacts will be saved in {artifacts_dir}')

    # Set up TensorBoard
    writer = SummaryWriter(log_dir=artifacts_dir)
    logging.info(f'TensorBoard logs will be saved in {artifacts_dir}. Run `tensorboard --logdir={artifacts_dir}` to view.')

    # Set up checkpoint directory within artifacts
    checkpoint_dir = artifacts_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    logging.info(f'Checkpoints will be saved in {checkpoint_dir}')

    data_dir = Path(args.data_dir)
    all_files = list(data_dir.glob('*.npz'))
    if not all_files:
        logging.error('No .npz files found in the directory')
        exit(1)

    random.shuffle(all_files)
    val_size = int(len(all_files) * args.val_ratio)
    train_files = all_files[val_size:]
    val_files = all_files[:val_size]

    train_dataset = NPZDataset(train_files)
    val_dataset = NPZDataset(val_files)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    input_size = 2049
    model = PositionClassifier(input_size, args.hidden_size, bidirectional=args.bidirectional, compress_sizes=compress_sizes, post_lstm_sizes=post_lstm_sizes).to(args.device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Handle resuming from checkpoint
    start_epoch = 0
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            checkpoint = torch.load(args.resume_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resuming training from epoch {start_epoch}")
        else:
            logging.error(f"Checkpoint file {args.resume_from} not found")
            exit(1)
    else:
        logging.info("Starting training from scratch")

    # Log the total number of trainable parameters
    total_params = count_parameters(model)
    logging.info(f"Total trainable parameters: {total_params}")
    
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
            model.eval()
            val_loss = 0.0
            total_samples = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for clip1, clip2, labels in val_loader:
                    clip1 = clip1.to(args.device).flatten(start_dim=0, end_dim=1)
                    clip2 = clip2.to(args.device).flatten(start_dim=0, end_dim=1)
                    labels = labels.to(args.device).flatten(start_dim=0, end_dim=1)
                    outputs = model(clip1, clip2)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item() * labels.size(0)
                    total_samples += labels.size(0)
            val_loss = val_loss / total_samples
            
            # Compute classification report string and dictionary
            report_str = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'], zero_division=0)
            report_dict = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'], zero_division=0, output_dict=True)

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

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f'Saved checkpoint at epoch {epoch+1} to {checkpoint_path}')

    writer.close()

if __name__ == "__main__":
    main()