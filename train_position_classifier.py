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

class NPZDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        clip1 = data['clip1'].astype(np.float32)  # Shape: (F, 2049)
        clip2 = data['clip2'].astype(np.float32)  # Shape: (F, 2049)
        label = data['label'].astype(np.float32)  # Scalar: 0 or 1
        return torch.from_numpy(clip1), torch.from_numpy(clip2), torch.tensor(label)

class PositionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PositionClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, 1)

    def forward(self, clip1, clip2):
        # clip1 and clip2: (batch, F, 2049)
        _, (h1, _) = self.lstm(clip1)  # h1: (1, batch, hidden_size)
        h1 = h1.squeeze(0)  # (batch, hidden_size)
        _, (h2, _) = self.lstm(clip2)
        h2 = h2.squeeze(0)  # (batch, hidden_size)
        combined = torch.cat((h1, h2), dim=1)  # (batch, 2*hidden_size)
        output = self.fc(combined)  # (batch, 1)
        return output

def main():
    parser = argparse.ArgumentParser(description="Train a position classifier on preprocessed video clip data.")
    parser.add_argument('data_dir', type=str, help='Directory containing .npz files with preprocessed data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of LSTM')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of data for validation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load and split data
    data_dir = Path(args.data_dir)
    all_files = list(data_dir.glob('*.npz'))
    if not all_files:
        logging.error('No .npz files found in the directory')
        exit(1)

    random.shuffle(all_files)
    val_size = int(len(all_files) * args.val_ratio)
    train_files = all_files[val_size:]
    val_files = all_files[:val_size]

    # Create datasets and dataloaders
    train_dataset = NPZDataset(train_files)
    val_dataset = NPZDataset(val_files)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer
    input_size = 2049  # 2048 ResNet50 features + 1 frame index
    model = PositionClassifier(input_size, args.hidden_size).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for clip1, clip2, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            clip1 = clip1.to(args.device)
            clip2 = clip2.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(clip1, clip2)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_dataset)
        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for clip1, clip2, labels in val_loader:
                clip1 = clip1.to(args.device)
                clip2 = clip2.to(args.device)
                labels = labels.to(args.device)
                outputs = model(clip1, clip2)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * labels.size(0)
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        report = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'])
        logging.info(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')
        logging.info(f'Classification Report:\n{report}')

        # Save model
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()