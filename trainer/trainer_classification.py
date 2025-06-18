from enum import Enum
from typing import Any, List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from collections import defaultdict

from config import Config
from sample_generator import get_sample_generator_class
from sample_generator.sample_generator_classification import ClassifierSampleGenerator
from utils import get_clip_indices_ending_at, get_video_file_path, get_video_fps_and_total_frames, load_image_features_from_disk, timecode_to_frames, frame_idx_to_timecode
from .interface import Trainer
from tqdm import tqdm


class ClassificationDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        clip1 = data['data_0'].astype(np.float32)
        clip2 = data['data_1'].astype(np.float32)
        label = data['data_2'].astype(np.float32)
        return torch.from_numpy(clip1), torch.from_numpy(clip2), torch.from_numpy(label)


class ClassificationTrainer(Trainer):
    ### Class Methods ###
    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        """Add command-line arguments specific to this trainer."""
        parser.add_argument('--bidirectional', action='store_true')
        parser.add_argument('--compress_sizes', type=str, default=None)
        parser.add_argument("--interaction_type", type=str,
                            default="dot", choices=["dot", "mlp"])
        parser.add_argument('--post_lstm_sizes', type=str, default=None)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--lr', type=float, default=0.001)  # Learning rate

        parser.add_argument('--image_feature_path', type=str, required=True)
        parser.add_argument('--add_position_feature',
                            action='store_true', default=False)
        parser.add_argument('--add_percent_completion_feature',
                            action='store_true', default=False)

    @staticmethod
    def from_args(args: Any, dataloader: DataLoader) -> 'Trainer':
        """Create an instance from parsed arguments and dataloader."""
        return ClassificationTrainer(args, dataloader)

    ### Initialization ###
    def __init__(self, args, dataloader=None, model=None, optimizer=None, device=None):
        """Initialize the trainer with args and optionally dataloader, model, optimizer, and device."""
        if args is None:
            raise ValueError("args must be provided and cannot be None.")
        self.args = args
        self.device = device if device is not None else args.device
        self.clip_length = getattr(args, 'clip_length', None)

        if model is not None and optimizer is not None:
            self.model = model
            self.optimizer = optimizer
            self.image_feature_path = args.image_feature_path
            self.add_position_feature = args.add_position_feature
            self.add_percent_completion_feature = args.add_percent_completion_feature
        else:
            if dataloader is None:
                raise ValueError(
                    "dataloader must be provided if model and optimizer are not.")
            compress_sizes = [int(x) for x in args.compress_sizes.split(
                ',')] if args.compress_sizes else []
            post_lstm_sizes = [int(x) for x in args.post_lstm_sizes.split(
                ',')] if args.post_lstm_sizes else []
            clip1, clip2, _ = next(iter(dataloader))
            _, B, F, input_size = clip1.shape
            self.clip_length = F
            self.image_feature_path = args.image_feature_path
            self.add_position_feature = args.add_position_feature
            self.add_percent_completion_feature = args.add_percent_completion_feature
            self.model = ClassifierModel(
                input_size=input_size,
                hidden_size=args.hidden_size,
                interaction_type=InteractionType(args.interaction_type),
                bidirectional=args.bidirectional,
                compress_sizes=compress_sizes,
                post_lstm_sizes=post_lstm_sizes,
                dropout=args.dropout
            ).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.lr)

    ### Instance Methods ###
    def save(self, dir: str, checkpoint_idx: int) -> None:
        """Save the trainer's state to a checkpoint file, including args and clip_length."""
        path = f"{dir}/checkpoint_{checkpoint_idx}.pt"
        args_dict = vars(self.args).copy()
        if self.clip_length is not None:
            args_dict['clip_length'] = self.clip_length
        additional_info = {'args': args_dict}
        self.model.save(path, checkpoint_idx, self.optimizer, additional_info)

    @staticmethod
    def load(checkpoint_path: str, device: str) -> Tuple['Trainer', int]:
        """Load the trainer from a checkpoint file, using saved args."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = checkpoint['model_config']
        interaction_type = InteractionType(model_config['interaction_type'])
        model = ClassifierModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            interaction_type=interaction_type,
            bidirectional=model_config['bidirectional'],
            compress_sizes=model_config['compress_sizes'],
            post_lstm_sizes=model_config['post_lstm_sizes'],
            dropout=model_config['dropout']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args_dict = checkpoint['args']
        args = Namespace(**args_dict)
        trainer = ClassificationTrainer(
            args=args, model=model, optimizer=optimizer, device=device)
        epoch = checkpoint['epoch']
        return trainer, epoch

    @staticmethod
    def get_dataloader(file_list: List[str], shuffle: bool, num_workers: int, worker_init_fn) -> DataLoader:
        """Create a dataloader from the file list."""
        dataset = ClassificationDataset(file_list)
        return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)

    @staticmethod
    def compute_metrics(preds, labels):
        """Compute classification metrics including precision, recall, F1 by class and macro averages."""
        preds = np.array(preds)
        labels = np.array(labels)
        classes = [0, 1]
        metrics = {}
        for c in classes:
            TP = np.sum((preds == c) & (labels == c))
            FP = np.sum((preds == c) & (labels != c))
            FN = np.sum((preds != c) & (labels == c))
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * precision * recall / \
                (precision + recall) if precision + recall > 0 else 0
            metrics[f'Precision/Class {c}'] = precision
            metrics[f'Recall/Class {c}'] = recall
            metrics[f'F1-score/Class {c}'] = f1
        # Macro averages
        metrics['Macro Avg/Precision'] = (
            metrics['Precision/Class 0'] + metrics['Precision/Class 1']) / 2
        metrics['Macro Avg/Recall'] = (
            metrics['Recall/Class 0'] + metrics['Recall/Class 1']) / 2
        metrics['Macro Avg/F1'] = (metrics['F1-score/Class 0'] +
                                   metrics['F1-score/Class 1']) / 2
        return metrics

    def fit(self, dataloader: DataLoader) -> Dict:
        """Perform a forward and backward pass for training, returning extended metrics."""
        self.model.train()
        total_loss = 0.0
        for clip1, clip2, label in tqdm(dataloader, total=len(dataloader)):
            clip1 = clip1.to(self.device).squeeze()
            clip2 = clip2.to(self.device).squeeze()
            label = label.to(self.device).squeeze()
            self.optimizer.zero_grad()
            output = self.model(clip1, clip2).squeeze()
            loss = nn.BCEWithLogitsLoss()(output, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        return {"Loss": avg_loss}

    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Perform a forward pass for evaluation, returning extended metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        with torch.no_grad():
            for clip1, clip2, label in tqdm(dataloader, total=len(dataloader)):
                clip1 = clip1.to(self.device).squeeze()
                clip2 = clip2.to(self.device).squeeze()
                label = label.to(self.device).squeeze()
                output = self.model(clip1, clip2).squeeze()
                loss = nn.BCEWithLogitsLoss()(output, label)
                total_loss += loss.item()
                pred = (torch.sigmoid(output) > 0.5).float()
                all_preds.append(pred.cpu().numpy())
                all_labels.append(label.cpu().numpy())
        avg_loss = total_loss / len(dataloader)
        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()
        metrics = ClassificationTrainer.compute_metrics(all_preds, all_labels)
        metrics['Loss'] = avg_loss
        return metrics

    def predict_timecodes(self, track_id: str, source_rider_id: str, source_timecodes: List[str], target_rider_id: str) -> List[str]:
        """Predict target timecodes corresponding to source timecodes using the trained model."""
        self.model.eval()
        sample_generator = ClassifierSampleGenerator(args=Namespace(**{
            'F': self.clip_length,
            'image_feature_path': self.image_feature_path,
            'add_position_feature': self.add_position_feature,
            'add_percent_completion_feature': self.add_percent_completion_feature,
        }))
        video_feature_cache = {}
        source_video_path = get_video_file_path(track_id, source_rider_id)
        source_fps, _ = get_video_fps_and_total_frames(source_video_path)
        target_video_path = get_video_file_path(track_id, target_rider_id)
        target_fps, target_total_frames = get_video_fps_and_total_frames(
            target_video_path)

        predictions = defaultdict(list)
        batch_size = 32
        samples = []

        # ignore the first split
        for source_timecode in source_timecodes:
            v1_frame_idx = timecode_to_frames(source_timecode, source_fps)
            for v2_frame_idx in range(target_total_frames):
                if v1_frame_idx < self.clip_length - 1 or v2_frame_idx < self.clip_length - 1:
                    continue
                row_dict = {
                    'track_id': track_id,
                    'v1_rider_id': source_rider_id,
                    'v2_rider_id': target_rider_id,
                    'v1_frame_idx': v1_frame_idx,
                    'v2_frame_idx': v2_frame_idx,
                    'label': 0.0,  # dummy value
                }
                clip1, clip2, _ = sample_generator.get_features(
                    video_feature_cache, **row_dict)
                samples.append((source_timecode, v2_frame_idx, clip1, clip2))
                if len(samples) == batch_size:
                    clip1_batch = torch.stack(
                        [torch.from_numpy(s[2]) for s in samples]).to(self.device)
                    clip2_batch = torch.stack(
                        [torch.from_numpy(s[3]) for s in samples]).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(clip1_batch, clip2_batch)
                        probabilities = torch.sigmoid(
                            outputs).squeeze().cpu().numpy()
                    for (stc, v2_idx, _, _), prob in zip(samples, probabilities):
                        predictions[stc].append((v2_idx, prob))
                    samples = []

        # Process remaining samples
        if samples:
            clip1_batch = torch.stack(
                [torch.from_numpy(s[2]) for s in samples]).to(self.device)
            clip2_batch = torch.stack(
                [torch.from_numpy(s[3]) for s in samples]).to(self.device)
            with torch.no_grad():
                outputs = self.model(clip1_batch, clip2_batch)
                probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            for (stc, v2_idx, _, _), prob in zip(samples, probabilities):
                predictions[stc].append((v2_idx, prob))

        # Get highest confidence prediction per source_timecode
        predicted_frame_indices = {}
        for stc, preds in predictions.items():
            if preds:
                v2_idx, _ = max(preds, key=lambda x: x[1])
                predicted_frame_indices[stc] = v2_idx
            else:
                predicted_frame_indices[stc] = None

        # Convert to timecodes
        predicted_timecodes = []
        for stc in source_timecodes:
            v2_frame_idx = predicted_frame_indices.get(stc)
            if v2_frame_idx is not None:
                timecode = frame_idx_to_timecode(v2_frame_idx, target_fps)
                predicted_timecodes.append(timecode)
            else:
                predicted_timecodes.append(None)

        return predicted_timecodes


class InteractionType(Enum):
    DOT = 'dot'
    MLP = 'mlp'


class ClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, interaction_type=InteractionType.MLP, bidirectional=False, compress_sizes=[], post_lstm_sizes=[], dropout=0.0):
        super(ClassifierModel, self).__init__()
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.compress_sizes = compress_sizes
        self.post_lstm_sizes = post_lstm_sizes
        self.dropout = dropout
        self.interaction_type = interaction_type

        # Compression layers (same for both interaction types)
        if compress_sizes:
            layers = []
            in_size = input_size
            for size in compress_sizes:
                layers.append(nn.Linear(in_size, size))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_size = size
            self.compression = nn.Sequential(*layers)
            lstm_input_size = compress_sizes[-1]
        else:
            self.compression = nn.Identity()
            lstm_input_size = input_size

        # LSTM layer (same for both interaction types)
        self.lstm = nn.LSTM(lstm_input_size, hidden_size,
                            batch_first=True, bidirectional=bidirectional)

        # Configure interaction-specific layers
        if self.interaction_type == InteractionType.MLP:
            lstm_output_size = 2 * hidden_size if bidirectional else hidden_size
            concat_size = 2 * lstm_output_size
            if post_lstm_sizes:
                layers = []
                in_size = concat_size
                for size in post_lstm_sizes:
                    layers.append(nn.Linear(in_size, size))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    in_size = size
                self.post_lstm = nn.Sequential(*layers)
                final_input_size = post_lstm_sizes[-1]
            else:
                self.post_lstm = nn.Identity()
                final_input_size = concat_size
            self.fc = nn.Linear(final_input_size, 1)
        elif self.interaction_type == InteractionType.DOT:
            self.fc = nn.Linear(1, 1)
        else:
            raise ValueError("Invalid interaction_type")

    def forward(self, clip1, clip2):
        # Compress input clips
        compressed_clip1 = self.compression(clip1)
        compressed_clip2 = self.compression(clip2)

        # Get LSTM outputs
        if self.bidirectional:
            _, (h_n, _) = self.lstm(compressed_clip1)
            h1 = torch.cat((h_n[-2], h_n[-1]), dim=1)
            _, (h_n, _) = self.lstm(compressed_clip2)
            h2 = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            _, (h1, _) = self.lstm(compressed_clip1)
            h1 = h1[-1]
            _, (h2, _) = self.lstm(compressed_clip2)
            h2 = h2[-1]

        # Handle interaction based on type
        if self.interaction_type == InteractionType.MLP:
            combined = torch.cat((h1, h2), dim=1)
            post_lstm_output = self.post_lstm(combined)
            output = self.fc(post_lstm_output)
        elif self.interaction_type == InteractionType.DOT:
            dot_product = torch.sum(h1 * h2, dim=1, keepdim=True)
            output = self.fc(dot_product)
        else:
            raise ValueError("Invalid interaction_type")
        return output

    def save(self, path, epoch, optimizer, additional_info=None):
        """Save the model state along with additional information."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_type': 'position',
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'bidirectional': self.bidirectional,
                'compress_sizes': self.compress_sizes,
                'post_lstm_sizes': self.post_lstm_sizes,
                'dropout': self.dropout,
                'interaction_type': self.interaction_type.value,
            }
        }
        if additional_info:
            checkpoint.update(additional_info)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model_config = checkpoint['model_config']
        interaction_type = InteractionType(
            model_config.pop('interaction_type'))
        model = cls(interaction_type=interaction_type,
                    **model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['epoch'], checkpoint['optimizer_state_dict']
