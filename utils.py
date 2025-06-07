import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import re

def pad_features_to_length(features, indices, F):
    """
    Pad or truncate the features array to ensure it has exactly F frames.
    
    Args:
        features (np.ndarray): Feature array of shape (num_frames, feature_dim)
        indices (list or np.ndarray): List of frame indices, length F
        F (int): Desired number of frames
    
    Returns:
        np.ndarray: Padded or truncated features array of shape (F, feature_dim)
    """
    num_frames = features.shape[0]
    feature_dim = features.shape[1]
    
    if num_frames == F:
        return features
    elif num_frames < F:
        # Pad by repeating the last frame's features
        padding = np.tile(features[-1:], (F - num_frames, 1))
        return np.vstack((features, padding))
    else:
        # Truncate to F frames if too long
        return features[:F]
    
def setup_logging(log_level="INFO"):
    logging.basicConfig(level=getattr(logging, log_level.upper()), format='%(levelname)s: %(message)s')

class PositionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=False, compress_sizes=[], post_lstm_sizes=[], dropout=0.0):
        super(PositionClassifier, self).__init__()
        self.bidirectional = bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.compress_sizes = compress_sizes
        self.post_lstm_sizes = post_lstm_sizes
        self.dropout = dropout
        
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
        
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
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

    def forward(self, clip1, clip2):
        compressed_clip1 = self.compression(clip1)
        compressed_clip2 = self.compression(clip2)
        
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
        
        combined = torch.cat((h1, h2), dim=1)
        post_lstm_output = self.post_lstm(combined)
        output = self.fc(post_lstm_output)
        return output

    def save(self, path, epoch, optimizer):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'bidirectional': self.bidirectional,
                'compress_sizes': self.compress_sizes,
                'post_lstm_sizes': self.post_lstm_sizes,
                'dropout': self.dropout,
            }
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model_config = checkpoint['model_config']
        model = cls(**model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['epoch'], checkpoint['optimizer_state_dict']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_clip_range(file_name):
    match = re.match(r"(\d+)_to_(\d+)_resnet50\.npy", file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def load_image_features_from_disk(track_id, rider_id, start_idx, end_idx, feature_base_path):
    F = end_idx - start_idx + 1
    if F <= 0:
        logging.error(f"Invalid frame range: start_idx {start_idx} > end_idx {end_idx}")
        return np.array([])

    feature_base_dir = Path(feature_base_path) / track_id / rider_id
    if not feature_base_dir.exists():
        logging.error(f"Feature directory {feature_base_dir} does not exist")
        return np.array([])

    clip_ranges = []
    for file_path in feature_base_dir.glob("*_resnet50.npy"):
        range_info = parse_clip_range(file_path.name)
        if range_info:
            clip_ranges.append((range_info[0], range_info[1], file_path))
        else:
            logging.error(f"Skipping invalid file name: {file_path.name}")

    if not clip_ranges:
        logging.error(f"No valid feature files found in {feature_base_dir}")
        return np.array([])

    overlapping_clips = [
        (clip_start, clip_end, file_path)
        for clip_start, clip_end, file_path in clip_ranges
        if clip_start <= end_idx and clip_end >= start_idx
    ]

    if not overlapping_clips:
        logging.error(f"No clips overlap with range [{start_idx}, {end_idx}]")
        return np.array([])

    features = []
    for clip_start, clip_end, file_path in overlapping_clips:
        try:
            clip_features = np.load(file_path)
            if clip_features.shape[1] != 2048:
                logging.error(f"Unexpected feature shape {clip_features.shape} in {file_path}")
                return np.array([])
        except Exception as e:
            logging.error(f"Error loading features from {file_path}: {e}")
            return np.array([])

        extract_start = max(clip_start, start_idx)
        extract_end = min(clip_end, end_idx)
        rel_start = extract_start - clip_start
        rel_end = extract_end - clip_start
        if rel_start <= rel_end:
            clip_features_subset = clip_features[rel_start:rel_end + 1]
            features.append(clip_features_subset)

    if not features:
        logging.error(f"No features loaded for range [{start_idx}, {end_idx}]")
        return np.array([])

    features = np.concatenate(features, axis=0)
    if features.shape[0] != F:
        logging.error(f"Loaded {features.shape[0]} frames, expected {F}")
        return np.array([])
    return features

def get_clip_indices_ending_at(end_idx, F):
    start = max(0, end_idx - F + 1)
    clip_indices = list(range(start, end_idx + 1))
    if len(clip_indices) < F:
        clip_indices += [clip_indices[-1]] * (F - len(clip_indices))
    return clip_indices[:F]

def save_batch(save_dir, batch_count, batch_clip1s, batch_clip2s, batch_labels):
    batch_clip_1_tensor = np.stack(batch_clip1s, axis=0)
    batch_clip_2_tensor = np.stack(batch_clip2s, axis=0)
    batch_label_tensor = np.array(batch_labels)
    np.savez(
        save_dir / f"batch_{batch_count:06d}.npz",
        clip1s=batch_clip_1_tensor,
        clip2s=batch_clip_2_tensor,
        labels=batch_label_tensor
    )