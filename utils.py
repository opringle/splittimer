from enum import Enum
import math
import cv2
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import re

def get_default_device_name():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def get_video_fps_and_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames

def get_frame(video_path, frame_idx):
    """
    Extract a specific frame from the video file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    else:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")

def timecode_to_frames(timecode, fps):
    parts = timecode.split(':')
    if len(parts) != 3:
        raise ValueError(f"Timecode must be in MM:SS:FF format, got '{timecode}'")
    MM, SS, FF = map(int, parts)
    time_in_seconds = MM * 60 + SS + FF / 24.0  # Convert to seconds, based on 24 FPS annotation in devinci resolve
    frame_index = int(time_in_seconds * fps)    # Frame index in the actual video
    return frame_index

def frame_idx_to_timecode(frame_index, fps):
    """
    Convert a frame index to a timecode in "MM:SS:FF" format based on the video's native frame rate.
    The timecode reflects how it would appear in DaVinci Resolve when loaded at 24.0 FPS.
    
    Args:
        frame_index (int): The frame index in the video.
        fps (float): The native frame rate of the video.
    
    Returns:
        str: The timecode in "MM:SS:FF" format.
    """
    # Calculate time in seconds from frame index
    time_in_seconds = frame_index / fps
    
    # Extract minutes
    MM = math.floor(time_in_seconds / 60)
    
    # Extract remaining seconds
    remaining_seconds = time_in_seconds - MM * 60
    SS = math.floor(remaining_seconds)
    
    # Extract fractional seconds and convert to frames at 24 FPS
    fractional_seconds = remaining_seconds - SS
    FF = math.floor(fractional_seconds * 24)
    
    # Format as MM:SS:FF with leading zeros
    timecode = f"{MM:02d}:{SS:02d}:{FF:02d}"
    return timecode

def setup_logging(log_level="INFO"):
    logging.basicConfig(level=getattr(logging, log_level.upper()), format='%(levelname)s: %(message)s')

class InteractionType(Enum):
    DOT = 'dot'
    MLP = 'mlp'

class PositionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, interaction_type=InteractionType.MLP, bidirectional=False, compress_sizes=[], post_lstm_sizes=[], dropout=0.0):
        super(PositionClassifier, self).__init__()
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
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        
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

    def save(self, path, epoch, optimizer):
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
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model_config = checkpoint['model_config']
        interaction_type = InteractionType(model_config.pop('interaction_type'))
        model = cls(interaction_type=interaction_type, **model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['epoch'], checkpoint['optimizer_state_dict']
    
class SequencePositionClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        layers = []
        in_size = 2 * input_size  # Concatenate two feature vectors
        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = size
        layers.append(nn.Linear(in_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, clip1, clip2):
        # Input shapes: (B, 1, input_size)
        clip1 = clip1.squeeze(1)  # (B, input_size)
        clip2 = clip2.squeeze(1)  # (B, input_size)
        combined = torch.cat((clip1, clip2), dim=1)  # (B, 2*input_size)
        output = self.model(combined)  # (B, 1)
        return output

    def save(self, path, epoch, optimizer):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_type': 'sequence',
            'model_config': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
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

def parse_clip_range(file_name, feature_type='individual', sequence_length=None):
    """
    Parse the frame range from a feature file name based on feature type.
    
    Args:
        file_name (str): Name of the feature file
        feature_type (str): 'individual' or 'sequence'
        sequence_length (int, optional): Length of sequence for sequence features
    
    Returns:
        tuple: (start_frame, end_frame) or None if parsing fails
    """
    if feature_type == 'individual':
        match = re.match(r"(\d+)_to_(\d+)_resnet50\.npy", file_name)
        if match:
            return int(match.group(1)), int(match.group(2))
    elif feature_type == 'sequence':
        if sequence_length is None:
            return None
        match = re.match(r"(\d+)_to_(\d+)_iconv3d_F{}.npy".format(sequence_length), file_name)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None

def load_image_features_from_disk(track_id, rider_id, start_idx, end_idx, feature_base_path, feature_type='individual', sequence_length=None):
    """
    Load image features from disk for a given track and rider.
    
    Args:
        track_id (str): Track identifier
        rider_id (str): Rider identifier
        start_idx (int): Starting frame index (for individual features)
        end_idx (int): Ending frame index
        feature_base_path (str): Base path to feature files
        feature_type (str): 'individual' or 'sequence'
        sequence_length (int, optional): Sequence length for sequence features
    
    Returns:
        np.ndarray: Features array; shape (F, feature_dim) for individual, (1, feature_dim) for sequence
    """
    if feature_type == 'individual':
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
            range_info = parse_clip_range(file_path.name, feature_type='individual')
            if range_info:
                clip_ranges.append((range_info[0], range_info[1], file_path))
            else:
                logging.error(f"Skipping invalid file name: {file_path.name}")

        if not clip_ranges:
            logging.error(f"No valid individual feature files found in {feature_base_dir}")
            return np.array([])

        overlapping_clips = [
            (clip_start, clip_end, file_path)
            for clip_start, clip_end, file_path in clip_ranges
            if clip_start <= end_idx and clip_end >= start_idx
        ]

        if not overlapping_clips:
            logging.error(f"No individual clips overlap with range [{start_idx}, {end_idx}]")
            return np.array([])

        features = []
        for clip_start, clip_end, file_path in overlapping_clips:
            try:
                clip_features = np.load(file_path)
                if clip_features.shape[1] != 2048:
                    logging.error(f"Unexpected individual feature shape {clip_features.shape} in {file_path}")
                    return np.array([])
            except Exception as e:
                logging.error(f"Error loading individual features from {file_path}: {e}")
                return np.array([])

            extract_start = max(clip_start, start_idx)
            extract_end = min(clip_end, end_idx)
            rel_start = extract_start - clip_start
            rel_end = extract_end - clip_start
            if rel_start <= rel_end:
                clip_features_subset = clip_features[rel_start:rel_end + 1]
                features.append(clip_features_subset)

        if not features:
            logging.error(f"No individual features loaded for range [{start_idx}, {end_idx}]")
            return np.array([])

        features = np.concatenate(features, axis=0)
        if features.shape[0] != F:
            logging.error(f"Loaded {features.shape[0]} individual frames, expected {F}")
            return np.array([])
        return features

    elif feature_type == 'sequence':
        if sequence_length is None:
            logging.error("sequence_length must be provided for sequence features")
            return np.array([])

        feature_base_dir = Path(feature_base_path) / track_id / rider_id
        if not feature_base_dir.exists():
            logging.error(f"Feature directory {feature_base_dir} does not exist")
            return np.array([])

        clip_ranges = []
        for file_path in feature_base_dir.glob(f"*_iconv3d_F{sequence_length}.npy"):
            range_info = parse_clip_range(file_path.name, feature_type='sequence', sequence_length=sequence_length)
            if range_info:
                clip_ranges.append((range_info[0], range_info[1], file_path))

        if not clip_ranges:
            logging.error(f"No valid sequence feature files found in {feature_base_dir} for F={sequence_length}")
            return np.array([])

        for clip_start, clip_end, file_path in clip_ranges:
            if clip_start <= end_idx <= clip_end:
                try:
                    clip_features = np.load(file_path)
                    rel_idx = end_idx - clip_start
                    if rel_idx < 0 or rel_idx >= clip_features.shape[0]:
                        logging.error(f"end_idx {end_idx} out of range for clip {clip_start} to {clip_end}")
                        return np.array([])
                    feature = clip_features[rel_idx]
                    return feature[None, :]  # Shape (1, feature_dim)
                except Exception as e:
                    logging.error(f"Error loading sequence feature from {file_path}: {e}")
                    return np.array([])

        logging.error(f"No sequence feature file found for end_idx {end_idx}")
        return np.array([])

    else:
        logging.error(f"Unknown feature_type: {feature_type}")
        return np.array([])

def get_clip_indices_ending_at(end_idx, F):
    start = max(0, end_idx - F + 1)
    clip_indices = list(range(start, end_idx + 1))
    assert len(clip_indices) == F, f"clip indices length {len(clip_indices)} != F ({F})"
    # if len(clip_indices) < F:
    #     # This just repeats the final frame
    #     # print(f"clip_indices before = {clip_indices}")
    #     clip_indices += [clip_indices[-1]] * (F - len(clip_indices))
    #     # print(f"clip_indices after = {clip_indices}")
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