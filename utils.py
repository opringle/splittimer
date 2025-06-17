from collections import defaultdict
import math
import random
from typing import List, Sequence
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
import re


def log_dict(prefix: str, metrics: dict) -> str:
    log_str = prefix
    for metric_name, metric_value in metrics.items():
        log_str += f'\n\t{metric_name} {metric_value:.3f}'
    logging.info(log_str)


def get_video_metadata(video_path, splits, rider_id, track_id):
    fps, total_frames = get_video_fps_and_total_frames(video_path)

    split_indices_raw = [timecode_to_frames(tc, fps) for tc in splits]
    split_indices = []
    for idx in split_indices_raw:
        assert idx < total_frames, f"Split index {idx} exceeds total frames {total_frames} for video {video_path}"
        assert idx >= 0, f"Split index {idx} is negative for video {video_path}"
        split_indices.append(idx)

    frame_indices = np.arange(total_frames)
    labels = np.zeros(total_frames)
    for idx in split_indices:
        labels[idx] = 1.0
    return frame_indices, labels, rider_id, track_id


def add_features_to_clip(image_features, frame_indices, total_frames=None, add_position=True, add_percent_completion=False):
    features = image_features.copy()
    if add_position:
        position_feature = np.array(frame_indices, dtype=np.float32)[:, None]
        features = np.concatenate([features, position_feature], axis=1)
    if add_percent_completion:
        if total_frames is None:
            raise ValueError(
                "total_frames must be provided to add percent completion feature")
        percent_completion_feature = (
            np.array(frame_indices, dtype=np.float32) / total_frames)[:, None]
        features = np.concatenate(
            [features, percent_completion_feature], axis=1)
    return features


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
        raise ValueError(
            f"Timecode must be in MM:SS:FF format, got '{timecode}'")
    MM, SS, FF = map(int, parts)
    # Convert to seconds, based on 24 FPS annotation in devinci resolve
    time_in_seconds = MM * 60 + SS + FF / 24.0
    # Frame index in the actual video
    frame_index = int(time_in_seconds * fps)
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


def get_video_file_path(track_id: str, rider_id: str):
    return Path("downloaded_videos") / track_id / rider_id / f"{track_id}_{rider_id}.mp4"


def setup_seed(seed):
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logging.info(f"Set random seed to {seed}")


def setup_logging(log_level="INFO"):
    logging.basicConfig(level=getattr(logging, log_level.upper()),
                        format='%(levelname)s: %(message)s')


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
        match = re.match(
            r"(\d+)_to_(\d+)_iconv3d_F{}.npy".format(sequence_length), file_name)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def load_image_features_from_disk(track_id, rider_id, start_idx, end_idx, feature_base_path):
    F = end_idx - start_idx + 1
    assert F > 0, f"Invalid frame range: start_idx {start_idx} > end_idx {end_idx}"

    feature_base_dir = Path(feature_base_path) / track_id / rider_id
    assert feature_base_dir.exists(
    ), f"Feature directory {feature_base_dir} does not exist"

    clip_ranges = []
    for file_path in feature_base_dir.glob("*_resnet50.npy"):
        range_info = parse_clip_range(
            file_path.name, feature_type='individual')
        if range_info:
            clip_ranges.append((range_info[0], range_info[1], file_path))
        else:
            logging.error(f"Skipping invalid file name: {file_path.name}")

    assert clip_ranges, f"No valid individual feature files found in {feature_base_dir}"

    overlapping_clips = [
        (clip_start, clip_end, file_path)
        for clip_start, clip_end, file_path in clip_ranges
        if clip_start <= end_idx and clip_end >= start_idx
    ]

    assert overlapping_clips, f"No individual clips overlap with range [{start_idx}, {end_idx}]"

    features = []
    for clip_start, clip_end, file_path in overlapping_clips:
        clip_features = np.load(file_path)
        assert clip_features.shape[
            1] == 2048, f"Unexpected individual feature shape {clip_features.shape} in {file_path}"
        extract_start = max(clip_start, start_idx)
        extract_end = min(clip_end, end_idx)
        rel_start = extract_start - clip_start
        rel_end = extract_end - clip_start
        if rel_start <= rel_end:
            clip_features_subset = clip_features[rel_start:rel_end + 1]
            features.append(clip_features_subset)

    assert features, f"No individual features loaded for range [{start_idx}, {end_idx}]"

    features = np.concatenate(features, axis=0)
    assert features.shape[0] == F, f"Loaded {features.shape[0]} individual frames, expected {F}"
    return features


def get_clip_indices_ending_at(end_idx, F):
    start = max(0, end_idx - F + 1)
    clip_indices = list(range(start, end_idx + 1))
    assert len(
        clip_indices) == F, f"clip indices length {len(clip_indices)} != F ({F})"
    return clip_indices[:F]


def save_batch(save_dir: str, samples: List[Sequence[np.ndarray]], batch_idx: int) -> None:
    """
    Saves a batch of size len(samples) to a npz file in save_dir/batch_{batch_idx}.npz.
    Each sample is a sequence of numpy arrays, concatenated by position across samples.

    Args:
        save_dir (str): Directory to save the NPZ file.
        batch_idx (int): Index of the batch for the filename.
        samples (List[Sequence[np.ndarray]]): List of sequences, each containing NumPy arrays.

    Raises:
        ValueError: If sequences have different lengths or arrays at the same position have incompatible shapes.
    """
    # Check for consistent sequence lengths
    seq_lengths = [len(s) for s in samples]
    if len(set(seq_lengths)) > 1:
        raise ValueError(
            "All sequences must have the same length for stacking.")

    # Group arrays by their position across all sequences
    idx_to_list_of_np_array = defaultdict(list)
    for s in samples:
        for idx, np_array in enumerate(s):
            idx_to_list_of_np_array[idx].append(np_array)

    # Stack arrays at each position into a single array
    batch_data = {}
    for idx, feature_list in idx_to_list_of_np_array.items():
        try:
            batch_data[f"data_{idx}"] = np.stack(feature_list, axis=0)
        except ValueError as e:
            raise ValueError(
                f"Arrays at position {idx} have incompatible shapes for stacking: {str(e)}")

    # Determine the file path and ensure the directory exists
    batch_file_path = Path(save_dir) / f"batch_{batch_idx}.npz"
    batch_file_path.parent.mkdir(parents=True, exist_ok=True)

    logging.debug(
        f"Saving batch {idx} with shapes: {[f.shape for f in batch_data.values()]}")

    # Save the stacked arrays to an NPZ file
    np.savez(batch_file_path, **batch_data)
