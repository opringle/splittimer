import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import re

def parse_clip_range(file_name):
    """Parse start_idx and end_idx from file name like '000000_to_000049_resnet50.npy'."""
    match = re.match(r"(\d+)_to_(\d+)_resnet50\.npy", file_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def load_image_features_from_disk(track_id, rider_id, start_idx, end_idx, feature_base_path):
    """Load precomputed ResNet50 features from disk for a frame range, handling multiple clips."""
    F = end_idx - start_idx + 1
    if F <= 0:
        logging.error(f"Invalid frame range: start_idx {start_idx} > end_idx {end_idx}")
        return np.array([])

    feature_base_dir = Path(feature_base_path) / track_id / rider_id
    if not feature_base_dir.exists():
        logging.error(f"Feature directory {feature_base_dir} does not exist")
        return np.array([])

    # List all .npy files and parse their ranges
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

    logging.debug(f"Found {len(clip_ranges)} clips for {track_id}/{rider_id}")

    # Find overlapping clips
    overlapping_clips = [
        (clip_start, clip_end, file_path)
        for clip_start, clip_end, file_path in clip_ranges
        if clip_start <= end_idx and clip_end >= start_idx
    ]

    logging.debug(f"Overlapping clips for [{start_idx}, {end_idx}]: {[(clip_start, clip_end) for clip_start, clip_end, _ in overlapping_clips]}")

    if not overlapping_clips:
        logging.error(f"No clips overlap with range [{start_idx}, {end_idx}]")
        return np.array([])

    # Load and extract features from overlapping clips
    features = []
    for clip_start, clip_end, file_path in overlapping_clips:
        try:
            clip_features = np.load(file_path)
            logging.debug(f"Loaded {file_path} with shape {clip_features.shape}")
            if clip_features.shape[1] != 2048:
                logging.error(f"Unexpected feature shape {clip_features.shape} in {file_path}, expected (N, 2048)")
                return np.array([])
        except Exception as e:
            logging.error(f"Error loading features from {file_path}: {e}")
            return np.array([])

        # Extract the subset of features within the requested range
        extract_start = max(clip_start, start_idx)
        extract_end = min(clip_end, end_idx)
        rel_start = extract_start - clip_start
        rel_end = extract_end - clip_start
        if rel_start <= rel_end:
            clip_features_subset = clip_features[rel_start:rel_end + 1]
            logging.debug(f"Extracted from {file_path}: frames [{extract_start}:{extract_end}] relative [{rel_start}:{rel_end+1}], shape {clip_features_subset.shape}")
            features.append(clip_features_subset)

    if not features:
        logging.error(f"No features loaded for range [{start_idx}, {end_idx}]")
        return np.array([])

    # Concatenate features into a single array
    features = np.concatenate(features, axis=0)
    logging.debug(f"Concatenated features shape: {features.shape}, expected F={F}")
    if features.shape[0] != F:
        logging.error(f"Loaded {features.shape[0]} frames, expected {F} for range {start_idx}:{end_idx}")
        return np.array([])

    return features

def get_clip_indices_ending_at(end_idx, F):
    """Compute frame indices for a clip ending at end_idx with length F, padding if necessary."""
    start = max(0, end_idx - F + 1)
    clip_indices = list(range(start, end_idx + 1))
    if len(clip_indices) < F:
        clip_indices += [clip_indices[-1]] * (F - len(clip_indices))
    return clip_indices[:F]  # Ensure exactly F indices

def save_batch(save_dir, batch_count, batch_clip1s, batch_clip2s, batch_labels):
    """Save a batch of clips and labels to an .npz file in the specified directory."""
    batch_clip_1_tensor = np.stack(batch_clip1s, axis=0)
    batch_clip_2_tensor = np.stack(batch_clip2s, axis=0)
    batch_label_tensor = np.array(batch_labels)
    np.savez(
        save_dir / f"batch_{batch_count:06d}.npz",
        clip1s=batch_clip_1_tensor,
        clip2s=batch_clip_2_tensor,
        labels=batch_label_tensor
    )
    logging.debug(f"Saved batch {batch_count} to {save_dir}. Clip1 features shape = {batch_clip_1_tensor.shape} Clip2 features shape = {batch_clip_2_tensor.shape}. Labels shape = {batch_label_tensor.shape}")

def main():
    parser = argparse.ArgumentParser(description="Generate batched training data using precomputed ResNet50 features.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file with 'set' column ('train' or 'val')")
    parser.add_argument("image_feature_path", type=str, help="Path to directory of clip features (image_feature_path/<trackId>/<riderId>/<start_idx>_<end_idx>_resnet50.npy)")
    parser.add_argument("output_dir", type=str, help="Directory to save .npz files (will create 'train' and 'val' subdirectories)")
    parser.add_argument("--F", type=int, default=50, help="Number of frames per clip")
    parser.add_argument("--batch_size", type=int, default=32, help="Samples per .npz file")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # Load and validate CSV
    df = pd.read_csv(args.csv_path)
    if 'set' not in df.columns:
        logging.error("CSV file must contain a 'set' column indicating 'train' or 'val'")
        exit(1)
    
    # Shuffle samples
    df = df.sample(frac=1, random_state=None, ignore_index=True)
    
    # Compute train and val sample counts
    num_train = (df['set'] == 'train').sum()
    num_val = (df['set'] == 'val').sum()
    logging.info(f"Loaded metadata for {num_train} training and {num_val} validation samples")

    # Create output directories
    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Initialize batch lists and counters
    train_batch_clip1s, train_batch_clip2s, train_batch_labels = [], [], []
    val_batch_clip1s, val_batch_clip2s, val_batch_labels = [], [], []
    train_batch_count = 0
    val_batch_count = 0

    # Process samples
    for row in tqdm(df.itertuples(), total=len(df), desc="Generating samples"):

        # Compute clip indices
        v1_indices = get_clip_indices_ending_at(row.v1_frame_idx, args.F)
        v2_indices = get_clip_indices_ending_at(row.v2_frame_idx, args.F)

        # Set start and end indices for loading features
        v1_start_idx = v1_indices[0]
        v1_end_idx = v1_indices[-1]
        v2_start_idx = v2_indices[0]
        v2_end_idx = v2_indices[-1]

        # Load precomputed features for v1
        v1_features = load_image_features_from_disk(row.track_id, row.v1_rider_id, v1_start_idx, v1_end_idx, args.image_feature_path)
        if v1_features.size == 0 or v1_features.shape[0] != (v1_end_idx - v1_start_idx + 1):
            logging.error(f"Failed to load features for v1 clip {v1_start_idx}:{v1_end_idx} in {row.Index}, expected {v1_end_idx - v1_start_idx + 1} frames, got {v1_features.shape[0] if v1_features.size > 0 else 0}")
            continue

        # Create padded features for v1
        v1_features_padded = np.array([v1_features[idx - v1_start_idx] for idx in v1_indices])
        if v1_features.shape[0] != v1_features_padded.shape[0]:
            logging.debug(f"padded features from shape = {v1_features.shape} to {v1_features_padded.shape}")

        # Load precomputed features for v2
        v2_features = load_image_features_from_disk(row.track_id, row.v2_rider_id, v2_start_idx, v2_end_idx, args.image_feature_path)
        if v2_features.size == 0 or v2_features.shape[0] != (v2_end_idx - v2_start_idx + 1):
            logging.error(f"Failed to load features for v2 clip {v2_start_idx}:{v2_end_idx} in {row.Index}, expected {v2_end_idx - v2_start_idx + 1} frames, got {v2_features.shape[0] if v2_features.size > 0 else 0}")
            continue

        # Create padded features for v2
        v2_features_padded = np.array([v2_features[idx - v2_start_idx] for idx in v2_indices])
        

        # Append absolute frame indices as the 2049th feature
        v1_features_with_pos = np.concatenate([v1_features_padded, np.array(v1_indices, dtype=np.float32)[:, None]], axis=1)
        v2_features_with_pos = np.concatenate([v2_features_padded, np.array(v2_indices, dtype=np.float32)[:, None]], axis=1)

        # Append to appropriate batch based on 'set'
        if row.set == 'train':
            train_batch_clip1s.append(v1_features_with_pos)
            train_batch_clip2s.append(v2_features_with_pos)
            train_batch_labels.append(row.label)
            if len(train_batch_clip1s) >= args.batch_size:
                save_batch(train_dir, train_batch_count, train_batch_clip1s, train_batch_clip2s, train_batch_labels)
                train_batch_clip1s, train_batch_clip2s, train_batch_labels = [], [], []
                train_batch_count += 1
        elif row.set == 'val':
            val_batch_clip1s.append(v1_features_with_pos)
            val_batch_clip2s.append(v2_features_with_pos)
            val_batch_labels.append(row.label)
            if len(val_batch_clip1s) >= args.batch_size:
                save_batch(val_dir, val_batch_count, val_batch_clip1s, val_batch_clip2s, val_batch_labels)
                val_batch_clip1s, val_batch_clip2s, val_batch_labels = [], [], []
                val_batch_count += 1
        else:
            logging.warning(f"Unknown set value: {row.set} for sample {row.Index}, skipping")

    # Save remaining batches
    if train_batch_clip1s:
        save_batch(train_dir, train_batch_count, train_batch_clip1s, train_batch_clip2s, train_batch_labels)
        train_batch_count += 1
    if val_batch_clip1s:
        save_batch(val_dir, val_batch_count, val_batch_clip1s, val_batch_clip2s, val_batch_labels)
        val_batch_count += 1

    # Log completion
    logging.info(f"Generated {num_train} training samples in {train_batch_count} batches in {train_dir}")
    logging.info(f"Generated {num_val} validation samples in {val_batch_count} batches in {val_dir}")

if __name__ == "__main__":
    main()