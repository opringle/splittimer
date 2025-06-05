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
            logging.warning(f"Skipping invalid file name: {file_path.name}")

    if not clip_ranges:
        logging.error(f"No valid feature files found in {feature_base_dir}")
        return np.array([])

    # Find overlapping clips
    overlapping_clips = [
        (clip_start, clip_end, file_path)
        for clip_start, clip_end, file_path in clip_ranges
        if clip_start <= end_idx and clip_end >= start_idx
    ]

    if not overlapping_clips:
        logging.error(f"No clips overlap with range [{start_idx}, {end_idx}]")
        return np.array([])

    # Load and extract features from overlapping clips
    features = []
    for clip_start, clip_end, file_path in overlapping_clips:
        try:
            clip_features = np.load(file_path)
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
            features.append(clip_features_subset)

    if not features:
        logging.error(f"No features loaded for range [{start_idx}, {end_idx}]")
        return np.array([])

    # Concatenate features into a single array
    features = np.concatenate(features, axis=0)
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

def main():
    parser = argparse.ArgumentParser(description="Generate batched training data using precomputed ResNet50 features.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser.add_argument("image_feature_path", type=str, help="Path to directory of clip features (image_feature_path/<trackId>/<riderId>/<start_idx>_<end_idx>_resnet50.npy)")
    parser.add_argument("output_dir", type=str, help="Directory to save .npz files")
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

    df = pd.read_csv(args.csv_path)
    logging.info(f"Loaded metadata for {len(df)} training samples")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_clip1s, batch_clip2s, batch_labels = [], [], []
    batch_count = 0

    for row in tqdm(df.itertuples(), total=len(df), desc="Generating samples"):
        # Compute clip indices directly
        v1_indices = get_clip_indices_ending_at(row.v1_frame_idx, args.F)
        v2_indices = get_clip_indices_ending_at(row.v2_frame_idx, args.F)

        if len(v1_indices) != args.F or len(v2_indices) != args.F:
            logging.error(f"Failed to compute {args.F} indices for sample {row.Index}, skipping")
            continue

        # Load precomputed features
        v1_start_idx = v1_indices[0]
        v1_end_idx = v1_indices[-1]
        v1_features = load_image_features_from_disk(row.track_id, row.v1_rider_id, v1_start_idx, v1_end_idx, args.image_feature_path)
        if v1_features.size == 0 or v1_features.shape[0] != args.F:
            logging.error(f"Failed to load features for v1 clip {v1_start_idx}:{v1_end_idx} in {row.Index}, skipping")
            continue

        v2_start_idx = v2_indices[0]
        v2_end_idx = v2_indices[-1]
        v2_features = load_image_features_from_disk(row.track_id, row.v2_rider_id, v2_start_idx, v2_end_idx, args.image_feature_path)
        if v2_features.size == 0 or v2_features.shape[0] != args.F:
            logging.error(f"Failed to load features for v2 clip {v2_start_idx}:{v2_end_idx} in {row.Index}, skipping")
            continue

        # Append absolute frame indices as 2049th feature
        v1_features_with_pos = np.concatenate([v1_features, np.array(v1_indices, dtype=np.float32)[:, None]], axis=1)
        v2_features_with_pos = np.concatenate([v2_features, np.array(v2_indices, dtype=np.float32)[:, None]], axis=1)

        # Add to batch
        batch_clip1s.append(v1_features_with_pos)
        batch_clip2s.append(v2_features_with_pos)
        batch_labels.append(row.label)

        # Save batch when full
        if len(batch_clip1s) >= args.batch_size:
            save_batch(output_dir, batch_count, batch_clip1s,batch_clip2s,batch_labels)
            batch_clip1s, batch_clip2s, batch_labels = [], [], []
            batch_count += 1

    # Save remaining samples
    if batch_clip1s:
        save_batch(output_dir, batch_count, batch_clip1s,batch_clip2s,batch_labels)

    logging.info(f"Generated {len(df)} samples in {batch_count + 1} batches in {output_dir}")

def save_batch(output_dir, batch_count, batch_clip1s, batch_clip2s, batch_labels):
    batch_clip_1_tensor = np.stack(batch_clip1s, axis=0)
    batch_clip_2_tensor = np.stack(batch_clip2s, axis=0)
    batch_label_tensor = np.array(batch_labels)
    np.savez(
            output_dir / f"batch_{batch_count:06d}.npz",
            clip1s=batch_clip_1_tensor,
            clip2s=batch_clip_2_tensor,
            labels=batch_label_tensor
        )
    logging.info(f"Saved batch {batch_count} to {output_dir}. Clip1 features shape = {batch_clip_1_tensor.shape} Clip2 features shape = {batch_clip_2_tensor.shape}. Labels shape = {batch_label_tensor.shape}")


if __name__ == "__main__":
    main()