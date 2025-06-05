import pandas as pd
import cv2
import numpy as np
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

def load_image_features_from_disk(track_id, rider_id, start_idx, end_idx, feature_base_path, clip_length):
    """Load precomputed ResNet50 features from disk for a frame range, handling multiple clips."""
    # Calculate expected number of frames
    F = end_idx - start_idx + 1
    if F <= 0:
        logging.error(f"Invalid frame range: start_idx {start_idx} > end_idx {end_idx}")
        return np.array([])

    # Determine which clips overlap with [start_idx, end_idx]
    feature_base_dir = Path(feature_base_path) / track_id / rider_id
    features = []
    current_idx = start_idx

    while current_idx <= end_idx:
        # Calculate clip boundaries
        clip_start = (current_idx // clip_length) * clip_length
        clip_end = min(clip_start + clip_length - 1, end_idx)
        feature_path = feature_base_dir / f"{clip_start:06d}_to_{clip_start + clip_length - 1:06d}_resnet50.npy"

        if not feature_path.exists():
            logging.error(f"Feature file {feature_path} does not exist for range {clip_start}:{clip_start + clip_length - 1}")
            return np.array([])

        try:
            clip_features = np.load(feature_path)
            if clip_features.shape[1] != 2048 or clip_features.shape[0] != clip_length:
                logging.error(f"Unexpected feature shape {clip_features.shape} in {feature_path}, expected ({clip_length}, 2048)")
                return np.array([])
        except Exception as e:
            logging.error(f"Error loading features from {feature_path}: {e}")
            return np.array([])

        # Extract relevant frames from this clip
        clip_relative_start = max(0, current_idx - clip_start)
        clip_relative_end = min(clip_length - 1, end_idx - clip_start)
        if clip_relative_start > clip_relative_end:
            logging.error(f"Invalid clip range {clip_relative_start}:{clip_relative_end} in {feature_path}")
            return np.array([])
        clip_features_subset = clip_features[clip_relative_start:clip_relative_end + 1]
        features.append(clip_features_subset)

        current_idx = clip_start + clip_relative_end + 2  # Move to next frame after clip_end

    # Concatenate features
    features = np.concatenate(features, axis=0)
    if features.shape[0] != F:
        logging.error(f"Loaded {features.shape[0]} frames, expected {F} for range {start_idx}:{end_idx}")
        return np.array([])

    return features

def get_clip_ending_at(video_path, end_idx, F, total_frames=None):
    """Extract F frames from video ending at end_idx, returning frames and indices."""
    # Validate starting frame
    start = max(0, end_idx - F + 1)
    if total_frames is not None and start >= total_frames:
        logging.warning(f"Start index {start} exceeds total frames {total_frames} in {video_path}, returning empty clip")
        return [], []

    clip_indices = list(range(start, end_idx + 1))
    if len(clip_indices) < F:
        logging.warning(f"Clip ending at {end_idx} in {video_path} has {len(clip_indices)} frames, padding to {F}")
        clip_indices += [clip_indices[-1]] * (F - len(clip_indices))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Cannot open video {video_path}")
        return [], []

    frames = []
    for idx in clip_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            logging.warning(f"Cannot read frame {idx} from {video_path}, padding with zeros")
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()

    return frames, clip_indices

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
        v1_path = Path("downloaded_videos") / row.track_id / row.v1_rider_id / f"{row.track_id}_{row.v1_rider_id}.mp4"
        v2_path = Path("downloaded_videos") / row.track_id / row.v2_rider_id / f"{row.track_id}_{row.v2_rider_id}.mp4"

        # Get total frames for validation
        cap1 = cv2.VideoCapture(str(v1_path))
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) if cap1.isOpened() else 0
        cap1.release()
        cap2 = cv2.VideoCapture(str(v2_path))
        total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) if cap2.isOpened() else 0
        cap2.release()

        # Extract frames and indices
        v1_frames, v1_indices = get_clip_ending_at(v1_path, row.v1_frame_idx, args.F, total_frames1)
        v2_frames, v2_indices = get_clip_ending_at(v2_path, row.v2_frame_idx, args.F, total_frames2)

        if len(v1_frames) != args.F or len(v2_frames) != args.F:
            logging.error(f"Failed to extract {args.F} frames for sample {row.Index}, skipping")
            continue

        # Load precomputed features
        v1_start_idx = v1_indices[0]
        v1_end_idx = v1_indices[-1]
        v1_features = load_image_features_from_disk(row.track_id, row.v1_rider_id, v1_start_idx, v1_end_idx, args.image_feature_path, args.F)
        if v1_features.size == 0 or v1_features.shape[0] != args.F:
            logging.error(f"Failed to load features for v1 clip {v1_start_idx}:{v1_end_idx} in {row.Index}, skipping")
            continue

        v2_start_idx = v2_indices[0]
        v2_end_idx = v2_indices[-1]
        v2_features = load_image_features_from_disk(row.track_id, row.v2_rider_id, v2_start_idx, v2_end_idx, args.image_feature_path, args.F)
        if v2_features.size == 0 or v2_features.shape[0] != args.F:
            logging.error(f"Failed to load features for v2 clip {v2_start_idx}:{v2_end_idx} in {row.Index}, skipping")
            continue

        # Append absolute frame indices as 2049th feature
        v1_features_with_pos = np.concatenate([v1_features, np.array(v1_indices, dtype=np.float32)[:, None]], axis=1)  # Shape: (F, 2049)
        v2_features_with_pos = np.concatenate([v2_features, np.array(v2_indices, dtype=np.float32)[:, None]], axis=1)  # Shape: (F, 2049)

        # Add to batch
        batch_clip1s.append(v1_features_with_pos)
        batch_clip2s.append(v2_features_with_pos)
        batch_labels.append(row.label)

        # Save batch when full
        if len(batch_clip1s) >= args.batch_size:
            np.savez(
                output_dir / f"batch_{batch_count:06d}.npz",
                clip1s=np.stack(batch_clip1s, axis=0),
                clip2s=np.stack(batch_clip2s, axis=0),
                labels=np.array(batch_labels)
            )
            logging.info(f"Saved batch {batch_count} with {len(batch_clip1s)} samples to {output_dir}")
            batch_clip1s, batch_clip2s, batch_labels = [], [], []
            batch_count += 1

    # Save remaining samples
    if batch_clip1s:
        np.savez(
            output_dir / f"batch_{batch_count:06d}.npz",
            clip1s=np.stack(batch_clip1s, axis=0),
            clip2s=np.stack(batch_clip2s, axis=0),
            labels=np.array(batch_labels)
        )
        logging.info(f"Saved final batch {batch_count} with {len(batch_clip1s)} samples to {output_dir}")

    logging.info(f"Generated {len(df)} samples in {batch_count + 1} batches in {output_dir}")

if __name__ == "__main__":
    main()