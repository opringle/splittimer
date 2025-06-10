import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
from utils import pad_features_to_length, setup_logging, parse_clip_range, load_image_features_from_disk, get_clip_indices_ending_at, save_batch

def main():
    parser = argparse.ArgumentParser(description="Generate batched training data using precomputed features.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file with 'set' column ('train' or 'val')")
    parser.add_argument("image_feature_path", type=str, help="Path to directory of clip features")
    parser.add_argument("output_dir", type=str, help="Directory to save .npz files")
    parser.add_argument("--F", type=int, default=50, help="Number of frames per clip (for individual features)")
    parser.add_argument("--sequence_length", type=int, default=10, help="Sequence length for sequence features")
    parser.add_argument("--feature_type", type=str, choices=['individual', 'sequence'], required=True, help="Type of features to use ('individual' or 'sequence')")
    parser.add_argument("--batch_size", type=int, default=32, help="Samples per .npz file")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    df = pd.read_csv(args.csv_path)
    if 'set' not in df.columns:
        logging.error("CSV file must contain a 'set' column")
        exit(1)
    
    df = df.sample(frac=1, random_state=None, ignore_index=True)
    num_train = (df['set'] == 'train').sum()
    num_val = (df['set'] == 'val').sum()
    logging.info(f"Loaded metadata for {num_train} training and {num_val} validation samples")

    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_batch_clip1s, train_batch_clip2s, train_batch_labels = [], [], []
    val_batch_clip1s, val_batch_clip2s, val_batch_labels = [], [], []
    train_batch_count = 0
    val_batch_count = 0

    for row in tqdm(df.itertuples(), total=len(df), desc="Generating samples"):
        if args.feature_type == 'individual':
            # Load individual frame features for a sequence of frames
            v1_indices = get_clip_indices_ending_at(row.v1_frame_idx, args.F)
            v2_indices = get_clip_indices_ending_at(row.v2_frame_idx, args.F)

            v1_start_idx, v1_end_idx = v1_indices[0], v1_indices[-1]
            v2_start_idx, v2_end_idx = v2_indices[0], v2_indices[-1]

            v1_features = load_image_features_from_disk(row.track_id, row.v1_rider_id, v1_start_idx, v1_end_idx, args.image_feature_path, feature_type='individual')
            if v1_features.size == 0 or v1_features.shape[0] != (v1_end_idx - v1_start_idx + 1):
                logging.error(f"Failed to load individual features for v1 clip {v1_start_idx}:{v1_end_idx}")
                continue

            v1_features_padded = pad_features_to_length(v1_features, v1_indices, args.F)

            v2_features = load_image_features_from_disk(row.track_id, row.v2_rider_id, v2_start_idx, v2_end_idx, args.image_feature_path, feature_type='individual')
            if v2_features.size == 0 or v2_features.shape[0] != (v2_end_idx - v2_start_idx + 1):
                logging.error(f"Failed to load individual features for v2 clip {v2_start_idx}:{v2_end_idx}")
                continue

            v2_features_padded = pad_features_to_length(v2_features, v2_indices, args.F)

            v1_features_with_pos = np.concatenate([v1_features_padded, np.array(v1_indices, dtype=np.float32)[:, None]], axis=1)
            v2_features_with_pos = np.concatenate([v2_features_padded, np.array(v2_indices, dtype=np.float32)[:, None]], axis=1)

        elif args.feature_type == 'sequence':
            # Load sequence features for the frame ending at the specified index
            v1_features = load_image_features_from_disk(row.track_id, row.v1_rider_id, row.v1_frame_idx, row.v1_frame_idx, args.image_feature_path, feature_type='sequence', sequence_length=args.sequence_length)
            if v1_features.size == 0:
                logging.error(f"Failed to load sequence features for v1 frame {row.v1_frame_idx}")
                continue

            v2_features = load_image_features_from_disk(row.track_id, row.v2_rider_id, row.v2_frame_idx, row.v2_frame_idx, args.image_feature_path, feature_type='sequence', sequence_length=args.sequence_length)
            if v2_features.size == 0:
                logging.error(f"Failed to load sequence features for v2 frame {row.v2_frame_idx}")
                continue

            # For sequence features, no padding is needed as it's a single feature vector
            v1_features_with_pos = np.concatenate([v1_features, np.array([row.v1_frame_idx], dtype=np.float32)[:, None]], axis=1)
            v2_features_with_pos = np.concatenate([v2_features, np.array([row.v2_frame_idx], dtype=np.float32)[:, None]], axis=1)

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
            logging.warning(f"Unknown set value: {row.set}")

    if train_batch_clip1s:
        save_batch(train_dir, train_batch_count, train_batch_clip1s, train_batch_clip2s, train_batch_labels)
        train_batch_count += 1
    if val_batch_clip1s:
        save_batch(val_dir, val_batch_count, val_batch_clip1s, val_batch_clip2s, val_batch_labels)
        val_batch_count += 1

    logging.info(f"Generated {num_train} training samples in {train_batch_count} batches")
    logging.info(f"Generated {num_val} validation samples in {val_batch_count} batches")

if __name__ == "__main__":
    main()