import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
from utils import setup_logging, load_image_features_from_disk, get_clip_indices_ending_at, save_batch, get_video_fps_and_total_frames

def compute_and_cache_video_features(video_path: str, feature_cache_dict):
    _, total_frames = get_video_fps_and_total_frames(video_path)
    frame_idx_array = np.arange(total_frames, dtype=np.float32)
    percent_through_clip_array = frame_idx_array / total_frames
    feature_cache_dict[video_path] = {"frame_idx_array": frame_idx_array, 'percent_through_clip_array': percent_through_clip_array}

def add_frame_index_feature(video_path: str, feature_cache_dict, start_frame_idx: int, end_frame_idx: int, features):
    clip_indices = feature_cache_dict[video_path]["frame_idx_array"][start_frame_idx:end_frame_idx+1]
    return np.concatenate([features, clip_indices[:, None]], axis=1)

def add_percent_through_video_feature(video_path: str, feature_cache_dict, start_frame_idx: int, end_frame_idx: int, features):
    percent_through_video = feature_cache_dict[video_path]["percent_through_clip_array"][start_frame_idx:end_frame_idx+1]
    return np.concatenate([features, percent_through_video[:, None]], axis=1)

def main():
    parser = argparse.ArgumentParser(description="Generate batched training data using precomputed features.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file with 'set' column ('train' or 'val')")
    parser.add_argument("image_feature_path", type=str, help="Path to directory of clip features")
    parser.add_argument("output_dir", type=str, help="Directory to save .npz files")
    parser.add_argument("--F", type=int, default=50, help="Number of frames per clip (for individual features)")
    parser.add_argument("--add_position_feature", action='store_true', help="Add a feature to each sample for end index")
    parser.add_argument("--add_percent_completion_feature", action='store_true', help="Add a feature to each sample for % completion")
    parser.add_argument("--batch_size", type=int, default=32, help="Samples per .npz file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible track splitting")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    df = pd.read_csv(args.csv_path)
    if 'set' not in df.columns:
        logging.error("CSV file must contain a 'set' column")
        exit(1)
    
    df = df.sample(frac=1, random_state=args.seed, ignore_index=True)
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

    video_feature_cache = {}

    for row in tqdm(df.itertuples(), total=len(df), desc="Generating samples"):
        video_path_v1 = Path("downloaded_videos") / row.track_id / row.v1_rider_id / f"{row.track_id}_{row.v1_rider_id}.mp4"
        video_path_v2 = Path("downloaded_videos") / row.track_id / row.v2_rider_id / f"{row.track_id}_{row.v2_rider_id}.mp4"

        if video_path_v1 not in video_feature_cache:
            compute_and_cache_video_features(video_path_v1, video_feature_cache)

        if video_path_v2 not in video_feature_cache:
            compute_and_cache_video_features(video_path_v2, video_feature_cache)

        # Load individual frame features for a sequence of frames
        v1_indices = get_clip_indices_ending_at(row.v1_frame_idx, args.F)
        v2_indices = get_clip_indices_ending_at(row.v2_frame_idx, args.F)

        v1_start_idx, v1_end_idx = v1_indices[0], v1_indices[-1]
        v2_start_idx, v2_end_idx = v2_indices[0], v2_indices[-1]

        v1_features = load_image_features_from_disk(row.track_id, row.v1_rider_id, v1_start_idx, v1_end_idx, args.image_feature_path)
        if v1_features.size == 0 or v1_features.shape[0] != (v1_end_idx - v1_start_idx + 1):
            logging.error(f"Failed to load individual features for v1 clip {v1_start_idx}:{v1_end_idx}")
            continue

        v2_features = load_image_features_from_disk(row.track_id, row.v2_rider_id, v2_start_idx, v2_end_idx, args.image_feature_path)
        if v2_features.size == 0 or v2_features.shape[0] != (v2_end_idx - v2_start_idx + 1):
            logging.error(f"Failed to load individual features for v2 clip {v2_start_idx}:{v2_end_idx}")
            continue       

        if args.add_position_feature:
            v1_features = add_frame_index_feature(video_path_v1, video_feature_cache, start_frame_idx=v1_start_idx, end_frame_idx=v1_end_idx, features=v1_features)
            v2_features = add_frame_index_feature(video_path_v2, video_feature_cache, start_frame_idx=v2_start_idx, end_frame_idx=v2_end_idx, features=v2_features)

        if args.add_percent_completion_feature:
            v1_features = add_percent_through_video_feature(video_path_v1, video_feature_cache, start_frame_idx=v1_start_idx, end_frame_idx=v1_end_idx, features=v1_features)
            v2_features = add_percent_through_video_feature(video_path_v2, video_feature_cache, start_frame_idx=v2_start_idx, end_frame_idx=v2_end_idx, features=v2_features)

        if row.set == 'train':
            train_batch_clip1s.append(v1_features)
            train_batch_clip2s.append(v2_features)
            train_batch_labels.append(row.label)
            if len(train_batch_clip1s) >= args.batch_size:
                save_batch(train_dir, train_batch_count, train_batch_clip1s, train_batch_clip2s, train_batch_labels)
                train_batch_clip1s, train_batch_clip2s, train_batch_labels = [], [], []
                train_batch_count += 1
        elif row.set == 'val':
            val_batch_clip1s.append(v1_features)
            val_batch_clip2s.append(v2_features)
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