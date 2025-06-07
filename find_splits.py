import torch
import numpy as np
import argparse
from pathlib import Path
import logging
import json
from tqdm import tqdm
import yaml
from utils import PositionClassifier, pad_features_to_length, setup_logging, load_image_features_from_disk, get_clip_indices_ending_at, parse_clip_range

def parse_timestamp(timestamp):
    """Parse a timestamp in MM:SS:FF format into minutes, seconds, and frames."""
    parts = timestamp.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {timestamp}. Expected MM:SS:FF")
    minutes = int(parts[0])
    seconds = int(parts[1])
    frames = int(parts[2])
    return minutes, seconds, frames

def timestamp_to_frame(timestamp, frame_rate):
    """Convert a timestamp in MM:SS:FF format to a frame index."""
    minutes, seconds, frames = parse_timestamp(timestamp)
    total_frames = (minutes * 60 + seconds) * frame_rate + frames
    return int(total_frames)

def load_source_splits(config_path, track_id, source_rider_id):
    """
    Load source splits from the video_config.yaml file for the given track and rider.

    Args:
        config_path (str): Path to the video_config.yaml file.
        track_id (str): Track identifier (e.g., 'loudenvielle_2025').
        source_rider_id (str): Source rider identifier (e.g., 'amaury_pierron').

    Returns:
        list: List of split timestamps in MM:SS:FF format.

    Exits:
        If no matching video is found, logs an error and exits with code 1.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find videos matching the track_id and source_rider_id
    matching_videos = [
        video for video in config.get('videos', [])
        if video.get('trackId') == track_id and video.get('riderId') == source_rider_id
    ]
    
    if not matching_videos:
        logging.error(f"No configuration found for track {track_id} and rider {source_rider_id} in {config_path}")
        exit(1)
    
    if len(matching_videos) > 1:
        logging.warning(f"Multiple videos found for track {track_id} and rider {source_rider_id}. Using the first one.")
    
    video = matching_videos[0]
    splits = video.get('splits', [])
    
    if not splits:
        logging.warning(f"No splits found for track {track_id} and rider {source_rider_id}")
    
    return splits

def main():
    parser = argparse.ArgumentParser(description="Find corresponding splits in target video based on source video splits.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to video_config.yaml file')
    parser.add_argument('--feature_base_path', type=str, required=True, help='Base directory containing trackId/riderId/clip_files')
    parser.add_argument('--trackId', type=str, required=True, help='Track identifier')
    parser.add_argument('--sourceRiderId', type=str, required=True, help='Source rider identifier')
    parser.add_argument('--targetRiderId', type=str, required=True, help='Target rider identifier')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--output_file', type=str, default='predicted_splits.json', help='Output JSON file for predicted splits')
    parser.add_argument('--F', type=int, default=50, help='Number of frames per clip')
    parser.add_argument('--frame_rate', type=float, required=True, help='Frame rate of the videos (frames per second)')
    parser.add_argument('--stride', type=int, default=1, help='Stride for generating candidate split points in the target video')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for considering a target clip as a split')
    args = parser.parse_args()

    setup_logging()

    # Set up directories
    source_video_dir = Path(args.feature_base_path) / args.trackId / args.sourceRiderId
    target_video_dir = Path(args.feature_base_path) / args.trackId / args.targetRiderId

    # Load model
    model, _, _ = PositionClassifier.load(args.checkpoint_path, args.device)
    model.eval()
    logging.info(f"Loaded model from {args.checkpoint_path}")

    # Load source split timestamps from YAML
    source_timestamps = load_source_splits(args.config_path, args.trackId, args.sourceRiderId)
    logging.info(f"Loaded {len(source_timestamps)} source splits from {args.config_path} for track {args.trackId} and rider {args.sourceRiderId}")

    # Convert timestamps to frame indices
    source_end_indices = [timestamp_to_frame(ts, args.frame_rate) for ts in source_timestamps]

    # Determine total frames in target video
    target_npy_files = list(target_video_dir.glob("*.npy"))
    if not target_npy_files:
        logging.error(f"No .npy files found in {target_video_dir}")
        exit(1)
    max_end_idx = max([parse_clip_range(file.name)[1] for file in target_npy_files])
    total_frames = max_end_idx + 1
    logging.info(f"Target video has {total_frames} frames")

    # Generate candidate end indices
    candidate_end_indices = list(range(args.F - 1, total_frames, args.stride))
    logging.info(f"Generated {len(candidate_end_indices)} candidate split points with stride {args.stride}")

    # Generate source split clips
    source_samples = {}
    for source_end_idx in source_end_indices:
        indices = get_clip_indices_ending_at(source_end_idx, args.F)
        start_idx = indices[0]
        features = load_image_features_from_disk(args.trackId, args.sourceRiderId, start_idx, source_end_idx, args.feature_base_path)
        if features.size == 0:
            logging.warning(f"Failed to load features for source split at {source_end_idx}")
            continue
        # Pad features to match F
        padded_features = pad_features_to_length(features, indices, args.F)
        features_with_pos = np.concatenate([padded_features, np.array(indices, dtype=np.float32)[:, None]], axis=1)
        source_samples[source_end_idx] = torch.from_numpy(features_with_pos).unsqueeze(0).to(args.device)
    logging.info(f"Generated {len(source_samples)} source split clips")

    if not source_samples:
        logging.error("No source split clips generated")
        exit(1)

    # Collect candidate pairs (target_end_idx, source_end_idx, score) where score > threshold
    candidates = []
    for end_idx in tqdm(candidate_end_indices, desc="Processing target frames"):
        indices = get_clip_indices_ending_at(end_idx, args.F)
        start_idx = indices[0]
        features = load_image_features_from_disk(args.trackId, args.targetRiderId, start_idx, end_idx, args.feature_base_path)
        if features.size == 0:
            continue
        # Pad features to match F
        padded_features = pad_features_to_length(features, indices, args.F)
        features_with_pos = np.concatenate([padded_features, np.array(indices, dtype=np.float32)[:, None]], axis=1)
        target_clip = torch.from_numpy(features_with_pos).unsqueeze(0).to(args.device)

        for source_end_idx, source_clip in source_samples.items():
            with torch.no_grad():
                output = model(source_clip, target_clip)
                score = torch.sigmoid(output).item()
            if score > args.threshold:
                candidates.append((end_idx, source_end_idx, score))

    # Sort candidates by score in descending order
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Assign best matches ensuring one prediction per source split and target frame
    assigned_target_indices = set()
    assigned_source_indices = set()
    predicted_splits = []
    for target_end_idx, source_end_idx, score in candidates:
        if target_end_idx not in assigned_target_indices and source_end_idx not in assigned_source_indices:
            predicted_splits.append({
                'target_end_idx': target_end_idx,
                'confidence': score,
                'source_end_idx': source_end_idx
            })
            assigned_target_indices.add(target_end_idx)
            assigned_source_indices.add(source_end_idx)

    # Sort splits by source_end_idx to maintain order
    predicted_splits.sort(key=lambda x: x['source_end_idx'])

    # Save predicted splits
    with open(args.output_file, 'w') as f:
        json.dump(predicted_splits, f, indent=4)
    logging.info(f"Saved {len(predicted_splits)} predicted splits to {args.output_file}")

if __name__ == "__main__":
    main()