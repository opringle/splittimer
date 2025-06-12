import torch
import numpy as np
import argparse
from pathlib import Path
import logging
import json
from tqdm import tqdm
import yaml
from utils import PositionClassifier, frame_idx_to_timecode, get_default_device_name, get_video_fps_and_total_frames, setup_logging, load_image_features_from_disk, get_clip_indices_ending_at, timecode_to_frames, add_features_to_clip

def load_source_splits(config_path, track_id, source_rider_id):
    """
    Load source splits from the video_config.yaml file for the given track and rider.

    Args:
        config_path (str): Path to the video_config.yaml file.
        track_id (str): Track identifier (e.g., 'loudenvielle_2025').
        source_rider_id (str): Source rider identifier (e.g., 'amaury_pierron').

    Returns:
        list: List of split timestamps in MM:SS:FF format.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
    parser.add_argument('config_path', type=str, help='Path to video_config.yaml file')
    parser.add_argument('image_feature_path', type=str, help="Path to directory of clip features")
    parser.add_argument('output_file', type=str, help="Path to output file")
    parser.add_argument('--trackId', type=str, required=True, help='Track identifier')
    parser.add_argument('--sourceRiderId', type=str, required=True, help='Source rider identifier')
    parser.add_argument('--targetRiderId', type=str, required=True, help='Target rider identifier')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--device', type=str, default=get_default_device_name(), help='Device to use (cuda or cpu)')
    parser.add_argument('--F', type=int, default=50, help='Number of frames per clip')
    parser.add_argument('--stride', type=int, default=1, help='Stride for generating candidate split points in the target video')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for considering a target clip as a split')
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument('--add_position_feature', action='store_true', default=False, help='Add position feature to clips')
    parser.add_argument('--add_percent_completion_feature', action='store_true', default=False, help='Add percent completion feature to clips')
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Set up directories
    source_video_dir = Path('downloaded_videos') / args.trackId / args.sourceRiderId
    target_video_dir = Path('downloaded_videos') / args.trackId / args.targetRiderId

    # Load model
    model, _, _ = PositionClassifier.load(args.checkpoint_path, args.device)
    model.eval()
    logging.info(f"Loaded model from {args.checkpoint_path}")

    # Load source split timestamps from YAML
    source_timestamps = load_source_splits(args.config_path, args.trackId, args.sourceRiderId)
    logging.info(f"Loaded {len(source_timestamps)} source splits from {args.config_path} for track {args.trackId} and rider {args.sourceRiderId}")

    # Convert timestamps to frame indices in the source video
    source_video_path = Path(source_video_dir) / f"{args.trackId}_{args.sourceRiderId}.mp4"
    source_fps, source_total_frames = get_video_fps_and_total_frames(source_video_path)
    source_end_indices = [timecode_to_frames(ts, source_fps) for idx, ts in enumerate(source_timestamps) if idx != 0]
    logging.debug(f"source_end_indices={source_end_indices}")

    # Determine total frames in target video
    target_video_path = Path(target_video_dir) / f"{args.trackId}_{args.targetRiderId}.mp4"
    target_fps, target_total_frames = get_video_fps_and_total_frames(target_video_path)
    logging.info(f"Target video has {target_total_frames} frames")

    # Generate candidate end indices
    candidate_end_indices = list(range(args.F - 1, target_total_frames, args.stride))
    logging.info(f"Generated {len(candidate_end_indices)} candidate split points with stride {args.stride}")

    # Generate source split clips
    source_samples = {}
    for source_end_idx in source_end_indices:
        indices = get_clip_indices_ending_at(source_end_idx, args.F)
        start_idx = indices[0]
        features = load_image_features_from_disk(args.trackId, args.sourceRiderId, start_idx, source_end_idx, args.image_feature_path)
        assert features.size > 0, f"Failed to load features for source split at {source_end_idx}"
        features_with_extras = add_features_to_clip(
            features, indices, total_frames=source_total_frames,
            add_position=args.add_position_feature,
            add_percent_completion=args.add_percent_completion_feature
        )
        source_samples[source_end_idx] = torch.from_numpy(features_with_extras).unsqueeze(0).to(args.device)
    logging.info(f"Generated {len(source_samples)} source split clips")

    if not source_samples:
        logging.error("No source split clips generated")
        exit(1)

    # Collect candidate pairs (target_end_idx, source_end_idx, score) where score > threshold
    candidates = []
    for end_idx in tqdm(candidate_end_indices, desc="Processing target frames"):
        indices = get_clip_indices_ending_at(end_idx, args.F)
        start_idx = indices[0]
        features = load_image_features_from_disk(args.trackId, args.targetRiderId, start_idx, end_idx, args.image_feature_path)
        assert features.size > 0, f"Failed to load features for target clip ending at {end_idx}"
        features_with_extras = add_features_to_clip(
            features, indices, total_frames=target_total_frames,
            add_position=args.add_position_feature,
            add_percent_completion=args.add_percent_completion_feature
        )
        target_clip = torch.from_numpy(features_with_extras).unsqueeze(0).to(args.device)

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
                'source_end_idx': source_end_idx,
                'source_timecode': frame_idx_to_timecode(source_end_idx, source_fps),
                'target_end_idx': target_end_idx,
                'target_timecode': frame_idx_to_timecode(target_end_idx, target_fps),
                'confidence': score,
            })
            assigned_target_indices.add(target_end_idx)
            assigned_source_indices.add(source_end_idx)

    # Sort splits by source_end_idx to maintain order
    predicted_splits.sort(key=lambda x: x['source_end_idx'])

    # Create output data with metadata
    output_data = {
        "trackId": args.trackId,
        "sourceRiderId": args.sourceRiderId,
        "targetRiderId": args.targetRiderId,
        "predicted_splits": predicted_splits
    }

    # Save output data to JSON
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    logging.info(f"Saved {len(predicted_splits)} predicted splits to {args.output_file}")

if __name__ == "__main__":
    main()