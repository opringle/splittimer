import numpy as np
import os
import argparse
from pathlib import Path
import random
import itertools
import logging
import pandas as pd

def get_clip_metadata(video_dir):
    """
    Extract frame indices, labels, riderId, and trackId from clip files without loading video data.
    Returns lists of frame indices, labels, and metadata.
    """
    labels = []
    frame_indices = []
    clip_ranges = []
    
    # Extract riderId and trackId from the directory path
    video_dir = Path(video_dir)
    parts = video_dir.parts
    if len(parts) < 2:
        raise ValueError(f"Directory path {video_dir} does not match expected structure <base_dir>/<trackId>/<riderId>")
    rider_id = parts[-1]  # e.g., "amaury_pierron"
    track_id = parts[-2]  # e.g., "loudenvielle_2025"
    
    # Find all clip files
    clip_files = sorted(video_dir.glob("*_x.npy"))
    if not clip_files:
        raise ValueError(f"No *_x.npy files found in directory: {video_dir}")
    
    # Process each clip file
    for npy_file in clip_files:
        # Parse frame range from filename (e.g., "000000_to_000124_x.npy")
        filename = npy_file.stem  # e.g., "000000_to_000124_x"
        frame_range = filename.replace('_x', '').split('_to_')
        assert len(frame_range) == 2, f"Invalid clip filename format {npy_file}."

        start_frame, end_frame = map(int, frame_range)  # e.g., 0, 124
        frames_in_clip = list(range(start_frame, end_frame + 1))
        
        # Load corresponding label file
        label_file = npy_file.with_name(filename.replace('_x', '_y') + '.npy')
        assert label_file.exists(), "no label file exists for clip"
        clip_labels = np.load(label_file)
        assert len(clip_labels) == len(frames_in_clip), f"Label length {len(clip_labels)} does not match frame count {len(frames_in_clip)} in {label_file}"
        labels.extend(clip_labels)
        
        frame_indices.extend(frames_in_clip)
        clip_ranges.append((start_frame, end_frame))
    
    frame_indices = np.array(frame_indices)
    labels = np.array(labels)
    logging.info(f"Extracted metadata for {len(frame_indices)} frames from {video_dir}")
    
    return frame_indices, labels, rider_id, track_id, clip_ranges

def generate_training_samples(v1_indices, v1_labels, v1_rider_id, v1_track_id,
                              v2_indices, v2_labels, v2_rider_id, v2_track_id,
                              max_negatives_per_positive=10, num_augmented_positives_per_segment=5):
    """
    Generate training samples with frame indices, riderId, and trackId from both videos.
    Returns metadata for samples without loading video data.
    """
    # Find split points
    v1_split_indices = v1_indices[v1_labels == 1.0]
    v2_split_indices = v2_indices[v2_labels == 1.0]
    logging.info(f"Found {len(v1_split_indices)} splits")
    
    assert len(v1_split_indices) != 0, "No split points found in v1_split_indices"
    assert len(v2_split_indices) != 0, "No split points found in v2_split_indices"
    assert len(v1_split_indices) == len(v2_split_indices), f"len(v1_split_indices) {len(v1_split_indices)} not equal to len(v2_split_indices) {len(v2_split_indices)}"
    
    v2_total_frames = v2_indices[-1] + 1 if len(v2_indices) > 0 else 0
    
    sample_labels = []
    sample_indices = []  # (v1_frame_idx, v2_frame_idx)
    sample_metadata = []  # List of dicts with riderId and trackId
    
    # Generate samples at split points
    for pos, v1_split_idx in enumerate(v1_split_indices):
        v2_split_idx = v2_split_indices[pos]
        
        # Positive samples
        sample_labels.append(1.0)
        sample_indices.append((v1_split_idx, v2_split_idx))
        sample_metadata.append({
            'v1_rider_id': v1_rider_id,
            'v1_track_id': v1_track_id,
            'v2_rider_id': v2_rider_id,
            'v2_track_id': v2_track_id,
        })
        
        # Negative samples
        neg_count = 0
        attempts = 0
        max_attempts = 50
        while neg_count < max_negatives_per_positive and attempts < max_attempts:
            # Choose a random frame in v2
            v2_idx = random.randint(0, v2_total_frames - 1)
            if v2_idx in v2_split_indices and v2_split_indices.index(v2_idx) == pos:
                # If the v2 split was selected, skip this iteration
                attempts += 1
                continue
            sample_labels.append(0.0)
            sample_indices.append((v1_split_idx, v2_idx))
            sample_metadata.append({
                'v1_rider_id': v1_rider_id,
                'v1_track_id': v1_track_id,
                'v2_rider_id': v2_rider_id,
                'v2_track_id': v2_track_id,
            })
            neg_count += 1
            attempts += 1
    
    # Augmented positive samples for segments between splits
    num_segments = len(v1_split_indices) - 1
    for seg in range(num_segments):
        v1_start_seg = v1_split_indices[seg]
        v1_end_seg = v1_split_indices[seg + 1]
        v2_start_seg = v2_split_indices[seg]
        v2_end_seg = v2_split_indices[seg + 1]
        
        v1_seg_length = v1_end_seg - v1_start_seg
        v2_seg_length = v2_end_seg - v2_start_seg
        
        # Select frames where full clips can be contained within the segment
        possible_idx1 = list(range(v1_start_seg, v1_end_seg + 1))
        
        # Generate relative positions with beta distribution
        num_samples = min(num_augmented_positives_per_segment, len(possible_idx1))
        rng = np.random.default_rng()
        relative_positions = rng.beta(a=0.5, b=0.5, size=num_samples)
        logging.debug(f"relative positions = {relative_positions}")
        
        # Map to frame indices
        min_idx = possible_idx1[0]
        max_idx = possible_idx1[-1]
        selected_idx1 = [int(min_idx + p * (max_idx - min_idx)) for p in relative_positions]
        selected_idx1 = [min(max(idx, min_idx), max_idx) for idx in selected_idx1]
        logging.debug(f"selected_idx1={selected_idx1}")
        
        for idx1 in selected_idx1:
            # Compute relative position in video 1 segment
            fraction_through_v1_segment = (idx1 - v1_start_seg) / v1_seg_length
            # Map to corresponding frame in video 2 segment
            idx2_float = v2_start_seg + fraction_through_v1_segment * v2_seg_length
            idx2 = int(round(idx2_float))
            if idx2 < v2_start_seg or idx2 > v2_end_seg:
                raise Exception(f"idx2 {idx2} is out of range [{v2_start_seg}, {v2_end_seg}]")
            
            sample_labels.append(1.0)
            sample_indices.append((idx1, idx2))
            sample_metadata.append({
                'v1_rider_id': v1_rider_id,
                'v1_track_id': v1_track_id,
                'v2_rider_id': v2_rider_id,
                'v2_track_id': v2_track_id,
            })
            
            # Generate negative samples for augmented positive sample
            neg_count = 0
            attempts = 0
            while neg_count < max_negatives_per_positive and attempts < max_attempts:
                random_idx2 = random.randint(0, v2_total_frames - 1)
                sample_labels.append(0.0)
                sample_indices.append((idx1, random_idx2))
                sample_metadata.append({
                    'v1_rider_id': v1_rider_id,
                    'v1_track_id': v1_track_id,
                    'v2_rider_id': v2_rider_id,
                    'v2_track_id': v2_track_id,
                })
                neg_count += 1
                attempts += 1
    
    return np.array(sample_labels), np.array(sample_indices), sample_metadata

def main():
    parser = argparse.ArgumentParser(description="Generate training metadata for all rider combinations on each track.")
    parser.add_argument("clips_dir", type=str, help="Base directory containing processed clips (processed_clips/<trackId>/<riderId>)")
    parser.add_argument("--clip_length", type=int, default=125, help="Number of frames per clip")
    parser.add_argument("--max_negatives_per_positive", type=int, default=10, help="Max negative samples per split point")
    parser.add_argument("--num_augmented_positives_per_segment", type=int, default=50, help="Number of augmented samples per segment")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()
    
    # Configure logging based on the provided log level
    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    clips_base_dir = Path(args.clips_dir)
    
    # Find all track directories
    track_dirs = [d for d in clips_base_dir.iterdir() if d.is_dir()]
    if not track_dirs:
        logging.error(f"No track directories found in {clips_base_dir}")
        return
    
    # Process each track
    dfs = []
    for track_dir in track_dirs:
        track_id = track_dir.name
        logging.info(f"Processing track: {track_id}")
        
        # Find all rider directories for this track
        rider_dirs = sorted([d for d in track_dir.iterdir() if d.is_dir()])
        if len(rider_dirs) < 2:
            logging.warning(f"Need at least two riders for track {track_id}, found {len(rider_dirs)}, skipping.")
            continue
        
        # Generate all possible ordered pairs of riders (rider1, rider2) and (rider2, rider1)
        rider_pairs = list(itertools.permutations(rider_dirs, 2))
        logging.info(f"Found {len(rider_dirs)} riders, generating samples for {len(rider_pairs)} rider pairs")
        
        # Process each rider pair
        for rider_dir1, rider_dir2 in rider_pairs:
            # Extract metadata for both riders
            logging.info(f"Extracting metadata for rider pair: {rider_dir1.name} and {rider_dir2.name}")
            v1_indices, v1_labels, v1_rider_id, v1_track_id, v1_clip_ranges = get_clip_metadata(rider_dir1)
            v2_indices, v2_labels, v2_rider_id, v2_track_id, v2_clip_ranges = get_clip_metadata(rider_dir2)
            
            # Generate training samples
            logging.info(f"Generating training samples for {v1_rider_id} and {v2_rider_id}")
            sample_labels, sample_indices, sample_metadata = generate_training_samples(
                v1_indices, v1_labels, v1_rider_id, v1_track_id,
                v2_indices, v2_labels, v2_rider_id, v2_track_id,
                max_negatives_per_positive=args.max_negatives_per_positive,
                num_augmented_positives_per_segment=args.num_augmented_positives_per_segment
            )
            
            # Check if sample generation failed
            if sample_labels is None:
                logging.warning(f"Failed to generate samples for pair {v1_rider_id} and {v2_rider_id}, skipping.")
                continue
            
            # Create DataFrame from sample data
            data = {
                'track_id': [meta['v1_track_id'] for meta in sample_metadata],
                'v1_rider_id': [meta['v1_rider_id'] for meta in sample_metadata],
                'v2_rider_id': [meta['v2_rider_id'] for meta in sample_metadata],
                'v1_frame_idx': [idx[0] for idx in sample_indices],
                'v2_frame_idx': [idx[1] for idx in sample_indices],
                'label': sample_labels,
            }
            df = pd.DataFrame(data)
            dfs.append(df)
            
            logging.info(f"Generated {len(sample_labels)} samples for pair {v1_rider_id} and {v2_rider_id}: "
                            f"{np.sum(sample_labels == 1.0)} positive, {np.sum(sample_labels == 0.0)} negative")

    df = pd.concat(dfs, axis=0)
    training_data_output_path = "training_data"
    os.makedirs(training_data_output_path, exist_ok=True)
    output_filename = f"training_metadata.csv"
    training_data_file_path = os.path.join(training_data_output_path, output_filename)
    df.to_csv(training_data_file_path, index=False)
    logging.info(f"Saved training metadata to {training_data_file_path}")
    

if __name__ == "__main__":
    main()