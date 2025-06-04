import numpy as np
import os
import argparse
from pathlib import Path
import random

def load_frames_and_labels(video_dir):
    """Load precomputed ResNet50 features, original frames, and corresponding labels from .npy files."""
    features = []
    frames = []
    labels = []
    frame_indices = []
    for npy_file in sorted(Path(video_dir).glob("*_x.npy")):
        # Load original frame clip
        clip_frames = np.load(npy_file)
        frames.extend(clip_frames)
        
        # Load corresponding ResNet50 feature file
        feature_file = npy_file.with_name(npy_file.stem + '_resnet50.npy')
        if not feature_file.exists():
            print(f"Warning: Feature file {feature_file} not found, skipping clip.")
            continue
        clip_features = np.load(feature_file)
        features.extend(clip_features)
        
        # Load corresponding label file
        label_file = npy_file.with_name(npy_file.stem.replace('_x', '_y') + '.npy')
        if label_file.exists():
            clip_labels = np.load(label_file)
            labels.extend(clip_labels)
        else:
            print(f"Warning: Label file {label_file} not found, using zeros.")
            labels.extend([0.0] * len(clip_frames))
        
        # Track global frame indices
        clip_idx = int(npy_file.stem.split('_')[0])
        frames_per_clip = len(clip_frames)
        frame_indices.extend(range(clip_idx * frames_per_clip, (clip_idx + 1) * frames_per_clip))
    
    if not frames:
        raise ValueError(f"No *_x.npy files found in directory: {video_dir}")
    
    frames = np.array(frames)
    features = np.array(features)
    labels = np.array(labels)
    frame_indices = np.array(frame_indices)
    
    print(f"Loaded {len(frames)} frames and {len(features)} features from {video_dir}")
    if len(frames) != len(features):
        print(f"Warning: Number of frames ({len(frames)}) does not match number of features ({len(features)})")
    
    return frames, features, labels, frame_indices

def get_clip_ending_at(frame_idx, frames, features, clip_length, total_frames):
    """Extract a clip of clip_length frames and features ending at frame_idx."""
    start_idx = max(0, frame_idx - clip_length + 1)
    end_idx = frame_idx + 1
    if start_idx >= end_idx or end_idx > total_frames:
        return None, None
    clip_frames = frames[start_idx:end_idx]
    clip_features = features[start_idx:end_idx]
    if len(clip_frames) < clip_length:
        # Pad with zeros if clip is too short
        padding_frames = np.zeros((clip_length - len(clip_frames), *clip_frames.shape[1:]), dtype=clip_frames.dtype)
        padding_features = np.zeros((clip_length - len(clip_features), *clip_features.shape[1:]), dtype=clip_features.dtype)
        clip_frames = np.concatenate([padding_frames, clip_frames], axis=0)
        clip_features = np.concatenate([padding_features, clip_features], axis=0)
    return clip_frames, clip_features

def generate_training_samples(v1_frames, v1_features, v1_labels, v1_indices, v2_frames, v2_features, v2_labels, v2_indices, clip_length=125, max_negatives_per_split=10):
    """Generate training samples with clips and features from both videos."""
    # Find split points
    v1_split_indices = v1_indices[v1_labels == 1.0]
    v2_split_indices = v2_indices[v2_labels == 1.0]
    
    if len(v1_split_indices) == 0 or len(v2_split_indices) == 0:
        print("Error: No split points found in one or both videos.")
        return None, None, None, None, None
    
    # Map split indices to positions
    v1_split_positions = {idx: pos for pos, idx in enumerate(v1_split_indices)}
    v2_split_positions = {idx: pos for pos, idx in enumerate(v2_split_indices)}
    
    v1_total_frames = len(v1_frames)
    v2_total_frames = len(v2_frames)
    
    v1_clips = []
    v1_clip_features = []
    v2_clips = []
    v2_clip_features = []
    sample_labels = []
    sample_indices = []
    
    # For each split in video 1
    for v1_split_idx in v1_split_indices:
        v1_pos = v1_split_positions[v1_split_idx]
        
        # Get video 1 clip ending at this split
        v1_clip, v1_features_clip = get_clip_ending_at(v1_split_idx, v1_frames, v1_features, clip_length, v1_total_frames)
        if v1_clip is None:
            continue
        
        # Positive sample: pair with video 2 clip at same split position
        if v1_pos < len(v2_split_indices):
            v2_split_idx = v2_split_indices[v1_pos]
            v2_clip, v2_features_clip = get_clip_ending_at(v2_split_idx, v2_frames, v2_features, clip_length, v2_total_frames)
            if v2_clip is not None:
                v1_clips.append(v1_clip)
                v1_clip_features.append(v1_features_clip)
                v2_clips.append(v2_clip)
                v2_clip_features.append(v2_features_clip)
                sample_labels.append(1.0)
                sample_indices.append((v1_split_idx, v2_split_idx))
        
        # Negative samples: pair with random video 2 clips
        neg_count = 0
        attempts = 0
        max_attempts = 50
        while neg_count < max_negatives_per_split and attempts < max_attempts:
            v2_idx = random.randint(0, v2_total_frames - 1)
            if v2_idx in v2_split_positions and v2_split_positions[v2_idx] == v1_pos:
                attempts += 1
                continue
            
            v2_clip, v2_features_clip = get_clip_ending_at(v2_idx, v2_frames, v2_features, clip_length, v2_total_frames)
            if v2_clip is not None:
                v1_clips.append(v1_clip)
                v1_clip_features.append(v1_features_clip)
                v2_clips.append(v2_clip)
                v2_clip_features.append(v2_features_clip)
                sample_labels.append(0.0)
                sample_indices.append((v1_split_idx, v2_idx))
                neg_count += 1
            attempts += 1
    
    return (np.array(v1_clips), np.array(v1_clip_features), np.array(v2_clips), 
            np.array(v2_clip_features), np.array(sample_labels), np.array(sample_indices))

def main():
    parser = argparse.ArgumentParser(description="Generate training data from preprocessed YouTube video clips.")
    parser.add_argument("video1_dir", type=str, help="Path to first video's processed clips directory")
    parser.add_argument("video2_dir", type=str, help="Path to second video's processed clips directory")
    parser.add_argument("--output_file", type=str, default="clip_training_data.npz", help="Output file for training data")
    parser.add_argument("--clip_length", type=int, default=125, help="Number of frames per clip")
    parser.add_argument("--max_negatives_per_split", type=int, default=10, help="Max negative samples per split point")
    args = parser.parse_args()
    
    # Load data from both videos
    print(f"Loading video 1 from {args.video1_dir}")
    v1_frames, v1_features, v1_labels, v1_indices = load_frames_and_labels(args.video1_dir)
    print(f"Loading video 2 from {args.video2_dir}")
    v2_frames, v2_features, v2_labels, v2_indices = load_frames_and_labels(args.video2_dir)
    
    # Generate training samples
    print("Generating training samples...")
    v1_clips, v1_clip_features, v2_clips, v2_clip_features, sample_labels, sample_indices = generate_training_samples(
        v1_frames, v1_features, v1_labels, v1_indices, v2_frames, v2_features, v2_labels, v2_indices,
        clip_length=args.clip_length, max_negatives_per_split=args.max_negatives_per_split
    )
    
    if v1_clips is None:
        print("Failed to generate samples.")
        return
    
    # Save training data
    training_data_output_path = "training_data"
    if not os.path.exists(training_data_output_path):
        os.makedirs(training_data_output_path)
    training_data_file_path = os.path.join(training_data_output_path, args.output_file)
    print(f"Saving training data to {training_data_file_path}")
    np.savez(
        training_data_file_path,
        video1_clips=v1_clips,
        video1_features=v1_clip_features,
        video2_clips=v2_clips,
        video2_features=v2_clip_features,
        labels=sample_labels,
        indices=sample_indices
    )
    
    print(f"Generated {len(sample_labels)} samples: {np.sum(sample_labels == 1.0)} positive, "
          f"{np.sum(sample_labels == 0.0)} negative")

if __name__ == "__main__":
    main()