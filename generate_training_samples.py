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

def generate_training_samples(v1_frames, v1_features, v1_labels, v1_indices, v2_frames, v2_features, v2_labels, v2_indices, clip_length=125, max_negatives_per_split=10, num_augmented_per_segment=5):
    """Generate training samples with clips and features from both videos, including augmented samples biased towards split points."""
    # Find split points
    v1_split_indices = v1_indices[v1_labels == 1.0]
    v2_split_indices = v2_indices[v2_labels == 1.0]
    
    if len(v1_split_indices) == 0 or len(v2_split_indices) == 0:
        print("Error: No split points found in one or both videos.")
        return None, None, None, None, None
    
    if len(v1_split_indices) != len(v2_split_indices):
        print("Error: Number of split points in video 1 and video 2 do not match.")
        return None, None, None, None, None
    
    v1_total_frames = len(v1_frames)
    v2_total_frames = len(v2_frames)
    
    v1_clips = []
    v1_clip_features = []
    v2_clips = []
    v2_clip_features = []
    sample_labels = []
    sample_indices = []
    
    # Generate samples at split points (original logic)
    for pos, v1_split_idx in enumerate(v1_split_indices):
        v2_split_idx = v2_split_indices[pos]
        
        v1_clip, v1_features_clip = get_clip_ending_at(v1_split_idx, v1_frames, v1_features, clip_length, v1_total_frames)
        v2_clip, v2_features_clip = get_clip_ending_at(v2_split_idx, v2_frames, v2_features, clip_length, v2_total_frames)
        
        if v1_clip is not None and v2_clip is not None:
            v1_clips.append(v1_clip)
            v1_clip_features.append(v1_features_clip)
            v2_clips.append(v2_clip)
            v2_clip_features.append(v2_features_clip)
            sample_labels.append(1.0)
            sample_indices.append((v1_split_idx, v2_split_idx))
        
        # Negative samples for split points
        neg_count = 0
        attempts = 0
        max_attempts = 50
        while neg_count < max_negatives_per_split and attempts < max_attempts:
            v2_idx = random.randint(0, v2_total_frames - 1)
            if v2_idx in v2_split_indices and v2_split_indices.index(v2_idx) == pos:
                attempts += 1
                continue
            v2_clip_neg, v2_features_clip_neg = get_clip_ending_at(v2_idx, v2_frames, v2_features, clip_length, v2_total_frames)
            if v2_clip_neg is not None:
                v1_clips.append(v1_clip)
                v1_clip_features.append(v1_features_clip)
                v2_clips.append(v2_clip_neg)
                v2_clip_features.append(v2_features_clip_neg)
                sample_labels.append(0.0)
                sample_indices.append((v1_split_idx, v2_idx))
                neg_count += 1
            attempts += 1
    
    # Generate augmented samples for segments between splits, biased towards split points
    num_segments = len(v1_split_indices) - 1
    for seg in range(num_segments):
        v1_start = v1_split_indices[seg]
        v1_end = v1_split_indices[seg + 1]
        v2_start = v2_split_indices[seg]
        v2_end = v2_split_indices[seg + 1]
        
        v1_seg_length = v1_end - v1_start
        v2_seg_length = v2_end - v2_start
        
        # Select frames where full clips can be contained within the segment
        possible_idx1 = list(range(v1_start + clip_length - 1, v1_end + 1))
        if len(possible_idx1) == 0:
            continue
        
        # Generate relative positions with beta distribution (biased towards 0 and 1)
        num_samples = min(num_augmented_per_segment, len(possible_idx1))
        relative_positions = np.random.beta(0.5, 0.5, num_samples)
        
        # Map to frame indices
        min_idx = possible_idx1[0]
        max_idx = possible_idx1[-1]
        selected_idx1 = [int(min_idx + p * (max_idx - min_idx)) for p in relative_positions]
        selected_idx1 = [min(max(idx, min_idx), max_idx) for idx in selected_idx1]
        
        for idx1 in selected_idx1:
            # Compute relative position in video 1 segment
            p = (idx1 - v1_start) / v1_seg_length
            # Map to corresponding frame in video 2 segment
            idx2_float = v2_start + p * v2_seg_length
            idx2 = int(round(idx2_float))
            if idx2 < v2_start or idx2 > v2_end or idx2 - clip_length + 1 < v2_start:
                continue
            
            v1_clip, v1_features_clip = get_clip_ending_at(idx1, v1_frames, v1_features, clip_length, v1_total_frames)
            v2_clip, v2_features_clip = get_clip_ending_at(idx2, v2_frames, v2_features, clip_length, v2_total_frames)
            if v1_clip is not None and v2_clip is not None:
                v1_clips.append(v1_clip)
                v1_clip_features.append(v1_features_clip)
                v2_clips.append(v2_clip)
                v2_clip_features.append(v2_features_clip)
                sample_labels.append(1.0)
                sample_indices.append((idx1, idx2))
                
                # Generate negative samples for augmented positive sample
                neg_count = 0
                attempts = 0
                while neg_count < max_negatives_per_split and attempts < max_attempts:
                    random_idx2 = random.randint(0, v2_total_frames - 1)
                    if abs(random_idx2 - idx2) < clip_length:
                        attempts += 1
                        continue
                    v2_clip_neg, v2_features_clip_neg = get_clip_ending_at(random_idx2, v2_frames, v2_features, clip_length, v2_total_frames)
                    if v2_clip_neg is not None:
                        v1_clips.append(v1_clip)
                        v1_clip_features.append(v1_features_clip)
                        v2_clips.append(v2_clip_neg)
                        v2_clip_features.append(v2_features_clip_neg)
                        sample_labels.append(0.0)
                        sample_indices.append((idx1, random_idx2))
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
    parser.add_argument("--num_augmented_per_segment", type=int, default=5, help="Number of augmented samples per segment")
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
        clip_length=args.clip_length, 
        max_negatives_per_split=args.max_negatives_per_split,
        num_augmented_per_segment=args.num_augmented_per_segment
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