import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from pprint import pprint

def load_frames_and_labels(video_dir):
    """Load precomputed ResNet50 features, original frames, and corresponding labels from .npy files."""
    features = []
    frames = []
    labels = []
    frame_indices = []
    for npy_file in sorted(Path(video_dir).glob("*_x.npy")):
        # Load original frame clip for display
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
            print(f"Warning: Label file {label_file} not found, skipping.")
            labels.extend([0.0] * len(clip_frames))
        
        # Track global frame indices
        clip_idx = int(npy_file.stem.split('_')[0])
        frames_per_clip = len(clip_frames)
        frame_indices.extend(range(clip_idx * frames_per_clip, (clip_idx + 1) * frames_per_clip))
    
    frames = np.array(frames)
    features = np.array(features)
    labels = np.array(labels)
    print(f"Loaded {len(frames)} frames and {len(features)} features from {video_dir}")
    if len(frames) != len(features):
        print(f"Warning: Number of frames ({len(frames)}) does not match number of features ({len(features)})")
    return frames, features, labels, frame_indices

def display_frames(frame1, frame2, title1, title2):
    """Display two frames side by side."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame1 = frame1.copy().transpose(1, 2, 0)
    frame2 = frame2.copy().transpose(1, 2, 0)
    frame1 = (frame1 * std + mean) * 255
    frame2 = (frame2 * std + mean) * 255
    frame1 = frame1.astype(np.uint8)
    frame2 = frame2.astype(np.uint8)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(frame1)
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(frame2)
    plt.title(title2)
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare split frames from two processed YouTube videos using precomputed ResNet50 features and evaluate predictions.")
    parser.add_argument("video_dir1", type=str, help="Path to first video's processed clips directory")
    parser.add_argument("video_dir2", type=str, help="Path to second video's processed clips directory")
    args = parser.parse_args()

    print(f"Loading frames, features, and labels from {args.video_dir1}")
    frames1, features1, labels1, indices1 = load_frames_and_labels(args.video_dir1)
    print(f"Loading frames, features, and labels from {args.video_dir2}")
    frames2, features2, labels2, indices2 = load_frames_and_labels(args.video_dir2)

    if len(frames1) == 0 or len(frames2) == 0 or len(features1) == 0 or len(features2) == 0:
        print("Error: No frames or features found in one or both directories.")
        return

    # Identify split frames (where labels == 1.0)
    split_indices1 = np.where(labels1 == 1.0)[0]
    split_indices2 = np.where(labels2 == 1.0)[0]
    if len(split_indices1) == 0:
        print("Error: No split frames (label 1.0) found in first video.")
        return
    if len(split_indices2) == 0:
        print("Error: No split frames (label 1.0) found in second video.")
        return
    if len(split_indices1) != len(split_indices2):
        print(f"Error: Number of split frames in video 1 ({len(split_indices1)}) does not match video 2 ({len(split_indices2)}).")
        return

    print(f"Found {len(split_indices1)} split frames in first video")
    print(f"Found {len(split_indices2)} split frames in second video")

    # Evaluate predictions
    correct_predictions = 0
    total_predictions = len(split_indices1)
    split_matches = []

    # Process each split frame
    for split_idx in tqdm(split_indices1, desc="Processing split frames"):
        selected_frame = frames1[split_idx]
        selected_feature = features1[split_idx]
        global_idx = indices1[split_idx]
        print(f"Processing split frame {global_idx} from first video")

        # Compute similarities with second video frames
        similarities = np.dot(features2, selected_feature.T).flatten()
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]

        # Check if the predicted frame is a split frame in video 2
        is_correct = labels2[max_similarity_idx] == 1.0
        if is_correct:
            correct_predictions += 1
        split_matches.append({
            'video1_idx': global_idx,
            'video2_predicted_idx': indices2[max_similarity_idx],
            'video2_actual_idx': indices2[split_indices2[split_indices1.tolist().index(split_idx)]],
            'similarity': max_similarity,
            'is_correct': is_correct
        })

        print(f"Most similar frame index: {indices2[max_similarity_idx]}, Similarity: {max_similarity:.4f}, Correct: {is_correct}")

        # Display the split frame and its most similar frame
        display_frames(
            selected_frame,
            frames2[max_similarity_idx],
            f"Split Frame (Video 1, idx {global_idx})",
            f"Most Similar Frame (Video 2, idx {indices2[max_similarity_idx]})"
        )

    # Compute and print evaluation metrics
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\nEvaluation Results:")
    print(f"Total split frames: {total_predictions}")
    print(f"Correctly predicted split frames: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Detailed match information
    print("\nDetailed Match Information:")
    for match in split_matches:
        pprint(match)

if __name__ == "__main__":
    main()