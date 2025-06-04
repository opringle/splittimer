import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import random
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def load_frames_from_dir(video_dir):
    """Load all frames from .npy files in the video directory."""
    frames = []
    for npy_file in sorted(Path(video_dir).glob("*.npy")):
        clip = np.load(npy_file)
        frames.extend(clip)
    return np.array(frames)

def extract_features(frames, model, device, batch_size=32):
    """Extract ResNet50 features for a batch of frames with batch processing."""
    model.eval()
    all_features = []
    
    # Convert frames to tensor and move to device in batches
    num_frames = len(frames)
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_frames, batch_size), desc="Extracting features"):
            end_idx = min(start_idx + batch_size, num_frames)
            batch_frames = frames[start_idx:end_idx]
            
            # Convert batch to tensor
            frames_tensor = torch.tensor(batch_frames, dtype=torch.float32, pin_memory=(device.type == "cuda"))
            frames_tensor = frames_tensor.to(device, non_blocking=True)
            
            # Extract features
            features = model(frames_tensor)
            # Flatten features to (batch_size, feature_dim)
            features = features.view(features.size(0), -1)
            # Normalize features for dot product similarity
            features = nn.functional.normalize(features, p=2, dim=1)
            
            # Collect features
            all_features.append(features.cpu().numpy())
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    print(f"Extracted features with shape {all_features.shape} (N, feature_dim)")
    return all_features

def display_frames(frame1, frame2, title1, title2):
    """Display two frames side by side."""
    # Denormalize frames for display (undo ResNet50 normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame1 = frame1.copy().transpose(1, 2, 0)  # (C, H, W) to (H, W, C)
    frame2 = frame2.copy().transpose(1, 2, 0)
    frame1 = (frame1 * std + mean) * 255  # Denormalize and scale to 0-255
    frame2 = (frame2 * std + mean) * 255
    frame1 = frame1.astype(np.uint8)
    frame2 = frame2.astype(np.uint8)

    # Create side-by-side plot
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
    parser = argparse.ArgumentParser(description="Compare frames from two processed YouTube videos using ResNet50.")
    parser.add_argument("video_dir1", type=str, help="Path to first video's processed clips directory")
    parser.add_argument("video_dir2", type=str, help="Path to second video's processed clips directory")
    args = parser.parse_args()

    # Load ResNet50 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    # Remove the final classification layer to get features
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)

    # Load frames from both videos
    print(f"Loading frames from {args.video_dir1}")
    frames1 = load_frames_from_dir(args.video_dir1)
    print(f"Loading frames from {args.video_dir2}")
    frames2 = load_frames_from_dir(args.video_dir2)

    if len(frames1) == 0 or len(frames2) == 0:
        print("Error: No frames found in one or both directories.")
        return

    # Select a random frame from the first video
    random_idx = random.randint(0, len(frames1) - 1)
    selected_frame = frames1[random_idx]
    print(f"Selected frame {random_idx} from first video")

    # Extract features for the selected frame and all frames in the second video
    selected_feature = extract_features(selected_frame[None, ...], model, device)  # Add batch dimension at index 0
    features2 = extract_features(frames2, model, device)
    
    # Compute dot product similarities
    similarities = np.dot(features2, selected_feature.T).flatten()
    max_similarity_idx = np.argmax(similarities)
    max_similarity = similarities[max_similarity_idx]

    print(f"Most similar frame index: {max_similarity_idx}, Similarity: {max_similarity:.4f}")

    # Display the selected frame and the most similar frame
    display_frames(
        selected_frame,
        frames2[max_similarity_idx],
        f"Selected Frame (Video 1, idx {random_idx})",
        f"Most Similar Frame (Video 2, idx {max_similarity_idx})"
    )

if __name__ == "__main__":
    main()