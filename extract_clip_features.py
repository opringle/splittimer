import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_features(frames, model, device, batch_size=32):
    """Extract ResNet50 features for a batch of frames with batch processing."""
    model.eval()
    all_features = []
    
    num_frames = len(frames)
    if num_frames == 0:
        print("Error: No frames provided for feature extraction.")
        return np.array([])
    
    with torch.no_grad():
        for start_idx in range(0, num_frames, batch_size):
            end_idx = min(start_idx + batch_size, num_frames)
            batch_frames = frames[start_idx:end_idx]
            
            frames_tensor = torch.tensor(batch_frames, dtype=torch.float32, pin_memory=(device.type == "cuda"))
            frames_tensor = frames_tensor.to(device, non_blocking=True)
            
            features = model(frames_tensor)
            features = features.view(features.size(0), -1)
            features = nn.functional.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    print(f"Extracted features with shape {all_features.shape} (N, feature_dim)")
    return all_features

def main():
    parser = argparse.ArgumentParser(description="Extract ResNet50 features for video clips in a nested directory structure.")
    parser.add_argument("clips_dir", type=str, help="Base directory containing processed clips (processed_clips/<trackId>/<riderId>/*_x.npy)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction")
    args = parser.parse_args()

    # Initialize ResNet50 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
    model = model.to(device)
    print(f"Using device: {device}")

    # Get list of clip files recursively
    clips_dir = Path(args.clips_dir)
    clip_files = sorted(clips_dir.rglob("*_x.npy"))  # Recursively find all *_x.npy files
    if not clip_files:
        print(f"Error: No clip files (*_x.npy) found in {args.clips_dir} or its subdirectories")
        return

    print(f"Found {len(clip_files)} clip files in {args.clips_dir} and subdirectories")

    # Process each clip file with a single progress bar
    for clip_file in tqdm(clip_files, desc="Processing clips"):
        print(f"Processing clip file: {clip_file}")
        clip = np.load(clip_file)
        
        # Extract features for the clip
        features = extract_features(clip, model, device, batch_size=args.batch_size)
        if len(features) == 0:
            print(f"Warning: No features extracted for {clip_file}, skipping.")
            continue
        
        # Save features to the same directory with updated naming
        # Expected clip_file name: <startFrame>_to_<endFrame>_x.npy
        base_name = clip_file.stem  # e.g., "000000_to_000124_x"
        feature_name = base_name.replace('_x', '_resnet50')  # e.g., "000000_to_000124_resnet50"
        feature_path = clip_file.with_name(f"{feature_name}.npy")
        np.save(feature_path, features)
        print(f"Saved features to {feature_path} with shape {features.shape}")

    print("Feature extraction complete!")

if __name__ == "__main__":
    main()