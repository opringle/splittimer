import pandas as pd
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
from pathlib import Path
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained ResNet50 model for feature extraction
resnet50 = models.resnet50(weights=models.resnet50.ResNet50_Weights.IMAGENET1K_V)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])  # Remove the last layer
resnet50.eval()

# Define preprocessing transform for frames
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(frame):
    """Extract ResNet50 features from a single frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = preprocess(frame_rgb).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = resnet50(frame_tensor).squeeze().numpy()
    return features

def get_clip(video_path, central_idx, F, total_frames):
    """Extract a clip of F frames centered around central_idx from the video."""
    start = max(0, min(central_idx - F // 2, total_frames - F))
    clip_indices = list(range(start, start + F))
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for idx in clip_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            logging.warning(f"Cannot read frame {idx} from {video_path}")
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))  # Placeholder
    cap.release()
    return frames

def main():
    parser = argparse.ArgumentParser(description="Generate training data for ML network from CSV.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser.add_argument("output_dir", type=str, help="Directory to save the .npz samples")
    parser.add_argument("--F", type=int, default=5, help="Number of frames in each clip")
    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(args.csv_path)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each row in the CSV
    for row in tqdm(df.itertuples(), total=len(df), desc="Generating samples"):
        # Construct video paths
        v1_path = Path("downloaded_videos") / row.track_id / row.v1_rider_id / f"{row.track_id}_{row.v1_rider_id}.mp4"
        v2_path = Path("downloaded_videos") / row.track_id / row.v2_rider_id / f"{row.track_id}_{row.v2_rider_id}.mp4"

        # Get total frames for each video
        cap1 = cv2.VideoCapture(str(v1_path))
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        cap1.release()
        cap2 = cv2.VideoCapture(str(v2_path))
        total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap2.release()

        # Extract clips
        v1_frames = get_clip(v1_path, row.v1_frame_idx, args.F, total_frames1)
        v2_frames = get_clip(v2_path, row.v2_frame_idx, args.F, total_frames2)

        # Extract features and add position index
        v1_features = []
        for pos, frame in enumerate(v1_frames):
            feat = extract_features(frame)
            feat_with_pos = np.append(feat, float(pos))
            v1_features.append(feat_with_pos)
        v1_clip = np.stack(v1_features, axis=0)

        v2_features = []
        for pos, frame in enumerate(v2_frames):
            feat = extract_features(frame)
            feat_with_pos = np.append(feat, float(pos))
            v2_features.append(feat_with_pos)
        v2_clip = np.stack(v2_features, axis=0)

        # Get the label
        label = row.label

        # Save the sample
        sample_id = f"sample_{row.Index:06d}"
        sample_path = output_dir / f"{sample_id}.npz"
        np.savez(sample_path, clip1=v1_clip, clip2=v2_clip, label=label)

    logging.info(f"Generated {len(df)} samples in {output_dir}")

if __name__ == "__main__":
    main()