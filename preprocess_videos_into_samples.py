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

# Set up device for GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize ResNet50 model
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
resnet50.to(device)
resnet50.eval()

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_batch(frames, batch_size=32):
    """Extract features from a list of frames in batches."""
    logging.debug(f"Extracting features from {len(frames)} frames with batch size {batch_size}")
    features = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        batch_tensors = [preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in batch_frames]
        batch_tensors = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            batch_features = resnet50(batch_tensors).squeeze().cpu().numpy()
        # Handle case where batch_features is 1D (single frame in batch)
        if len(batch_features.shape) == 1:
            batch_features = batch_features.reshape(1, -1)
        features.append(batch_features)
    return np.vstack(features)

def get_clip(video_path, central_idx, F, total_frames):
    """Extract F frames from a video starting around central_idx."""
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
            logging.warning(f"Cannot read frame {idx} from {video_path}. Padding instead")
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames

def main():
    # 15 seconds per sample should preprocess all samples in 24 hours
    parser = argparse.ArgumentParser(description="Generate batched training data with optimized feature extraction.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser.add_argument("output_dir", type=str, help="Directory to save .npz files")
    parser.add_argument("--F", type=int, default=50, help="Number of frames per clip")
    parser.add_argument("--batch_size", type=int, default=32, help="Samples per .npz file")
    parser.add_argument("--feature_batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    df = pd.read_csv(args.csv_path)
    logging.info(f"Loaded metadata for {len(df)} training samples")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_clip1s, batch_clip2s, batch_labels = [], [], []
    batch_count = 0

    for row in tqdm(df.itertuples(), total=len(df), desc="Generating samples"):
        v1_path = Path("downloaded_videos") / row.track_id / row.v1_rider_id / f"{row.track_id}_{row.v1_rider_id}.mp4"
        v2_path = Path("downloaded_videos") / row.track_id / row.v2_rider_id / f"{row.track_id}_{row.v2_rider_id}.mp4"

        # Get total frames for each video
        cap1 = cv2.VideoCapture(str(v1_path))
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        cap1.release()
        cap2 = cv2.VideoCapture(str(v2_path))
        total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap2.release()

        # Extract frames
        v1_frames = get_clip(v1_path, row.v1_frame_idx, args.F, total_frames1)
        v2_frames = get_clip(v2_path, row.v2_frame_idx, args.F, total_frames2)

        # Extract features in batches and append position
        v1_features = extract_features_batch(v1_frames, batch_size=args.feature_batch_size)
        v1_features_with_pos = [np.append(feat, float(pos)) for pos, feat in enumerate(v1_features)]
        v1_clip = np.stack(v1_features_with_pos, axis=0)

        v2_features = extract_features_batch(v2_frames, batch_size=args.feature_batch_size)
        v2_features_with_pos = [np.append(feat, float(pos)) for pos, feat in enumerate(v2_features)]
        v2_clip = np.stack(v2_features_with_pos, axis=0)

        # Add to batch
        batch_clip1s.append(v1_clip)
        batch_clip2s.append(v2_clip)
        batch_labels.append(row.label)

        # Save batch when full
        if len(batch_clip1s) >= args.batch_size:
            np.savez(
                output_dir / f"batch_{batch_count:06d}.npz",
                clip1s=np.stack(batch_clip1s, axis=0),
                clip2s=np.stack(batch_clip2s, axis=0),
                labels=np.array(batch_labels)
            )
            batch_clip1s, batch_clip2s, batch_labels = [], [], []
            batch_count += 1

    # Save remaining samples
    if batch_clip1s:
        np.savez(
            output_dir / f"batch_{batch_count:06d}.npz",
            clip1s=np.stack(batch_clip1s, axis=0),
            clip2s=np.stack(batch_clip2s, axis=0),
            labels=np.array(batch_labels)
        )

    logging.info(f"Generated {len(df)} samples in {batch_count + 1} batches in {output_dir}")

if __name__ == "__main__":
    main()