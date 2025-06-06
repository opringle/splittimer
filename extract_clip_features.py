import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import cv2
import math

def extract_features(frames, model, device, batch_size=32):
    """Extract ResNet50 features for a batch of frames."""
    model.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    all_features = []

    if len(frames) == 0:
        logging.error("No frames provided for feature extraction")
        return np.array([])

    with torch.no_grad():
        for start_idx in range(0, len(frames), batch_size):
            end_idx = min(start_idx + batch_size, len(frames))
            batch_frames = frames[start_idx:end_idx]

            # Preprocess frames
            batch_tensors = [preprocess(frame) for frame in batch_frames]
            batch_tensors = torch.stack(batch_tensors).to(device)

            # Extract features
            features = model(batch_tensors).squeeze(-1).squeeze(-1)  # Shape: (batch_size, 2048)
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    logging.debug(f"Extracted features with shape {all_features.shape}")
    return all_features

def main():
    parser = argparse.ArgumentParser(description="Extract ResNet50 features for video clips in a nested directory structure.")
    parser.add_argument("videos_dir", type=str, help="Base directory containing videos (videos_dir/<trackId>/<riderId>/*.mp4)")
    parser.add_argument("output_dir", type=str, help="Base directory to save feature .npy files (output_dir/<trackId>/<riderId>/*.npy)")
    parser.add_argument("--feature-extraction-batch-size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--clip-length", type=int, default=50, help="Length of each clip")
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

    # Initialize ResNet50 model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
    model.to(device)
    logging.info(f"Using device: {device}")

    # Get list of video files recursively
    videos_dir = Path(args.videos_dir)
    clip_files = sorted(videos_dir.rglob("*.mp4"))
    if not clip_files:
        logging.error(f"No video files (*.mp4) found in {videos_dir} or its subdirectories")
        return

    logging.info(f"Found {len(clip_files)} video files in {videos_dir}")

    # Calculate total number of clips for progress bar
    total_clips = 0
    for clip_file in clip_files:
        cap = cv2.VideoCapture(str(clip_file))
        if not cap.isOpened():
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        total_clips += math.ceil(total_frames / args.clip_length)

    output_dir = Path(args.output_dir)

    # Process clips with a single progress bar
    with tqdm(total=total_clips, desc="Processing clips") as pbar:
        for clip_file in clip_files:
            # Extract trackId and riderId from path
            parts = clip_file.parts
            if len(parts) < 3:
                logging.error(f"Invalid path structure for {clip_file}, skipping")
                pbar.update(math.ceil(total_frames / args.clip_length))
                continue
            rider_id = parts[-2]
            track_id = parts[-3]

            # Load the video
            cap = cv2.VideoCapture(str(clip_file))
            if not cap.isOpened():
                logging.error(f"Cannot open video {clip_file}, skipping")
                pbar.update(math.ceil(total_frames / args.clip_length))
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.debug(f"Video {clip_file} has {total_frames} frames")

            # Process frames in clips of length C
            clip_length = args.clip_length
            for start_idx in range(0, total_frames, clip_length):
                end_idx = min(start_idx + clip_length, total_frames)
                first_frame = start_idx
                last_frame = end_idx - 1
                output_clip_dir = output_dir / track_id / rider_id
                feature_name = f"{first_frame:06d}_to_{last_frame:06d}_resnet50.npy"
                feature_path = output_clip_dir / feature_name

                # Check if feature file already exists
                if feature_path.exists():
                    logging.info(f"Feature file already exists for clip {first_frame} to {last_frame} in {clip_file}, skipping")
                    pbar.update(1)
                    continue

                # Read frames for the current clip
                clip_frames = []
                clip_indices = list(range(start_idx, end_idx))
                for idx in clip_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        clip_frames.append(frame)
                    else:
                        logging.warning(f"Cannot read frame {idx} from {clip_file}, padding")
                        clip_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                        clip_indices.append(clip_indices[-1] if clip_indices else 0)

                # Pad partial clips
                if len(clip_frames) != clip_length:
                    logging.debug(f"Partial clip at {start_idx}:{end_idx} with {len(clip_frames)} frames, padding")
                    while len(clip_frames) < clip_length:
                        clip_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                        clip_indices.append(clip_indices[-1] if clip_indices else 0)

                # Extract features
                features = extract_features(clip_frames, model, device, batch_size=args.feature_extraction_batch_size)
                if features.size == 0:
                    logging.error(f"No features extracted for clip {start_idx}:{end_idx} in {clip_file}, skipping")
                    pbar.update(1)
                    continue

                # Save features
                output_clip_dir.mkdir(parents=True, exist_ok=True)
                np.save(feature_path, features)
                logging.info(f"Saved features to {feature_path} with shape {features.shape}")

                # Update progress bar
                pbar.update(1)

            cap.release()

    logging.info("Feature extraction complete!")

if __name__ == "__main__":
    main()