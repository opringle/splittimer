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

from utils import get_default_device_name, get_video_fps_and_total_frames, setup_logging


def extract_individual_features(preprocessed_frames, model, device, batch_size=32):
    """Extract ResNet50 features for a batch of preprocessed frames."""
    logging.debug(f"Computing resnet features...")
    model.eval()
    all_features = []

    with torch.no_grad():
        for start_idx in range(0, len(preprocessed_frames), batch_size):
            end_idx = min(start_idx + batch_size, len(preprocessed_frames))
            batch_tensors = torch.stack(
                preprocessed_frames[start_idx:end_idx]).to(device)
            # Shape: (batch_size, 2048)
            features = model(batch_tensors).squeeze(-1).squeeze(-1)
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    logging.debug(
        f"Extracted individual features with shape {all_features.shape}")
    return all_features


def preprocess_frame(frame):
    """Preprocess a single frame for feature extraction."""
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    return preprocess(frame)


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual features for video frames.")
    parser.add_argument("videos_dir", type=str,
                        help="Base directory containing videos (videos_dir/<trackId>/<riderId>/*.mp4)")
    parser.add_argument("output_dir", type=str,
                        help="Base directory to save feature .npy files (output_dir/<trackId>/<riderId>/*.npy)")
    parser.add_argument("--feature-extraction-batch-size", type=int,
                        default=16, help="Batch size for feature extraction")
    parser.add_argument("--clip-length", type=int, default=50,
                        help="Length of each clip for batching features")
    parser.add_argument('--device', type=str, default=get_default_device_name(),
                        help='Device to use (cuda or cpu)')
    parser.add_argument("--log-level", type=str, default="INFO", choices=[
                        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Initialize model
    device = torch.device(args.device)
    individual_model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove final classification layer
    individual_model = nn.Sequential(*list(individual_model.children())[:-1])
    individual_model.to(device)
    logging.info(f"Using device: {device}")

    # Get list of video files recursively
    videos_dir = Path(args.videos_dir)
    gopro_videos = sorted(videos_dir.rglob("*.mp4"))
    if not gopro_videos:
        logging.error(
            f"No video files (*.mp4) found in {videos_dir} or its subdirectories")
        return

    logging.info(f"Found {len(gopro_videos)} video files in {videos_dir}")

    # Calculate total number of clips for progress bar
    total_clips = 0
    for gopro_video in gopro_videos:
        _, total_frames = get_video_fps_and_total_frames(str(gopro_video))
        total_clips += math.ceil(total_frames / args.clip_length)

    output_dir = Path(args.output_dir)

    # Process clips with a single progress bar
    with tqdm(total=total_clips, desc="Processing clips") as pbar:
        for gopro_video in gopro_videos:
            # Extract trackId and riderId from path
            parts = gopro_video.parts
            if len(parts) < 3:
                logging.error(
                    f"Invalid path structure for {gopro_video}, skipping")
                pbar.update(math.ceil(total_frames / args.clip_length))
                continue
            rider_id = parts[-2]
            track_id = parts[-3]

            # Load the video
            cap = cv2.VideoCapture(str(gopro_video))
            _, total_frames = get_video_fps_and_total_frames(str(gopro_video))
            logging.debug(f"Video {gopro_video} has {total_frames} frames")

            # Process frames in clips of length C
            clip_length = args.clip_length
            for start_idx in range(0, total_frames, clip_length):
                end_idx = min(start_idx + clip_length - 1, total_frames - 1)
                first_frame = start_idx
                last_frame = end_idx
                output_clip_dir = output_dir / track_id / rider_id

                # Define output path
                individual_feature_path = output_clip_dir / \
                    f"{first_frame:06d}_to_{last_frame:06d}_resnet50.npy"

                # Skip if file exists
                if individual_feature_path.exists():
                    logging.info(
                        f"Individual feature file exists for clip {first_frame} to {last_frame}, skipping")
                    pbar.update(1)
                    continue

                # Read frames from start_idx to end_idx
                frames = []
                frame_indices = list(range(start_idx, end_idx + 1))
                logging.debug(
                    f"Constructing clip buffer from {len(frame_indices)} frame indices...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                for i in range(len(frame_indices)):
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        logging.warning(
                            f"Cannot read frame {start_idx + i} from {gopro_video}, padding with zeros")
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                logging.debug(f"Done")

                # Preprocess frames
                logging.debug(f"Preprocessing {len(frames)} frames.")
                preprocessed_frames = [
                    preprocess_frame(frame) for frame in frames]
                logging.debug(f"Done.")

                # Extract individual features
                individual_features = extract_individual_features(
                    preprocessed_frames, individual_model, device, batch_size=args.feature_extraction_batch_size)
                if individual_features.size == 0:
                    pbar.update(1)
                    continue
                output_clip_dir.mkdir(parents=True, exist_ok=True)
                np.save(individual_feature_path, individual_features)
                logging.info(
                    f"Saved individual features to {individual_feature_path} with shape {individual_features.shape}")

                pbar.update(1)

            cap.release()

    logging.info("Feature extraction complete!")


if __name__ == "__main__":
    main()
