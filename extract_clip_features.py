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

from utils import get_default_device_name, get_video_fps_and_total_frames

class R3D18FeatureExtractor(nn.Module):
    """Feature extractor for R3D-18 model, removing the final classification layer."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.flatten(1)
        return x

def extract_individual_features(preprocessed_frames, model, device, batch_size=32):
    """Extract ResNet50 features for a batch of preprocessed frames."""
    logging.debug(f"Computing resnet features...")
    model.eval()
    all_features = []

    with torch.no_grad():
        for start_idx in range(0, len(preprocessed_frames), batch_size):
            end_idx = min(start_idx + batch_size, len(preprocessed_frames))
            batch_tensors = torch.stack(preprocessed_frames[start_idx:end_idx]).to(device)
            features = model(batch_tensors).squeeze(-1).squeeze(-1)  # Shape: (batch_size, 2048)
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    logging.debug(f"Extracted individual features with shape {all_features.shape}")
    return all_features

def create_padded_sequence(preprocessed_frames, start_clip, i, F):
    """Create a padded sequence of F frames ending at position i."""
    sequence_start = max(start_clip, i - F + 1)
    sequence_end = i + 1
    sequence = preprocessed_frames[sequence_start - start_clip : sequence_end - start_clip]
    pad_length = F - len(sequence)
    if pad_length > 0:
        zero_tensor = torch.zeros_like(preprocessed_frames[0])
        padded_sequence = [zero_tensor] * pad_length + sequence
    else:
        padded_sequence = sequence[-F:]
    return torch.stack(padded_sequence)

def extract_sequence_features(preprocessed_frames, start_clip, start_idx, end_idx, model, device, F, batch_size=32):
    """Extract 3D CNN features for sequences ending at each frame in the clip."""
    model.eval()
    all_features = []

    sequences = []
    for i in range(start_idx, end_idx + 1):
        sequence = create_padded_sequence(preprocessed_frames, start_clip, i, F)
        sequences.append(sequence)

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            end = min(start + batch_size, len(sequences))
            batch_sequences = sequences[start:end]
            batch_tensor = torch.stack(batch_sequences)  # Shape: (B, F, 3, H, W)
            batch_tensor = batch_tensor.permute(0, 2, 1, 3, 4)  # Shape: (B, 3, F, H, W)
            features = model(batch_tensor.to(device))
            all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    logging.debug(f"Extracted sequence features with shape {all_features.shape}")
    return all_features

def preprocess_frame(frame):
    """Preprocess a single frame for feature extraction."""
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(frame)

def main():
    parser = argparse.ArgumentParser(description="Extract features for video frames and sequences.")
    parser.add_argument("videos_dir", type=str, help="Base directory containing videos (videos_dir/<trackId>/<riderId>/*.mp4)")
    parser.add_argument("output_dir", type=str, help="Base directory to save feature .npy files (output_dir/<trackId>/<riderId>/*.npy)")
    parser.add_argument("--feature-extraction-batch-size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--clip-length", type=int, default=50, help="Length of each clip for batching features")
    parser.add_argument("--sequence-length", type=int, default=10, help="Number of frames in each sequence for 3D CNN features")
    parser.add_argument("--feature-types", type=str, choices=['individual', 'sequence', 'both'], default='individual', help="Types of features to extract")
    parser.add_argument('--device', type=str, default=get_default_device_name(), help='Device to use (cuda or cpu)')
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # Initialize models
    device = torch.device(args.device)
    if args.feature_types in ['individual', 'both']:
        individual_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        individual_model = nn.Sequential(*list(individual_model.children())[:-1])  # Remove final classification layer
        individual_model.to(device)
    if args.feature_types in ['sequence', 'both']:
        sequence_model = models.video.r3d_18(weights='KINETICS400_V1')
        sequence_model = R3D18FeatureExtractor(sequence_model)
        sequence_model.to(device)
    logging.info(f"Using device: {device}")

    # Get list of video files recursively
    videos_dir = Path(args.videos_dir)
    gopro_videos = sorted(videos_dir.rglob("*.mp4"))
    if not gopro_videos:
        logging.error(f"No video files (*.mp4) found in {videos_dir} or its subdirectories")
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
                logging.error(f"Invalid path structure for {gopro_video}, skipping")
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

                # Define output paths
                individual_feature_path = output_clip_dir / f"{first_frame:06d}_to_{last_frame:06d}_resnet50.npy"
                sequence_feature_path = output_clip_dir / f"{first_frame:06d}_to_{last_frame:06d}_iconv3d_F{args.sequence_length}.npy"

                # Skip if files exist
                skip_clip = False
                if args.feature_types in ['individual', 'both'] and individual_feature_path.exists():
                    logging.info(f"Individual feature file exists for clip {first_frame} to {last_frame}, skipping")
                    skip_clip = True
                if args.feature_types in ['sequence', 'both'] and sequence_feature_path.exists():
                    logging.info(f"Sequence feature file exists for clip {first_frame} to {last_frame}, skipping")
                    skip_clip = True
                if skip_clip and (args.feature_types == 'individual' or args.feature_types == 'sequence' or \
                                 (args.feature_types == 'both' and individual_feature_path.exists() and sequence_feature_path.exists())):
                    pbar.update(1)
                    continue

                # Compute start_clip for sequence features
                start_clip = max(0, start_idx - args.sequence_length + 1)

                # Read frames from start_clip to end_idx
                frames = []
                frame_indices = list(range(start_clip, end_idx + 1))
                logging.debug(f"Constructing clip buffer from {len(frame_indices)} frame indices...")
                num_frames_to_read = end_idx - start_clip + 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_clip)
                for i in range(num_frames_to_read):
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    else:
                        logging.warning(f"Cannot read frame {start_clip + i} from {gopro_video}, padding with zeros")
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                logging.debug(f"Done")

                # Preprocess frames
                logging.debug(f"Preprocessing {len(frames)} frames.")
                preprocessed_frames = [preprocess_frame(frame) for frame in frames]
                logging.debug(f"Done.")

                # Extract individual features for frames start_idx to end_idx
                if args.feature_types in ['individual', 'both']:
                    individual_preprocessed = preprocessed_frames[start_idx - start_clip : end_idx - start_clip + 1]
                    individual_features = extract_individual_features(individual_preprocessed, individual_model, device, batch_size=args.feature_extraction_batch_size)
                    if individual_features.size == 0:
                        pbar.update(1)
                        continue
                    output_clip_dir.mkdir(parents=True, exist_ok=True)
                    np.save(individual_feature_path, individual_features)
                    logging.info(f"Saved individual features to {individual_feature_path} with shape {individual_features.shape}")

                # Extract sequence features for positions start_idx to end_idx
                if args.feature_types in ['sequence', 'both']:
                    sequence_features = extract_sequence_features(preprocessed_frames, start_clip, start_idx, end_idx, sequence_model, device, args.sequence_length, batch_size=args.feature_extraction_batch_size)
                    if sequence_features.size == 0:
                        pbar.update(1)
                        continue
                    output_clip_dir.mkdir(parents=True, exist_ok=True)
                    np.save(sequence_feature_path, sequence_features)
                    logging.info(f"Saved sequence features to {sequence_feature_path} with shape {sequence_features.shape}")

                pbar.update(1)

            cap.release()

    logging.info("Feature extraction complete!")

if __name__ == "__main__":
    main()