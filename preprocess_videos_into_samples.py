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

logging.basicConfig(level=logging.INFO)

resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
resnet50.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = preprocess(frame_rgb).unsqueeze(0)
    with torch.no_grad():
        features = resnet50(frame_tensor).squeeze().numpy()
    return features

def get_clip(video_path, central_idx, F, total_frames):
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
    parser = argparse.ArgumentParser(description="Generate batched training data.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file")
    parser.add_argument("output_dir", type=str, help="Directory to save .npz files")
    parser.add_argument("--F", type=int, default=50, help="Number of frames per clip")
    parser.add_argument("--batch_size", type=int, default=128, help="Samples per .npz file")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_clip1s, batch_clip2s, batch_labels = [], [], []
    batch_count = 0

    for row in tqdm(df.itertuples(), total=len(df), desc="Generating samples"):
        v1_path = Path("downloaded_videos") / row.track_id / row.v1_rider_id / f"{row.track_id}_{row.v1_rider_id}.mp4"
        v2_path = Path("downloaded_videos") / row.track_id / row.v2_rider_id / f"{row.track_id}_{row.v2_rider_id}.mp4"

        cap1 = cv2.VideoCapture(str(v1_path))
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        cap1.release()
        cap2 = cv2.VideoCapture(str(v2_path))
        total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        cap2.release()

        v1_frames = get_clip(v1_path, row.v1_frame_idx, args.F, total_frames1)
        v2_frames = get_clip(v2_path, row.v2_frame_idx, args.F, total_frames2)

        v1_features = [np.append(extract_features(frame), float(pos)) for pos, frame in enumerate(v1_frames)]
        v1_clip = np.stack(v1_features, axis=0)
        v2_features = [np.append(extract_features(frame), float(pos)) for pos, frame in enumerate(v2_frames)]
        v2_clip = np.stack(v2_features, axis=0)

        batch_clip1s.append(v1_clip)
        batch_clip2s.append(v2_clip)
        batch_labels.append(row.label)

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