import pandas as pd
import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def get_frame(video_path, frame_idx):
    """
    Extract a specific frame from the video file.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
    else:
        raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Inspect training data by displaying random positive and negative samples.")
    parser.add_argument("csv_path", type=str, help="Path to the training data CSV")
    parser.add_argument("--num_samples_per_class", type=int, default=3, help="Number of positive and negative samples to display")
    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(args.csv_path)

    # Filter positive and negative samples
    df_pos = df[df['label'] == 1.0]
    df_neg = df[df['label'] == 0.0]

    # Determine the number of samples to display (handle cases where there are fewer samples than requested)
    num_pos = min(args.num_samples_per_class, len(df_pos))
    num_neg = min(args.num_samples_per_class, len(df_neg))

    # Sample random positive and negative samples
    pos_samples = df_pos.sample(n=num_pos)
    neg_samples = df_neg.sample(n=num_neg)

    # Total number of rows for the plot
    total_rows = num_pos + num_neg

    # Create a figure with subplots: each row has two images (one for each frame in the pair)
    _, axs = plt.subplots(nrows=total_rows, ncols=2, figsize=(10, 5 * total_rows))

    # Combine positive and negative samples
    samples = list(pos_samples.itertuples()) + list(neg_samples.itertuples())
    labels = ['Positive'] * num_pos + ['Negative'] * num_neg

    for i, (row, label_type) in enumerate(zip(samples, labels)):
        track_id = row.track_id
        v1_rider_id = row.v1_rider_id
        v2_rider_id = row.v2_rider_id
        v1_frame_idx = row.v1_frame_idx
        v2_frame_idx = row.v2_frame_idx

        # Construct video paths
        v1_path = Path("downloaded_videos") / track_id / v1_rider_id / f"{track_id}_{v1_rider_id}.mp4"
        v2_path = Path("downloaded_videos") / track_id / v2_rider_id / f"{track_id}_{v2_rider_id}.mp4"

        # Extract frames
        frame1 = get_frame(v1_path, v1_frame_idx)
        frame2 = get_frame(v2_path, v2_frame_idx)

        # Plot the frames
        axs[i, 0].imshow(frame1)
        axs[i, 0].set_title(f"{label_type} sample: {v1_rider_id} frame {v1_frame_idx}")
        axs[i, 0].axis('off')
        axs[i, 1].imshow(frame2)
        axs[i, 1].set_title(f"{v2_rider_id} frame {v2_frame_idx}")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()