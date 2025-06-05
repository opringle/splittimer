import pandas as pd
import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import webbrowser
import logging
import numpy as np
import shutil

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
    parser = argparse.ArgumentParser(description="Inspect training data by generating an HTML page with random positive and negative samples.")
    parser.add_argument("csv_path", type=str, help="Path to the training data CSV")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of positive and negative samples to display")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the CSV file
    df = pd.read_csv(args.csv_path)

    # Filter positive and negative samples
    df_pos = df[df['label'] == 1.0]
    df_neg = df[df['label'] == 0.0]

    # Determine the number of samples to display
    num_pos = min(args.num_samples, len(df_pos))
    num_neg = min(args.num_samples, len(df_neg))

    # Sample random positive and negative samples
    pos_samples = df_pos.sample(n=num_pos)
    neg_samples = df_neg.sample(n=num_neg)

    # Combine samples
    samples = list(pos_samples.itertuples()) + list(neg_samples.itertuples())
    labels = ['Positive'] * num_pos + ['Negative'] * num_neg

    # Clean up and create temporary directory
    temp_dir = Path("training_data_inspection")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logging.info(f"Removed existing directory {temp_dir}")
    temp_dir.mkdir(exist_ok=True)
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Initialize HTML content
    html = """
    <html>
    <head>
    <title>Training Data Inspection</title>
    <style>
    .sample { margin-bottom: 20px; }
    .sample img { width: 45%; margin-right: 5%; }
    </style>
    </head>
    <body>
    <h1>Training Data Inspection</h1>
    """

    # Placeholder image for missing frames
    placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
    placeholder[:, :, 0] = 255  # Red square

    for i, (row, label_type) in enumerate(zip(samples, labels)):
        track_id = row.track_id
        v1_rider_id = row.v1_rider_id
        v2_rider_id = row.v2_rider_id
        v1_frame_idx = row.v1_frame_idx
        v2_frame_idx = row.v2_frame_idx

        # Construct video paths
        v1_path = Path("downloaded_videos") / track_id / v1_rider_id / f"{track_id}_{v1_rider_id}.mp4"
        v2_path = Path("downloaded_videos") / track_id / v2_rider_id / f"{track_id}_{v2_rider_id}.mp4"

        # Get frames or use placeholder
        try:
            frame1 = get_frame(v1_path, v1_frame_idx)
        except ValueError as e:
            logging.warning(f"Could not load frame for {v1_rider_id} frame {v1_frame_idx}: {e}")
            frame1 = placeholder

        try:
            frame2 = get_frame(v2_path, v2_frame_idx)
        except ValueError as e:
            logging.warning(f"Could not load frame for {v2_rider_id} frame {v2_frame_idx}: {e}")
            frame2 = placeholder

        # Save frames
        frame1_path = frames_dir / f"sample_{i}_v1.png"
        frame2_path = frames_dir / f"sample_{i}_v2.png"
        plt.imsave(frame1_path, frame1)
        plt.imsave(frame2_path, frame2)

        # Add to HTML
        html += f"""
        <div class="sample">
        <h3>{label_type} sample: {v1_rider_id} frame {v1_frame_idx} and {v2_rider_id} frame {v2_frame_idx}</h3>
        <img src="frames/sample_{i}_v1.png">
        <img src="frames/sample_{i}_v2.png">
        </div>
        """

    # Close HTML
    html += """
    </body>
    </html>
    """

    # Save HTML file
    html_path = temp_dir / "index.html"
    with open(html_path, "w") as f:
        f.write(html)

    # Open in default browser
    webbrowser.open(str(html_path))

    # Inform user
    logging.info(f"Inspection HTML generated at {html_path}. You can delete the directory '{temp_dir}' when done.")

if __name__ == "__main__":
    main()