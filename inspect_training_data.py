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
from tqdm import tqdm

from utils import get_frame, get_video_file_path, setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Inspect training data by generating an HTML page with random samples per sample type.")
    parser.add_argument("csv_path", type=str,
                        help="Path to the training data CSV")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to display per sample type")
    parser.add_argument("--sample_types", type=str, nargs='*',
                        help="Optional list of sample types to include (e.g., type1 type2)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    df = pd.read_csv(args.csv_path)

    assert 'sample_type' in df.columns, "CSV file does not contain 'sample_type' column"

    # Filter by sample_types if provided
    if args.sample_types:
        df = df[df['sample_type'].isin(args.sample_types)]
        if df.empty:
            logging.error(
                f"No samples found for specified sample types: {args.sample_types}")
            return

    # Sample N samples per sample_type (no grouping by label)
    sampled_df = df.groupby('sample_type').apply(lambda x: x.sample(
        min(len(x), args.num_samples)), include_groups=False).reset_index()
    sampled_df = sampled_df.sort_values('sample_type')

    # Clean up and create temporary directory with error handling
    temp_dir = Path("training_data_inspection")
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logging.info(f"Removed existing directory {temp_dir}")
    except Exception as e:
        logging.error(f"Failed to remove existing directory {temp_dir}: {e}")
        return
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

    # Initialize sample index
    sample_idx = 0

    # Process samples with progress bar
    current_sample_type = None
    for row in tqdm(sampled_df.itertuples(), total=len(sampled_df), desc="Processing samples"):
        if row.sample_type != current_sample_type:
            html += f"<h2>Sample Type: {row.sample_type}</h2>"
            current_sample_type = row.sample_type

        track_id = row.track_id
        v1_rider_id = row.v1_rider_id
        v2_rider_id = row.v2_rider_id
        v1_frame_idx = row.v1_frame_idx
        v2_frame_idx = row.v2_frame_idx
        sample_type = row.sample_type
        label = row.label

        v1_path = get_video_file_path(track_id, v1_rider_id)
        v2_path = get_video_file_path(track_id, v2_rider_id)

        frame1 = get_frame(v1_path, v1_frame_idx)
        frame2 = get_frame(v2_path, v2_frame_idx)

        frame1_path = frames_dir / f"sample_{sample_idx}_v1.png"
        frame2_path = frames_dir / f"sample_{sample_idx}_v2.png"
        plt.imsave(frame1_path, frame1)
        plt.imsave(frame2_path, frame2)

        # Add to HTML with track_id and sample_type
        html += f"""
        <div class="sample">
        <h3>Label: {label}, Track ID: {track_id}, Sample Type: {sample_type}, {v1_rider_id} frame {v1_frame_idx} and {v2_rider_id} frame {v2_frame_idx}</h3>
        <img src="frames/sample_{sample_idx}_v1.png">
        <img src="frames/sample_{sample_idx}_v2.png">
        </div>
        """
        sample_idx += 1

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
    logging.info(
        f"Inspection HTML generated at {html_path}. You can delete the directory '{temp_dir}' when done.")


if __name__ == "__main__":
    main()
