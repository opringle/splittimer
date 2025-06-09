import yaml
import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
import webbrowser
import logging
import numpy as np
import shutil
from datetime import datetime

from utils import get_frame, timecode_to_frames

def main():
    parser = argparse.ArgumentParser(description="Inspect split times by generating an HTML page with frames for each track and rider.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load the YAML configuration file
    try:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config file {args.config_path}: {e}")
        return

    # Validate config
    if not config.get('videos'):
        logging.error("Config file does not contain 'videos' key")
        return

    # Clean up and create temporary directory
    temp_dir = Path("split_times_inspection")
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
    <title>Split Times Inspection</title>
    <style>
    .track { margin-bottom: 40px; }
    .rider { margin-bottom: 20px; }
    .split { display: inline-block; margin-right: 10px; text-align: center; }
    .split img { width: 200px; height: auto; }
    .split p { margin: 5px 0; }
    </style>
    </head>
    <body>
    <h1>Split Times Inspection</h1>
    """

    # Placeholder image for missing frames
    placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
    placeholder[:, :, 0] = 255  # Red square

    # Group videos by trackId
    tracks = {}
    for video in config['videos']:
        track_id = video['trackId']
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append(video)

    # Initialize frame counter for unique file names
    frame_counter = 0

    # Process each track
    for track_id in sorted(tracks.keys()):
        html += f"<h2>Track: {track_id}</h2>"
        for video in tracks[track_id]:
            rider_id = video['riderId']
            splits = video['splits']
            html += f"<h3>Rider: {rider_id}</h3><div class='rider'>"
            
            for split_idx, split_timecode in enumerate(splits, 1):
                # Construct video path
                video_path = Path("downloaded_videos") / track_id / rider_id / f"{track_id}_{rider_id}.mp4"

                # Convert split time to frames
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video {video_path}")
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_idx = timecode_to_frames(split_timecode, fps)    
                
                # Get frame or use placeholder
                try:
                    frame = get_frame(video_path, frame_idx)
                except ValueError as e:
                    logging.warning(f"Could not load frame for {rider_id} at {split_timecode}: {e}")
                    frame = placeholder
                
                # Save frame with unique counter
                frame_path = frames_dir / f"frame_{frame_counter}.png"
                plt.imsave(frame_path, frame)
                
                # Add to HTML with unique counter
                html += f"""
                <div class='split'>
                <p>Split {split_idx}: {split_timecode}</p>
                <img src='frames/frame_{frame_counter}.png'>
                </div>
                """
                frame_counter += 1
            
            html += "</div>"

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