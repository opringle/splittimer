import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import logging
import shutil

from utils import get_frame, get_video_file_path, get_video_fps_and_total_frames, timecode_to_frames

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an HTML page to display predicted split frames from source and target videos.")
    parser.add_argument('--predictions_json', required=True,
                        help="Path to the predictions JSON file containing trackId, sourceRiderId, sourceTimecodes, and predictions")
    return parser.parse_args()


def load_predictions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def add_label(frame, text):
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White text
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2  # Center horizontally
    text_y = height - 10  # Near bottom
    cv2.putText(frame, text, (text_x, text_y),
                font, font_scale, color, thickness)
    return frame


def main():
    args = parse_args()
    json_path = Path(args.predictions_json)
    predictions_data = load_predictions(json_path)

    # Extract metadata from JSON
    trackId = predictions_data.get('trackId')
    sourceRiderId = predictions_data.get('sourceRiderId')
    sourceTimecodes = predictions_data.get('sourceTimecodes', [])
    predictions = predictions_data.get('predictions', {})

    # Check if required keys are present
    if not all([trackId, sourceRiderId, sourceTimecodes, predictions]):
        logging.error(
            "Missing required keys in predictions JSON: trackId, sourceRiderId, sourceTimecodes, predictions")
        exit(1)

    if not isinstance(sourceTimecodes, list):
        logging.error(
            "sourceTimecodes is not a list or is missing in the JSON")
        exit(1)

    if not sourceTimecodes:
        logging.warning("No source timecodes found in the JSON file")
        exit(0)

    # Create directory for images and clean up if it already exists
    images_dir = json_path.parent / (json_path.stem + "_images")
    if images_dir.exists():
        shutil.rmtree(images_dir)  # Remove existing directory and its contents
    images_dir.mkdir()  # Create a new empty directory

    # Get source video path and FPS
    source_video_path = get_video_file_path(trackId, sourceRiderId)
    source_fps, _ = get_video_fps_and_total_frames(source_video_path)

    if not Path(source_video_path).exists():
        logging.error(f"Source video file {source_video_path} does not exist")
        exit(1)

    html_sections = []

    # Process each target rider
    for target_rider_id, predicted_timecodes in predictions.items():
        if not isinstance(predicted_timecodes, list) or len(predicted_timecodes) != len(sourceTimecodes):
            logging.warning(
                f"Invalid or mismatched predicted timecodes for target rider {target_rider_id}")
            continue

        target_video_path = get_video_file_path(trackId, target_rider_id)
        target_fps, _ = get_video_fps_and_total_frames(target_video_path)

        if not Path(target_video_path).exists():
            logging.warning(
                f"Target video file {target_video_path} does not exist for rider {target_rider_id}")
            continue

        snippets = []
        for i, (source_timecode, predicted_timecode) in enumerate(zip(sourceTimecodes, predicted_timecodes)):
            if predicted_timecode is None:
                continue  # Skip if no prediction
            try:
                source_frame_idx = timecode_to_frames(
                    source_timecode, source_fps)
                target_frame_idx = timecode_to_frames(
                    predicted_timecode, target_fps)
                source_frame = get_frame(source_video_path, source_frame_idx)
                target_frame = get_frame(target_video_path, target_frame_idx)
                if source_frame is None or target_frame is None:
                    logging.warning(
                        f"Could not retrieve frames for split {i+1} for rider {target_rider_id}")
                    continue
                labeled_source = add_label(
                    source_frame, f"Source: {source_timecode}")
                labeled_target = add_label(
                    target_frame, f"Target: {predicted_timecode}")
                pair = np.hstack([labeled_source, labeled_target])
                image_path = images_dir / f"split_{target_rider_id}_{i+1}.png"
                cv2.imwrite(str(image_path), pair)

                caption = f"Source Timecode: {source_timecode}, Predicted Target Timecode: {predicted_timecode}"
                snippet = f"""
                <div class="split-container">
                    <img src="{images_dir.name}/split_{target_rider_id}_{i+1}.png" alt="Split {i+1}" class="split-image">
                    <p class="caption">{caption}</p>
                </div>
                """
                snippets.append(snippet)
            except Exception as e:
                logging.error(
                    f"Error processing split {i+1} for rider {target_rider_id}: {e}")

        if snippets:
            section = f"""
            <h2>Target Rider: {target_rider_id}</h2>
            {''.join(snippets)}
            """
            html_sections.append(section)

    if html_sections:
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Predicted Splits</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    text-align: center;
                }}
                .split-container {{
                    margin: 20px 0;
                }}
                .split-image {{
                    max-width: 80%;
                    height: auto;
                }}
                .caption {{
                    margin-top: 10px;
                    font-size: 16px;
                }}
            </style>
        </head>
        <body>
            <h1>Predicted Splits for Track: {trackId}, Source: {sourceRiderId}</h1>
            {''.join(html_sections)}
        </body>
        </html>
        """
        html_path = json_path.parent / (json_path.stem + "_splits.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        logging.info(f"Saved HTML file to {html_path}")
    else:
        logging.warning("No valid splits to display")


if __name__ == "__main__":
    main()
