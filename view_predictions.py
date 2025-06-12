import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import logging
import shutil

from utils import get_frame

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate an HTML page to display predicted split frames from source and target videos.")
    parser.add_argument('--predictions_json', required=True, help="Path to the predictions JSON file containing trackId, sourceRiderId, targetRiderId, and predicted_splits")
    return parser.parse_args()

def load_predictions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def get_video_path(trackId, riderId):
    return f"downloaded_videos/{trackId}/{riderId}/{trackId}_{riderId}.mp4"

def add_label(frame, text):
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White text
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2  # Center horizontally
    text_y = height - 10  # Near bottom
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    return frame

def main():
    args = parse_args()
    json_path = Path(args.predictions_json)
    predictions_data = load_predictions(json_path)
    
    # Extract metadata from JSON
    trackId = predictions_data.get('trackId')
    sourceRiderId = predictions_data.get('sourceRiderId')
    targetRiderId = predictions_data.get('targetRiderId')
    predicted_splits = predictions_data.get('predicted_splits', [])
    
    # Check if required keys are present
    if not all([trackId, sourceRiderId, targetRiderId]):
        logging.error("Missing required keys in predictions JSON: trackId, sourceRiderId, targetRiderId")
        exit(1)
    
    if not isinstance(predicted_splits, list):
        logging.error("predicted_splits is not a list or is missing in the JSON")
        exit(1)
    
    if not predicted_splits:
        logging.warning("No predicted splits found in the JSON file")
        exit(0)
    
    # Create directory for images and clean up if it already exists
    images_dir = json_path.parent / (json_path.stem + "_images")
    if images_dir.exists():
        shutil.rmtree(images_dir)  # Remove existing directory and its contents
    images_dir.mkdir()  # Create a new empty directory
    
    source_video_path = get_video_path(trackId, sourceRiderId)
    target_video_path = get_video_path(trackId, targetRiderId)
    
    # Check if video files exist
    if not Path(source_video_path).exists():
        logging.error(f"Source video file {source_video_path} does not exist")
        exit(1)
    if not Path(target_video_path).exists():
        logging.error(f"Target video file {target_video_path} does not exist")
        exit(1)
    
    html_snippets = []
    
    # Process each prediction
    for i, pred in enumerate(predicted_splits):
        source_frame = get_frame(source_video_path, pred['source_end_idx'])
        target_frame = get_frame(target_video_path, pred['target_end_idx'])
        if source_frame is None or target_frame is None:
            logging.warning(f"Could not retrieve frames for split {i+1}")
            continue
        labeled_source = add_label(source_frame, f"Source Split {i+1}: frame {pred['source_end_idx']}")
        labeled_target = add_label(target_frame, f"Target Prediction {i+1}: frame {pred['target_end_idx']}")
        pair = np.hstack([labeled_source, labeled_target])
        image_path = images_dir / f"split_{i+1}.png"
        cv2.imwrite(str(image_path), pair)
        
        caption = f"Source Frame: {pred['source_end_idx']}, Target Frame: {pred['target_end_idx']}"
        if 'confidence' in pred:
            caption += f", Confidence: {pred['confidence']:.2f}"
        
        snippet = f"""
        <div class="split-container">
            <img src="{images_dir.name}/split_{i+1}.png" alt="Split {i+1}" class="split-image">
            <p class="caption">{caption}</p>
        </div>
        """
        html_snippets.append(snippet)
    
    if html_snippets:
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
            <h1>Predicted Splits for Track: {trackId}, Source: {sourceRiderId}, Target: {targetRiderId}</h1>
            {''.join(html_snippets)}
        </body>
        </html>
        """
        html_path = json_path.parent / (json_path.stem + "_splits.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        logging.info(f"Saved HTML file to {html_path}")
    else:
        logging.warning("No splits to display")

if __name__ == "__main__":
    main()