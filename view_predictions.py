import argparse
import json
import cv2
import numpy as np
from pathlib import Path
import logging

from utils import get_frame

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Display predicted split frames from source and target videos.")
    parser.add_argument('--predictions_json', required=True, help="Path to the predictions JSON file")
    parser.add_argument('--trackId', required=True, help="Track identifier")
    parser.add_argument('--sourceRiderId', required=True, help="Source rider identifier")
    parser.add_argument('--targetRiderId', required=True, help="Target rider identifier")
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
    predictions = load_predictions(args.predictions_json)
    source_video_path = get_video_path(args.trackId, args.sourceRiderId)
    target_video_path = get_video_path(args.trackId, args.targetRiderId)
    
    # Check if video files exist
    if not Path(source_video_path).exists():
        logging.error(f"Source video file {source_video_path} does not exist")
        exit(1)
    if not Path(target_video_path).exists():
        logging.error(f"Target video file {target_video_path} does not exist")
        exit(1)
    
    # Process each prediction
    for i, pred in enumerate(predictions):
        source_frame = get_frame(source_video_path, pred['source_end_idx'])
        target_frame = get_frame(target_video_path, pred['target_end_idx'])
        if source_frame is not None and target_frame is not None:
            labeled_source = add_label(source_frame, f"Source Split {i+1}: frame {pred['source_end_idx']}")
            labeled_target = add_label(target_frame, f"Target Prediction {i+1}: frame {pred['target_end_idx']}")
            pair = np.hstack([labeled_source, labeled_target])
            cv2.imshow(f"Split {i+1}", pair)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Split {i+1}")

if __name__ == "__main__":
    main()