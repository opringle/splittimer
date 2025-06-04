import cv2
import numpy as np
import os
from torchvision import transforms
import torch
import argparse
import yt_dlp
import re

def download_youtube_video(url, output_path):
    """Download a YouTube video to the specified output path."""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, 'temp_video.mp4'),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = os.path.join(output_path, 'temp_video.mp4')
            return video_path, info['title']
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, None

def preprocess_frame(frame, preprocess):
    """Preprocess a single video frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return preprocess(frame_rgb).numpy()

def process_and_save_clips(video_path, output_dir, preprocess, clip_duration_seconds=5):
    """Process the video into clips and save them as numpy arrays."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    assert video_fps == 25.0, f"Video fps {video_fps} is not 25.0, cannot process."
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = int(clip_duration_seconds * video_fps)
    
    clip_idx = 0
    frame_buffer = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = preprocess_frame(frame, preprocess)
        frame_buffer.append(processed_frame)
        
        if len(frame_buffer) >= frames_per_clip:
            clip_array = np.array(frame_buffer)
            clip_path = os.path.join(output_dir, f'clip_{clip_idx:04d}.npy')
            np.save(clip_path, clip_array)
            print(f"Saved clip {clip_idx} to {clip_path} with shape {clip_array.shape}")
            frame_buffer = []
            clip_idx += 1
        
        frame_idx += 1
    
    # handle the last clip
    if frame_buffer:
        clip_array = np.array(frame_buffer)
        clip_path = os.path.join(output_dir, f'clip_{clip_idx:04d}.npy')
        np.save(clip_path, clip_array)
        print(f"Saved clip {clip_idx} to {clip_path} with shape {clip_array.shape}")
    
    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess YouTube video for deep learning models.")
    parser.add_argument("url", type=str, help="YouTube video URL")
    parser.add_argument("--resolution", type=int, nargs=2, default=[224,224], help="Resolution for preprocessing, width and height")
    parser.add_argument("--output_dir", type=str, default="processed_clips", help="Output directory for processed clips")
    args = parser.parse_args()
    
    print(f"Using resolution: {args.resolution[0]}x{args.resolution[1]}")
    
    resize_size = tuple(args.resolution)
    preprocess = transforms.Compose([
        # Converting it to a PIL Image for compatibility with spatial transformations.
        transforms.ToPILImage(),
        # Resizing it to the model’s required dimensions.
        transforms.Resize(resize_size),
        # Transforming it into a PyTorch tensor for computation.
        transforms.ToTensor(),
        # Normalizing it to match the data distribution the model was trained on (e.g., ImageNet).
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Set up the video download directory
    video_output_path = "downloaded_videos"
    if not os.path.exists(video_output_path):
        os.makedirs(video_output_path)
    
    # Download the video to downloaded_videos directory
    video_path, title = download_youtube_video(args.url, video_output_path)
    
    if video_path and title:
        # Prepare the output directory for clips
        processed_title = re.sub(r'[^a-zA-Z0-9 ]', '', title).lower().replace(' ', '_')
        full_output_dir = os.path.join(args.output_dir, processed_title)
        print(f"Saving clips to {full_output_dir}")
        
        # Process the video and save clips
        process_and_save_clips(video_path, full_output_dir, preprocess)
        
        # Clean up the temporary video file
        os.remove(video_path)
        print("Processing complete!")
    else:
        print("Failed to download video")

if __name__ == "__main__":
    main()