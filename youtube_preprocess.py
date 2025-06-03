import cv2
import numpy as np
from pytube import YouTube
import os
from torchvision import transforms
import torch
import argparse

def download_youtube_video(url, output_path):
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not stream:
            raise Exception("No suitable video stream found")
        stream.download(output_path=output_path, filename='temp_video.mp4')
        return os.path.join(output_path, 'temp_video.mp4')
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def preprocess_frame(frame, preprocess):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return preprocess(frame_rgb).numpy()

def process_and_save_clips(video_path, output_dir, preprocess, clip_duration=5, fps=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps
    frames_per_clip = int(clip_duration * fps)
    
    clip_idx = 0
    frame_buffer = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % int(video_fps / fps) == 0:  # Sample frames to match target fps
            processed_frame = preprocess_frame(frame, preprocess)
            frame_buffer.append(processed_frame)
        
        if len(frame_buffer) >= frames_per_clip:
            clip_array = np.array(frame_buffer)
            clip_path = os.path.join(output_dir, f'clip_{clip_idx:04d}.npy')
            np.save(clip_path, clip_array)
            print(f"Saved clip {clip_idx} to {clip_path}")
            frame_buffer = []
            clip_idx += 1
        
        frame_idx += 1
    
    # Save remaining frames as a final clip
    if frame_buffer:
        clip_array = np.array(frame_buffer)
        clip_path = os.path.join(output_dir, f'clip_{clip_idx:04d}.npy')
        np.save(clip_path, clip_array)
        print(f"Saved clip {clip_idx} to {clip_path}")
    
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
        transforms.ToPILImage(),
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    video_path = download_youtube_video(args.url, ".")
    
    if video_path:
        print("Processing video...")
        process_and_save_clips(video_path, args.output_dir, preprocess)
        os.remove(video_path)  # Clean up temporary video file
        print("Processing complete!")
    else:
        print("Failed to download video")

if __name__ == "__main__":
    main()