import cv2
import numpy as np
import os
from torchvision import transforms
import torch
import argparse
import yt_dlp
import re

def download_youtube_video(url, output_path):
    """Download a YouTube video to the specified output path using the video title."""
    try:
        with yt_dlp.YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False)  # Get info without downloading
            title = info['title']
            processed_title = re.sub(r'[^a-zA-Z0-9 ]', '', title).lower().replace(' ', '_')
            video_path = os.path.join(output_path, f'{processed_title}.mp4')
            
            if os.path.exists(video_path):
                print(f"Video file already exists at {video_path}, skipping download.")
                return video_path, processed_title
            
            # Updated options for maximum quality
            ydl_opts = {
                'format': 'bestvideo+bestaudio/best',  # Select highest quality video and audio
                'merge_output_format': 'mp4',          # Merge into MP4 for compatibility
                'outtmpl': video_path,                 # Output file path
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                return video_path, processed_title
    except Exception as e:
        print(f"Error downloading video or retrieving metadata: {e}")
        return None, None
    
def preprocess_frame(frame, preprocess):
    """Preprocess a single video frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return preprocess(frame_rgb).numpy()

def process_and_save_clips(video_path, output_dir, preprocess, clip_duration_seconds=5, split_indices=None):
    """Process the video into clips and save them as numpy arrays with corresponding label files."""
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
    
    if split_indices is None:
        split_indices = []
    split_set = set(split_indices)
    print(f"Split indices: {split_indices}")
    
    clip_idx = 0
    frame_buffer = []
    label_buffer = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = preprocess_frame(frame, preprocess)
        frame_buffer.append(processed_frame)
        
        label = 1.0 if frame_idx in split_set else 0.0
        label_buffer.append(label)
        
        if len(frame_buffer) >= frames_per_clip:
            clip_array = np.array(frame_buffer[:frames_per_clip])
            label_array = np.array(label_buffer[:frames_per_clip], dtype=np.float32)
            
            clip_path = os.path.join(output_dir, f'{clip_idx:04d}_x.npy')
            label_path = os.path.join(output_dir, f'{clip_idx:04d}_y.npy')
            
            np.save(clip_path, clip_array)
            np.save(label_path, label_array)
            
            print(f"Saved clip {clip_idx} to {clip_path} with shape {clip_array.shape}")
            print(f"Saved labels {clip_idx} to {label_path} with shape {label_array.shape}")
            
            frame_buffer = frame_buffer[frames_per_clip:]
            label_buffer = label_buffer[frames_per_clip:]
            clip_idx += 1
        
        frame_idx += 1
    
    if frame_buffer:
        clip_array = np.array(frame_buffer)
        label_array = np.array(label_buffer, dtype=np.float32)
        
        clip_path = os.path.join(output_dir, f'{clip_idx:04d}_x.npy')
        label_path = os.path.join(output_dir, f'{clip_idx:04d}_y.npy')
        
        np.save(clip_path, clip_array)
        np.save(label_path, label_array)
        
        print(f"Saved clip {clip_idx} to {clip_path} with shape {clip_array.shape}")
        print(f"Saved labels {clip_idx} to {label_path} with shape {label_array.shape}")
    
    cap.release()

def timecode_to_frames(timecode, fps):
    """Convert a timecode in HH:MM:SS:FF format to a frame index based on fps."""
    parts = timecode.split(':')
    if len(parts) != 3:
        raise ValueError(f"Timecode must be in MM:SS:FF format but got '{timecode}'")
    MM, SS, FF = map(int, parts)
    total_frames = (MM * 60 + SS) * int(fps) + FF
    return total_frames

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess YouTube video for deep learning models.")
    parser.add_argument("url", type=str, help="YouTube video URL")
    parser.add_argument("--resolution", type=int, nargs=2, default=[224,224], help="Resolution for preprocessing, width and height")
    parser.add_argument("--output_dir", type=str, default="processed_clips", help="Output directory for processed clips")
    parser.add_argument("--split-times", type=str, nargs='*', default=[], help="Timestamps in MM:SS:FF format to mark as split points (labeled 1.0)")
    parser.add_argument("--keep-video", action="store_true", help="Keep the downloaded video file after processing")
    args = parser.parse_args()
    
    print(f"Using resolution: {args.resolution[0]}x{args.resolution[1]}")
    print(f"Split timestamps: {args.split_times}")
    print(f"Keep video file: {args.keep_video}")
    
    resize_size = tuple(args.resolution)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    video_output_path = "downloaded_videos"
    if not os.path.exists(video_output_path):
        os.makedirs(video_output_path)
    
    video_path, title = download_youtube_video(args.url, video_output_path)
    print(f"title={title}")
    
    if video_path and title:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file for FPS check")
            return
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        assert video_fps == 25.0, f"Video fps {video_fps} is not 25.0, cannot process."
        cap.release()
        
        try:
            split_indices = [timecode_to_frames(tc, video_fps) for tc in args.split_times]
        except ValueError as e:
            print(f"Error in split_times: {e}")
            return
        
        full_output_dir = os.path.join(args.output_dir, title)
        print(f"Saving clips and labels to {full_output_dir}")
        
        process_and_save_clips(video_path, full_output_dir, preprocess, split_indices=split_indices)
        
        if not args.keep_video:
            os.remove(video_path)
            print(f"Deleted video file: {video_path}")
        else:
            print(f"Kept video file at: {video_path}")
        
        print("Processing complete!")
    else:
        print("Failed to download video or retrieve metadata")

if __name__ == "__main__":
    main()