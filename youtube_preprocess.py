import cv2
import numpy as np
import os
from torchvision import transforms
import argparse
import yt_dlp
import yaml

def download_youtube_video(url, output_path, track_id, rider_id):
    """Download a YouTube video using track_id and rider_id for naming."""
    video_path = os.path.join(output_path, f'{track_id}_{rider_id}.mp4')
    if os.path.exists(video_path):
        print(f"Video file already exists at {video_path}, skipping download.")
        return video_path
    try:
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': video_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return video_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def preprocess_frame(frame, preprocess):
    """Preprocess a single video frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return preprocess(frame_rgb).numpy()

def process_and_save_clips(video_path, output_dir, preprocess, clip_duration_seconds=5, split_indices=None):
    """Process the video into clips and save them with frame-range naming."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    assert video_fps == 25.0, f"Video fps {video_fps} is not 25.0"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = int(clip_duration_seconds * video_fps)
    
    if split_indices is None:
        split_indices = []
    split_set = set(split_indices)
    print(f"Split indices: {split_indices}")
    
    clip_start = 0
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
            
            starting_frame = clip_start
            ending_frame = clip_start + frames_per_clip - 1
            clip_path = os.path.join(output_dir, f'{starting_frame:06d}_to_{ending_frame:06d}_x.npy')
            label_path = os.path.join(output_dir, f'{starting_frame:06d}_to_{ending_frame:06d}_y.npy')
            
            np.save(clip_path, clip_array)
            np.save(label_path, label_array)
            print(f"Saved clip from {starting_frame:06d} to {ending_frame:06d} to {clip_path}")
            
            clip_start += frames_per_clip
            frame_buffer = frame_buffer[frames_per_clip:]
            label_buffer = label_buffer[frames_per_clip:]
        
        frame_idx += 1
    
    if frame_buffer:
        clip_array = np.array(frame_buffer)
        label_array = np.array(label_buffer, dtype=np.float32)
        
        starting_frame = clip_start
        ending_frame = clip_start + len(clip_array) - 1
        clip_path = os.path.join(output_dir, f'{starting_frame:06d}_to_{ending_frame:06d}_x.npy')
        label_path = os.path.join(output_dir, f'{starting_frame:06d}_to_{ending_frame:06d}_y.npy')
        
        np.save(clip_path, clip_array)
        np.save(label_path, label_array)
        print(f"Saved clip from {starting_frame:06d} to {ending_frame:06d} to {clip_path}")
    
    cap.release()

def timecode_to_frames(timecode, fps):
    """Convert MM:SS:FF timecode to frame index."""
    parts = timecode.split(':')
    if len(parts) != 3:
        raise ValueError(f"Timecode must be in MM:SS:FF format, got '{timecode}'")
    MM, SS, FF = map(int, parts)
    return (MM * 60 + SS) * int(fps) + FF

def main():
    parser = argparse.ArgumentParser(description="Process YouTube videos from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resolution", type=int, nargs=2, default=[224, 224], help="Resolution for preprocessing")
    parser.add_argument("--output_dir", type=str, default="processed_clips", help="Base output directory")
    parser.add_argument("--keep-video", action="store_true", help="Keep downloaded video files")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    videos = config['videos']
    
    resize_size = tuple(args.resolution)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    video_output_path = "downloaded_videos"
    os.makedirs(video_output_path, exist_ok=True)
    
    for video in videos:
        rider_id = video['riderId']
        track_id = video['trackId']
        url = video['url']
        splits = video['splits']
        
        print(f"\nProcessing video: {track_id}/{rider_id}")
        video_path = download_youtube_video(url, video_output_path, track_id, rider_id)
        
        if not video_path:
            print(f"Failed to download video for {track_id}/{rider_id}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            continue
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        assert video_fps == 25.0, f"Video fps {video_fps} is not 25.0 for {track_id}/{rider_id}"
        cap.release()
        
        try:
            split_indices = [timecode_to_frames(tc, video_fps) for tc in splits]
        except ValueError as e:
            print(f"Error in splits for {track_id}/{rider_id}: {e}")
            continue
        
        full_output_dir = os.path.join(args.output_dir, track_id, rider_id)
        print(f"Saving clips to {full_output_dir}")
        process_and_save_clips(video_path, full_output_dir, preprocess, split_indices=split_indices)
        
        if not args.keep_video:
            os.remove(video_path)
            print(f"Deleted video file: {video_path}")
        else:
            print(f"Kept video file at: {video_path}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()