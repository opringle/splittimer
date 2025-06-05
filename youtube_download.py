import os
import argparse
import yt_dlp
import yaml
from pathlib import Path
import logging

def download_youtube_video(url, output_path, track_id, rider_id):
    """
    Download a YouTube video using track_id and rider_id for naming.
    Saves the video in the format: downloaded_videos/$trackid/$riderid/trackid_riderid.mp4
    """
    video_dir = Path(output_path) / track_id / rider_id
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f'{track_id}_{rider_id}.mp4'
    
    if video_path.exists():
        logging.info(f"Video file already exists at {video_path}, skipping download.")
        return str(video_path)
    try:
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'outtmpl': str(video_path),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logging.info(f"Downloaded video to {video_path}")
        return str(video_path)
    except Exception as e:
        logging.error(f"Error downloading video: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download YouTube videos from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    args = parser.parse_args()
    
    # Configure logging with the specified level
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    # Load the YAML config with error handling
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return
    
    videos = config['videos']
    
    video_output_path = "downloaded_videos"
    Path(video_output_path).mkdir(exist_ok=True)
    
    for video in videos:
        rider_id = video['riderId']
        track_id = video['trackId']
        url = video['url']
        
        logging.info(f"Processing video: {track_id}/{rider_id}")
        video_path = download_youtube_video(url, video_output_path, track_id, rider_id)
        if video_path is None:
            logging.warning(f"Failed to download video for {track_id}/{rider_id}")
    
    logging.info("Download complete!")

if __name__ == "__main__":
    main()