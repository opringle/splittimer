from pathlib import Path
import yaml
from collections import defaultdict


class Config:
    def __init__(self, config_path):
        """
        Initialize the Config class by loading the YAML file from the given path.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.videos = self.config["videos"]

    def get_video_path(self, video):
        """
        Generate the file path for a given video entry.

        Args:
            video (dict): A dictionary containing 'trackId' and 'riderId'.

        Returns:
            Path: The constructed video file path.
        """
        track_id = video["trackId"]
        rider_id = video["riderId"]
        return Path("downloaded_videos") / track_id / rider_id / f"{track_id}_{rider_id}.mp4"

    def get_videos_by_track(self):
        """
        Group all videos by their track ID.

        Returns:
            defaultdict: A dictionary mapping track IDs to lists of video entries.
        """
        track_videos = defaultdict(list)
        for video in self.videos:
            track_videos[video["trackId"]].append(video)
        return track_videos

    def get_unique_track_ids(self):
        return list(set(video["trackId"] for video in self.videos))
