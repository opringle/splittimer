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

    def get_trackid_to_video_metadata(self):
        """
        Group all videos by their track ID.

        Returns:
            defaultdict: A dictionary mapping track IDs to lists of video entries.
        """
        trackid_to_video_metadatas = defaultdict(list)
        for video_metadata in self.videos:
            trackid_to_video_metadatas[video_metadata["trackId"]].append(
                video_metadata)
        return trackid_to_video_metadatas

    def get_unique_track_ids(self):
        return list(set(video["trackId"] for video in self.videos))
