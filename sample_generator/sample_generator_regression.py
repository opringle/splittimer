from pathlib import Path
from typing import Sequence
import numpy as np
from .interface import SampleGenerator
from utils import get_clip_indices_ending_at, get_video_file_path, get_video_fps_and_total_frames, get_video_metadata, load_image_features_from_disk


class RegressionSampleGenerator(SampleGenerator):
    @classmethod
    def add_args(cls, parser):
        """Add command-line arguments specific to this preprocessor."""
        parser.add_argument("--F", type=int, default=50,
                            help="Number of frames per clip (for individual features)")
        parser.add_argument("--add_position_feature", action='store_true',
                            help="Add a feature to each sample for end index")
        parser.add_argument("--add_percent_completion_feature", action='store_true',
                            help="Add a feature to each sample for % completion")

    @staticmethod
    def from_args(args):
        """Create an instance from parsed arguments"""
        return RegressionSampleGenerator(args)

    def __init__(self, args):
        """Initialize the sample generator with arguments."""
        self.args = args
        self.samples = []

    def _compute_and_cache_video_features(self, video_path: str, feature_cache: dict):
        _, total_frames = get_video_fps_and_total_frames(video_path)
        frame_idx_array = np.arange(total_frames, dtype=np.float32)
        percent_through_clip_array = frame_idx_array / total_frames
        feature_cache[video_path] = {
            "frame_idx_array": frame_idx_array, 'percent_through_clip_array': percent_through_clip_array}

    def _add_frame_index_feature(self, video_path: str, feature_cache_dict, start_frame_idx: int, end_frame_idx: int, features):
        clip_indices = feature_cache_dict[video_path]["frame_idx_array"][start_frame_idx:end_frame_idx+1]
        return np.concatenate([features, clip_indices[:, None]], axis=1)

    def _add_percent_through_video_feature(self, video_path: str, feature_cache_dict, start_frame_idx: int, end_frame_idx: int, features):
        percent_through_video = feature_cache_dict[video_path][
            "percent_through_clip_array"][start_frame_idx:end_frame_idx+1]
        return np.concatenate([features, percent_through_video[:, None]], axis=1)

    def get_features(self, video_feature_cache: dict, **kwargs) -> Sequence[np.ndarray]:
        track_id = kwargs["track_id"]
        v1_rider_id = kwargs["v1_rider_id"]
        v2_rider_id = kwargs["v2_rider_id"]
        v1_frame_idx = kwargs["v1_frame_idx"]
        v2_frame_idx = kwargs["v2_frame_idx"]
        label = np.array(kwargs["label"])

        video_path_v1 = get_video_file_path(track_id, v1_rider_id)
        video_path_v2 = get_video_file_path(track_id, v2_rider_id)

        if video_path_v1 not in video_feature_cache:
            self._compute_and_cache_video_features(
                video_path_v1, video_feature_cache)

        if video_path_v2 not in video_feature_cache:
            self._compute_and_cache_video_features(
                video_path_v2, video_feature_cache)

        # Load individual frame features for a sequence of frames
        v1_indices = get_clip_indices_ending_at(v1_frame_idx, self.args.F)
        v2_indices = get_clip_indices_ending_at(v2_frame_idx, self.args.F)

        v1_start_idx, v1_end_idx = v1_indices[0], v1_indices[-1]
        v2_start_idx, v2_end_idx = v2_indices[0], v2_indices[-1]

        v1_features = load_image_features_from_disk(
            track_id, v1_rider_id, v1_start_idx, v1_end_idx, self.args.image_feature_path)

        v2_features = load_image_features_from_disk(
            track_id, v2_rider_id, v2_start_idx, v2_end_idx, self.args.image_feature_path)

        if self.args.add_position_feature:
            v1_features = self._add_frame_index_feature(
                video_path_v1, video_feature_cache, start_frame_idx=v1_start_idx, end_frame_idx=v1_end_idx, features=v1_features)
            v2_features = self._add_frame_index_feature(
                video_path_v2, video_feature_cache, start_frame_idx=v2_start_idx, end_frame_idx=v2_end_idx, features=v2_features)

        if self.args.add_percent_completion_feature:
            v1_features = self._add_percent_through_video_feature(
                video_path_v1, video_feature_cache, start_frame_idx=v1_start_idx, end_frame_idx=v1_end_idx, features=v1_features)
            v2_features = self._add_percent_through_video_feature(
                video_path_v2, video_feature_cache, start_frame_idx=v2_start_idx, end_frame_idx=v2_end_idx, features=v2_features)

        return v1_features, v2_features, label
