from typing import Set
import numpy as np
import random
import itertools
import logging
import pandas as pd
from .interface import Preprocessor, SplitType
from utils import get_video_file_path, get_video_metadata


class RegressionPreprocessor(Preprocessor):
    @classmethod
    def add_args(cls, parser):
        """Add command-line arguments specific to this preprocessor."""
        parser.add_argument("--clip-length", type=int, required=True,
                            help="Number of frames to use per clip")
        parser.add_argument("--alpha_split_0", type=float, default=0.5,
                            help="Beta distribution alpha value at split 0")
        parser.add_argument("--alpha", type=float, default=0.5,
                            help="Beta distribution alpha value at split > 0")
        parser.add_argument("--beta_split_0", type=float, default=0.5,
                            help="Beta distribution beta value at split 0")
        parser.add_argument("--beta", type=float, default=0.5,
                            help="Beta distribution beta value at split > 0")
        parser.add_argument("--num_augmented_positives_per_segment", type=int,
                            default=50, help="Number of augmented samples per segment")
        parser.add_argument("--num_non_overlapping_samples_per_positive", type=int,
                            default=5, help="Number of non-overlapping samples per positive sample")
        parser.add_argument("--ignore_first_split", action="store_true",
                            help="Ignore the first split when generating training data")

    @staticmethod
    def from_args(args, config):
        """Create an instance from parsed arguments, config, and track-to-set mapping."""
        return RegressionPreprocessor(args, config)

    def __init__(self, args, config):
        """Initialize the preprocessor with arguments, configuration, and track assignments."""
        self.args = args
        self.config = config

    def generate_training_samples(self, v1_indices, v1_labels, v1_rider_id, v1_track_id,
                                  v2_indices, v2_labels, v2_rider_id, v2_track_id,
                                  num_augmented_positives_per_segment, num_non_overlapping_samples_per_positive):
        """Generate training samples for a pair of videos."""
        assert v1_track_id == v2_track_id, "Track IDs must be the same for v1 and v2"
        v1_split_indices = v1_indices[v1_labels == 1.0].astype(int)
        v2_split_indices = v2_indices[v2_labels == 1.0].astype(int)
        logging.debug(f"Found {len(v1_split_indices)} splits")

        assert len(
            v1_split_indices) > 0, "No split points found in v1_split_indices"
        assert len(
            v2_split_indices) > 0, "No split points found in v2_split_indices"
        assert len(v1_split_indices) == len(
            v2_split_indices), "Mismatch in split counts"

        v1_total_frames = v1_indices[-1] + 1 if len(v1_indices) > 0 else 0
        v2_total_frames = v2_indices[-1] + 1 if len(v2_indices) > 0 else 0
        v1_min_idx = v1_split_indices[0] + self.args.clip_length - 1
        v2_min_idx = v2_split_indices[0] + self.args.clip_length - 1

        metadata_base = {
            "v1_rider_id": v1_rider_id,
            "v1_track_id": v1_track_id,
            "v2_rider_id": v2_rider_id,
            "v2_track_id": v2_track_id,
        }

        sample_labels = []
        sample_indices = []
        sample_metadata = []

        start_split = 1 if self.args.ignore_first_split else 0
        rng = np.random.default_rng(seed=self.args.seed)

        def generate_samples_around_positive(v1_idx, v2_true_idx, sample_type_prefix):
            samples = []
            # Overlapping samples: |v1_idx - v2_idx| < clip_length
            for offset in range(- (self.args.clip_length - 1), self.args.clip_length):

                if offset == 0:
                    continue  # Skip the positive sample itself
                v2_idx = v2_true_idx + offset
                logging.debug(
                    f"Generating offset samples. v1_idx={v1_idx}.  v2_true_idx={v2_true_idx}. current_v2_idx={v2_idx}. offset={offset}")
                if v2_idx < self.args.clip_length - 1 or v2_idx >= v2_total_frames:
                    continue
                samples.append((float(offset), (v1_idx, v2_idx),
                               {**metadata_base, "sample_type": f"{sample_type_prefix}_overlapping"}))

            # Non-overlapping samples: |v1_idx - v2_idx| >= clip_length
            for _ in range(num_non_overlapping_samples_per_positive):
                while True:
                    # select a random position in video 2
                    v2_idx = rng.integers(
                        self.args.clip_length - 1, v2_total_frames)
                    # compute the offset
                    offset_value = v2_idx - v2_true_idx
                    if abs(offset_value) >= self.args.clip_length:
                        # cap offset value (it is not possible to know the offset between two clips that don't overlap)
                        # TODO: this doesn't make sense. there should be a single value to represent no overlap, not 2 (pos or neg)
                        if offset_value >= 0:
                            capped_offset_value = self.args.clip_length
                        else:
                            capped_offset_value = - self.args.clip_length
                        logging.debug(
                            f"Generated non overlapping sample. v1_idx={v1_idx}.  v2_true_idx={v2_true_idx}. current_v2_idx={v2_idx}. offset_value={offset_value}. capped_offset_value={capped_offset_value}")
                        samples.append((float(capped_offset_value), (v1_idx, v2_idx),
                                       {**metadata_base, "sample_type": f"{sample_type_prefix}_non_overlapping"}))
                        break

            return samples

        # Positive samples at split points (offset=0)
        for split_number in range(start_split, len(v1_split_indices)):
            v1_split_idx = v1_split_indices[split_number]
            v2_true_idx = v2_split_indices[split_number]
            if v1_split_idx < self.args.clip_length - 1 or v2_true_idx < self.args.clip_length - 1:
                continue
            sample_labels.append(0.0)
            sample_indices.append((v1_split_idx, v2_true_idx))
            sample_metadata.append(
                {**metadata_base, "sample_type": "split_offset"})

            # Generate overlapping and non-overlapping samples
            additional_samples = generate_samples_around_positive(
                v1_split_idx, v2_true_idx, "split")
            for label, idx, meta in additional_samples:
                sample_labels.append(label)
                sample_indices.append(idx)
                sample_metadata.append(meta)

        # Augmented positive samples within segments (offset=0)
        num_segments = len(v1_split_indices) - 1
        for seg in range(num_segments):
            v1_start_seg = max(v1_split_indices[seg], v1_min_idx)
            v1_end_seg = v1_split_indices[seg + 1]
            possible_idx1 = list(range(v1_start_seg, v1_end_seg + 1))
            num_samples = min(
                num_augmented_positives_per_segment, len(possible_idx1))
            if num_samples == 0:
                continue
            alpha = self.args.alpha_split_0 if seg == 0 else self.args.alpha
            beta = self.args.beta_split_0 if seg == 0 else self.args.beta
            relative_positions = rng.beta(a=alpha, b=beta, size=num_samples)
            min_idx = possible_idx1[0]
            max_idx = possible_idx1[-1]
            selected_idx1 = [int(min_idx + p * (max_idx - min_idx))
                             for p in relative_positions]
            selected_idx1 = [min(max(idx, min_idx), max_idx)
                             for idx in selected_idx1]
            for idx1 in selected_idx1:
                fraction = (
                    idx1 - v1_split_indices[seg]) / (v1_split_indices[seg + 1] - v1_split_indices[seg])
                v2_true_idx_float = v2_split_indices[seg] + fraction * (
                    v2_split_indices[seg + 1] - v2_split_indices[seg])
                v2_true_idx = int(round(v2_true_idx_float))
                if v2_true_idx < self.args.clip_length - 1 or v2_true_idx >= v2_total_frames:
                    continue
                sample_labels.append(0.0)
                sample_indices.append((idx1, v2_true_idx))
                sample_metadata.append(
                    {**metadata_base, "sample_type": "augmented_offset"})

                # Generate overlapping and non-overlapping samples
                additional_samples = generate_samples_around_positive(
                    idx1, v2_true_idx, "augmented")
                for label, idx, meta in additional_samples:
                    sample_labels.append(label)
                    sample_indices.append(idx)
                    sample_metadata.append(meta)

        return np.array(sample_labels), np.array(sample_indices), sample_metadata

    def generate_training_metadata(self, track_ids: Set[str], split_type: SplitType) -> 'pd.DataFrame':
        """Generate training metadata and return it as a DataFrame."""
        track_videos = self.config.get_trackid_to_video_metadata()()
        if len(track_videos) < 2:
            logging.error(
                f"Need at least two tracks, found {len(track_videos)}")
            return pd.DataFrame()

        dfs = []
        for track_id in track_ids:
            track_videos_list = self.config.get_trackid_to_video_metadata()()[
                track_id]
            if len(track_videos_list) < 2:
                logging.warning(
                    f"Need at least two riders for track {track_id}, found {len(track_videos_list)}, skipping.")
                continue
            video_pairs = list(itertools.permutations(track_videos_list, 2))
            logging.info(
                f"Found {len(track_videos_list)} riders for track {track_id}, generating samples for {len(video_pairs)} pairs")
            for video1, video2 in video_pairs:
                rider_id1 = video1["riderId"]
                rider_id2 = video2["riderId"]
                video_path1 = get_video_file_path(track_id, rider_id1)
                video_path2 = get_video_file_path(track_id, rider_id2)
                if not video_path1.exists() or not video_path2.exists():
                    logging.warning(
                        f"Video missing for pair {rider_id1} and {rider_id2}, skipping.")
                    continue
                v1_indices, v1_labels, _, _ = get_video_metadata(
                    str(video_path1), video1["splits"], rider_id1, track_id)
                v2_indices, v2_labels, _, _ = get_video_metadata(
                    str(video_path2), video2["splits"], rider_id2, track_id)
                num_augmented = self.args.num_augmented_positives_per_segment if split_type != SplitType.VAL else 0
                num_non_overlapping = self.args.num_non_overlapping_samples_per_positive if split_type != SplitType.VAL else 0
                sample_labels, sample_indices, sample_metadata = self.generate_training_samples(
                    v1_indices, v1_labels, rider_id1, track_id,
                    v2_indices, v2_labels, rider_id2, track_id,
                    num_augmented, num_non_overlapping
                )
                data = {
                    "track_id": [meta["v1_track_id"] for meta in sample_metadata],
                    "v1_rider_id": [meta["v1_rider_id"] for meta in sample_metadata],
                    "v2_rider_id": [meta["v2_rider_id"] for meta in sample_metadata],
                    "v1_frame_idx": [idx[0] for idx in sample_indices],
                    "v2_frame_idx": [idx[1] for idx in sample_indices],
                    "label": sample_labels,
                    "sample_type": [meta["sample_type"] for meta in sample_metadata],
                    "set": [split_type] * len(sample_labels),
                }
                df = pd.DataFrame(data)
                dfs.append(df)
                logging.debug(
                    f"Generated {len(sample_labels)} samples for pair {rider_id1} and {rider_id2}")

        if dfs:
            df = pd.concat(dfs, axis=0)
            return df
        logging.info("No training samples generated.")
        return pd.DataFrame()
