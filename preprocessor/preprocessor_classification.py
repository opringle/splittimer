import numpy as np
import random
import logging
import itertools
import pandas as pd
from .interface import Preprocessor, SplitType
from utils import get_video_file_path, get_video_metadata
from typing import Set


class ClassifierPreprocessor(Preprocessor):
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
        parser.add_argument("--max_negatives_per_positive", type=int,
                            default=10, help="Max negative samples per split point")
        parser.add_argument("--num_augmented_positives_per_segment", type=int,
                            default=50, help="Number of augmented samples per segment")
        parser.add_argument("--ignore_first_split", action="store_true",
                            help="Ignore the first split when generating training data")

    @staticmethod
    def from_args(args, config):
        """Create an instance from parsed arguments, config, and track-to-set mapping."""
        return ClassifierPreprocessor(args, config)

    def __init__(self, args, config):
        """Initialize the preprocessor with arguments, configuration, and track assignments."""
        self.args = args
        self.config = config

    def generate_training_metadata(self, track_ids: Set[str], split_type: SplitType) -> 'pd.DataFrame':
        """Generate training metadata for the specified track IDs."""
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
                sample_labels, sample_indices, sample_metadata = self.generate_training_samples(
                    v1_indices, v1_labels, rider_id1, track_id,
                    v2_indices, v2_labels, rider_id2, track_id,
                    num_augmented
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
                    f"Generated {len(sample_labels)} samples for pair {rider_id1} and {rider_id2}: "
                    f"{np.sum(sample_labels == 1.0)} positive, {np.sum(sample_labels == 0.0)} negative"
                )
        if dfs:
            return pd.concat(dfs, axis=0)
        return pd.DataFrame()

    def generate_training_samples(self, v1_indices, v1_labels, v1_rider_id, v1_track_id,
                                  v2_indices, v2_labels, v2_rider_id, v2_track_id,
                                  num_augmented_positives_per_segment):
        """Generate positive and negative training samples for a pair of videos."""
        clip_length = self.args.clip_length
        max_negatives_per_positive = self.args.max_negatives_per_positive
        ignore_first_split = self.args.ignore_first_split
        alpha_split_0 = self.args.alpha_split_0
        alpha = self.args.alpha
        beta_split_0 = self.args.beta_split_0
        beta = self.args.beta
        seed = self.args.seed

        assert v1_track_id == v2_track_id, "Track IDs must match for v1 and v2"
        v1_split_indices = v1_indices[v1_labels == 1.0]
        v2_split_indices = v2_indices[v2_labels == 1.0]
        v2_frame_idx_to_split_number = {
            int(idx): i for i, idx in enumerate(v2_split_indices)}
        logging.debug(f"Found {len(v1_split_indices)} splits")

        assert len(v1_split_indices) > 0, "No split points in v1_split_indices"
        assert len(v2_split_indices) > 0, "No split points in v2_split_indices"
        assert len(v1_split_indices) == len(
            v2_split_indices), "Mismatch in split counts"

        v1_total_frames = v1_indices[-1] + 1 if len(v1_indices) > 0 else 0
        v2_total_frames = v2_indices[-1] + 1 if len(v2_indices) > 0 else 0
        v1_min_idx = v1_split_indices[0] + clip_length - 1
        v2_min_idx = v2_split_indices[0] + clip_length - 1

        metadata_base = {
            "v1_rider_id": v1_rider_id,
            "v1_track_id": v1_track_id,
            "v2_rider_id": v2_rider_id,
            "v2_track_id": v2_track_id,
        }

        sample_labels = []
        sample_indices = []
        sample_metadata = []

        start_split = 1 if ignore_first_split else 0

        # Positive samples at split points
        for split_number in range(start_split, len(v1_split_indices)):
            v1_split_idx = v1_split_indices[split_number]
            v2_split_idx = v2_split_indices[split_number]
            sample_labels.append(1.0)
            sample_indices.append((v1_split_idx, v2_split_idx))
            sample_metadata.append({**metadata_base, "sample_type": "split"})

            # Negative samples for split points
            negative_labels, negative_indices, negative_metadata = self.generate_negative_samples(
                v1_split_idx, v2_split_idx, v2_total_frames, clip_length,
                max_negatives_per_positive, metadata_base, v2_min_idx,
                v2_frame_idx_to_split_number, split_number
            )
            sample_labels.extend(negative_labels)
            sample_indices.extend(negative_indices)
            sample_metadata.extend(negative_metadata)

        # Augmented positive samples between split points
        num_segments = len(v1_split_indices) - 1
        for seg in range(num_segments):
            v1_start_seg = max(v1_split_indices[seg], v1_min_idx)
            v1_end_seg = v1_split_indices[seg + 1]
            v2_start_seg = max(v2_split_indices[seg], v2_min_idx)
            v2_end_seg = v2_split_indices[seg + 1]

            possible_idx1 = list(range(v1_start_seg, v1_end_seg + 1))
            num_samples = min(
                num_augmented_positives_per_segment, len(possible_idx1))

            rng = np.random.default_rng(seed=seed)
            alpha_val = alpha_split_0 if seg == 0 else alpha
            beta_val = beta_split_0 if seg == 0 else beta
            relative_positions = rng.beta(
                a=alpha_val, b=beta_val, size=num_samples)
            min_idx = possible_idx1[0]
            max_idx = possible_idx1[-1]
            selected_idx1 = [int(min_idx + p * (max_idx - min_idx))
                             for p in relative_positions]
            selected_idx1 = [min(max(idx, min_idx), max_idx)
                             for idx in selected_idx1]

            for idx1 in selected_idx1:
                fraction = (idx1 - v1_start_seg) / (v1_end_seg - v1_start_seg)
                idx2 = int(round(v2_start_seg + fraction *
                           (v2_end_seg - v2_start_seg)))
                if not (v2_start_seg <= idx2 <= v2_end_seg):
                    raise ValueError(
                        f"idx2 {idx2} out of range [{v2_start_seg}, {v2_end_seg}]")
                sample_labels.append(1.0)
                sample_indices.append((idx1, idx2))
                sample_metadata.append(
                    {**metadata_base, "sample_type": "augmented"})

                # Negative samples for augmented positives
                negative_labels, negative_indices, negative_metadata = self.generate_negative_samples(
                    idx1, idx2, v2_total_frames, clip_length,
                    max_negatives_per_positive, metadata_base, v2_min_idx
                )
                sample_labels.extend(negative_labels)
                sample_indices.extend(negative_indices)
                sample_metadata.extend(negative_metadata)

        return np.array(sample_labels), np.array(sample_indices), sample_metadata

    @staticmethod
    def generate_negative_samples(
        v1_idx, positive_v2_idx, v2_total_frames, clip_length, max_negatives,
        metadata_base, v2_min_idx, v2_frame_idx_to_split_number=None, split_number=None
    ):
        """Generate negative samples avoiding positive split points."""
        if v2_min_idx > v2_total_frames - 1:
            logging.warning(
                f"Cannot generate negatives: v2_min_idx {v2_min_idx} > v2_total_frames - 1")
            return [], [], []

        negative_labels = []
        negative_indices = []
        negative_metadata = []
        attempts = 0
        max_attempts = 50

        while len(negative_labels) < max_negatives and attempts < max_attempts:
            v2_idx = random.randint(v2_min_idx, v2_total_frames - 1)
            if split_number is not None and v2_frame_idx_to_split_number is not None:
                is_false_negative = (
                    v2_idx in v2_frame_idx_to_split_number and
                    v2_frame_idx_to_split_number[v2_idx] == split_number
                )
                if is_false_negative:
                    attempts += 1
                    continue
            elif v2_idx == positive_v2_idx:
                attempts += 1
                continue
            negative_labels.append(0.0)
            negative_indices.append((v1_idx, v2_idx))
            negative_metadata.append(
                {**metadata_base, "sample_type": "negative"})
            attempts += 1

        return negative_labels, negative_indices, negative_metadata
