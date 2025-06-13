import numpy as np
import os
import argparse
from pathlib import Path
import random
import itertools
import logging
import pandas as pd
import yaml
from collections import defaultdict

from utils import get_video_fps_and_total_frames, timecode_to_frames


def get_video_metadata(video_path, splits, rider_id, track_id):
    fps, total_frames = get_video_fps_and_total_frames(video_path)

    split_indices_raw = [timecode_to_frames(tc, fps) for tc in splits]
    split_indices = []
    for idx in split_indices_raw:
        assert idx < total_frames, f"Split index {idx} exceeds total frames {total_frames} for video {video_path}"
        assert idx >= 0, f"Split index {idx} is negative for video {video_path}"
        split_indices.append(idx)

    frame_indices = np.arange(total_frames)
    labels = np.zeros(total_frames)
    for idx in split_indices:
        labels[idx] = 1.0
    return frame_indices, labels, rider_id, track_id


def generate_negative_samples(
    v1_idx,
    positive_v2_idx,
    v2_total_frames,
    clip_length,
    max_negatives,
    metadata_base,
    v2_min_idx,
    v2_frame_idx_to_split_number=None,
    split_number=None,
):
    if v2_min_idx > v2_total_frames - 1:
        logging.warning(
            f"Cannot generate negative samples: v2_min_idx {v2_min_idx} > v2_total_frames - 1 {v2_total_frames - 1}")
        return [], [], []

    negative_labels = []
    negative_indices = []
    negative_metadata = []
    attempts = 0
    max_attempts = 50

    while len(negative_labels) < max_negatives and attempts < max_attempts:
        v2_idx = random.randint(v2_min_idx, v2_total_frames - 1)
        # For split points, check if v2_idx is the positive split index
        if split_number is not None and v2_frame_idx_to_split_number is not None:
            is_false_negative = v2_idx in v2_frame_idx_to_split_number and v2_frame_idx_to_split_number[
                v2_idx] == split_number
            if is_false_negative:
                attempts += 1
                continue
        # For augmented positives, just ensure v2_idx != positive_v2_idx
        elif v2_idx == positive_v2_idx:
            attempts += 1
            continue

        negative_labels.append(0.0)
        negative_indices.append((v1_idx, v2_idx))
        negative_metadata.append({**metadata_base, "sample_type": "negative"})
        attempts += 1

    return negative_labels, negative_indices, negative_metadata


def generate_training_samples(
    seed,
    v1_indices,
    v1_labels,
    v1_rider_id,
    v1_track_id,
    v2_indices,
    v2_labels,
    v2_rider_id,
    v2_track_id,
    clip_length,
    max_negatives_per_positive=10,
    num_augmented_positives_per_segment=5,
    ignore_first_split=False,
    alpha_split_0=0.5,
    alpha=0.5,
    beta_split_0=0.5,
    beta=0.5,
):
    assert v1_track_id == v2_track_id, "Track IDs must be the same for v1 and v2"
    v1_split_indices = v1_indices[v1_labels == 1.0]
    v2_split_indices = v2_indices[v2_labels == 1.0]
    v2_frame_idx_to_split_number = {
        int(idx): i for i, idx in enumerate(v2_split_indices)}
    logging.debug(f"Found {len(v1_split_indices)} splits")

    assert len(v1_split_indices) != 0, "No split points found in v1_split_indices"
    assert len(v2_split_indices) != 0, "No split points found in v2_split_indices"
    assert len(v1_split_indices) == len(
        v2_split_indices
    ), f"len(v1_split_indices) {len(v1_split_indices)} not equal to len(v2_split_indices) {len(v2_split_indices)}"

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

    # Generate samples at split points (positive samples, type "split")
    for split_number in range(start_split, len(v1_split_indices)):
        v1_split_idx = v1_split_indices[split_number]
        v2_split_idx = v2_split_indices[split_number]

        sample_labels.append(1.0)
        sample_indices.append((v1_split_idx, v2_split_idx))
        sample_metadata.append({**metadata_base, "sample_type": "split"})

        # Generate negative samples for split points
        negative_labels, negative_indices, negative_metadata = generate_negative_samples(
            v1_split_idx,
            v2_split_idx,
            v2_total_frames,
            clip_length,
            max_negatives_per_positive,
            metadata_base,
            v2_min_idx,
            v2_frame_idx_to_split_number,
            split_number,
        )
        sample_labels.extend(negative_labels)
        sample_indices.extend(negative_indices)
        sample_metadata.extend(negative_metadata)

    # Augmented positive samples for segments (type "augmented")
    num_segments = len(v1_split_indices) - 1
    for seg in range(num_segments):
        v1_start_seg = max(v1_split_indices[seg], v1_min_idx)
        v1_end_seg = v1_split_indices[seg + 1]
        v2_start_seg = max(v2_split_indices[seg], v2_min_idx)
        v2_end_seg = v2_split_indices[seg + 1]

        possible_idx1 = list(range(v1_start_seg, v1_end_seg + 1))

        num_samples = min(num_augmented_positives_per_segment,
                          len(possible_idx1))

        rng = np.random.default_rng(seed=seed)
        # if first split concentrate probability near end of split (start)
        # else concentrate at both start and end
        alpha = alpha_split_0 if seg == 0 else alpha
        beta = beta_split_0 if seg == 0 else beta
        relative_positions = rng.beta(a=alpha, b=beta, size=num_samples)
        min_idx = possible_idx1[0]
        max_idx = possible_idx1[-1]
        selected_idx1 = [int(min_idx + p * (max_idx - min_idx))
                         for p in relative_positions]
        selected_idx1 = [min(max(idx, min_idx), max_idx)
                         for idx in selected_idx1]

        for idx1 in selected_idx1:
            fraction_through_v1_segment = (
                idx1 - v1_start_seg) / (v1_end_seg - v1_start_seg)
            idx2_float = v2_start_seg + fraction_through_v1_segment * \
                (v2_end_seg - v2_start_seg)
            idx2 = int(round(idx2_float))
            if idx2 < v2_start_seg or idx2 > v2_end_seg:
                raise Exception(
                    f"idx2 {idx2} is out of range [{v2_start_seg}, {v2_end_seg}]")

            sample_labels.append(1.0)
            sample_indices.append((idx1, idx2))
            sample_metadata.append(
                {**metadata_base, "sample_type": "augmented"})

            # Generate negative samples for augmented positives
            negative_labels, negative_indices, negative_metadata = generate_negative_samples(
                idx1,
                idx2,
                v2_total_frames,
                clip_length,
                max_negatives_per_positive,
                metadata_base,
                v2_min_idx,
            )
            sample_labels.extend(negative_labels)
            sample_indices.extend(negative_indices)
            sample_metadata.extend(negative_metadata)

    return np.array(sample_labels), np.array(sample_indices), sample_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate training metadata for all rider combinations on each track.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument(
        "--clip-length",
        type=int,
        required=True,
        help="Number of frames to use per clip",
    )
    parser.add_argument(
        "--alpha_split_0",
        type=float,
        default=0.5,
        help="Beta distribution alpha value at split 0",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Beta distribution alpha value at split > 0",
    )
    parser.add_argument(
        "--beta_split_0",
        type=float,
        default=0.5,
        help="Beta distribution beta value at split 0",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Beta distribution beta value at split > 0",
    )
    parser.add_argument(
        "--max_negatives_per_positive",
        type=int,
        default=10,
        help="Max negative samples per split point",
    )
    parser.add_argument(
        "--num_augmented_positives_per_segment",
        type=int,
        default=50,
        help="Number of augmented samples per segment",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Ratio of tracks to use for validation (between 0 and 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--ignore_first_split",
        action="store_true",
        help="Ignore the first split when generating training data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible track splitting",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    if args.ignore_first_split:
        logging.info(
            "Ignoring the first split for generating positive samples at split points")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logging.info(
            f"Set random seed to {args.seed} for reproducible track splitting")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    videos = config["videos"]

    track_videos = defaultdict(list)
    for video in videos:
        track_videos[video["trackId"]].append(video)

    track_ids = list(track_videos.keys())
    if len(track_ids) < 2:
        logging.error(
            f"Need at least two tracks for splitting, found {len(track_ids)}")
        exit(1)
    random.shuffle(track_ids)
    num_val_tracks = max(1, int(args.val_ratio * len(track_ids)))
    val_tracks = track_ids[:num_val_tracks]
    val_tracks_set = set(val_tracks)
    logging.info(
        f"Assigned {len(val_tracks)} tracks to validation: {', '.join(val_tracks)}")
    logging.info(
        f"Assigned {len(track_ids) - len(val_tracks)} tracks to training: {', '.join(set(track_ids) - val_tracks_set)}")

    dfs = []
    for track_id, track_videos_list in track_videos.items():
        if len(track_videos_list) < 2:
            logging.warning(
                f"Need at least two riders for track {track_id}, found {len(track_videos_list)}, skipping.")
            continue
        set_type = "val" if track_id in val_tracks_set else "train"
        video_pairs = list(itertools.permutations(track_videos_list, 2))
        logging.info(
            f"Found {len(track_videos_list)} riders for track {track_id}, generating samples for {len(video_pairs)} pairs")
        for video1, video2 in video_pairs:
            rider_id1 = video1["riderId"]
            rider_id2 = video2["riderId"]
            video_path1 = Path("downloaded_videos") / track_id / \
                rider_id1 / f"{track_id}_{rider_id1}.mp4"
            video_path2 = Path("downloaded_videos") / track_id / \
                rider_id2 / f"{track_id}_{rider_id2}.mp4"

            assert video_path1.exists(
            ), f"Video file {video_path1} does not exist, skipping pair {rider_id1} and {rider_id2}"
            assert video_path2.exists(
            ), f"Video file {video_path2} does not exist, skipping pair {rider_id1} and {rider_id2}"

            v1_indices, v1_labels, v1_rider_id, v1_track_id = get_video_metadata(
                str(video_path1), video1["splits"], rider_id1, track_id)
            v2_indices, v2_labels, v2_rider_id, v2_track_id = get_video_metadata(
                str(video_path2), video2["splits"], rider_id2, track_id)

            # do not generate augmented samples in validation
            num_augmented_positives_per_segment = args.num_augmented_positives_per_segment if set_type != "val" else 0

            logging.debug(
                f"Generating training samples for {v1_rider_id} and {v2_rider_id}")
            sample_labels, sample_indices, sample_metadata = generate_training_samples(
                args.seed,
                v1_indices,
                v1_labels,
                v1_rider_id,
                v1_track_id,
                v2_indices,
                v2_labels,
                v2_rider_id,
                v2_track_id,
                args.clip_length,
                max_negatives_per_positive=args.max_negatives_per_positive,
                num_augmented_positives_per_segment=num_augmented_positives_per_segment,
                ignore_first_split=args.ignore_first_split,
                alpha_split_0=args.alpha_split_0,
                alpha=args.alpha,
                beta_split_0=args.beta_split_0,
                beta=args.beta,
            )

            data = {
                "track_id": [meta["v1_track_id"] for meta in sample_metadata],
                "v1_rider_id": [meta["v1_rider_id"] for meta in sample_metadata],
                "v2_rider_id": [meta["v2_rider_id"] for meta in sample_metadata],
                "v1_frame_idx": [idx[0] for idx in sample_indices],
                "v2_frame_idx": [idx[1] for idx in sample_indices],
                "label": sample_labels,
                "sample_type": [meta["sample_type"] for meta in sample_metadata],
                "set": [set_type] * len(sample_labels),
            }
            df = pd.DataFrame(data)
            dfs.append(df)

            logging.debug(
                f"Generated {len(sample_labels)} samples for pair {v1_rider_id} and {v2_rider_id}: "
                f"{np.sum(sample_labels == 1.0)} positive, {np.sum(sample_labels == 0.0)} negative"
            )

    if dfs:
        df = pd.concat(dfs, axis=0)
        training_data_output_path = "training_data"
        os.makedirs(training_data_output_path, exist_ok=True)
        output_filename = "training_metadata.csv"
        training_data_file_path = os.path.join(
            training_data_output_path, output_filename)
        df.to_csv(training_data_file_path, index=False)
        logging.info(
            f"Saved {len(df)} training metadata to {training_data_file_path}. {df['set'].value_counts().get('train', 0)} training, {df['set'].value_counts().get('val', 0)} validation"
        )
    else:
        logging.info("No training samples generated.")


if __name__ == "__main__":
    main()
