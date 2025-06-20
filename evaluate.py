from collections import defaultdict
import logging
import random
from typing import List
import torch
import argparse
from tqdm import tqdm
from config import Config
from sample_generator import get_sample_generator_class
from trainer import get_trainer_class
from utils import get_default_device_name, get_video_file_path, get_video_fps_and_total_frames, log_dict, setup_logging, setup_seed, timecode_to_frames, timecode_to_seconds
import json


def worker_init_fn(worker_id):
    """Initialize random seed for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    setup_seed(worker_seed)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model's performance at finding splits in target videos for multiple tracks")
    parser.add_argument('config_path', type=str,
                        help='Path to video_config.yaml file')
    parser.add_argument('image_feature_path', type=str,
                        help="Path to directory of clip features")
    parser.add_argument('output_file', type=str, help="Path to output file")
    parser.add_argument('--trackIds', type=str, nargs='+', required=True,
                        help='List of track identifiers')
    parser.add_argument('--checkpoint_path', type=str,
                        required=True, help='Path to the model checkpoint file')
    parser.add_argument('--device', type=str, default=get_default_device_name(),
                        help='Device to use (cuda or cpu)')
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument('--trainer_type', type=str, required=True,
                        help='Type of model to train')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args, _ = parser.parse_known_args()
    TrainerClass = get_trainer_class(args.trainer_type)
    TrainerClass.add_args(parser)

    args = parser.parse_args()

    setup_logging(args.log_level)
    setup_seed(args.seed)
    config = Config(args.config_path)

    trainer, _ = TrainerClass.load(args.checkpoint_path, args.device)

    # Step 1: Precompute track-to-target rider IDs and necessary data
    track_data = {}
    for trackId in args.trackIds:
        riders_for_track = config.get_rider_ids(trackId)
        assert len(
            riders_for_track) >= 2, f"Only {len(riders_for_track)} riders have runs on track {trackId}"
        random.shuffle(riders_for_track)
        source_rider_id = riders_for_track[0]
        target_rider_ids = riders_for_track[1:]
        source_timecodes = config.get_timecodes(trackId, source_rider_id)
        # Ignore the first split
        source_timecodes_sliced = source_timecodes[1:]
        target_rid_to_timecodes = {rid: config.get_timecodes(
            # Ignore the first split
            trackId, rid)[1:] for rid in target_rider_ids}
        track_data[trackId] = {
            'source_rider_id': source_rider_id,
            'source_timecodes_sliced': source_timecodes_sliced,
            'target_rid_to_timecodes': target_rid_to_timecodes
        }

    # Step 2: Compute total number of target riders to predict for
    total_target_riders = sum(
        len(data['target_rid_to_timecodes']) for data in track_data.values())

    # Step 3: Set up a single tqdm progress bar for all predictions
    pbar = tqdm(total=total_target_riders,
                desc="Overall target rider predictions")

    output_data_list = []
    mae_track_list = []

    # Step 4: Process each track and update the progress bar
    for trackId, data in track_data.items():
        source_rider_id = data['source_rider_id']
        source_timecodes_sliced = data['source_timecodes_sliced']
        target_rid_to_timecodes = data['target_rid_to_timecodes']

        predictions_dict = {}
        mean_absolute_error_sum = 0
        for target_rider_id, actual_timecodes_sliced in target_rid_to_timecodes.items():
            logging.info(
                f"{trackId}: source_rider_id {source_rider_id} target_rider_id {target_rider_id}")
            target_video_path = get_video_file_path(trackId, target_rider_id)
            target_fps, _ = get_video_fps_and_total_frames(target_video_path)

            predicted_timecodes = trainer.predict_timecodes(
                trackId, source_rider_id, source_timecodes_sliced, target_rider_id)
            predictions_dict[target_rider_id] = predicted_timecodes

            mae = mean_absolute_error_seconds(
                predicted_timecodes, actual_timecodes_sliced, target_fps)
            mean_absolute_error_sum += mae

            # Update the progress bar for each prediction
            pbar.update(1)

        mae_track = mean_absolute_error_sum / len(target_rid_to_timecodes)
        logging.info(f"For track {trackId}, MAE: {mae_track:.4f} seconds")
        mae_track_list.append(mae_track)

        output_data = {
            "trackId": trackId,
            "sourceRiderId": source_rider_id,
            "sourceTimecodes": source_timecodes_sliced,
            "predictions": predictions_dict
        }
        output_data_list.append(output_data)

    pbar.close()

    # Step 5: Compute and log overall MAE
    overall_mae = sum(mae_track_list) / len(mae_track_list)
    logging.info(f"Overall MAE across all tracks: {overall_mae:.4f} seconds")

    with open(args.output_file, 'w') as f:
        json.dump(output_data_list, f, indent=4)
    logging.info(f"Predictions written to {args.output_file}")


def mean_absolute_error_seconds(predicted_timecodes: List[str], actual_timecodes: List[str], fps: float) -> float:
    """Compute the mean absolute error in seconds between two lists of timecodes."""
    logging.debug(
        f"Computing MAE (seconds) between predictions: {predicted_timecodes} & labels: {actual_timecodes}")
    # Convert timecodes to total seconds
    predicted_seconds = [timecode_to_seconds(
        tc, fps) if tc is not None else None for tc in predicted_timecodes]
    actual_seconds = [timecode_to_seconds(tc, fps) for tc in actual_timecodes]

    # Filter out None predictions
    valid_pairs = [(p, a) for p, a in zip(
        predicted_seconds, actual_seconds) if p is not None]
    if not valid_pairs:
        return 0.0
    abs_diff_seconds = [abs(p - a) for p, a in valid_pairs]
    return sum(abs_diff_seconds) / len(abs_diff_seconds)


if __name__ == "__main__":
    main()
