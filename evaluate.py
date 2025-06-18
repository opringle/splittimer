from collections import defaultdict
import logging
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
        description="Evaluate a model's performance at finding splits in a target video'")
    parser.add_argument('config_path', type=str,
                        help='Path to video_config.yaml file')
    parser.add_argument('image_feature_path', type=str,
                        help="Path to directory of clip features")
    parser.add_argument('output_file', type=str, help="Path to output file")

    parser.add_argument('--trackId', type=str,
                        required=True, help='Track identifier')
    parser.add_argument('--sourceRiderId', type=str,
                        required=True, help='Source rider identifier')
    parser.add_argument('--targetRiderIds', type=str, nargs='+',
                        required=True, help='Riders to find splits for')
    parser.add_argument('--checkpoint_path', type=str,
                        required=True, help='Path to the model checkpoint file')
    parser.add_argument('--device', type=str, default=get_default_device_name(),
                        help='Device to use (cuda or cpu)')
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument('--trainer_type', type=str, required=True,
                        help='Type of model to train')

    args, _ = parser.parse_known_args()
    TrainerClass = get_trainer_class(args.trainer_type)
    TrainerClass.add_args(parser)

    args = parser.parse_args()

    setup_logging(args.log_level)
    config = Config(args.config_path)

    trainer, _ = TrainerClass.load(args.checkpoint_path, args.device)

    source_timecodes = config.get_timecodes(args.trackId, args.sourceRiderId)
    source_timecodes_sliced = source_timecodes[1:]  # Ignore the first split
    target_rid_to_timecodes = {rid: config.get_timecodes(
        args.trackId, rid) for rid in args.targetRiderIds}

    predictions_dict = {}
    mean_absolute_error_sum = 0
    for target_rider_id, actual_timecodes in tqdm(target_rid_to_timecodes.items(), desc="Target rider predictions"):
        target_video_path = get_video_file_path(args.trackId, target_rider_id)
        target_fps, _ = get_video_fps_and_total_frames(target_video_path)

        # Ignore the first split
        actual_timecodes_sliced = actual_timecodes[1:]

        predicted_timecodes = trainer.predict_timecodes(
            args.trackId, args.sourceRiderId, source_timecodes_sliced, target_rider_id)
        predictions_dict[target_rider_id] = predicted_timecodes

        mean_absolute_error_sum += mean_absolute_error_seconds(
            predicted_timecodes, actual_timecodes_sliced, target_fps)

    metrics = {'macro_mean_absolute_error_seconds': mean_absolute_error_sum /
               len(target_rid_to_timecodes)}
    log_dict(f"Model achieves metrics:", metrics)

    output_data = {
        "trackId": args.trackId,
        "sourceRiderId": args.sourceRiderId,
        "sourceTimecodes": source_timecodes_sliced,
        "predictions": predictions_dict
    }
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
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
