from collections import defaultdict
import logging
from typing import List
import torch
import argparse
from tqdm import tqdm
from config import Config
from sample_generator import get_sample_generator_class
from trainer import get_trainer_class
from utils import get_default_device_name, get_video_file_path, get_video_fps_and_total_frames, log_dict, setup_logging, setup_seed, timecode_to_frames


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
    parser.add_argument('--sample_generator_type', type=str, required=True,
                        help='Type of generator')

    args, _ = parser.parse_known_args()
    TrainerClass = get_trainer_class(args.trainer_type)
    TrainerClass.add_args(parser)

    SampleGeneratorClass = get_sample_generator_class(
        args.sample_generator_type)
    SampleGeneratorClass.add_args(parser)

    args = parser.parse_args()

    setup_logging(args.log_level)
    config = Config(args.config_path)

    trainer = TrainerClass.load(args.checkpoint_path, args.device)

    source_timecodes = config.get_timecodes(args.trackId, args.sourceRiderId)
    target_rid_to_timecodes = {rid: config.get_timecodes(
        args.trackId, rid) for rid in args.targetRiderIds}

    mean_absolute_error_sum = 0
    for target_rider_id, actual_timecodes in tqdm(target_rid_to_timecodes.items(), desc="Target rider predictions"):
        predicted_timecodes = trainer.predict_timecodes(
            args.trackId, args.sourceRiderId, source_timecodes, target_rider_id)

        mean_absolute_error_seconds = mean_absolute_error_seconds(
            predicted_timecodes, actual_timecodes)
        mean_absolute_error_sum += mean_absolute_error_seconds

    metrics = {'macro_mean_absolute_error_seconds': mean_absolute_error_sum /
               len(target_rid_to_timecodes)}
    log_dict(f"Model achieves metrics:", metrics)


def mean_absolute_error_seconds(predicted_timecodes: List[str], actual_timecodes: List[str]) -> float:
    # TODO: compute the mean absolute error between 2 equal length arrays of timecodes in format MM:SS:FF
    pass


if __name__ == "__main__":
    main()
