import logging
from typing import List
import torch
import argparse
from tqdm import tqdm
from config import Config
from trainer import get_trainer_class
from utils import get_default_device_name, log_dict, setup_logging, setup_seed


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
    # TODO: make this argument a list of strings instead of string type
    parser.add_argument('--targetRiderIds', type=str,
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
    setup_seed(args.seed)
    config = Config(args.config)

    trainer = TrainerClass.load(args.checkpoint_path, args.device)

    source_timecodes = config.get_timecodes(args.trackId, args.sourceRiderId)
    target_rider_id_to_timecodes = {}
    for target_rider_id in args.targetRiderIds:
        target_rider_id_to_timecodes[target_rider_id] = config.get_timecodes(
            args.trackId, target_rider_id)

    results = []
    mean_absolute_error = 0
    for target_rider_id in tqdm(args.targetRiderIds, desc="Target rider predictions"):
        # TODO: use SampleGenerator to create a sample

        # TODO: use trainer to make predictions
        predicted_timecodes = trainer.predict_splits(
            args.trackId, args.sourceRiderId, target_rider_id)
        actual_timecodes = config.get_timecodes(args.trackId, target_rider_id)
        mean_absolute_error_rider = get_mae(
            predicted_timecodes, actual_timecodes)
        mean_absolute_error += mean_absolute_error_rider
        results.append({
            'trackId': args.trackId,
            'sourceRiderId': args.sourceRiderId,
            'sourceTimecodes': source_timecodes,
            'targetRiderId': target_rider_id,
            'targetTimecodes':  actual_timecodes,
            'targetPredictedTimecodes': predicted_timecodes,
            'meanAbsoluteErrorSeconds': mean_absolute_error_rider,
        })
    metrics = {'mean_absolute_error_seconds': mean_absolute_error /
               len(args.targetRiderIds)}
    log_dict(f"Model achieves metrics,", metrics)


def get_mae(predicted_timecodes: List[str], actual_timecodes: List[str]) -> float:
    # TODO: compute the mean absolute error between 2 equal length arrays of timecodes in format MM:SS:FF
    pass


if __name__ == "__main__":
    main()
