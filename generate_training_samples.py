# generate_training_samples.py
import argparse
import logging
from pathlib import Path
import pandas as pd
import os
import random
import numpy as np
from config import Config  # Import the new Config class
from preprocessor import get_preprocessor_class
from preprocessor.interface import SplitType
from utils import setup_logging, setup_seed


def main():
    # Create parser with common arguments
    parser = argparse.ArgumentParser(description="Generate training metadata.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--val_ratio", type=float,
                        default=0.2, help="Validation track ratio")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output_path", type=str, default="training_data/training_metadata.csv",
                        help="Path to save the output CSV file")
    parser.add_argument("--preprocessor_type", type=str,
                        default="default", help="Preprocessor type")

    # Parse known arguments to get preprocessor_type
    args, _ = parser.parse_known_args()
    PreprocessorClass = get_preprocessor_class(args.preprocessor_type)
    PreprocessorClass.add_args(parser)
    args = parser.parse_args()

    setup_logging(args.log_level)
    setup_seed(args.seed)
    config = Config(args.config)
    preprocessor = PreprocessorClass.from_args(args, config)

    # Determine train/validation tracks
    track_ids = config.get_unique_track_ids()
    if len(track_ids) < 2:
        logging.error(f"Need at least two tracks, found {len(track_ids)}")
        exit(1)
    random.shuffle(track_ids)
    num_val_tracks = max(1, int(args.val_ratio * len(track_ids)))
    val_tracks = track_ids[:num_val_tracks]
    train_tracks = [tid for tid in track_ids if tid not in val_tracks]
    logging.info(f"Validation tracks: {', '.join(val_tracks)}")
    logging.info(
        f"Training tracks: {', '.join(train_tracks)}")

    train_df = preprocessor.generate_training_metadata(
        train_tracks, split_type=SplitType.TRAIN)
    val_df = preprocessor.generate_training_metadata(
        val_tracks, split_type=SplitType.VAL)
    df = pd.concat([train_df, val_df], axis=0)

    # Save the generated metadata
    if not df.empty:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(
            f"Saved {len(df)} metadata to {output_path}. Train: {df['set'].value_counts().get('train', 0)}, Val: {df['set'].value_counts().get('val', 0)}")
    else:
        logging.error("No training samples generated.")


if __name__ == "__main__":
    main()
