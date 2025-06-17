import shutil
import pandas as pd
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
from sample_generator import get_sample_generator_class
from utils import save_batch, setup_logging, setup_seed


def main():
    parser = argparse.ArgumentParser(
        description="Generate batched training data using precomputed features.")
    parser.add_argument(
        "csv_path", type=str, help="Path to the CSV file with 'set' column ('train' or 'val')")
    parser.add_argument("image_feature_path", type=str,
                        help="Path to directory of clip features")
    parser.add_argument("output_dir", type=str,
                        help="Directory to save .npz files")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible track splitting")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Samples per .npz file")
    parser.add_argument("--sample_generator_type", type=str, default=None,
                        help="Type of sample generator to use")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args, _ = parser.parse_known_args()
    SampleGeneratorClass = get_sample_generator_class(
        args.sample_generator_type)
    SampleGeneratorClass.add_args(parser)
    args = parser.parse_args()

    setup_logging(args.log_level)
    setup_seed(args.seed)
    sample_generator = SampleGeneratorClass.from_args(args)

    df = pd.read_csv(args.csv_path)
    df = df.sample(frac=1, random_state=args.seed, ignore_index=True)
    train_df = df[df['set'] == 'train']
    val_df = df[df['set'] == 'val']
    logging.info(
        f"Loaded metadata for {len(train_df)} training and {len(val_df)} validation samples")

    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    if train_dir.exists():
        if not train_dir.is_dir():
            logging.error(
                f"{train_dir} exists but is not a directory. Please remove or rename it.")
            return
        logging.info(f"Removing existing training directory: {train_dir}")
        shutil.rmtree(train_dir)
    val_dir = output_dir / 'val'
    if val_dir.exists():
        if not val_dir.is_dir():
            logging.error(
                f"{val_dir} exists but is not a directory. Please remove or rename it.")
            return
        logging.info(f"Removing existing validation directory: {val_dir}")
        shutil.rmtree(val_dir)
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    video_feature_cache = {}
    train_batch_count = 0
    val_batch_count = 0

    # Process training samples
    train_samples = []
    for idx, row in enumerate(tqdm(train_df.itertuples(), total=len(train_df), desc="Generating training samples")):
        row_dict = row._asdict()
        sample = sample_generator.get_features(
            video_feature_cache, **row_dict)
        train_samples.append(sample)

        logging.debug(f"Train sample shapes: {[f.shape for f in sample]}")

        if len(train_samples) >= args.batch_size or idx == len(train_df) - 1:
            save_batch(train_dir, train_samples, idx)
            train_samples = []
            train_batch_count += 1
    # TODO: handle remaining samples

    # Process validation samples
    val_samples = []
    for idx, row in enumerate(tqdm(val_df.itertuples(), total=len(val_df), desc="Generating validation samples")):
        row_dict = row._asdict()
        val_samples.append(sample_generator.get_features(
            video_feature_cache, **row_dict))

        if len(val_samples) >= args.batch_size or idx == len(val_df) - 1:
            save_batch(val_dir, val_samples, idx)
            val_samples = []
            val_batch_count += 1
    # TODO: handle remaining samples

    logging.info(
        f"Generated {len(train_df)} training samples in {train_batch_count} batches")
    logging.info(
        f"Generated {len(val_df)} validation samples in {val_batch_count} batches")


if __name__ == "__main__":
    main()
