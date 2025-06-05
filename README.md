
## Prerequisites

- install pyenv
- configure python environment

```bash
pyenv install 3.12

pyenv virtualenv 3.12 splittimer

pyenv local splittimer

pip install -r ./requirements.txt
```

- install ffmpeg

```bash
brew install ffmpeg
```

## Preprocess GoPro runs

Download youtube videos based on config file

```bash
python youtube_download.py --config video_config.yaml
```

Compute image features from frames and save to disk

```bash
python extract_clip_features.py downloaded_videos video_features --feature-extraction-batch-size=5 --clip-length=50 --log-level DEBUG
```

Generate generalized training data for identifying splits

```bash
python generate_training_samples.py --config video_config.yaml --max_negatives_per_positive 10 --num_augmented_positives_per_segment 50 --log-level DEBUG
```

Inspect the training data

```bash
python inspect_training_data.py training_data/training_metadata.csv --num_samples 10 && open ./training_data_inspection/index.html
```

Preprocess videos into training samples and save to disk

```bash
python preprocess_videos_into_samples.py training_data/training_metadata.csv video_features training_data --F=50 --batch_size=32 --log-level DEBUG
```

Train and evaluate a model on the data

```bash
python train_position_classifier.py training_data --bidirectional --compress_sizes 1024,512 --hidden_size 256 --post_lstm_sizes 256,128
```