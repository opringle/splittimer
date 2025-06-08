## TODO

- label 25 videos from 5 tracks
- achieve validation F1 score on positive class > 0.95

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

Generate labels to train any model

```bash
python generate_training_samples.py --config video_config.yaml --max_negatives_per_positive 1 --num_augmented_positives_per_segment 50 --log-level DEBUG
```

Inspect the labels

```bash
python inspect_training_data.py training_data/training_metadata.csv --num_samples 10 && open ./training_data_inspection/index.html
```

Compute features for each frame index and save to disk

```bash
python extract_clip_features.py downloaded_videos video_features --feature-extraction-batch-size=5 --clip-length=50  --feature-types individual --log-level DEBUG
```

Or compute sequence features leading up to each frame index and save to disk

```bash
python extract_clip_features.py downloaded_videos video_features --feature-extraction-batch-size=5 --clip-length=50 --sequence-length=50 --feature-types sequence --log-level DEBUG
```

Preprocess videos into training samples and save to disk

```bash
rm -rf ./training_data/train && rm -rf ./training_data/val && \
```

```bash
python preprocess_videos_into_samples.py training_data/training_metadata.csv video_features training_data --F=50 --batch_size=32 --log-level DEBUG --feature_type individual
```

```bash
python preprocess_videos_into_samples.py training_data/training_metadata.csv video_features training_data --F=50 --batch_size=32 --log-level DEBUG --feature_type sequence --sequence_length 50
```

Train and evaluate a model on the data

```bash
python train_position_classifier.py training_data --bidirectional --compress_sizes 1024,512 --hidden_size 256 --post_lstm_sizes 256,128 --learning_rate 0.0001 --dropout 0.0 --eval_interval 1 --checkpoint_interval 1
```

```bash
TODO
```

Find splits in a target video using the model

```bash
python find_splits.py --config_path video_config.yaml --feature_base_path ./video_features --trackId loudenvielle_2025 --sourceRiderId amaury_pierron --targetRiderId vali_holl --checkpoint_path artifacts/experiment_20250607_071920/checkpoints/checkpoint_epoch_1.pth --frame_rate=25.0
```

View the predictions
```bash
python view_predictions.py --predictions_json ./predicted_splits.json --trackId loudenvielle_2025 --sourceRiderId amaury_pierron --targetRiderId vali_holl
```