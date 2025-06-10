## TODO

- achieve perfect validation score??? Seems an easy task. This should be pos
- update preview labels to look at 2 video clips and a label

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

Manually add splits in the config file using Davinci Resolve with 24.0 FPS. You can inspect annotations like this:

```bash
python inspect_splits.py video_config.yaml && \
open ./split_times_inspection/index.html
```

Generate positive and negative labels to train any model type

```bash
python generate_training_samples.py --config video_config.yaml --clip-length 50 --ignore_first_split --max_negatives_per_positive 1 --num_augmented_positives_per_segment 50
```

Inspect the labels

```bash
python inspect_training_data.py training_data/training_metadata.csv --num_samples=15 --sample_types augmented && \
open ./training_data_inspection/index.html 
```

Compute features for each frame index and save to disk

```bash
python extract_clip_features.py downloaded_videos video_features --feature-extraction-batch-size=5 --clip-length=50 --feature-types individual --log-level DEBUG
```

Or compute sequence features leading up to each frame index and save to disk

```bash
python extract_clip_features.py downloaded_videos video_features --feature-extraction-batch-size=5 --clip-length=50 --sequence-length=50 --feature-types sequence --log-level DEBUG
```

Preprocess videos into training samples and save to disk

```bash
rm -rf ./training_data/train && rm -rf ./training_data/val
```


```bash
python preprocess_videos_into_samples.py training_data/training_metadata.csv video_features training_data --F=50 --batch_size=32 --feature_type individual --log-level DEBUG
```

```bash
python preprocess_videos_into_samples.py training_data/training_metadata.csv video_features training_data --F=50 --batch_size=32 --log-level DEBUG --feature_type sequence --sequence_length 50
```

Train and evaluate a model on the data

```bash
# no compression before lstm. dot product after
python train_position_classifier.py training_data --bidirectional --hidden_size 512 --interaction_type dot --learning_rate 0.01 --dropout 0.0 --eval_interval 1

# compress clips before lstm. dot product after
python train_position_classifier.py training_data --bidirectional --compress_sizes 512,128 --interaction_type dot --hidden_size 64 --learning_rate 0.0001 --dropout 0.0 --eval_interval 1 --checkpoint_interval 1

# best yet: 0.8806 macro average F1 score on an unseen track
python train_position_classifier.py training_data --bidirectional --compress_sizes 128 --interaction_type mlp --hidden_size 128 --post_lstm_sizes 64 --learning_rate 0.0001 --dropout 0.5 --eval_interval 1 --checkpoint_interval 1

python train_position_classifier.py training_data --bidirectional --compress_sizes 128 --interaction_type mlp --hidden_size 128 --post_lstm_sizes 64 --learning_rate 0.0001 --dropout 0.6 --eval_interval 1 --checkpoint_interval 1
```

```bash
TODO train on I3D features instead of sequence of resnet features
```

Find splits in a target video using the model

```bash
python find_splits.py --config_path video_config.yaml --feature_base_path ./video_features --trackId loudenvielle_2025 --sourceRiderId amaury_pierron --targetRiderId vali_holl --checkpoint_path artifacts/experiment_20250607_071920/checkpoints/checkpoint_epoch_1.pth --frame_rate=25.0
```

View the predictions
```bash
python view_predictions.py --predictions_json ./predicted_splits.json --trackId loudenvielle_2025 --sourceRiderId amaury_pierron --targetRiderId vali_holl
```