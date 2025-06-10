## TODO

- ensure training results are reproducible with random seeds
- annotate more videos (ews runs too)

```bash
# best values
./run_pipeline.sh --alpha_split_0 0.7 --alpha 0.7 --beta_split_0 0.7 --beta 0.7 --clip_length 100 --num_augmented 50
```

- update `inspect_training_data.py` to show videos side by side and verify that the training data quality is good

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

## Train position classifiers

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
python generate_training_samples.py --config video_config.yaml --clip-length 50 --ignore_first_split --max_negatives_per_positive 1 --num_augmented_positives_per_segment 50 --alpha_split_0 0.5 --alpha 0.5 --beta_split_0 0.5 --beta 0.5
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
python preprocess_videos_into_samples.py training_data/training_metadata.csv video_features training_data --F=50 --add_position_feature --add_percent_completion_feature --batch_size=32 --feature_type individual --log-level DEBUG
```

```bash
python preprocess_videos_into_samples.py training_data/training_metadata.csv video_features training_data --F=50 --batch_size=32 --log-level DEBUG --feature_type sequence --sequence_length 50
```

Train and evaluate a model on the data

```bash
python train_position_classifier.py training_data --bidirectional --compress_sizes 128 --interaction_type mlp --hidden_size 128 --post_lstm_sizes 64 --learning_rate 0.0001 --dropout 0.5 --eval_interval 1 --checkpoint_interval 1
```

```bash
TODO train on I3D features instead of sequence of resnet features
```

Find splits in a target video using the model

```bash
python find_splits.py video_config.yaml video_features --trackId leogang_2025 --F 100 --sourceRiderId asa_vermette --targetRiderId jordan_williams --frame_rate=100 --checkpoint_path artifacts/alpha0_0_7_alpha_0_7_beta0_0_7_beta_0_7_frames_100_augmented_50/checkpoints/checkpoint_epoch_7.pth
```

View the predictions
```bash
python view_predictions.py --predictions_json ./predicted_splits.json --trackId loudenvielle_2025 --sourceRiderId amaury_pierron --targetRiderId vali_holl
```