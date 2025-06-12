## Thoughts

Predicting the offset between the same split index in 2 different clips (instead of a binary label) gives more signal.

I think the offset should be less than the clip length for all samples. If there is no overlap between 2 clips then it is not possible to predict the offset.

For the frame index of each split in video 1:
    Find the frame index of the same split in video 2
    For each value in range(video_2_frame_idx - F, video_2_frame_idx + F):
        Create a sample.

In this case, when generating samples I would do so around matching indices. For example, align the frames then iterate in both directions generating labels.


## TODO

- find best hyperparams on validation data
- visually insepct the predictions on a test sample
- annotate more videos (ews runs too)
- reframe as ML problem
- update `inspect_training_data.py` to show videos side by side and verify that the training data quality is good

```bash
# best values
./run_pipeline.sh --alpha_split_0 0.7 --alpha 0.7 --beta_split_0 0.7 --beta 0.7 --clip_length 100 --num_augmented 50

./hyperparameter_search.sh \
  --alpha_split_0_range 0.5:0.1:0.7 \
  --alpha_range 0.5:0.1:0.7 \
  --beta_split_0_range 0.5:0.1:0.7 \
  --beta_range 0.5:0.1:0.7 \
  --clip_length_range 50:50:150 \
  --num_augmented_range 25:25:75 \
  --add_position_feature_values true false \
  --add_percent_completion_feature_values true false
```


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
python generate_training_samples.py --config video_config.yaml --clip-length 50 --ignore_first_split --max_negatives_per_positive 1 --num_augmented_positives_per_segment 50 --alpha_split_0 0.5 --alpha 0.5 --beta_split_0 0.5 --beta 0.5 --seed 1

python generate_training_samples_regression.py --config video_config.yaml --clip-length 50 --ignore_first_split --num_augmented_positives_per_segment 50 --alpha_split_0 0.5 --alpha 0.5 --beta_split_0 0.5 --beta 0.5 --seed 1
```

Inspect the labels

```bash
python inspect_training_data.py training_data/training_metadata.csv --num_samples=15 --sample_types augmented && \
open ./training_data_inspection/index.html 

python inspect_training_data_regression.py training_data/training_metadata_regression.csv --num_samples=15 --sample_types split_offset && \
open ./training_data_inspection/index.html 
```

Compute features for each frame index and save to disk

```bash
python extract_clip_features.py downloaded_videos video_features --feature-extraction-batch-size=5 --clip-length=50 --log-level DEBUG
```

Preprocess videos into training samples and save to disk

```bash
rm -rf ./training_data/train && rm -rf ./training_data/val
```


```bash
python preprocess_videos_into_samples.py training_data/training_metadata.csv video_features training_data --F=50 --add_position_feature --add_percent_completion_feature --batch_size=32 --log-level DEBUG

python preprocess_videos_into_samples_regression.py training_data/training_metadata_regression.csv video_features training_data_regression --F=50 --add_position_feature --add_percent_completion_feature --batch_size=32 --log-level DEBUG
```

Train and evaluate a model on the data

```bash
python train_position_classifier.py training_data --bidirectional --compress_sizes 128 --interaction_type mlp --hidden_size 128 --post_lstm_sizes 64 --learning_rate 0.0001 --dropout 0.5 --eval_interval 1 --checkpoint_interval 1

python train_position_classifier_regression.py training_data_regression --bidirectional --compress_sizes 128 --interaction_type mlp --hidden_size 128 --post_lstm_sizes 64 --learning_rate 0.0001 --dropout 0.5 --eval_interval 1 --checkpoint_interval 1
```

Find splits in a target video using the model

```bash
python find_splits.py video_config.yaml video_features predictions.json --trackId leogang_2025 --F 100 --sourceRiderId asa_vermette --targetRiderId jordan_williams --add_position_feature --add_percent_completion_feature --checkpoint_path artifacts/alpha0_0_7_alpha_0_7_beta0_0_7_beta_0_7_frames_100_augmented_50_20250611_142150/checkpoints/checkpoint_epoch_3.pth
```

View the predictions
```bash
python view_predictions.py --predictions_json ./predictions.json && open ./predictions_splits.html
```