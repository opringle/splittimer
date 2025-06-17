## Thoughts

Could I efficiently construct batches on the fly? That way I wouldn't need duplicate features stored on disk.
Alternatively I could have each clips features stored once. Then with some smarter dataloading I could read from disk.

## TODO

- refactor python files to interface implementations
- rewrite pipeline and hyperopt for new code
- annotate more videos (ews runs too)

```bash
# best combo: step 7 (epoch 8) 0.905 macro F1 score on validation data
./run_pipeline.sh --alpha_split_0 0.5 --alpha 0.5 --beta_split_0 0.5 --beta 0.5 --clip_length 50 --num_augmented 50 --no-add_position_feature --no-add_percent_completion_feature

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
# classification
python generate_training_samples.py \
    --config video_config.yaml \
    --val_ratio 0.2 \
    --log-level INFO \
    --seed 42 \
    --output_path training_data/metadata_classification.csv \
    --preprocessor_type classifier \
    --clip-length 50 \
    --alpha_split_0 0.5 \
    --alpha 0.5 \
    --beta_split_0 0.5 \
    --beta 0.5 \
    --max_negatives_per_positive 1 \
    --num_augmented_positives_per_segment 50 \
    --ignore_first_split

# regression
python generate_training_samples.py \
    --config video_config.yaml \
    --val_ratio 0.2 \
    --log-level INFO \
    --seed 42 \
    --output_path training_data/metadata_regression.csv \
    --preprocessor_type regressor \
    --clip-length 50 \
    --alpha_split_0 0.5 \
    --alpha 0.5 \
    --beta_split_0 0.5 \
    --beta 0.5 \
    --num_augmented_positives_per_segment 3 \
    --num_non_overlapping_samples_per_positive 0 \
    --ignore_first_split
```

Inspect the labels

```bash
# classification
python inspect_training_data.py training_data/metadata_classification.csv --num_samples=3 && \
open ./training_data_inspection/index.html

# regression
python inspect_training_data.py training_data/metadata_regression.csv --num_samples=3 && \
open ./training_data_inspection/index.html
```

Compute features for each frame index and save to disk

```bash
python extract_clip_features.py downloaded_videos video_features --feature-extraction-batch-size=5 --clip-length=50 --log-level DEBUG
```

Preprocess videos into training samples and save to disk

```bash
python preprocess_videos_into_samples.py \
    training_data/metadata_classification.csv \
    video_features \
    training_data_classification \
    --seed=42 \
    --batch_size=32 \
    --sample_generator_type classifier \
    --log-level DEBUG \
    --F=50 \
    --add_position_feature \
    --add_percent_completion_feature

python preprocess_videos_into_samples.py \
    training_data/metadata_regression.csv \
    video_features \
    training_data_regression \
    --seed=42 \
    --batch_size=32 \
    --sample_generator_type regressor \
    --log-level DEBUG \
    --F=50 \
    --add_position_feature \
    --add_percent_completion_feature
```

Train and evaluate a model on the data

```bash
# classification
python train_model.py \
    training_data_classification \
    --eval_interval 1 \
    --checkpoint_interval 1 \
    --learning_rate 0.0001 \
    --trainer_type classifier \
    --bidirectional \
    --compress_sizes 128 \
    --interaction_type mlp \
    --hidden_size 128 \
    --post_lstm_sizes 64 \
    --dropout 0.5 \
    --image_feature_path video_features \
    --add_position_feature \
    --add_percent_completion_feature

# regression
python train_model.py \
    training_data_regression \
    --eval_interval 1 \
    --checkpoint_interval 1 \
    --learning_rate 0.0001 \
    --trainer_type regressor \
    --bidirectional \
    --compress_sizes 128 \
    --interaction_type mlp \
    --hidden_size 128 \
    --post_lstm_sizes 64 \
    --dropout 0.5
```

Evaluate

```bash
# classification
python evaluate.py \
    video_config.yaml \
    video_features \
    predictions.json \
    --trackId leogang_2025 \
    --sourceRiderId asa_vermette \
    --targetRiderIds jordan_williams \
    --checkpoint_path ./artifacts/experiment_20250617_063046/checkpoints/checkpoint_0.pt \
    --log-level INFO \
    --trainer_type classifier \
    --sample_generator_type classifier

# regressor
python evaluate.py \




python find_splits.py video_config.yaml video_features predictions.json --trackId leogang_2025 --F 50 --sourceRiderId asa_vermette --targetRiderId jordan_williams --checkpoint_path artifacts/alpha0_0_5_alpha_0_5_beta0_0_5_beta_0_5_frames_50_augmented_50_nopos_nopct_20250611_205932/checkpoints/checkpoint_epoch_8.pth

python find_splits.py video_config.yaml video_features predictions.json --trackId loudenvielle_2025 --F 50 --sourceRiderId amaury_pierron --targetRiderId joe_breeden --checkpoint_path artifacts/alpha0_0_5_alpha_0_5_beta0_0_5_beta_0_5_frames_50_augmented_50_nopos_nopct_20250611_205932/checkpoints/checkpoint_epoch_8.pth

python find_splits_regression.py video_config.yaml video_features predictions.json --trackId leogang_2025 --F 50 --sourceRiderId asa_vermette --targetRiderId jordan_williams --checkpoint_path artifacts/experiment_20250612_061757/checkpoints/checkpoint_epoch_2.pth
```

View the predictions

```bash
python view_predictions.py --predictions_json ./predictions.json && open ./predictions_splits.html
```
