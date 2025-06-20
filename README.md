## Thoughts

- Disk storage is limitation for frame offset model.
- I could write features for each sample at every index, then construct batches on the fly
- That way I wouldn't need duplicate sample features stored on disk
- I should only do this if the regressor performs better with more data

## TODO

- annotate more videos (ews runs too)
- consider LLM for video encoding

```bash
# best classifier:
# Val mont_sainte_anne_2024: mean average split prediction error (seconds) = 63.257
# Train loudenvielle_2025: mean average split prediction error (seconds) = 114.744
# Train leogang_2025: mean average split prediction error (seconds) = 29.110
# Train poland_2025: mean average split prediction error (seconds) = 45.766
# Train val_di_sole_2024: mean average split prediction error (seconds) = 46.440
./scripts/run_pipeline_classifier.sh --alpha_split_0 0.5 --alpha 0.5 --beta_split_0 0.5 --beta 0.5 --clip_length 50 --num_augmented 50 --no-add_position_feature --no-add_percent_completion_feature

# best regressor:
# Val mont_sainte_anne_2024: mean average split prediction error (seconds) = 88.074
# Train loudenvielle_2025: mean average split prediction error (seconds) = 91.060
# Train leogang_2025: mean average split prediction error (seconds) = 81.542
# Train poland_2025: mean average split prediction error (seconds) = 59.225
# Train val_di_sole_2024: mean average split prediction error (seconds) = 82.391

./scripts/run_pipeline_regressor.sh --alpha_split_0 0.5 --alpha 0.5 --beta_split_0 0.5 --beta 0.5 --clip_length 50 --num_augmented 4 --no-add_position_feature --no-add_percent_completion_feature
```

Search for best hyperparams over entire training pipeline

```bash
./scripts/hyperparameter_search_classifier.sh \
  --alpha_split_0_range 0.5:0.1:0.7 \
  --alpha_range 0.5:0.1:0.7 \
  --beta_split_0_range 0.5:0.1:0.7 \
  --beta_range 0.5:0.1:0.7 \
  --clip_length_range 50:50:150 \
  --num_augmented_range 25:25:75 \
  --add_position_feature_values true false \
  --add_percent_completion_feature_values true false

./scripts/hyperparameter_search_regressor.sh \
  --alpha_split_0_range 0.5:0.1:0.5 \
  --alpha_range 0.5:0.1:0.5 \
  --beta_split_0_range 0.5:0.1:0.5 \
  --beta_range 0.5:0.1:0.5 \
  --clip_length_range 50:1:50 \
  --num_augmented_range 4:1:4 \
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
    --F=50 \
    --add_position_feature \
    --add_percent_completion_feature \
    --log-level DEBUG

python preprocess_videos_into_samples.py \
    training_data/metadata_regression.csv \
    video_features \
    training_data_regression \
    --seed=42 \
    --batch_size=32 \
    --sample_generator_type regressor \
    --F=50 \
    --add_position_feature \
    --add_percent_completion_feature \
    --log-level DEBUG
```

Train a model on the data

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
    --add_percent_completion_feature \
    --log-level DEBUG

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
    --dropout 0.5 \
    --image_feature_path video_features \
    --add_position_feature \
    --add_percent_completion_feature \
    --log-level DEBUG
```

Evaluate

```bash
# classification
python evaluate.py \
    video_config.yaml \
    video_features \
    predictions.json \
    --trackIds val_di_sole_2024 leogang_2025 \
    --checkpoint_path ./artifacts/alpha0_0_5_alpha_0_5_beta0_0_5_beta_0_5_frames_50_augmented_50_nopos_nopct_20250620_154155/checkpoints/checkpoint_4.pt \
    --trainer_type classifier \
    --image_feature_path video_features \
    --seed 1 \
    --log-level DEBUG

# regressor
python evaluate.py \
    video_config.yaml \
    video_features \
    predictions.json \
    --trackIds val_di_sole_2024 leogang_2025 \
    --checkpoint_path ./artifacts/alpha0_0_5_alpha_0_5_beta0_0_5_beta_0_5_frames_50_augmented_4_nopos_nopct_20250620_155805/checkpoints/checkpoint_0.pt \
    --trainer_type regressor \
    --image_feature_path video_features \
    --seed 1 \
    --log-level DEBUG
```

View the predictions

```bash
python view_predictions.py --predictions_json ./predictions.json && open ./predictions_splits.html
```
