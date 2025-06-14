## Thoughts

What's the next step?

I should be able to evaluate end to end on the actual task - no matter how I frame the ML problem. I should refactor the code to handle this with flexibility for multiple approaches.

Step 1: download gopro videos (universal)
Step 2: annotate splits for each trackId and riderId (universal)
Step 3: Generate sample indices (should use a class)

```python
# generate_training_samples.py
train_tracks = ...
val_tracks = ...
dfs = []
preprocessor = get_preprocessor(args.preprocessor_type)
for track_id, track_videos_list in track_videos.items():
  samples = preprocessor.generate_samples(track_videos_list)
  dfs.append(pd.DataFrame(samples))
df = pd.concat(dfs, axis=0)
...

# extract_clip_features.py
preprocessor = get_preprocessor(args.preprocessor_type)
for gopro_video_path in gopro_videos:
  preprocessor.write_video_features(gopro_video_path, output_feature_path)

# preprocess_videos_into_samples.py
df = pd.read_csv(args.csv_path)
preprocessor = get_preprocessor(args.preprocessor_type)
sample_metadatas = []
idx = 0
for sample_metadata in tqdm(df.itertuples(), total=len(df), desc="Generating samples"):
  if idx >= args.batch_size or idx == len(df) - 1:
    preprocessor.write_sample(sample_metadatas)
    sample_metadatas = [sample_metadata]
  else:
    sample_metadatas.append(sample_metadata)
  idx += 1

# train_model.py
model_cls = get_model_class(args.model_type)
if args.resume_from and os.path.isfile(args.resume_from):
  model_cls.load(args.resume_from)
else:
  model = model_cls()
model.train(train_dir, val_dir)

# validate_model.py
predictions = []
for videoPath in args.validation_videos:
  trackId, riderId = ...
  true_split_timecodes = load_splits_from_config(trackId, riderId)
  predicted_split_timecodes = model.predict(trackId: str, videoPath: str)
  predictions.append({'trackId': trackId, 'riderId': riderId, 'true_split_timecodes': true_split_timecodes, 'predicted_split_timecodes': predicted_split_timecodes})

eval_metrics = compute_eval_metrics(predictions)
results = {'metrics': eval_metrics, 'predictions': predictions}
```

Classifiers do poorly on the target task. Should I improve the data augmentation for the regressor to handle clips that don't overlap???

## TODO

- reframe as ML problem
- annotate more videos (ews runs too)
- update `inspect_training_data.py` to show videos side by side and verify that the training data quality is good

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
    --num_augmented_positives_per_segment 50 \
    --num_non_overlapping_samples_per_positive 50 \
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
rm -rf ./training_data/train && rm -rf ./training_data/val

rm -rf ./training_data_regression
```

```bash
python preprocess_videos_into_samples.py training_data/training_metadata.csv video_features training_data --F=50 --add_position_feature --add_percent_completion_feature --batch_size=32 --log-level DEBUG

python preprocess_videos_into_samples_regression.py training_data/training_metadata_regression.csv video_features training_data_regression --F=50 --batch_size=32 --log-level DEBUG
```

Train and evaluate a model on the data

```bash
python train_position_classifier.py training_data --bidirectional --compress_sizes 128 --interaction_type mlp --hidden_size 128 --post_lstm_sizes 64 --learning_rate 0.0001 --dropout 0.5 --eval_interval 1 --checkpoint_interval 1

python train_position_classifier_regression.py training_data_regression --bidirectional --compress_sizes 128 --interaction_type mlp --hidden_size 128 --post_lstm_sizes 64 --learning_rate 0.0001 --eval_interval 1 --dropout 0.5 --checkpoint_interval 1
```

Find splits in a target video using the model

```bash
python find_splits.py video_config.yaml video_features predictions.json --trackId leogang_2025 --F 50 --sourceRiderId asa_vermette --targetRiderId jordan_williams --checkpoint_path artifacts/alpha0_0_5_alpha_0_5_beta0_0_5_beta_0_5_frames_50_augmented_50_nopos_nopct_20250611_205932/checkpoints/checkpoint_epoch_8.pth

python find_splits.py video_config.yaml video_features predictions.json --trackId loudenvielle_2025 --F 50 --sourceRiderId amaury_pierron --targetRiderId joe_breeden --checkpoint_path artifacts/alpha0_0_5_alpha_0_5_beta0_0_5_beta_0_5_frames_50_augmented_50_nopos_nopct_20250611_205932/checkpoints/checkpoint_epoch_8.pth

python find_splits_regression.py video_config.yaml video_features predictions.json --trackId leogang_2025 --F 50 --sourceRiderId asa_vermette --targetRiderId jordan_williams --checkpoint_path artifacts/experiment_20250612_061757/checkpoints/checkpoint_epoch_2.pth
```

View the predictions

```bash
python view_predictions.py --predictions_json ./predictions.json && open ./predictions_splits.html
```
