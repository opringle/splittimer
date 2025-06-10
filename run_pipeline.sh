#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 alpha_split_0 alpha beta_split_0 beta"
    exit 1
fi

# Assign arguments to variables
ALPHA_SPLIT_0=$1
ALPHA=$2
BETA_SPLIT_0=$3
BETA=$4

# Base command parameters
CONFIG="video_config.yaml"
CLIP_LENGTH=50
MAX_NEGATIVES=1
NUM_AUGMENTED=50
FEATURE_TYPE="individual"
F=50
BATCH_SIZE=32
BIDIRECTIONAL="--bidirectional"
COMPRESS_SIZES=128
INTERACTION_TYPE="mlp"
HIDDEN_SIZE=128
POST_LSTM_SIZES=64
LEARNING_RATE=0.0001
DROPOUT=0.5
EVAL_INTERVAL=1

# Log the parameters being used
echo "Running pipeline with alpha_split_0=$ALPHA_SPLIT_0, alpha=$ALPHA, beta_split_0=$BETA_SPLIT_0, beta=$BETA"

# Run the first Python script: generate training samples
python generate_training_samples.py \
    --config "$CONFIG" \
    --clip-length $CLIP_LENGTH \
    --ignore_first_split \
    --max_negatives_per_positive $MAX_NEGATIVES \
    --num_augmented_positives_per_segment $NUM_AUGMENTED \
    --alpha_split_0 $ALPHA_SPLIT_0 \
    --alpha $ALPHA \
    --beta_split_0 $BETA_SPLIT_0 \
    --beta $BETA

# Clean up training data directories
rm -rf ./training_data/train
rm -rf ./training_data/val

# Run the second Python script: preprocess videos
python preprocess_videos_into_samples.py \
    training_data/training_metadata.csv \
    video_features \
    training_data \
    --F=$F \
    --batch_size=$BATCH_SIZE \
    --feature_type $FEATURE_TYPE

# Run the third Python script: train classifier
python train_position_classifier.py \
    training_data \
    $BIDIRECTIONAL \
    --compress_sizes $COMPRESS_SIZES \
    --interaction_type $INTERACTION_TYPE \
    --hidden_size $HIDDEN_SIZE \
    --post_lstm_sizes $POST_LSTM_SIZES \
    --learning_rate $LEARNING_RATE \
    --dropout $DROPOUT \
    --eval_interval $EVAL_INTERVAL

echo "Pipeline completed"