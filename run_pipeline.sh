#!/bin/bash

# Initialize variables with empty values
ALPHA_SPLIT_0=""
ALPHA=""
BETA_SPLIT_0=""
BETA=""
CLIP_LENGTH=""
NUM_AUGMENTED=""
ADD_POSITION_FEATURE="true"
ADD_PERCENT_COMPLETION_FEATURE="true"

# Function to display usage
usage() {
    echo "Usage: $0 --alpha_split_0 <value> --alpha <value> --beta_split_0 <value> --beta <value> --clip_length <value> --num_augmented <value> [--add_position_feature] [--add_percent_completion_feature]"
    exit 1
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --alpha_split_0)
            ALPHA_SPLIT_0="$2"
            shift 2
            ;;
        --alpha)
            ALPHA="$2"
            shift 2
            ;;
        --beta_split_0)
            BETA_SPLIT_0="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        --clip_length)
            CLIP_LENGTH="$2"
            shift 2
            ;;
        --num_augmented)
            NUM_AUGMENTED="$2"
            shift 2
            ;;
        --add_position_feature)
            ADD_POSITION_FEATURE="true"
            shift 1
            ;;
        --no-add_position_feature)
            ADD_POSITION_FEATURE="false"
            shift 1
            ;;
        --add_percent_completion_feature)
            ADD_PERCENT_COMPLETION_FEATURE="true"
            shift 1
            ;;
        --no-add_percent_completion_feature)
            ADD_PERCENT_COMPLETION_FEATURE="false"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if all required arguments are provided
if [ -z "$ALPHA_SPLIT_0" ] || [ -z "$ALPHA" ] || [ -z "$BETA_SPLIT_0" ] || [ -z "$BETA" ] || [ -z "$CLIP_LENGTH" ] || [ -z "$NUM_AUGMENTED" ]; then
    echo "Error: All required arguments (--alpha_split_0, --alpha, --beta_split_0, --beta, --clip_length, --num_augmented) are required."
    usage
fi

# Validate clip_length and num_augmented are positive integers
if ! [[ "$CLIP_LENGTH" =~ ^[0-9]+$ ]] || [ "$CLIP_LENGTH" -le 0 ]; then
    echo "Error: --clip_length must be a positive integer."
    exit 1
fi
if ! [[ "$NUM_AUGMENTED" =~ ^[0-9]+$ ]] || [ "$NUM_AUGMENTED" -le 0 ]; then
    echo "Error: --num_augmented must be a positive integer."
    exit 1
fi

# Base command parameters
CONFIG="video_config.yaml"
MAX_NEGATIVES=1
SEED=1
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
echo "Running pipeline with alpha_split_0=$ALPHA_SPLIT_0, alpha=$ALPHA, beta_split_0=$BETA_SPLIT_0, beta=$BETA, clip_length=$CLIP_LENGTH, num_augmented=$NUM_AUGMENTED, add_position_feature=$ADD_POSITION_FEATURE, add_percent_completion_feature=$ADD_PERCENT_COMPLETION_FEATURE"

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
    --beta $BETA \
    --seed $SEED

# Clean up training data directories
rm -rf ./training_data/train
rm -rf ./training_data/val

# Build the preprocess command with optional feature flags
PREPROCESS_CMD="python preprocess_videos_into_samples.py \
    training_data/training_metadata.csv \
    video_features \
    training_data \
    --F=$CLIP_LENGTH \
    --batch_size=$BATCH_SIZE \
    --seed $SEED"

if [ "$ADD_POSITION_FEATURE" = "true" ]; then
    PREPROCESS_CMD="$PREPROCESS_CMD --add_position_feature"
fi
if [ "$ADD_PERCENT_COMPLETION_FEATURE" = "true" ]; then
    PREPROCESS_CMD="$PREPROCESS_CMD --add_percent_completion_feature"
fi

# Run the second Python script: preprocess videos
$PREPROCESS_CMD

# Format alpha and beta values for directory name (replace . with _)
ALPHA_SPLIT_0_DIR=${ALPHA_SPLIT_0//./_}
ALPHA_DIR=${ALPHA//./_}
BETA_SPLIT_0_DIR=${BETA_SPLIT_0//./_}
BETA_DIR=${BETA//./_}

# Convert boolean flags to short strings for directory name
POS_FEATURE=$([ "$ADD_POSITION_FEATURE" = "true" ] && echo "pos" || echo "nopos")
PCT_FEATURE=$([ "$ADD_PERCENT_COMPLETION_FEATURE" = "true" ] && echo "pct" || echo "nopct")

# Generate timestamp in YYYYMMDD_HHMMSS format
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Construct artifacts directory path with timestamp and feature flags
ARTIFACTS_DIR="artifacts/alpha0_${ALPHA_SPLIT_0_DIR}_alpha_${ALPHA_DIR}_beta0_${BETA_SPLIT_0_DIR}_beta_${BETA_DIR}_frames_${CLIP_LENGTH}_augmented_${NUM_AUGMENTED}_${POS_FEATURE}_${PCT_FEATURE}_${TIMESTAMP}"

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
    --eval_interval $EVAL_INTERVAL \
    --artifacts_dir $ARTIFACTS_DIR \
    --checkpoint_interval 1 \
    --seed $SEED

echo "Pipeline completed"