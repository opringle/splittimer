#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --alpha_split_0_range <start:step:end> --alpha_range <start:step:end> --beta_split_0_range <start:step:end> --beta_range <start:step:end> --clip_length_range <start:step:end> --num_augmented_range <start:step:end> --add_position_feature_values <value1 value2 ...> --add_percent_completion_feature_values <value1 value2 ...>"
    echo "Example: $0 --alpha_split_0_range 0.1:0.1:0.5 --alpha_range 0.1:0.1:0.5 --beta_split_0_range 0.1:0.1:0.5 --beta_range 0.1:0.1:0.5 --clip_length_range 10:10:30 --num_augmented_range 1:1:3 --add_position_feature_values true false --add_percent_completion_feature_values true"
    exit 1
}

# Function to validate range format
validate_range() {
    local range=$1
    local param=$2
    IFS=':' read -r start step end <<< "$range"
    if [ -z "$start" ] || [ -z "$step" ] || [ -z "$end" ]; then
        echo "Error: Invalid range format for $param: $range. Expected start:step:end"
        exit 1
    fi
    # Check if step is positive
    if (( $(echo "$step <= 0" | bc -l) )); then
        echo "Error: Step must be positive for $param: $range"
        exit 1
    fi
}

# Initialize variables
alpha_split_0_range=""
alpha_range=""
beta_split_0_range=""
beta_range=""
clip_length_range=""
num_augmented_range=""
add_position_feature_values=()
add_percent_completion_feature_values=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --alpha_split_0_range)
            alpha_split_0_range="$2"
            shift 2
            ;;
        --alpha_range)
            alpha_range="$2"
            shift 2
            ;;
        --beta_split_0_range)
            beta_split_0_range="$2"
            shift 2
            ;;
        --beta_range)
            beta_range="$2"
            shift 2
            ;;
        --clip_length_range)
            clip_length_range="$2"
            shift 2
            ;;
        --num_augmented_range)
            num_augmented_range="$2"
            shift 2
            ;;
        --add_position_feature_values)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                add_position_feature_values+=("$1")
                shift
            done
            ;;
        --add_percent_completion_feature_values)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                add_percent_completion_feature_values+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if all required arguments are provided
if [ -z "$alpha_split_0_range" ] || [ -z "$alpha_range" ] || [ -z "$beta_split_0_range" ] || [ -z "$beta_range" ] || [ -z "$clip_length_range" ] || [ -z "$num_augmented_range" ] || [ ${#add_position_feature_values[@]} -eq 0 ] || [ ${#add_percent_completion_feature_values[@]} -eq 0 ]; then
    echo "Error: All range arguments are required."
    usage
fi

# Validate ranges
validate_range "$alpha_split_0_range" "--alpha_split_0_range"
validate_range "$alpha_range" "--alpha_range"
validate_range "$beta_split_0_range" "--beta_split_0_range"
validate_range "$beta_range" "--beta_range"
validate_range "$clip_length_range" "--clip_length_range"
validate_range "$num_augmented_range" "--num_augmented_range"

# Validate flag values
for val in "${add_position_feature_values[@]}"; do
    if [ "$val" != "true" ] && [ "$val" != "false" ]; then
        echo "Error: Invalid value for --add_position_feature_values: $val. Must be 'true' or 'false'"
        exit 1
    fi
done
for val in "${add_percent_completion_feature_values[@]}"; do
    if [ "$val" != "true" ] && [ "$val" != "false" ]; then
        echo "Error: Invalid value for --add_percent_completion_feature_values: $val. Must be 'true' or 'false'"
        exit 1
    fi
done

# Generate values arrays
IFS=':' read -r start step end <<< "$alpha_split_0_range"
alpha_split_0_values=( $(seq $start $step $end) )
IFS=':' read -r start step end <<< "$alpha_range"
alpha_values=( $(seq $start $step $end) )
IFS=':' read -r start step end <<< "$beta_split_0_range"
beta_split_0_values=( $(seq $start $step $end) )
IFS=':' read -r start step end <<< "$beta_range"
beta_values=( $(seq $start $step $end) )
IFS=':' read -r start step end <<< "$clip_length_range"
clip_length_values=( $(seq $start $step $end) )
IFS=':' read -r start step end <<< "$num_augmented_range"
num_augmented_values=( $(seq $start $step $end) )

# Nested loops to run the script for each combination
for alpha_split_0 in "${alpha_split_0_values[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        for beta_split_0 in "${beta_split_0_values[@]}"; do
            for beta in "${beta_values[@]}"; do
                for clip_length in "${clip_length_values[@]}"; do
                    for num_augmented in "${num_augmented_values[@]}"; do
                        for add_position_feature in "${add_position_feature_values[@]}"; do
                            for add_percent_completion_feature in "${add_percent_completion_feature_values[@]}"; do
                                # Construct the command
                                cmd="./run_pipeline.sh --alpha_split_0 $alpha_split_0 --alpha $alpha --beta_split_0 $beta_split_0 --beta $beta --clip_length $clip_length --num_augmented $num_augmented"
                                if [ "$add_position_feature" = "true" ]; then
                                    cmd="$cmd --add_position_feature"
                                else
                                    cmd="$cmd --no-add_position_feature"
                                fi
                                if [ "$add_percent_completion_feature" = "true" ]; then
                                    cmd="$cmd --add_percent_completion_feature"
                                else
                                    cmd="$cmd --no-add_percent_completion_feature"
                                fi
                                echo "Running: $cmd"
                                # Execute the command
                                $cmd
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All runs completed"