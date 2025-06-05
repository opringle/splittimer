import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import random

def denormalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a frame for visualization.
    
    Args:
        frame (np.ndarray): Normalized frame of shape (C, H, W).
        mean (list): Mean values used for normalization.
        std (list): Std dev values used for normalization.
    
    Returns:
        np.ndarray: Denormalized frame of shape (H, W, C) in [0,1].
    """
    frame = frame.copy()
    for c in range(3):
        frame[c] = frame[c] * std[c] + mean[c]
    frame = np.clip(frame, 0, 1)
    frame = np.transpose(frame, (1, 2, 0))
    return frame

def main():
    parser = argparse.ArgumentParser(description="Visualize training data samples by showing final frames side by side.")
    parser.add_argument("npz_file", type=str, help="Path to the .npz file containing training data (e.g., training_data/clip_training_data.npz)")
    parser.add_argument("--num_positive", type=int, default=5, help="Number of positive samples to view")
    parser.add_argument("--num_negative", type=int, default=5, help="Number of negative samples to view")
    parser.add_argument("--save_dir", type=str, help="Directory to save the figures (optional)")
    args = parser.parse_args()
    
    # Load the .npz file
    print(f"Loading data from {args.npz_file}")
    data = np.load(args.npz_file)
    video1_clips = data['video1_clips']
    video2_clips = data['video2_clips']
    labels = data['labels']
    sample_indices = data['indices']
    
    # Find positive and negative sample indices
    positive_indices = np.where(labels == 1.0)[0]
    negative_indices = np.where(labels == 0.0)[0]
    
    # Select random samples
    num_positive = min(args.num_positive, len(positive_indices))
    num_negative = min(args.num_negative, len(negative_indices))
    selected_positive = random.sample(list(positive_indices), num_positive) if num_positive > 0 else []
    selected_negative = random.sample(list(negative_indices), num_negative) if num_negative > 0 else []
    
    # Function to plot samples
    def plot_samples(selected_indices, title_prefix):
        if not selected_indices:
            print(f"No {title_prefix.lower()} samples to display.")
            return
        for idx in selected_indices:
            v1_frame = denormalize(video1_clips[idx, -1])
            v2_frame = denormalize(video2_clips[idx, -1])
            v1_idx, v2_idx = sample_indices[idx]
            label = labels[idx]
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(v1_frame)
            axes[0].set_title(f"Video 1, Frame: {v1_idx}")
            axes[0].axis('off')
            axes[1].imshow(v2_frame)
            axes[1].set_title(f"Video 2, Frame: {v2_idx}")
            axes[1].axis('off')
            
            fig.suptitle(f"{title_prefix} Sample - Label: {int(label)} (Frame {v1_idx} vs {v2_idx})")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                save_path = os.path.join(args.save_dir, f"{title_prefix.lower().replace(' ', '_')}_{v1_idx}_{v2_idx}.png")
                fig.savefig(save_path)
                print(f"Saved {title_prefix} figure to {save_path}")
                plt.close(fig)
            else:
                plt.show()
    
    # Plot positive and negative samples
    plot_samples(selected_positive, "Positive")
    plot_samples(selected_negative, "Negative")

if __name__ == "__main__":
    main()