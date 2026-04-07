import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_and_visualize_losses(checkpoint_path, save_plots=True, output_dir="/kaggle/working/"):
    """
    Load GAN training losses from checkpoint and create comprehensive visualizations
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        save_plots (bool): Whether to save plots to disk
        output_dir (str): Directory to save plots
    
    Returns:
        dict: Dictionary containing loss data and statistics
    """
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    
    try:
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract loss data
        g_losses = checkpoint.get('g_losses', [])
        d_losses = checkpoint.get('d_losses', [])
        epoch = checkpoint.get('epoch', len(g_losses))
        
        if not g_losses or not d_losses:
            print("Warning: No loss data found in checkpoint")
            return None
        
        print(f"Loaded {len(g_losses)} epochs of training data")
        print(f"Training completed up to epoch: {epoch}")
        
        # Convert to numpy arrays for easier manipulation
        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)
        epochs = np.arange(1, len(g_losses) + 1)
        
        # Create comprehensive visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GAN Training Loss Analysis', fontsize=16, fontweight='bold')
        
        # 1. Basic Loss Curves
        axes[0, 0].plot(epochs, g_losses, label='Generator Loss', color='blue', linewidth=2)
        axes[0, 0].plot(epochs, d_losses, label='Discriminator Loss', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('(a)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Smoothed Loss Curves (moving average)
        window_size = max(1, len(g_losses) // 20)  # 5% of total epochs
        if len(g_losses) > window_size:
            g_smooth = np.convolve(g_losses, np.ones(window_size)/window_size, mode='valid')
            d_smooth = np.convolve(d_losses, np.ones(window_size)/window_size, mode='valid')
            smooth_epochs = epochs[window_size-1:]
            
            axes[0, 1].plot(smooth_epochs, g_smooth, label=f'Generator (MA-{window_size})', color='blue', linewidth=2)
            axes[0, 1].plot(smooth_epochs, d_smooth, label=f'Discriminator (MA-{window_size})', color='red', linewidth=2)
        else:
            axes[0, 1].plot(epochs, g_losses, label='Generator Loss', color='blue', linewidth=2)
            axes[0, 1].plot(epochs, d_losses, label='Discriminator Loss', color='red', linewidth=2)
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('(b)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Loss Ratio Analysis
        # Avoid division by zero
        d_losses_safe = np.where(d_losses == 0, 1e-8, d_losses)
        loss_ratio = g_losses / d_losses_safe
        
        axes[1, 0].plot(epochs, loss_ratio, color='purple', linewidth=2)
        axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Perfect Balance')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Generator Loss / Discriminator Loss')
        axes[1, 0].set_title('(c)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Loss Distribution
        axes[1, 1].hist(g_losses, bins=30, alpha=0.7, label='Generator', color='blue', density=True)
        axes[1, 1].hist(d_losses, bins=30, alpha=0.7, label='Discriminator', color='red', density=True)
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('(d)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(output_dir, "gan_training_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
        
        plt.show()
        
        # Calculate and display statistics
        print("\n" + "="*50)
        print("TRAINING STATISTICS")
        print("="*50)
        
        # Recent performance (last 10% of training)
        recent_period = max(1, len(g_losses) // 10)
        recent_g = g_losses[-recent_period:]
        recent_d = d_losses[-recent_period:]
        
        stats = {
            'total_epochs': len(g_losses),
            'final_epoch': epoch,
            'generator_stats': {
                'final_loss': g_losses[-1],
                'mean_loss': np.mean(g_losses),
                'std_loss': np.std(g_losses),
                'min_loss': np.min(g_losses),
                'max_loss': np.max(g_losses),
                'recent_mean': np.mean(recent_g),
                'recent_std': np.std(recent_g)
            },
            'discriminator_stats': {
                'final_loss': d_losses[-1],
                'mean_loss': np.mean(d_losses),
                'std_loss': np.std(d_losses),
                'min_loss': np.min(d_losses),
                'max_loss': np.max(d_losses),
                'recent_mean': np.mean(recent_d),
                'recent_std': np.std(recent_d)
            },
            'convergence_analysis': {
                'final_loss_ratio': g_losses[-1] / max(d_losses[-1], 1e-8),
                'mean_loss_ratio': np.mean(loss_ratio),
                'recent_loss_ratio': np.mean(recent_g) / max(np.mean(recent_d), 1e-8)
            }
        }
        
        print(f"Total Epochs Trained: {stats['total_epochs']}")
        print(f"Checkpoint Saved at Epoch: {stats['final_epoch']}")
        print()
        
        print("GENERATOR PERFORMANCE:")
        print(f"  Final Loss: {stats['generator_stats']['final_loss']:.6f}")
        print(f"  Mean Loss: {stats['generator_stats']['mean_loss']:.6f} ± {stats['generator_stats']['std_loss']:.6f}")
        print(f"  Range: [{stats['generator_stats']['min_loss']:.6f}, {stats['generator_stats']['max_loss']:.6f}]")
        print(f"  Recent Performance: {stats['generator_stats']['recent_mean']:.6f} ± {stats['generator_stats']['recent_std']:.6f}")
        print()
        
        print("DISCRIMINATOR PERFORMANCE:")
        print(f"  Final Loss: {stats['discriminator_stats']['final_loss']:.6f}")
        print(f"  Mean Loss: {stats['discriminator_stats']['mean_loss']:.6f} ± {stats['discriminator_stats']['std_loss']:.6f}")
        print(f"  Range: [{stats['discriminator_stats']['min_loss']:.6f}, {stats['discriminator_stats']['max_loss']:.6f}]")
        print(f"  Recent Performance: {stats['discriminator_stats']['recent_mean']:.6f} ± {stats['discriminator_stats']['recent_std']:.6f}")
        print()
        
        print("CONVERGENCE ANALYSIS:")
        print(f"  Final G/D Ratio: {stats['convergence_analysis']['final_loss_ratio']:.3f}")
        print(f"  Mean G/D Ratio: {stats['convergence_analysis']['mean_loss_ratio']:.3f}")
        print(f"  Recent G/D Ratio: {stats['convergence_analysis']['recent_loss_ratio']:.3f}")
        
        # Training quality assessment
        print()
        print("TRAINING QUALITY ASSESSMENT:")
        final_ratio = stats['convergence_analysis']['final_loss_ratio']
        if 0.5 <= final_ratio <= 2.0:
            print("  ✅ Good balance between Generator and Discriminator")
        elif final_ratio < 0.5:
            print("  ⚠️  Generator may be overpowering Discriminator")
        else:
            print("  ⚠️  Discriminator may be overpowering Generator")
        
        if stats['generator_stats']['recent_std'] < stats['generator_stats']['std_loss']:
            print("  ✅ Generator loss is stabilizing")
        else:
            print("  ⚠️  Generator loss may still be unstable")
        
        return stats
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def compare_multiple_checkpoints(checkpoint_paths, labels=None):
    """
    Compare losses from multiple checkpoints
    
    Args:
        checkpoint_paths (list): List of checkpoint file paths
        labels (list): Optional labels for each checkpoint
    """
    if labels is None:
        labels = [f"Checkpoint {i+1}" for i in range(len(checkpoint_paths))]
    
    plt.figure(figsize=(15, 6))
    
    # Generator losses comparison
    plt.subplot(1, 2, 1)
    for i, (path, label) in enumerate(zip(checkpoint_paths, labels)):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            g_losses = checkpoint.get('g_losses', [])
            if g_losses:
                epochs = np.arange(1, len(g_losses) + 1)
                plt.plot(epochs, g_losses, label=label, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.title('Generator Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Discriminator losses comparison
    plt.subplot(1, 2, 2)
    for i, (path, label) in enumerate(zip(checkpoint_paths, labels)):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            d_losses = checkpoint.get('d_losses', [])
            if d_losses:
                epochs = np.arange(1, len(d_losses) + 1)
                plt.plot(epochs, d_losses, label=label, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Loss')
    plt.title('Discriminator Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Single checkpoint analysis
    checkpoint_path = "/kaggle/input/tg-gan-checkpoints/gan_transformer_checkpoint.pth"
    
    # Alternative paths you might have:
    # checkpoint_path = "/kaggle/input/my-gan-checkpoints/gan_transformer_checkpoint.pth"
    # checkpoint_path = "/kaggle/working/gan_transformer_checkpoint_epoch_50.pth"
    
    print("Analyzing GAN training checkpoint...")
    loss_stats = load_and_visualize_losses(checkpoint_path)
    
    # If you have multiple checkpoints to compare:
    # checkpoint_paths = [
    #     "/kaggle/working/gan_transformer_checkpoint_epoch_25.pth",
    #     "/kaggle/working/gan_transformer_checkpoint_epoch_50.pth",
    #     "/kaggle/working/gan_transformer_checkpoint_epoch_75.pth"
    # ]
    # labels = ["Epoch 25", "Epoch 50", "Epoch 75"]
    # compare_multiple_checkpoints(checkpoint_paths, labels)
