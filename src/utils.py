import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seeds(seed_value=42):
    """
    Sets the seeds for reproducibility of experiments across all relevant libraries.

    Args:
        seed_value (int): The seed value to use for all random number generators.
    """
    # This function is crucial for scientific experiments. By setting a seed,
    # we ensure that any process that uses random numbers (like initializing
    # network weights or choosing random actions) will produce the exact same
    # sequence of numbers every time we run the script with the same seed.
    # This makes our results reproducible and allows fair comparisons.

    # Set seed for Python's built-in random module
    random.seed(seed_value)
    # Set seed for NumPy's random number generator
    np.random.seed(seed_value)
    # Set seed for PyTorch's CPU random number generator
    torch.manual_seed(seed_value)
    
    # We keep the GPU seeding part as good practice, but it will only run if a
    # compatible GPU is detected, preventing errors if it's not configured.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_metrics(results, title, window_size=100):
    """
    Plots multiple metrics from a results dictionary, showing both raw data
    and a smoothed moving average for better trend visualization.
    """
    # --- 1. Data Cleaning ---
    # As discussed, for FrozenLake, Total Reward and Success Rate are the same metric.
    # We remove one to avoid redundancy and make the final plot cleaner.
    if "Total Reward" in results:
        del results["Total Reward"]

    # --- 2. Plot Initialization ---
    num_metrics = len(results)
    # Create a figure with a separate subplot for each metric, arranged vertically.
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 6 * num_metrics), sharex=True)
    # If there's only one metric, `subplots` returns a single axis object, not a list.
    # We wrap it in a list to handle both cases uniformly and avoid errors.
    if num_metrics == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16)

    # --- 3. Plotting Loop ---
    # Iterate through each metric (e.g., 'Success Rate', 'Training Loss') and plot it.
    for i, (metric_name, values) in enumerate(results.items()):
        ax = axes[i]
        
        # Plot 1: Raw Data
        # We plot the raw, noisy data with high transparency (alpha=0.3).
        # This shows the underlying variance and noise in the training process.
        ax.plot(values, alpha=0.3, color='gray', linewidth=0.5, label='Raw Data')
        
        # Plot 2: Smoothed Moving Average
        # We plot a moving average to clearly see the learning trend over time.
        if len(values) >= window_size:
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            # The x-axis for the smoothed line must be shifted to align correctly with the raw data.
            x_smooth = np.arange(window_size - 1, len(values))
            ax.plot(x_smooth, moving_avg, linewidth=2, label=f'{window_size}-ep Moving Avg')

        # --- 4. Formatting ---
        # Add titles and labels for clarity. These new labels are more precise.
        ax.set_title(metric_name, fontsize=14)
        ax.set_ylabel(f'{metric_name} ({window_size}-ep avg)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    # Set the x-axis label only on the bottom-most plot to avoid repetition.
    axes[-1].set_xlabel('Episode', fontsize=12)

    # --- 5. Saving and Displaying ---
    filename = title.replace(" ", "_").lower() + ".png"
    # Save with higher DPI and tight bounding box for better quality in the report.
    plt.savefig(f'results/plots/{filename}', dpi=150, bbox_inches='tight')
    print(f"\n📈 Plots saved to results/plots/{filename}")
    
    # Adjust layout to prevent titles and labels from overlapping.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

