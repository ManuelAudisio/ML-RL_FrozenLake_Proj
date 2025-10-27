import torch
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seeds(seed_value=42):
    """
    Sets the seeds for reproducibility of experiments across all relevant libraries.

    This function is crucial for scientific experiments. By setting a single seed,
    it ensures that any process using random numbers (e.g., network weight
    initialization, epsilon-greedy exploration, replay buffer sampling,
    environment's stochasticity) will produce the exact same sequence of
    numbers every time the script is run with the same seed.
    This allows for fair, reproducible comparisons between different experiments.

    Args:
        seed_value (int): the integer value to use for all random number generators.
    """
    
    # Set seed for Python's built-in random module (used for `random.sample`)
    random.seed(seed_value)
    
    # Set seed for NumPy (used for `np.random.uniform`, `np.random.choice`, and Gymnasium's env)
    np.random.seed(seed_value)
    
    # Set seed for PyTorch (used for network weight initialization)
    torch.manual_seed(seed_value)
    
def plot_metrics(results, title, window_size=100):
    """
    Plots multiple training metrics from a results dictionary and saves the figure.
    
    This function is designed to be robust:
    1. It handles redundant metrics (e.g., Total Reward vs. Success Rate).
    2. It creates separate subplots for each metric.
    3. It correctly handles metrics of different lengths (per-episode vs. per-step).
    4. It plots both raw data (for variance) and a smoothed moving average (for trend).
    
    Args:
        results (dict): a dictionary where keys are metric names (str) and values are the lists of metric data (list).
        title (str): the main title for the plot and the base for the save filename.
        window_size (int): the number of episodes to average over for the smoothed line.
    """
    # 1. Data Cleaning
    # For the FrozenLake environment, Total Reward and Success Rate are identical.
    # This check removes one of them to avoid a redundant plot and keep the
    # final visual clean and focused.
    if "Total Reward" in results:
        del results["Total Reward"]

    # 2. Plot Initialization 
    num_metrics = len(results)
    
    # Create a figure with `num_metrics` subplots, arranged vertically.
    # `figsize` is set for good readability.
    # `sharex=False` is CRITICAL: it allows each subplot to have
    # its own independent x-axis scale (e.g., 0-50000 for episodes, 0-2M for loss steps).
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 6 * num_metrics), sharex=False)
    
    # If there's only one metric, `plt.subplots` returns a single axis object,
    # not a list. This line wraps it in a list to make the code below
    # work uniformly for any number of metrics.
    if num_metrics == 1:
        axes = [axes]
    
    fig.suptitle(title, fontsize=16)

    # 3. Plotting Loop 
    # Iterate through each metric (e.g., 'Success Rate', 'Training Loss') and plot it.
    for i, (metric_name, values) in enumerate(results.items()):
        ax = axes[i] # select the correct subplot for this metric
        
        # Determine the correct x-axis label based on the metric's name.
        if "Loss" in metric_name:
            x_label = "Training Steps (Replay Calls)"
            # If the loss data is huge, we downsample it to make the plot readable.
            if len(values) > 50000:
                values = values[::len(values)//50000]
        else:
            x_label = "Episode"
        
        # Plot 1: Raw Data
        # Plot the original, noisy data with high transparency (alpha=0.3).
        # This shows the underlying variance and noise, as seen in scientific papers.
        ax.plot(values, alpha=0.3, color='gray', linewidth=0.5, label='Raw Data')
        
        # Plot 2: Smoothed Moving Average
        # This helps to visualize the actual learning trend.
        if len(values) >= window_size:
            # `np.convolve` is a standard way to calculate a moving average.
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            
            # The x-axis for the smoothed line must be shifted to be centered
            # in its window, aligning it correctly with the raw data.
            x_smooth = np.arange(window_size - 1, len(values))
            ax.plot(x_smooth, moving_avg, linewidth=2, label=f'{window_size}-ep Moving Avg')

        # 4. Formatting
        # Add titles and labels for clarity
        ax.set_title(metric_name, fontsize=14)
        ax.set_ylabel(f'{metric_name} ({window_size}-ep avg)', fontsize=12)
        ax.set_xlabel(x_label, fontsize=12) # Use the dynamically determined label.
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    # 5. Saving and Displaying
    # create a valid filename from the title
    filename = title.replace(" ", "_").lower() + ".png"
    
    # Save with higher DPI (dots per inch) for better quality in the report.
    # `bbox_inches='tight'` removes excess white space.
    plt.savefig(f'results/plots/{filename}', dpi=150, bbox_inches='tight')
    print(f"\n📈 Plots saved to results/plots/{filename}")
    
    # Adjust layout to prevent titles and labels from overlapping.
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

