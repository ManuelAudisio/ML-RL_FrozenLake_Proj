import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# 1. SCRIPT CONFIGURATION
# Defines the default directories where data will be read from and plots saved to.
# This script is intended to be run *after* `run_suite.py` has generated the data.
DATA_DIR = "results/final_analysis/data"
PLOT_DIR = "results/final_analysis/plots"
WINDOW_SIZE = 500

def load_experiment_data(data_dir):
    """
    Loads all experiment result .json files from the specified data directory.

    Args:
        data_dir (str): The path to the directory containing the .json files.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              data from one experiment's .json file.
    """
    # Use glob to find all files ending with .json in the target directory.
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    # Safety check in case the script is run before the data exists.
    if not json_files:
        print(f"Error: No .json files found in '{data_dir}'.")
        return []
    all_data = []
    # Loop through each found file, open it, and load its JSON content.
    for file_path in json_files:
        with open(file_path, 'r') as f:
            all_data.append(json.load(f))
    print(f"Loaded {len(all_data)} experiment data files from '{data_dir}'.")
    return all_data

def analyze_and_plot(all_data, window_size):
    """
    Analyzes the collected experiment data to compute mean and standard deviation
    across different seeds for each agent, then generates the final comparative plot
    and prints a summary table.
    """
    
    # 2. DATA GROUPING
    # Group all loaded experiment runs by the agent type.
    results_by_agent = {
        "Tabular Q-Learning": [],
        "Deep Q-Network (DQN)": []
    }
    for data in all_data:
        # Determine the agent type by looking for "Tabular" in the experiment name.
        # This is more robust than checking for "DQN".
        agent_key = "Tabular Q-Learning" if "Tabular" in data["config"]["experiment_name"] else "Deep Q-Network (DQN)"
        results_by_agent[agent_key].append(data)

    # 3. PLOT INITIALIZATION 
    # Create a figure with two subplots, arranged vertically (2 rows, 1 column).
    # `figsize` controls the overall size of the figure.
    # `sharey=True` is CRITICAL: it forces both subplots to use the same Y-axis scale,
    # ensuring a fair and direct visual comparison of performance.
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharey=True)
    fig.suptitle("Tabular Q-Learning vs. DQN: Performance Comparison", fontsize=20)

    # Dictionary to store the final evaluation results for the summary table.
    summary_table = {}
    
    # Define colors to be consistent
    colors = {"Tabular Q-Learning": "tab:blue", "Deep Q-Network (DQN)": "tab:orange"}
    agent_names = ["Tabular Q-Learning", "Deep Q-Network (DQN)"] # Define order

    # 4. DATA PROCESSING AND PLOTTING LOOP 
    # Iterate over each agent type to process and plot its data.
    for i, agent_name in enumerate(agent_names):
        ax = axes[i]
        runs = results_by_agent[agent_name]
        
        if not runs:
            ax.set_title(f"{agent_name} - No data found", fontsize=16)
            continue

        # Extract the list of success rates from each run (e.g., [run1_rates, run2_rates, run3_rates])
        success_rates_list = [run["training_metrics"]["Success Rate"] for run in runs]
        
        # Determine the number of episodes for this agent (e.g., 50000 for Tabular)
        num_episodes = len(success_rates_list[0])
        
        # Statistical Calculation 
        # Calculate the mean and standard deviation across all runs (axis=0).
        # This gives us a mean curve and a std dev curve, both of length `num_episodes`.
        mean_sr = np.mean(success_rates_list, axis=0)
        std_sr = np.std(success_rates_list, axis=0)
        
        # Smooth the mean and std dev curves for a cleaner plot.
        mean_smooth = np.convolve(mean_sr, np.ones(window_size)/window_size, mode='valid')
        std_smooth = np.convolve(std_sr, np.ones(window_size)/window_size, mode='valid')
        
        # Create the corresponding x-axis for the smoothed data.
        x_smooth = np.arange(window_size - 1, num_episodes)
        
        # --- Plotting ---
        color = colors[agent_name]
        ax.plot(x_smooth, mean_smooth, linewidth=2.5, color=color, label=f'Mean Success Rate')
      
        # Plot the standard deviation as a shaded band around the mean.  
        ax.fill_between(
            x_smooth,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            color=color,
            alpha=0.2,
            label=f'Standard Deviation'
        )
        
        # --- Formatting ---
        ax.set_title(agent_name, fontsize=16)
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_xlim(0, num_episodes)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- Final Evaluation Statistics ---
        # Extract the single scalar value for evaluation success rate from each run
        eval_scores = [run["final_evaluation"]["evaluation_success_rate"] for run in runs]
        summary_table[agent_name] = f"{np.mean(eval_scores):.2%} ± {np.std(eval_scores):.2%}"
    
    # 5. FINAL FIGURE FORMATTING & SAVING
    # Add a single, shared y-axis label for the entire figure for a cleaner look.
    fig.text(-0.01, 0.5, f"Success Rate ({window_size}-ep Moving Avg)", va='center', rotation='vertical', fontsize=14)
    # Set the y-axis limit for both plots (since they are shared).
    ax.set_ylim(0, 1.05)
    
    # Adjust layout to prevent titles/labels from overlapping.
    plt.tight_layout(rect=[0.03, 0, 1, 0.95]) # Adjust layout

    # --- Saving the Final Plot ---
    os.makedirs(PLOT_DIR, exist_ok=True)
    final_plot_path = os.path.join(PLOT_DIR, "final_comparison_subplots.png")
    plt.savefig(final_plot_path, dpi=200)
    print(f"\n📈 Final comparative plot saved to: {final_plot_path}")
    plt.show()

    # 6. PRINTING THE SUMMARY TABLE
    # Print the final, aggregated results to the console.
    print("\n--- Final Evaluation Summary (Mean ± Std Dev across seeds) ---")
    for agent, result in summary_table.items():
        print(f"{agent}: {result}")

# 7. SCRIPT EXECUTION
# This block runs only when the script is called directly.
if __name__ == "__main__":
    # Load all .json data from the specified directory.
    all_data = load_experiment_data(DATA_DIR)
    if all_data:
        analyze_and_plot(all_data, window_size=WINDOW_SIZE)
