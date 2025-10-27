import copy
import time
import os
from main import run_experiment

# SCRIPT CONFIGURATION 
# This is the central control panel for the entire experimental suite.
# It defines all parameters and experiments to be run for the final analysis.

# 1. Define the final output directories for this analysis.
#    This keeps the final, multi-seed results separate from preliminary tests.
FINAL_RESULTS_DIR = "results/final_analysis"

# 2. Define the seeds for the robustness check.
#    Running on multiple seeds allows for statistical analysis (mean, std dev).
SEEDS = [42, 123, 999]

# 3. Define the base configurations for our two main agents.
#    These are the "template" configurations for each agent type.
TABULAR_BASE_CONFIG = {
    "agent_type": "tabular",
    "training": {
        "num_episodes": 50000,
        "max_steps_per_episode": 500,
        "epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 0.99995,
    },
    "tabular_params": {"discount_factor": 0.99}
}

DQN_BASE_CONFIG = {
    "agent_type": "dqn",
    "training": {
        "num_episodes": 20000,
        "max_steps_per_episode": 500,
        "epsilon_start": 1.0, "epsilon_end": 0.01, "epsilon_decay": 0.9999,
    },
    "dqn_params": {
        "discount_factor": 0.99, "learning_rate": 0.0005,
        "batch_size": 64, "replay_buffer_size": 10000,
    }
}

# 4. List of all experiment configurations to run.
#    This script will loop over this list and run each experiment for each seed.
experiments_to_run = [
    ("Tabular_Agent_Final", TABULAR_BASE_CONFIG),
    ("DQN_Agent_Final", DQN_BASE_CONFIG)
]

# MAIN EXECUTION BLOCK
# This block runs only when the script is executed directly.
if __name__ == "__main__":
    start_time = time.time()
    
    print("--- 🚀 Starting Full Experiment Suite for Final Analysis ---")
    
    # Outer Loop: Iterate over Agent Types 
    # This loop takes each configuration defined in 'experiments_to_run'.
    for agent_name, base_config in experiments_to_run:
        
        # Inner Loop: Iterate over Seeds
        # For each agent, this loop runs the experiment for each seed in the SEEDS list.
        for seed in SEEDS:
            # Create a deep copy of the base config to avoid modifying the template.
            run_config = copy.deepcopy(base_config)
            
            # --- Dynamically create the full configuration for this specific run ---
            # a. Add general environment and evaluation parameters
            run_config.update({
                "env_id": "FrozenLake-v1",
                "env_config": {"is_slippery": True, "map_name": "8x8"},
                "evaluation": {
                    "num_episodes": 500,
                    "max_steps_per_episode": base_config["training"]["max_steps_per_episode"]
                },
                # b. Add the specific save paths for the final analysis folders
                "save_paths": {
                    "model_dir": os.path.join(FINAL_RESULTS_DIR, "models"),
                    "data_dir": os.path.join(FINAL_RESULTS_DIR, "data"),
                    "plot_dir": os.path.join(FINAL_RESULTS_DIR, "plots"),
                }
            })
            
            # c. Set the specific seed for this run
            run_config["seed"] = seed
            # d. Create a unique experiment name for saving files
            run_config["experiment_name"] = f"{agent_name}_seed_{seed}"
            
            print(f"\n{'='*50}")
            print(f"--- 🏃 Running: {run_config['experiment_name']} ---")
            print(f"{'='*50}")
            
            # --- Execute the Experiment ---
            # Call the 'run_experiment' function from 'main.py'
            # and pass it the complete, unique configuration for this run.
            run_experiment(run_config)
            
            print(f"--- ✅ Finished: {run_config['experiment_name']} ---")
            
    # Once all loops are finished, calculate and print the total time.
    end_time = time.time()
    print(f"\n--- 🎉 Full Experiment Suite Completed in {(end_time - start_time) / 60:.2f} minutes ---")

