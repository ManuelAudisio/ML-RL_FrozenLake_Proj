import gymnasium as gym
import numpy as np
import torch
from collections import deque
import json
import time

from tabular_agent import TabularQAgent
from dqn_agent import DQNAgent
from utils import plot_metrics, set_seeds

# --- CONFIGURATION DICTIONARY ---
# This is our central control panel for experiments. All hyperparameters are defined here.
CONFIG = {
    "experiment_name": "DQN_stochastic_lr_0.001_seed_42", # A unique name to identify this specific run
    "seed": 42, # The random seed for reproducibility
    "agent_type": "dqn", # Choose "tabular" or "dqn" to switch between agents
    
    "env_id": "FrozenLake-v1",
    "env_config": {"is_slippery": True, "map_name": "8x8"},
    
    "training": {
        "num_episodes": 20000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9999,
    },
    
    "evaluation": {
        "num_episodes": 500, # Number of episodes to run for the final, separate evaluation
    },
    
    "tabular_params": {
        "discount_factor": 0.99,
    },
    
    "dqn_params": {
        "discount_factor": 0.99,
        "learning_rate": 0.001,
        "batch_size": 64,
        "replay_buffer_size": 10000,
    }
}

def evaluate_agent(agent, env, num_episodes):
    """
    Runs the trained agent in the environment for a number of episodes
    with no exploration (epsilon=0) to evaluate its final, deterministic performance.

    Args:
        agent: The trained agent to evaluate.
        env: The Gymnasium environment.
        num_episodes (int): The number of episodes to run for evaluation.

    Returns:
        dict: A dictionary containing the average success rate and steps per episode.
    """
    successes = []
    steps = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        step_count = 0
        while not done:
            # Agent chooses the BEST action based on its learned policy (no randomness).
            action = agent.choose_action(state, epsilon=0)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            step_count += 1
        
        successes.append(1 if reward > 0 else 0)
        steps.append(step_count)
        
    return {
        "evaluation_success_rate": np.mean(successes),
        "evaluation_avg_steps": np.mean(steps)
    }

def run_experiment(config):
    """Initializes and runs a full training and evaluation experiment."""
    
    # --- 1. SETUP ---
    # Set all random seeds for reproducibility across all libraries.
    set_seeds(config["seed"])
    
    print(f"--- Starting Experiment: {config['experiment_name']} ---")
    env = gym.make(config["env_id"], **config["env_config"])

    # --- 2. AGENT INITIALIZATION ---
    if config["agent_type"] == "tabular":
        agent = TabularQAgent(
            state_space_size=env.observation_space.n,
            action_space_size=env.action_space.n,
            **config["tabular_params"]
        )
    elif config["agent_type"] == "dqn":
        agent = DQNAgent(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            **config["dqn_params"]
        )
        agent.memory = deque(maxlen=config["dqn_params"]["replay_buffer_size"])
    else:
        raise ValueError(f"Agent type '{config['agent_type']}' not recognized.")
    
    print(f"{agent.__class__.__name__} created.")

    # --- 3. TRAINING PHASE ---
    training_successes = []
    training_steps = []
    training_losses = []
    epsilon = config["training"]["epsilon_start"]

    print(f"Starting training for {config['training']['num_episodes']} episodes...")
    for episode in range(config['training']['num_episodes']):
        state, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            if config["agent_type"] == "tabular":
                agent.update(state, action, reward, next_state)
            elif config["agent_type"] == "dqn":
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay(config["dqn_params"]["batch_size"])
                if loss is not None:
                    training_losses.append(loss)
            
            state = next_state
            step_count += 1
            total_reward += reward

        epsilon = max(config["training"]["epsilon_end"], epsilon * config["training"]["epsilon_decay"])
        
        training_steps.append(step_count)
        training_successes.append(1 if total_reward > 0 else 0)

        if (episode + 1) % 1000 == 0:
            avg_loss = np.mean(training_losses[-100:]) if training_losses else 0
            # BUG FIX #1: Correctly reference the number of episodes from the 'training' sub-dictionary.
            print(f"  ... Episode {episode + 1}/{config['training']['num_episodes']} | Epsilon: {epsilon:.4f} | Avg Loss (last 100): {avg_loss:.4f}")
    
    print("✅ Training finished.")
    
    # --- 4. EVALUATION PHASE ---
    print("\n--- Evaluating final agent performance... ---")
    evaluation_results = evaluate_agent(agent, env, config["evaluation"]["num_episodes"])
    print(f"Evaluation Success Rate: {evaluation_results['evaluation_success_rate']:.4f}")
    print(f"Evaluation Avg Steps: {evaluation_results['evaluation_avg_steps']:.2f}")

    env.close()

    # --- 5. SAVING RESULTS ---
    # BUG FIX #2: Use correct file extensions and clearer filenames for saving models.
    if config["agent_type"] == "tabular":
        # np.save automatically adds the .npy extension, so we build the path without it.
        model_path = f"results/models/{config['experiment_name']}_qtable.npy"
        np.save(model_path, agent.q_table)
    elif config["agent_type"] == "dqn":
        model_path = f"results/models/{config['experiment_name']}_dqn.pth"
        torch.save(agent.q_network.state_dict(), model_path)
    print(f"\n🧠 Agent brain saved to {model_path}")
    
    full_results = {
        "config": config,
        "training_metrics": {
            "Success Rate": training_successes,
            "Steps per Episode": training_steps,
            "Training Loss": training_losses,
        },
        "final_evaluation": evaluation_results
    }
    results_path = f"results/data/{config['experiment_name']}.json"
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=4)
    print(f"📊 Full experiment data saved to {results_path}")

    # --- 6. PLOTTING ---
    # The plot title is now the unique experiment name for clarity.
    plot_metrics(full_results["training_metrics"], title=config["experiment_name"])


if __name__ == "__main__":
    # The main entry point. It runs one experiment based on the CONFIG dictionary.
    run_experiment(CONFIG)

