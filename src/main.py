import gymnasium as gym
import numpy as np
import torch
from collections import deque
import json
import time

from tabular_agent import TabularQAgent
from dqn_agent import DQNAgent
from utils import plot_metrics, set_seeds

# CONFIGURATION DICTIONARY 
# This dictionary is the central control panel for running a SINGLE experiment.
# All hyperparameters are defined here to be easily viewed and modified.
CONFIG = {
    # A unique name for this specific run. Used for naming saved files.
    "experiment_name": "DQN_stochastic_lr_0.0005_seed_42", 
    "seed": 42, # The random seed for reproducibility
    "agent_type": "dqn", # Choose "tabular" or "dqn" to switch between agents
    
    # Environment settings
    "env_id": "FrozenLake-v1",
    # env_config is passed to gym.make()
    "env_config": {"is_slippery": True, "map_name": "8x8"},
    
    # Training loop settings
    "training": {
        "num_episodes": 20000,
        # Safety limit to prevent infinite loops in a single episode. It happened during tests :/
        "max_steps_per_episode": 500, 
        "epsilon_start": 1.0, # Exploration rate starts at 100%
        "epsilon_end": 0.01, # Exploration rate ends at 1%
        "epsilon_decay": 0.9999, # Multiplicative decay factor per episode
    },
    
    # Evaluation loop settings
    "evaluation": {
        # Number of episodes to run for the final performance evaluation
        "num_episodes": 500, 
        "max_steps_per_episode": 500, # Safety limit as above
    },
    
    # Agent-specific hyperparameters for Tabular Q-Learning
    "tabular_params": {
        "discount_factor": 0.99,
    },
    
    # Agent-specific hyperparameters for DQN
    "dqn_params": {
        "discount_factor": 0.99,
        "learning_rate": 0.0005, # 0.0005 was the most stable LR of preliminary tests
        "batch_size": 64,
        "replay_buffer_size": 10000,
    }
}

def evaluate_agent(agent, env, num_episodes):
    """
    Runs the trained agent in the environment for a set number of episodes
    with no exploration (epsilon=0) to evaluate its final, learned policy.

    Args:
        agent: The trained agent (Tabular or DQN) to evaluate.
        env: The Gymnasium environment used for evaluation.
        num_episodes (int): The number of episodes to run for evaluation.

    Returns:
        dict: A dictionary containing the average success rate and average steps per episode.
    """
    successes = []
    steps = []
    
    # Loop for the specified number of evaluation episodes
    for _ in range(num_episodes):
        state, info = env.reset()
        done = False
        step_count = 0
        
        # Run one full episode
        while not done:
            # CRITICAL: Epsilon is 0
            # The agent is forced to "exploit" its learned knowledge,
            # choosing the best action it thinks exists. No random exploration.
            action = agent.choose_action(state, epsilon=0)
            
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            step_count += 1
            
        # Store the result of this episode
        # In FrozenLake, any reward > 0 means success (reaching the goal)
        successes.append(1 if reward > 0 else 0)
        steps.append(step_count)
        
    # Return the average performance over all evaluation episodes    
    return {
        "evaluation_success_rate": np.mean(successes),
        "evaluation_avg_steps": np.mean(steps)
    }

def run_experiment(config):
    """
    Initializes and runs a full training and evaluation experiment
    based on the provided configuration dictionary.
    """    
    # 1. SETUP
    # Set all random seeds (for Python, NumPy, PyTorch) to ensure
    # that this experiment is reproducible. This is a critical step for
    # scientific validity, ensuring that "luck" is the same between runs.
    set_seeds(config["seed"])
    
    print(f"--- Starting Experiment: {config['experiment_name']} ---")
    
    # Create the training environment with the specified step limit
    # This environment will be used for the agent's learning phase.
    train_env = gym.make(
        config["env_id"], 
        **config["env_config"],
        max_episode_steps=config["training"]["max_steps_per_episode"]
    )
    print(f"Training environment created with max_steps={config['training']['max_steps_per_episode']}.")
    
    # Create a separate, identical environment for the final evaluation.
    # This is good practice to ensure the evaluation is clean and isolated from training.
    eval_env = gym.make(
        config["env_id"], 
        **config["env_config"],
        max_episode_steps=config["evaluation"]["max_steps_per_episode"]
    )
    print(f"Evaluation environment created with max_steps={config['evaluation']['max_steps_per_episode']}.")

    # 2. AGENT INITIALIZATION
    # This block acts as a switch, creating the correct agent
    # based on the 'agent_type' string in the config.
    if config["agent_type"] == "tabular":
        agent = TabularQAgent(
            state_space_size = train_env.observation_space.n,
            action_space_size = train_env.action_space.n,
            **config["tabular_params"] # unpacks 'discount_factor'
        )
    elif config["agent_type"] == "dqn":
        agent = DQNAgent(
            state_size = train_env.observation_space.n,
            action_size = train_env.action_space.n,
            **config["dqn_params"] # Unpacks 'discount_factor', 'learning_rate', etc.
        )
        # The replay buffer is part of the agent but its size is defined
        # in the experiment config, so it is set here.
        agent.memory = deque(maxlen=config["dqn_params"]["replay_buffer_size"])
    else:
        # Fail fast if the agent type in the config is unknown.
        raise ValueError(f"Agent type '{config['agent_type']}' not recognized.")
    
    print(f"{agent.__class__.__name__} created.")

    # 3. TRAINING PHASE
    # Initialize lists to store metrics from each episode for later plotting.
    training_successes = []
    training_steps = []
    training_losses = []
    # Initialize the exploration rate (epsilon) to its starting value.
    epsilon = config["training"]["epsilon_start"]

    print(f"Starting training for {config['training']['num_episodes']} episodes...")
    
    # This is the main training loop, which iterates over each game (episode).
    for episode in range(config['training']['num_episodes']):
        # Start a new game by resetting the environment
        state, info = train_env.reset()
        done = False
        step_count = 0
        total_reward = 0
        
        # This inner loop runs for a single game, step by step, until it ends
        # (or hits the max_steps_per_episode limit).
        while not done:
            # 1. Agent chooses an action based on its current policy (epsilon-greedy).
            action = agent.choose_action(state, epsilon)
            
            # 2. The environment executes the action and returns the outcome.
            next_state, reward, done, truncated, info = train_env.step(action)

            # 3. The agent learns from this experience (state, action, reward, next_state).
            # The learning logic is different for each agent.
            if config["agent_type"] == "tabular":
                # The Tabular agent learns immediately from the experience.
                agent.update(state, action, reward, next_state)
                
            elif config["agent_type"] == "dqn":
                # The DQN agent first stores the experience in its memory.
                agent.remember(state, action, reward, next_state, done)
                # Then, it learns from a random batch of past experiences
                loss = agent.replay(config["dqn_params"]["batch_size"])
                if loss is not None:
                    training_losses.append(loss)
                    
            # Move to the next state for the next step in the episode
            state = next_state
            step_count += 1
            total_reward += reward
            
        # End of Episode
        # After each game, decay epsilon slightly to gradually shift
        # from 100% exploration to 100% exploitation.
        epsilon = max(config["training"]["epsilon_end"], epsilon * config["training"]["epsilon_decay"])
        
        # Log the metrics for this episode.
        training_steps.append(step_count)
        training_successes.append(1 if total_reward > 0 else 0)

        # Print a progress update every 1000 episodes.
        if (episode + 1) % 1000 == 0:
            avg_loss = np.mean(training_losses[-100:]) if training_losses else 0
            # Correctly references 'num_episodes' from the 'training' sub-dictionary.
            print(f"  ... Episode {episode + 1}/{config['training']['num_episodes']} | Epsilon: {epsilon:.4f} | Avg Loss (last 100): {avg_loss:.4f}")
    
    print("✅ Training finished.")
    
    # 4. EVALUATION PHASE
    # Now that training is done, call our evaluation function to get the
    # final, objective performance score using the *evaluation* environment.
    print("\n--- Evaluating final agent performance... ---")
    
    evaluation_results = evaluate_agent(agent, eval_env, config["evaluation"]["num_episodes"])
    print(f"Evaluation Success Rate: {evaluation_results['evaluation_success_rate']:.4f}")
    print(f"Evaluation Avg Steps: {evaluation_results['evaluation_avg_steps']:.2f}")

    eval_env.close()

    # 5. SAVING RESULTS
    # This section archives the experiment for later analysis.
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
    # 'json.dump' writes the dictionary to a text file in a human-readable format.
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=4)
    print(f"📊 Full experiment data saved to {results_path}")

    # --- 6. PLOTTING ---
    # The plot title is now the unique experiment name for clarity.
    plot_metrics(full_results["training_metrics"], title=config["experiment_name"])


if __name__ == "__main__":
    # The main entry point. It runs one experiment based on the CONFIG dictionary.
    run_experiment(CONFIG)

