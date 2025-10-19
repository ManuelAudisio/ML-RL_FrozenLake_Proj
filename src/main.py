import gymnasium as gym
import numpy as np
from collections import deque
from tabular_agent import TabularQAgent
from dqn_agent import DQNAgent
from utils import plot_metrics

# --- CONFIGURATION DICTIONARY ---
# This is our central control panel for experiments.
CONFIG = {
    "agent_type": "dqn", # Choose here: "tabular" or "dqn"
    "env_id": "FrozenLake-v1",
    "env_config": {"is_slippery": True, "map_name": "8x8"}, # We are now on the main challenge
    "num_episodes": 20000,     

    # --- Shared Parameters ---
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.9999,

    # --- DQN-Specific Parameters ---
    #"dqn_learning_rate": 0.001, # Our baseline learning rate
    #"dqn_learning_rate": 0.0005, # Test 1
    "dqn_learning_rate": 0.0001, # Test 2
    "dqn_batch_size": 64,
    "dqn_replay_buffer_size": 10000
}

def run_experiment():
    print(f"--- Starting Experiment For Agent: {CONFIG['agent_type'].upper()} ---")
    env = gym.make(CONFIG["env_id"], **CONFIG["env_config"])

    # --- Conditional agent initialization ---
    if CONFIG["agent_type"] == "tabular":
        agent = TabularQAgent(
            state_space_size=env.observation_space.n,
            action_space_size=env.action_space.n,
            discount_factor=CONFIG["discount_factor"]
        )
    elif CONFIG["agent_type"] == "dqn":
        agent = DQNAgent(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            discount_factor=CONFIG["discount_factor"],
            learning_rate=CONFIG["dqn_learning_rate"]
        )
        agent.memory = deque(maxlen=CONFIG["dqn_replay_buffer_size"])
    else:
        raise ValueError(f"Agent type '{CONFIG['agent_type']}' not recognized.")
    
    print(f"{agent.__class__.__name__} created.")

    # --- Lists to store metrics ---
    rewards_per_episode = []
    steps_per_episode = []
    success_rate = []
    losses = [] # New list for tracking loss
    epsilon = CONFIG["epsilon_start"]

    print(f"Starting training for {CONFIG['num_episodes']} episodes...")
    for episode in range(CONFIG["num_episodes"]):
        state, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            # --- Conditional learning logic ---
            if CONFIG["agent_type"] == "tabular":
                agent.update(state, action, reward, next_state)
            elif CONFIG["agent_type"] == "dqn":
                agent.remember(state, action, reward, next_state, done)
                # The replay method now returns the loss
                loss = agent.replay(CONFIG["dqn_batch_size"])
                if loss is not None:
                    losses.append(loss)

            state = next_state
            total_reward += reward
            step_count += 1

        epsilon = max(CONFIG["epsilon_end"], epsilon * CONFIG["epsilon_decay"])
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(step_count)
        success_rate.append(1 if total_reward > 0 else 0)

        if (episode + 1) % 1000 == 0:
            avg_loss = np.mean(losses[-1000:]) if losses else 0 # Average recent loss
            print(f"  ... Episode {episode + 1}/{CONFIG['num_episodes']}. Epsilon: {epsilon:.4f}. Avg Loss: {avg_loss:.4f}")

    env.close()
    print("✅ Training finished.")
    
    # Create a dynamic filename for the model
    model_name = f"{CONFIG['agent_type']}_on_{CONFIG['env_id']}.pth"
    model_path = f"results/models/{model_name}"

    if CONFIG["agent_type"] == "tabular":
        # For the tabular agent, we save the entire Q-table numpy array
        np.save(model_path, agent.q_table)
    elif CONFIG["agent_type"] == "dqn":
        # For the DQN agent, we save the network's learned weights and biases
        torch.save(agent.q_network.state_dict(), model_path)
    
    print(f"🧠 Agent brain saved to {model_path}")
    
    # --- Plotting all metrics ---
    results = {
        "Total Reward": rewards_per_episode,
        "Success Rate": success_rate,
        "Steps per Episode": steps_per_episode,
        "Training Loss": losses
    }
    # Create a dynamic title to avoid overwriting plot files
    plot_title = f"{CONFIG['agent_type']}_stochastic_lr_{CONFIG['dqn_learning_rate']}"
    plot_metrics(results, title=plot_title)


if __name__ == "__main__":
    run_experiment()
