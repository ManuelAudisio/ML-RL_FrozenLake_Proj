import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os

from dqn_agent import DQNAgent
from tabular_agent import TabularQAgent

# 1. GLOBAL CONFIGURATION

# A mapping from the environment's integer actions to visual arrows.
# This makes the final plot human-readable.
ACTION_MAP = { 
              0: "←",
              1: "↓",
              2: "→",
              3: "↑" }
MAP_SIZE = 8

def visualize_policy(model_path):
    """
    Loads a trained agent's model file (Q-table or DQN weights) and
    generates a visual plot of its learned policy for the 8x8 FrozenLake.
    """
    
    # 2. LOAD AGENT BRAIN
    
    # Safety check: ensure the specified model file actually exists.
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return

    print(f"--- 🧠 Loading agent brain from: {model_path} ---")

    # Determine which agent to load based on the file extension.
    # .pth is the standard extension for PyTorch models (DQN).
    # .npy is the standard extension for NumPy arrays (Q-table).
    agent_type = "dqn" if model_path.endswith(".pth") else "tabular"
    
    # Create a temporary environment just to get its properties
    # (state and action space dimensions).
    dummy_env = gym.make("FrozenLake-v1", map_name=f"{MAP_SIZE}x{MAP_SIZE}")
    state_size = dummy_env.observation_space.n
    action_size = dummy_env.action_space.n
    dummy_env.close()

    # Load the agent based on the detected type
    if agent_type == "tabular":
        # Create a new, untrained TabularQAgent instance
        agent = TabularQAgent(state_size, action_size, discount_factor=0.99)
        # Load the saved Q-table array from the .npy file
        agent.q_table = np.load(model_path)
        print("Tabular Q-Learning agent loaded.")
    else: # dqn
        # Create a new, untrained DQNAgent instance.
        # Note: i pass dummy values for lr/gamma as they are not needed for inference.
        agent = DQNAgent(state_size, action_size, learning_rate=0.001, discount_factor=0.99)
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.q_network.eval()
        print("DQN agent loaded.")

    # 3. POLICY EXTRACTION 
    # Now that the agent is loaded, we query it to find its optimal policy.
    
    # Create an 8x8 grid to store the best action (0-3) for each state.
    policy_grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)
    for state in range(state_size):
        # Ask the agent: "What is the best possible action from this state?"
        # We set epsilon=0 to force the agent to exploit its knowledge, not explore.
        action = agent.choose_action(state, epsilon=0)
        # Convert the 1D state index (e.g., 27) to 2D grid coordinates (e.g., row 3, col 3)
        row, col = divmod(state, MAP_SIZE)
        # Store the best action in our policy grid
        policy_grid[row, col] = action

    print("--- 🗺️ Policy extracted. Generating visualization... ---")
    
    # 4. PLOT VISUALIZATION 
    
    # Load the environment's map description (S=Start, F=Frozen, H=Hole, G=Goal)
    # This is used to color the background of the grid.

    desc = gym.make("FrozenLake-v1", map_name=f"{MAP_SIZE}x{MAP_SIZE}").desc.astype(str)
    color_map = np.zeros((MAP_SIZE, MAP_SIZE, 3)) # Create an empty RGB map
    color_map[desc == 'S'] = [0.6, 0.8, 1.0]  # Start = Light Blue
    color_map[desc == 'F'] = [1.0, 1.0, 1.0]  # Frozen = White
    color_map[desc == 'H'] = [0.6, 0.6, 0.6]  # Hole = Gray
    color_map[desc == 'G'] = [0.8, 1.0, 0.8]  # Goal = Light Green

    fig, ax = plt.subplots(figsize=(10, 10))
    # Draw the colored background map
    ax.imshow(color_map)

    # Loop through the grid and draw the arrows
    for r in range(MAP_SIZE):
        for c in range(MAP_SIZE):
            # Get the best action from our extracted policy
            action = policy_grid[r, c]
            # Convert the action number (0-3) to an arrow string ("←", "↓", ...)
            arrow = ACTION_MAP[action]
            
            # Highlight bad decisions: if the agent is on a Hole, draw the arrow in red.
            # Note: The agent will never actually take this action (episode ends),
            # but it shows what the agent *would* do from that terminal state.
            text_color = "red" if desc[r, c] == 'H' else "black"
            
            # Draw the arrow text on the correct cell
            ax.text(c, r, arrow, ha='center', va='center', fontsize=20, color=text_color)

    # 5. PLOT FORMATTING & SAVING
    ax.set_xticks(np.arange(-.5, MAP_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-.5, MAP_SIZE, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.tick_params(axis='both', which='major', labelsize=0)
    
    plot_title = os.path.basename(model_path).replace('.npy', '').replace('.pth', '')
    ax.set_title(f"Learned Policy: {plot_title}", fontsize=16)
    
    # Define the save directory (and create it if it doesn't exist)
    save_dir = "results/final_analysis/plots"
    os.makedirs(save_dir, exist_ok=True)
    # Save the final plot as a high-DPI image
    save_path = os.path.join(save_dir, f"policy_{plot_title.lower()}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"--- ✅ Policy visualization saved to: {save_path} ---")
    plt.show()

# 6. SCRIPT EXECUTION
# This block runs only when the script is called from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the learned policy of a trained agent.")
    parser.add_argument("model_path", type=str, help="Path to the saved model file (.npy for Q-table, .pth for DQN).")
    
    args = parser.parse_args()
    visualize_policy(args.model_path)

