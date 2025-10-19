# src/watch_agent.py
import gymnasium as gym
import numpy as np
import torch
import time
from collections import deque

# Importiamo le classi degli agenti e la configurazione dal nostro script principale
from dqn_agent import DQNAgent
from tabular_agent import TabularQAgent
from main import CONFIG # Usiamo la stessa configurazione per coerenza

def watch_game(num_games=5):
    """
    Loads a trained agent and renders its gameplay in the terminal.
    """
    print("--- Loading agent to watch gameplay ---")
    
    # --- 1. Load the trained agent ---
    model_name = f"{CONFIG['agent_type']}_on_{CONFIG['env_id']}.pth"
    model_path = f"results/models/{model_name}"

    env = gym.make(CONFIG["env_id"], **CONFIG["env_config"], render_mode='ansi')
    
    # Initialize a new agent, same as in training
    if CONFIG["agent_type"] == "tabular":
        agent = TabularQAgent(
            state_space_size=env.observation_space.n,
            action_space_size=env.action_space.n,
            discount_factor=CONFIG["discount_factor"]
        )
        # Load the saved Q-table
        agent.q_table = np.load(model_path)
    elif CONFIG["agent_type"] == "dqn":
        agent = DQNAgent(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            discount_factor=CONFIG["discount_factor"]
        )
        # Load the saved network weights
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.q_network.eval() # Set the network to evaluation mode
    else:
        raise ValueError("Agent type not recognized in CONFIG.")
        
    print(f"🧠 Agent brain loaded from {model_path}")

    # --- 2. Run and Render a few games ---
    for i in range(num_games):
        state, info = env.reset()
        done = False
        print(f"\n--- 🎮 Starting Game #{i+1} ---")
        time.sleep(1) # Pause before starting
        
        step = 0
        while not done:
            # Render the current state to the terminal
            print(env.render())
            
            # --- AGENT CHOOSES BEST ACTION (NO EXPLORATION) ---
            # We set epsilon to 0 to see the optimal policy the agent has learned
            action = agent.choose_action(state, epsilon=0)
            
            # Environment takes a step
            next_state, reward, done, truncated, info = env.step(action)
            
            # Print info about the step
            print(f"Step: {step+1}")
            print(f"Action taken: {action}")
            print(f"Reward received: {reward}")
            print("-" * 20)
            
            state = next_state
            step += 1
            time.sleep(0.5) # Pause between steps to make it watchable
        
        # Render the final state
        print(env.render())
        if reward > 0:
            print("🏆 Agent Won!")
        else:
            print("💀 Agent Lost!")
        time.sleep(2) # Pause before the next game
        
    env.close()

if __name__ == "__main__":
    # Assicurati che il CONFIG in main.py sia impostato sull'agente che vuoi guardare!
    watch_game()