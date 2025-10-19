import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 1. DEFINING THE NEURAL NETWORK ARCHITECTURE ---
class QNetwork(nn.Module):
    """Neural Network to approximate the Q-function."""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Simple architecture: Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output
        self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, action_size)

    def forward(self, state):
        """Defines the forward pass of the network."""
        # ReLU is a standard activation function that works well in most cases.
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        # The output layer is linear because Q-values are not probabilities.
        return self.layer3(x)

# --- 2. DEFINING THE DQN AGENT ---
class DQNAgent:
    """Agent that learns using a Deep Q-Network with Experience Replay."""
    def __init__(self, state_size, action_size, discount_factor=0.99, learning_rate=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor

        # The core of our agent: the Q-Network
        self.q_network = QNetwork(self.state_size, self.action_size)

        # Adam is a robust and widely used optimizer.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # Mean Squared Error is the standard loss function for regression problems like Q-learning.
        self.loss_fn = nn.MSELoss()

        # The replay buffer, implemented as a deque for efficient appends and pops.
        self.memory = deque(maxlen=10000)

    def choose_action(self, state, epsilon):
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.uniform(0, 1) < epsilon:
            # Explore: choose a random action
            return np.random.choice(self.action_size)
        else:
            # Exploit: choose the best action predicted by the network
            # 1. Convert state to a one-hot encoded tensor
            state_tensor = torch.zeros(self.state_size)
            state_tensor[state] = 1.0
            
            # We use torch.no_grad() to disable gradient calculation for inference,
            # which speeds up the process as we are not training here.
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            # Return the action with the highest Q-value
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Trains the network by sampling from the replay buffer."""
        # Don't train if the memory is not filled enough
        if len(self.memory) < batch_size:
            return None # Return None if no training happened

        # Sample a random minibatch of experiences from memory
        minibatch = random.sample(self.memory, batch_size)

        # Unzip the batch into separate lists
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert lists to numpy arrays for vectorized operations
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # --- PREPARE DATA FOR PYTORCH ---
        # One-hot encode states and next_states
        states_one_hot = np.zeros((batch_size, self.state_size))
        states_one_hot[np.arange(batch_size), states] = 1
        
        next_states_one_hot = np.zeros((batch_size, self.state_size))
        next_states_one_hot[np.arange(batch_size), next_states] = 1

        # Convert all data to PyTorch tensors
        states_tensor = torch.FloatTensor(states_one_hot)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states_one_hot)
        dones_tensor = torch.BoolTensor(dones)
        
        # --- THE CORE DQN UPDATE LOGIC ---
        # 1. Get the current Q-values predicted by the network for the batch of states.
        #    We use .gather() to select only the Q-values for the actions that were actually taken.
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # 2. Get the maximum Q-value for the next states.
        #    We use .detach() to prevent gradients from flowing into this calculation,
        #    as it is only used to compute the target.
        max_next_q_values = self.q_network(next_states_tensor).detach().max(1)[0]
        # We set the Q-value for terminal states to 0.
        max_next_q_values[dones_tensor] = 0.0

        # 3. Calculate the target Q-values using the Bellman equation.
        target_q_values = rewards_tensor + (self.gamma * max_next_q_values)

        # 4. Calculate the loss (MSE) between the current and target Q-values.
        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        # 5. Perform backpropagation to update the network's weights.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item() # Return the loss value for tracking
