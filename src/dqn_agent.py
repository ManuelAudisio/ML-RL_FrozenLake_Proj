import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 1. DEFINING THE NEURAL NETWORK ARCHITECTURE ---
class QNetwork(nn.Module):
    """
    This class defines the structure of our function approximator, a simple Neural Network.
    It will take a state as input and output the Q-values for each possible action.
    """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # The architecture is simple as requested: Input -> Hidden 1 -> Hidden 2 -> Output.
        # We use 32 neurons per hidden layer as a good starting point.
        self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, action_size)

    def forward(self, state):
        """Defines the forward pass of the network."""
        # We use the ReLU activation function for the hidden layers. It is a standard
        # and effective choice that helps with non-linearities.
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        # The output layer is linear because Q-values are not probabilities and can be any real number.
        return self.layer3(x)

# --- 2. DEFINING THE DQN AGENT ---
class DQNAgent:
    """
    This class implements the DQN agent, including the Q-Network,
    the experience replay mechanism, and the learning logic.
    """
    def __init__(self, state_size, action_size, learning_rate, discount_factor, **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor

        # The agent's "brain": the neural network that approximates Q-values.
        self.q_network = QNetwork(self.state_size, self.action_size)

        # We use the Adam optimizer, which is a robust choice for training neural networks.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # The Mean Squared Error loss measures the difference between our predicted Q-values
        # and the "target" Q-values from the Bellman equation.
        self.loss_fn = nn.MSELoss()

        # The replay buffer. It will be initialized as a deque from the main script.
        self.memory = None

    def choose_action(self, state, epsilon):
        """Chooses an action using an epsilon-greedy policy."""
        # With probability epsilon, we explore. Otherwise, we exploit.
        if np.random.uniform(0, 1) < epsilon:
            # Exploration: choose a random action from the available actions.
            return np.random.choice(self.action_size)
        else:
            # Exploitation: ask the network for the best action.
            # Convert the single integer state into a one-hot encoded tensor.
            state_tensor = torch.zeros(self.state_size)
            state_tensor[state] = 1.0
            
            # Set the network to evaluation mode. This is good practice for inference.
            self.q_network.eval()
            # `torch.no_grad()` disables gradient calculation, which is unnecessary for inference
            # and makes the process faster.
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            # Set the network back to training mode for the next learning step.
            self.q_network.train()
            
            # Return the action (index) with the highest Q-value.
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple (s, a, r, s', done) in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Trains the Q-Network by sampling a random batch from the replay buffer."""
        # Do not attempt to train if the buffer doesn't have enough experiences yet.
        if len(self.memory) < batch_size:
            return None

        # --- 1. Sample Experiences ---
        # Sample a random minibatch of experiences from the replay buffer.
        minibatch = random.sample(self.memory, batch_size)
        # Unzip the batch into separate lists of states, actions, etc.
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert the batch data to numpy arrays for efficient processing.
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # --- 2. Prepare Data for PyTorch ---
        # Convert integer states to one-hot encoding vectors.
        states_one_hot = np.zeros((batch_size, self.state_size))
        states_one_hot[np.arange(batch_size), states] = 1
        
        next_states_one_hot = np.zeros((batch_size, self.state_size))
        next_states_one_hot[np.arange(batch_size), next_states] = 1

        # Convert all numpy arrays to PyTorch tensors.
        states_tensor = torch.FloatTensor(states_one_hot)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states_one_hot)
        dones_tensor = torch.BoolTensor(dones)
        
        # --- 3. The Core DQN Update Logic ---
        # Get the Q-values that the network *currently* predicts for the states in the batch.
        # Then, use .gather() to select only the Q-values for the actions that were actually taken.
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # Calculate the target Q-values. First, get the max Q-value for the *next* states.
        # .detach() is used because this is part of the target calculation, not the training graph.
        with torch.no_grad():
            max_next_q_values = self.q_network(next_states_tensor).max(1)[0]
        # For terminal states (where the game ended), the future reward is 0 by definition.
        max_next_q_values[dones_tensor] = 0.0

        # Calculate the target Q-value using the Bellman equation: Target = R + γ * max_a' Q(s', a')
        target_q_values = rewards_tensor + (self.gamma * max_next_q_values)

        # Calculate the Mean Squared Error between our current predictions and the targets.
        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        # --- 4. Perform Backpropagation ---
        self.optimizer.zero_grad() # Reset gradients from the previous step.
        loss.backward()            # Calculate new gradients based on the loss.
        
        # --- BEST PRACTICE: GRADIENT CLIPPING ---
        # This prevents the gradients from becoming too large (exploding),
        # which can destabilize the training process. It acts as a safety net.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()      # Update the network's weights using the (clipped) gradients.
        
        # Return the loss value for logging and analysis.
        return loss.item()

