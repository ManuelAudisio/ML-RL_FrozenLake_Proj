import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 1. Q-Network Architecture Definition ---
class QNetwork(nn.Module):
    """
    This class defines the neural network architecture for the Q-function approximator.
    It is a fully-connected network as specified by the project requirements.
    """
    def __init__(self, state_size, action_size):
        """
        Initializes the layers of the neural network.

        Args:
            state_size (int): the dimension of the input state (e.g., 64 for one-hot 8x8 grid).
            action_size (int): the dimension of the output (number of possible actions, e.g., 4).
        """
        super(QNetwork, self).__init__()
        # Defines the architecture: Input -> Hidden 1 (32) -> Hidden 2 (32) -> Output
        # 32 neurons per hidden layer is an effective baseline.
        self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, action_size)

    def forward(self, state):
        """
        Defines the forward pass of data through the network.

        Args:
            state (torch.Tensor): the input state tensor.
        
        Returns:
            torch.Tensor: the output Q-values for each action.
        """
        # The ReLU activation function is applied to hidden layers to introduce non-linearity,
        # which is essential for learning complex value functions.
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        
        # The final output layer is linear. This is crucial because Q-values are
        # unbounded (can be positive or negative) and do not represent probabilities.
        return self.layer3(x)

# --- 2. DQN Agent Implementation ---
class DQNAgent:
    """
    This class implements the Deep Q-Network (DQN) agent.
    It manages the Q-Network, the replay buffer (memory), and the learning process.
    """
    def __init__(self, state_size, action_size, learning_rate, discount_factor):
        """
        Initializes the agent, its network, optimizer, and loss function.

        Args:
            state_size (int): the dimension of the input state.
            action_size (int): the dimension of the output (# of actions).
            learning_rate (float): the learning rate (alpha) for the Adam optimizer.
            discount_factor (float): the discount factor (gamma) for future rewards.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor

        # The agent's "brain": the Q-Network that estimates state-action values.
        # This is the "online" network that is actively trained.
        self.q_network = QNetwork(self.state_size, self.action_size)

        # The Adam optimizer is a robust choice that adapts the learning rate
        # and works well for most deep learning tasks.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Mean Squared Error (MSE) is the standard loss function for DQN.
        # It measures the squared difference between the network's predicted Q-values
        # and the "target" Q-values derived from the Bellman equation.
        self.loss_fn = nn.MSELoss()

        # The replay buffer. It will be initialized as a deque (a fixed-size queue)
        # from the main script. Storing it as 'None' here ensures it's set by the runner.
        self.memory = None

    def choose_action(self, state, epsilon):
        """
        Selects an action using the Epsilon-Greedy strategy.
        This balances exploration (trying random actions) vs. exploitation (using known info).
        
        Args:
            state (int): the agent's current state (as an integer index).
            epsilon (float): the probability of choosing a random (explore) action.

        Returns:
            int: The index of the chosen action.
        """
        if np.random.uniform(0, 1) < epsilon:
            # Exploration:
            # With probability 'epsilon', a random action is chosen.
            # This is vital for discovering new, potentially better state-action pairs.
            return np.random.choice(self.action_size)
        else:
            # Exploitation: 
            # The agent uses its current knowledge to pick the best action.
            
            # 1. Convert the integer state into a one-hot tensor representation.
            #    This is the format the neural network expects as input.
            state_tensor = torch.zeros(self.state_size)
            state_tensor[state] = 1.0
            
            # 2. Set the network to evaluation mode (e.g., disables dropout if it were used).
            self.q_network.eval()
            
            # 3. Use `torch.no_grad()` to disable gradient calculations.
            #    This is purely for inference (prediction), not training, so
            #    this step saves computation and memory.
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
            # 4. Set the network back to training mode for the next learning step.
            self.q_network.train()
            
            # 5. Return the action (index) with the highest predicted Q-value.
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Saves a single transition (an "experience") to the replay buffer.
        The buffer is a deque, which automatically discards the oldest experience
        when it reaches its maximum size.
        """        
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Trains the Q-Network by sampling and learning from a batch of past experiences.
        This is the core learning step of the agent.
        
        Args:
            batch_size (int): the number of experiences to sample from the buffer.
        
        Returns:
            float or None: the calculated loss value for this training step, or None if
                           training did not occur.
        """        
        # A guard clause: do not attempt to train if the memory buffer doesn't
        # have enough experiences to form a full batch.
        if len(self.memory) < batch_size:
            return None

        # 1. Sample Experiences 
        # Randomly sample a `batch_size` of experiences from the replay buffer.
        # This random sampling breaks the temporal correlation between consecutive
        # steps, which is crucial for stabilizing the training of the neural network.
        minibatch = random.sample(self.memory, batch_size)
        
        # Unzip the batch into separate lists of states, actions, etc.
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert the lists into numpy arrays for efficient batch processing.
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # 2. Prepare Data Tensors
        # Convert the integer states (e.g., 27) into one-hot encoded vectors
        # (e.g., a vector of 64 zeros with a '1' at index 27).
        states_one_hot = np.zeros((batch_size, self.state_size))
        states_one_hot[np.arange(batch_size), states] = 1
        
        next_states_one_hot = np.zeros((batch_size, self.state_size))
        next_states_one_hot[np.arange(batch_size), next_states] = 1

        # Convert all numpy arrays to PyTorch tensors for processing by the network.
        states_tensor = torch.FloatTensor(states_one_hot)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states_one_hot)
        dones_tensor = torch.BoolTensor(dones)
        
        # 3. Calculate Target Q-Values
        
        # First, get the Q-values that the network *currently* predicts for this batch of states.
        # Then, use `.gather()` to select only the Q-values for the *actions that were actually taken*.
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # Next, calculate the target values using the Bellman equation.
        # We need the max Q-value for the *next* state.
        with torch.no_grad():
            max_next_q_values = self.q_network(next_states_tensor).max(1)[0]
            
        # If a state was a terminal state (done=True), its future value is 0 by definition.
        max_next_q_values[dones_tensor] = 0.0

        # The Bellman Target: Target = immediate_reward + discounted_future_value
        target_q_values = rewards_tensor + (self.gamma * max_next_q_values)

        # 4. Compute Loss and Perform Backpropagation
        
        # Calculate the Mean Squared Error loss between what the network *predicted*
        # (current_q_values) and what it *should have* predicted (target_q_values).
        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        # Clear any old gradients from the previous training step.
        self.optimizer.zero_grad()
        # Calculate the new gradients (i.e., how much each weight contributed to the loss).
        loss.backward()
        
         # --- BEST PRACTICE: GRADIENT CLIPPING ---
        # This line prevents the gradients from becoming excessively large (exploding),
        # which can destabilize the training (it happened during tests). It "clips" their norm to a max value.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        # Tell the optimizer to update all of the network's weights using the
        # calculated (and clipped) gradients.
        self.optimizer.step()      
        
        # Return the scalar value of the loss for logging and analysis.
        return loss.item()

