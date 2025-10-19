import numpy as np

class TabularQAgent:
    """
    This class implements the Tabular Q-Learning agent.
    It learns by updating a table of Q-values for each state-action pair.
    """
    def __init__(self, state_space_size, action_space_size, discount_factor, **kwargs):
        """
        Initializes the agent.

        Args:
            state_space_size (int): The number of states (e.g., 64 for 8x8 FrozenLake).
            action_space_size (int): The number of actions (e.g., 4).
            discount_factor (float): Gamma, the importance of future rewards.
        """
        # The Q-table is the agent's "brain", a matrix of size [states x actions].
        # It's initialized with zeros as the agent knows nothing initially.
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.gamma = discount_factor

        # A matrix to count visits to each state-action pair for dynamic learning rate.
        self.visits = np.zeros((state_space_size, action_space_size), dtype=int)

    def choose_action(self, state, epsilon):
        """Chooses an action using an epsilon-greedy policy."""
        # With probability epsilon, explore. Otherwise, exploit.
        if np.random.uniform(0, 1) < epsilon:
            # Exploration: Choose a random action.
            action = np.random.choice(self.q_table.shape[1])
        else:
            # Exploitation: Choose the best-known action for the current state.
            action = np.argmax(self.q_table[state, :])
        return action

    def update(self, state, action, reward, next_state):
        """Updates the Q-table using the Q-learning rule with a dynamic learning rate."""
        
        # 1. Increment the visit count for this state-action pair.
        self.visits[state, action] += 1
        
        # 2. Calculate the dynamic learning rate (alpha). It decreases as we visit a pair more often.
        alpha = 1.0 / self.visits[state, action]

        # 3. Find the best Q-value for the state we landed in (max_a' Q(s', a')).
        max_next_q_value = np.max(self.q_table[next_state, :])

        # 4. Calculate the "target" value: our new, better estimate for Q(s, a).
        target = reward + self.gamma * max_next_q_value

        # 5. Get the old Q-value.
        current_q_value = self.q_table[state, action]

        # 6. Update the Q-value by moving it a small step (alpha) towards the target.
        self.q_table[state, action] = current_q_value + alpha * (target - current_q_value)
