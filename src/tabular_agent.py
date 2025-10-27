import numpy as np

class TabularQAgent:
    """
    Implements a Q-Learning agent that uses a lookup table (the "Q-table")
    to learn the optimal action-selection policy.

    This agent is "tabular" because it stores a value for every possible
    state-action pair in a discrete environment. It does not generalize.
    """
    def __init__(self, state_space_size, action_space_size, discount_factor):
        """
        Initializes the agent's internal knowledge structures

        Args:
            state_space_size (int): total # of states (e.g., 64 for 8x8 FrozenLake).
            action_space_size (int): # of possible actions (e.g., 4 directions).
            discount_factor (float): Gamma, the importance of future rewards.
        """
        # The Q-table is the agent's "brain".
        # It's a matrix where:
        # - Rows correspond to states (S)
        # - Columns correspond to actions (A)
        # - Each cell Q[s, a] stores the agent's current estimate of the total future reward it can get by taking action 'a' in state 's'.        
        # It's initialized with zeros as the agent knows nothing initially.
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.gamma = discount_factor

        # A matrix to count visits to each state-action pair.
        # This is crucial for implementing a dynamic, adaptive learning rate (alpha),
        # which helps to stabilize learning in stochastic environments.
        self.visits = np.zeros((state_space_size, action_space_size), dtype=int)

    def choose_action(self, state, epsilon):
        """
        Selects an action using the Epsilon-Greedy strategy.

        This strategy provides a simple solution to the "exploration vs. exploitation" dilemma:
        - Exploit: choose the best action we currently know.
        - Explore: choose a random action to discover potentially better paths.

        Args:
            state (int): the agent's current state.
            epsilon (float): the probability of choosing a random (explore) action.

        Returns:
            int: The index of the chosen action.
        """
        # With probability epsilon, explore. Otherwise, exploit.
        if np.random.uniform(0, 1) < epsilon:
            # Exploration:
            # With probability 'epsilon', choose a random action.
            # This allows the agent to try new things and avoid getting stuck in a sub-optimal policy.
            action = np.random.choice(self.q_table.shape[1])
        else:
            # Exploitation:
            # With probability '1 - epsilon', we choose the best-known action.
            # We look at the Q-table row for the current 'state' and use
            # np.argmax() to find the index (action) of the highest Q-value.
            action = np.argmax(self.q_table[state, :])
            
        return action

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-value for the experienced (state, action) pair using the
        Bellman equation, which is the core of the Q-Learning algorithm.
        
        Args:
            state (int): the state where the action was taken.
            action (int): the action that was taken.
            reward (float): the immediate reward received from the environment.
            next_state (int): the new state the agent transitioned to.
        """        
        # 1. Dynamic Learning Rate (Alpha)
        
        # Increment the visit count for this state-action pair.
        self.visits[state, action] += 1
        
        # Calculate the dynamic learning rate (alpha). 
        # Alpha = 1 / N(s, a), where N is the visit count.
        # This means the agent learns a lot from the first visit (alpha=1.0)
        # and less and less from subsequent visits (e.g., alpha=0.01 after 99 visits).
        # This is a common technique to ensure the Q-values eventually converge,
        # especially in stochastic environments.
        alpha = 1.0 / self.visits[state, action]

        # 2. The Q-Learning Target (Bellman Equation)
        # First, I find the best possible value I can get from the *next* state.
        # This is the "max_a' Q(s', a')" part of the equation.
        # I look at all actions in 'next_state' and pick the highest Q-value.
        max_next_q_value = np.max(self.q_table[next_state, :])

        # Next, I calculate the 'target' value. This is my new, improved
        # estimate for Q(s, a). It's the immediate reward plus the discounted value of the future.
        # Target = r + γ * max_a' Q(s', a')
        target = reward + self.gamma * max_next_q_value

        # 3. The Q-Value Update
        # Get the old Q-value we are about to replace.
        current_q_value = self.q_table[state, action]

        # Update the Q-table by moving the old value a small step (alpha)
        # in the direction of the new target value.
        # Q(s, a) <- Q(s, a) + α * (Target - Q(s, a))
        self.q_table[state, action] = current_q_value + alpha * (target - current_q_value)
