import numpy as np

class TabularQAgent:
    def __init__(self, state_space_size, action_space_size, discount_factor):
        self.q_table = np.zeros((state_space_size, action_space_size))
        # We no longer need self.lr, as it will be dynamic. We can comment it out or remove it.
        # self.lr = learning_rate 
        self.gamma = discount_factor

        # --- NUOVA AGGIUNTA ---
        # A matrix to count how many times we have visited each state-action pair.
        # This is essential for the dynamic learning rate.
        self.visits = np.zeros((state_space_size, action_space_size), dtype=int)

    def choose_action(self, state, epsilon):
        # This method remains exactly the same.
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.q_table.shape[1])
        else:
            action = np.argmax(self.q_table[state, :])
        return action

    def update(self, state, action, reward, next_state):
        # --- LOGICA DI AGGIORNAMENTO MODIFICATA ---

        # 1. Increment the visit count for the state-action pair we just experienced.
        self.visits[state, action] += 1

        # 2. Calculate the dynamic learning rate (alpha) for this specific update.
        alpha = 1.0 / self.visits[state, action]

        # 3. The rest of the update rule is the same, but uses the new dynamic alpha.
        max_next_q_value = np.max(self.q_table[next_state, :])
        target = reward + self.gamma * max_next_q_value
        current_q_value = self.q_table[state, action]
        
        # We use the new dynamic 'alpha' here instead of the old 'self.lr'
        self.q_table[state, action] = current_q_value + alpha * (target - current_q_value)