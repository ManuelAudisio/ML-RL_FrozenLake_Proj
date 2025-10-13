# RL Project: Q-Learning vs. DQN on FrozenLake

This project is an academic assignment for the Machine Learning course. It involves implementing and comparing two Reinforcement Learning algorithms, Tabular Q-Learning and Deep Q-Network (DQN), to solve the FrozenLake-v1 environment from the Gymnasium library.

## Project Goal

The main objective is to analyze and compare the performance of a classic tabular method against a deep learning-based approach on the same discrete, stochastic environment. The analysis will focus on learning speed, final performance, and stability.

## Project Structure

- `/src`: Contains all the Python source code.
  - `main.py`: The main script to run experiments.
  - `tabular_agent.py`: Implementation of the Tabular Q-Learning agent.
  - `dqn_agent.py`: Implementation of the Deep Q-Network agent.
  - `utils.py`: Utility functions for plotting and logging.
- `/report`: Contains the final PDF report.
- `/notebooks`: Jupyter notebooks for experimentation and visualization.
- `/results`: (Git-ignored) Directory for saving generated plots and models.
- `/final_showcase`: Directory for presenting final plots and models.

## How to Run

1.  **Set up the environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Run an experiment:**
    ```bash
    python src/main.py
    ```