import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(results, title, window_size=500): # Aumentiamo la finestra per smussare di più
    """
    Plots multiple metrics from a results dictionary.
    """
    # Crea una figura con un sotto-grafico per ogni metrica
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 8 * len(results)))
    fig.suptitle(title, fontsize=16)

    # Itera attraverso le metriche e plottale
    for i, (metric_name, values) in enumerate(results.items()):
        ax = axes[i]
        moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        ax.plot(moving_avg)
        ax.set_title(metric_name)
        ax.set_xlabel(f'Episodes (averaged over {window_size} episodes)')
        ax.set_ylabel(f'Average {metric_name}')
        ax.grid(True)

    filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(f'results/plots/{filename}')
    print(f"\n📈 Plots saved to results/plots/{filename}")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta il layout
    plt.show()

"""
def plot_rewards(rewards_per_episode, title, window_size=100):
    # aggiungi virgolette se togli quelle del blocco 
    Plots the agent's performance with a moving average to smooth the curve.

    Args:
        rewards_per_episode (list): A list containing the total reward for each episode.
        title (str): The title for the plot and the filename.
        window_size (int): The number of episodes to average over for smoothing.
    # aggiungi virgolette se togli quelle del blocco 
    
    # A moving average helps to visualize the underlying trend in performance,
    # smoothing out the noise from individual episode results.
    moving_avg = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg)
    plt.title(title)
    plt.xlabel(f'Episodes (averaged over {window_size} episodes)')
    plt.ylabel('Average Reward per Episode')
    plt.grid(True)

    # We save the plot automatically to the results folder. This is crucial for the report.
    # We create a safe filename from the title.
    filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(f'results/plots/{filename}')
    print(f"\n📈 Plot saved to results/plots/{filename}")

    # Display the plot
    plt.show()
    
"""