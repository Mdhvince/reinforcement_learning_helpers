
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    plt.style.use('seaborn-paper')
    result = np.fromfile("training_results/FCQ_results.dat")

    n_episodes = 1000
    result = result.reshape(n_episodes, 3)

    fig, axs = plt.subplots(3)

    axs[0].plot(list(range(n_episodes)), result[:, 0], linewidth=1)
    axs[0].set_title("Steps per episode")

    axs[1].plot(list(range(n_episodes)), result[:, 1], linewidth=1)
    axs[1].set_title("Mean last 100 rewards per episode (training)")

    axs[2].plot(list(range(n_episodes)), result[:, 2], linewidth=1)
    axs[2].set_title("Mean last 100 rewards per episode (Evaluation)")

    config_arrow = dict(headwidth=5, facecolor='red', shrink=0.05, width=2)
    axs[2].annotate(
        "Last save - Best agent", xy=(783, 480), xycoords='data',
        xytext=(0.75, 0.5), textcoords='axes fraction',
        arrowprops=config_arrow,
        horizontalalignment='right', verticalalignment='top',
    )

    plt.xlabel('Episodes')
    plt.show()