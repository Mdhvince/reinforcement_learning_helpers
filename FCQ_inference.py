from pathlib import Path
from itertools import count

import gym
import torch
import matplotlib.pyplot as plt

from FCQ_training import FCQ
from action_selection import GreedyStrategy



if __name__ == "__main__":
    model_dir = Path("models")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")

    env = gym.make("CartPole-v1")
    nS, nA = env.observation_space.shape[0], env.action_space.n

    net = FCQ(device, nS, nA, hidden_dims=(512, 128))
    net.load_state_dict(torch.load(model_dir / "model.498.pt"))
    net.eval()

    strategy = GreedyStrategy()

    R = []
    n_episodes = 3
    for _ in range(n_episodes):
        state, done = env.reset()[0], False
        R.append(0)
        for _ in count():
            a = strategy.select_action(net, state)

            plt.figure(3)
            plt.clf()
            plt.imshow(env.env.render())

            state, reward, done, *_ = env.step(a)
            R[-1] += reward

            if done: break
    
    env.close()