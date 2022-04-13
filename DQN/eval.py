from Wrapper import InitEvalEnv
from Agent import DQNAgent, LQNAgent
import gym
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
def save_frames_as_gif(frames, path='./', filename='enduro.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    # FuncAnimation: A class that exposes a callable interface to animate a matplotlib figure.
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
    # https://matplotlib.org/gallery/animation/simple_anim.html#sphx-glr-gallery-animation-simple-anim-py
    #anim.save(path + filename, writer='imagemagick', fps=60)
    anim.save(path + filename.replace('.gif', '.mp4'), writer='ffmpeg', fps=30)


def Evaluation(epoch=0, directory = 'temp/',eval_episodes = 20, model = 'dqn'):
    env = InitEvalEnv()
    if model == 'DQN' or model == 'DDQN':
        agent = DQNAgent(
            input_shape=env.observation_space.shape,
            action_shape=env.action_space.n,
            checkpoint_dir=directory,
            memory_size=10000
        )
    else:
        agent =LQNAgent(
            input_shape=env.observation_space.shape,
            action_shape=env.action_space.n,
            checkpoint_dir=directory,
            memory_size=10000
        )
    agent.load_networks()

    scores = []
    
    for episode in range(eval_episodes):
        done = False
        score = 0
        observation = env.reset()
        frames = []
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            observation = new_observation
            frames.append(env.render(mode="rgb_array"))

        scores.append(score)
        print(f"Ep: {episode} | Score: {score}")
        save_frames_as_gif(frames, path = directory, filename=str(epoch) + '_' + str(episode) + '.gif')
    mean = np.mean(scores)
    var = np.std(scores, ddof = 1)
    print(f"Epoch: {epoch} | Mean: {mean} | Var: {var}")
    





