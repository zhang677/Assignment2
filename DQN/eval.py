from Wrapper import InitEvalEnv
from Agent import DQNAgent
import gym
import os
import gym
import numpy as np
import matplotlib.pyplot as plt

def Evaluation(epoch=0, directory = 'temp/',eval_episodes = 20):
    env = InitEvalEnv()
    agent = DQNAgent(
        input_shape=env.observation_space.shape,
        action_shape=env.action_space.n,
        checkpoint_dir=directory
    )
    agent.load_networks()

    scores = []
    for episode in range(eval_episodes):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            observation = new_observation

        scores.append(score)
        print(f"Ep: {episode} | Score: {score}")
    
    mean = np.mean(scores)
    var = np.std(scores, ddof = 1)
    print(f"Epoch: {epoch} | Mean: {mean} | Var: {var}")





