from Wrapper import InitEnv
import gym
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from Agent import DQNAgent, LQNAgent
from eval import Evaluation
import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 2')
    # Model name
    parser.add_argument('--model', default='DQN', type=str, help='LQN, DLQN, DQN, DDQN')
    # Env name
    parser.add_argument('--env', default='ALE/Enduro-v5', type=str, help='ALE/Enduro-v5, ALE/SpaceInvaders-v5')
    # Model dir name
    parser.add_argument('--dir', default='temp-0410/', type=str, help='Directoty storing the model')
    # Training parameters
    parser.add_argument('--start', default=50000, type=int, help='Replay start size')
    parser.add_argument('--memory', default=1000000, type=int, help='Replay memory size')
    parser.add_argument('--max_eps', default=1, type=float, help='Initial exploration')
    parser.add_argument('--min_eps', default=0.01, type=float, help='Final exploration')
    parser.add_argument('--dec_eps', default=1000000, type=int, help='Final exploration frame')
    parser.add_argument('--freq', default=10000, type=int, help='Target network update frequency')
    parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--lr', default=0.00025, type=float, help='Learning rate')
    parser.add_argument('--games', default=200, type=int, help='Total episodes')
    parser.add_argument('--eval', action='store_true', help='Whether do eval during training')
    parser.add_argument('--epoch', default=1000, type=int, help='Eval epoch')
    parser.add_argument('--leps', default=0.01, type=float, help='RMSLoss eps')
    parser.add_argument('--m', default=0.95, type=float, help='RMSLoss momentum')
    args = parser.parse_args()

    if args.env == 'ALE/Enduro-v5':
        env = InitEnv('ALE/Enduro-v5')
    elif args.env == 'ALE/SpaceInvaders-v5':
        env = InitEnv('ALE/SpaceInvaders-v5')
    else:
        raise KeyError(f'{args.env} not tried yet')

    if not os.path.exists(args.dir):
        raise NotADirectoryError('Folder specified to save plots and models does not exist')   

 
    # Initialize Agent
    epsilon_decrement = (args.max_eps-args.min_eps)/args.dec_eps

    if args.model == 'DQN' or args.model == 'DDQN':
        agent = DQNAgent(
            input_shape=env.observation_space.shape,
            action_shape=env.action_space.n,
            gamma=args.gamma,
            epsilon=args.max_eps,
            learning_rate=args.lr,
            batch_size=32,
            memory_size=args.memory,
            epsilon_minimum=args.min_eps,
            epsilon_decrement=epsilon_decrement,
            target_replace_frequency=args.freq,
            replay_start_size = args.start,
            leps = args.leps,
            momentum = args.m,
            checkpoint_dir=args.dir,

        )
    elif args.model == 'LQN' or args.model == 'DLQN':
        agent = LQNAgent(
            input_shape=env.observation_space.shape,
            action_shape=env.action_space.n,
            gamma=args.gamma,
            epsilon=args.max_eps,
            learning_rate=args.lr,
            batch_size=32,
            memory_size=args.memory,
            epsilon_minimum=args.min_eps,
            epsilon_decrement=epsilon_decrement,
            target_replace_frequency=args.freq,
            replay_start_size = args.start,
            leps = args.leps,
            momentum = args.m,
            checkpoint_dir=args.dir,

        )      
    else:
        raise KeyError(f'{args.model} not implemented yet')

    # Populate the replay memory
    while agent.replay_memory.memory_counter < agent.replay_start_size:
        print(f"Populate replay memory {agent.replay_memory.memory_counter}/{agent.replay_start_size}")
        done = False
        observation = env.reset()
        while not done:
            action = agent.random_action()
            new_observation, reward, done, info = env.step(action)
            agent.save_to_memory(observation, action, reward, new_observation, done)
            observation = new_observation

    # Initialize training records
    plot_name = f'{args.dir}{args.model}_agent_enduro_plot.png'
    rolling_average_n = args.games
    best_score = 0
    scores, steps, rolling_means, epsilons = [], [], [], []
    current_step = 0

    
    # Train and Evaluation and Visualization
    vis_eps = []
    vis_scores = []
    for episode in range(args.games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward

            agent.save_to_memory(observation, action, reward, new_observation, done)
            if args.model == 'DQN' or args.model == 'LQN':
                agent.learn()
            if args.model == 'DDQN' or args.model == 'DLQN':
                agent.double_learn()
            observation = new_observation
            current_step += 1
            

        scores.append(score)
        steps.append(current_step)
        

        rolling_mean = np.mean(scores[-rolling_average_n:])
        rolling_means.append(rolling_mean)

        print(f"Ep: {episode} | Score: {score} | Avg: {rolling_mean:.1f} | Best: {best_score:.1f}")
        vis_eps.append(episode)
        vis_scores.append(score)
        if (score > best_score) or score == best_score:
            best_score = score
            agent.save_networks()
        if args.eval:
                if (episode % args.epoch) == 0:
                    Evaluation(epoch=episode, directory=args.dir, eval_episodes=5, model = args.model)
    plt.figure()
    plt.plot(vis_eps, vis_scores)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.savefig(args.dir + '/' + args.model + str(time.time()) + '.png')

    

