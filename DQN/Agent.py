import numpy as np
import torch

from model import DeepQNetwork
from Memory import Memory


class DQNAgent:
    def __init__(
            self, input_shape, action_shape, gamma=0.99, epsilon=0.1, learning_rate=0.00025,
            batch_size=32, memory_size=1000000, epsilon_minimum=0.01,
            epsilon_decrement=9e-7, target_replace_frequency=10000, replay_start_size=50000,
            leps=0.01, momentum=0.95, checkpoint_dir='temp/'
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_minimum = epsilon_minimum
        self.epsilon_decrement = epsilon_decrement
        self.target_replace_frequency = target_replace_frequency
        self.checkpoint_dir = checkpoint_dir
        self.replay_start_size = replay_start_size

        self.action_space = [i for i in range(action_shape)]
        self.batch_space = [i for i in range(self.batch_size)]
        self.current_step = 0

        self.replay_memory = Memory(memory_size, input_shape)
        self.eval_network, self.target_network = self.create_networks(
            input_shape, action_shape, learning_rate, leps, momentum
        )

    def create_networks(self, *args, **kwargs):
        return (
            DeepQNetwork(*args, **kwargs, checkpoint_file=self.checkpoint_dir + 'dqn_eval'),
            DeepQNetwork(*args, **kwargs, checkpoint_file=self.checkpoint_dir + 'dqn_target')
        )

    def random_action(self):
        return np.random.choice(self.action_space)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)

        self.eval_network.eval()
        with torch.no_grad():
            state = torch.tensor(np.array([observation]), dtype=torch.float)
            state = state.to(self.eval_network.device)
            actions = self.eval_network.forward(state)
        return torch.argmax(actions).item()

    def replace_target_network(self):
        if self.current_step % self.target_replace_frequency == 0:
            self.target_network.load_state_dict(self.eval_network.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_minimum:
            self.epsilon -= self.epsilon_decrement
        else:
            self.epsilon = self.epsilon_minimum

    def save_networks(self):
        self.target_network.save_checkpoint()
        self.eval_network.save_checkpoint()

    def load_networks(self):
        self.target_network.load_checkpoint()
        self.eval_network.load_checkpoint()

    def save_to_memory(self, state, action, reward, new_state, done):
        self.replay_memory.save(state, action, reward, new_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.replay_memory.sample(self.batch_size)
        return (
            self.eval_network.to_tensor(state),
            self.eval_network.to_tensor(action),
            self.eval_network.to_tensor(reward),
            self.eval_network.to_tensor(new_state),
            self.eval_network.to_tensor(done)
        )

    def learn(self):

        # Fill all the replay memory before starting
        if self.replay_memory.memory_counter < self.replay_start_size:
            return

        self.replace_target_network()
        states, actions, rewards, next_states, done_flags = self.sample_memory() 

        self.target_network.eval()
        with torch.no_grad():
            action_values_next = self.target_network.forward(next_states)
            action_values_next = action_values_next.max(dim=1)[0]
            action_values_next[done_flags] = 0.0
            action_value_target = rewards + self.gamma * action_values_next

        # Propagate errors and step
        self.eval_network.train()
        self.eval_network.optimizer.zero_grad()
        action_values = self.eval_network.forward(states)[self.batch_space, actions]

        self.eval_network.backward(action_value_target, action_values)
        self.decrement_epsilon()
        self.current_step += 1

    def double_learn(self):
        # Fill all the replay memory before starting
        if self.replay_memory.memory_counter < self.replay_start_size:
            return

        self.replace_target_network()
        states, actions, rewards, next_states, done_flags = self.sample_memory()

        
        next_actions = self.choose_action(next_states)
        self.target_network.eval()
        with torch.no_grad():
            action_values_next = self.target_network.forward(next_states)[self.batch_space,next_actions]
            action_values_next[done_flags] = 0.0
            action_value_target = rewards + self.gamma * action_values_next
        
        self.eval_network.train()
        self.eval_network.optimizer.zero_grad()
        action_values = self.eval_network.forward(states)[self.batch_space, actions]

        # Propagate errors and step
        self.eval_network.backward(action_value_target, action_values)
        self.decrement_epsilon()
        self.current_step += 1


