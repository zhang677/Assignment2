import logging

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, learning_rate, leps, momentum, checkpoint_file):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.checkpoint_file = checkpoint_file

        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4,bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        flattened_shape = self.calculate_flattened_shape(self.input_shape)

        self.fc1 = nn.Linear(flattened_shape, 512)
        self.fc2 = nn.Linear(512, output_shape)

        self.loss = f.huber_loss 
        #self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate, eps=leps, momentum=momentum)#eps=0.01, momentum=0.95)
        '''
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        '''
        self.device = self.get_device()
        self.to(self.device)

    @staticmethod
    def get_device():
        device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_name)
        logging.info(f'Using device: {device}')
        return device

    def calculate_flattened_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def save_checkpoint(self):
        logging.info('Saving checkpoint')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        logging.info('Loading checkpoint')
        print(f'Checkout file: {self.checkpoint_file}')
        self.load_state_dict(torch.load(self.checkpoint_file))

    def to_tensor(self, inputs):
        return torch.tensor(inputs).to(self.device)

    def forward(self, inputs):
        # Convolutions
        x = f.relu(self.conv1(inputs))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        # Flatten
        x = x.view(x.size()[0], -1)
        # Linear layers
        x = f.relu(self.fc1(x))
        return self.fc2(x)

    def backward(self, target, value):
        loss = self.loss(target, value, reduction='mean').to(self.device)
        loss.backward()
        self.optimizer.step()

class LinearQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, learning_rate, leps, momentum, checkpoint_file):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.checkpoint_file = checkpoint_file

        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4,bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)

        flattened_shape = self.calculate_flattened_shape(self.input_shape)

        self.fc1 = nn.Linear(flattened_shape, 512)
        self.fc2 = nn.Linear(512, output_shape)

        self.loss = f.huber_loss 
        #self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate, eps=leps, momentum=momentum)#eps=0.01, momentum=0.95)
        self.device = self.get_device()
        self.to(self.device)

    @staticmethod
    def get_device():
        device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_name)
        logging.info(f'Using device: {device}')
        return device

    def calculate_flattened_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    def save_checkpoint(self):
        logging.info('Saving checkpoint')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        logging.info('Loading checkpoint')
        print(f'Checkout file: {self.checkpoint_file}')
        self.load_state_dict(torch.load(self.checkpoint_file))

    def to_tensor(self, inputs):
        return torch.tensor(inputs).to(self.device)

    def forward(self, inputs):
        # Convolutions
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        # Flatten
        x = x.view(x.size()[0], -1)
        # Linear layers
        x = self.fc1(x)
        return self.fc2(x)

    def backward(self, target, value):
        loss = self.loss(target, value, reduction='mean').to(self.device)
        loss.backward()
        self.optimizer.step()