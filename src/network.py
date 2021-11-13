import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, pool=True):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def fanin_initialize(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, input_channels, output_actions):
        super(Actor, self).__init__()
        self.conv1 = ConvBlock(input_channels, 64, normalize=False)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 512)
        self.conv4 = ConvBlock(512, 512)
        self.conv5 = ConvBlock(512, 512)
        self.pool = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(2049, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, output_actions)
        self.tanh = nn.Tanh()
        self.init_w = 3e-4
        self.initialize_weights(self.init_w)

    def initialize_weights(self, init_w):
        self.fc1.weight.data = fanin_initialize(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_initialize(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, obstacle_area):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, obstacle_area], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self, input_channels, action_dim=1):
        super(Critic, self).__init__()
        self.conv1 = ConvBlock(input_channels, 64, normalize=False)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 512)
        self.conv4 = ConvBlock(512, 512)
        self.conv5 = ConvBlock(512, 512)
        self.pool = nn.MaxPool2d(3)

        self.fc1 = nn.Linear(2049 + action_dim, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.init_w = 3e-4
        self.initialize_weights(self.init_w)

    def initialize_weights(self, init_w):
        self.fc1.weight.data = fanin_initialize(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_initialize(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        
    def forward(self, x, obstacle_area, in_action):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, obstacle_area, in_action], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__=="__main__":

    x = torch.randn(2, 3, 200, 200)
    x_obstacle = torch.randn(2, 1)
    x_action = torch.randn(2, 1)

    actor = Actor(3, 1)
    out_action = actor(x, x_obstacle)
    critic = Critic(3)
    out = critic(x, x_obstacle, out_action)
    print(out.shape)