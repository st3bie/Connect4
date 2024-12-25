"""
This module implements a Clipped Double Deep Q-Learning model using PyTorch. 
Includes a convolutional DQN model, DQN agent, and replay memory buffer.
"""
from collections import deque
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from engine.model_config import *

class ConvDQN(nn.Module):
    """
    Deep Q-Network (DQN) with two convolutional layers and and two fully connected layers.
    """
    def __init__(self, action_size):
        super(ConvDQN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        """
        Defines the forward pass of the ConvDQN.

        Args:
            x (torch.Tensor): Input tensor of the current board state.
        Returns:
            torch.Tensor: Q-values for each action.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x


class DQNAgent:
    """
    Double Deep Q-Learning agent.
    """
    def __init__(self, device="cpu"):
        self.action_size = ACTION_SIZE
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.device = device

        self.q_net1 = ConvDQN(self.action_size).to(device)
        self.q_net2 = ConvDQN(self.action_size).to(device)

        self.optimizer1 = optim.Adam(self.q_net1.parameters(), lr=LR)
        self.optimizer2 = optim.Adam(self.q_net2.parameters(), lr=LR)

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        """
        Selects an action based on the q-network using epsilon-greedy exploration.
        """
        self.steps_done += 1

        epsilon = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1.0 * self.steps_done / EPS_DECAY)

        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state,
                                            dtype=torch.float32,
                                            device=self.device).unsqueeze(0)
                return self.q_net1(state_tensor).argmax(dim=1).item()

    def compute_loss(self):
        """
        Computes the loss for both Q-networks using the Bellman equation and sampled 
        transitions from the replay memory. Uses Clipped Double Q-learning for loss
        computation.
        """
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        with torch.no_grad():
            next_actions1 = self.q_net1(next_states).argmax(dim=1, keepdim=True)
            next_actions2 = self.q_net2(next_states).argmax(dim=1, keepdim=True)

            q1_next = self.q_net1(next_states).gather(1, next_actions1).squeeze(1)
            q2_next = self.q_net2(next_states).gather(1, next_actions2).squeeze(1)
            q_next = torch.min(q1_next, q2_next)

        q1_current = self.q_net1(states).gather(1, actions).squeeze(1)
        q2_current = self.q_net2(states).gather(1, actions).squeeze(1)

        target = rewards + self.gamma * q_next * (1 - dones)

        loss1 = F.mse_loss(q1_current, target)
        loss2 = F.mse_loss(q2_current, target)
        return loss1, loss2

    def optimize(self):
        """
        Optimization based on the computed loss for both Q-networks.
        """
        if len(self.memory) < self.batch_size:
            return

        loss1, loss2 = self.compute_loss()

        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition into the replay memory for future optimizations.
        """
        self.memory.push(state, action, reward, next_state, done)

    def save_model(self, filename="models.pth"):
        """
        Saves checkpoint in a file with provided filename.
        """
        checkpoint = {
            "q_net1": self.q_net1.state_dict(),
            "q_net2": self.q_net2.state_dict(),
            "optimizer1": self.optimizer1.state_dict(),
            "optimizer2": self.optimizer2.state_dict(),
            "steps_done": self.steps_done
        }
        torch.save(checkpoint, filename)

    def load_model(self, filename="models.pth"):
        """
        Loads checkpoint from the provided file.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_net1.load_state_dict(checkpoint["q_net1"])
        self.q_net2.load_state_dict(checkpoint["q_net2"])
        self.optimizer1.load_state_dict(checkpoint["optimizer1"])
        self.optimizer2.load_state_dict(checkpoint["optimizer2"])
        self.steps_done = checkpoint["steps_done"]

class ReplayMemory:
    """
    Replay memory is used to store previous transitions for sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Push a transition into the memory buffer.
        """
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the memory.
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        """
        Returns the current size of the memory.
        """
        return len(self.memory)
