import math
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from engine.ModelConfig import *

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN_AI():
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.gamma = GAMMA
        self.epsilon_start = EPS_START
        self.epsilon_end = EPS_END
        self.epsilon_decay = EPS_DECAY

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        self.steps_processed = 0

    # Select an action based on the 
    def select_action(self, state):
        random_sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_processed / self.epsilon_decay)
        self.steps_processed += 1
        
        if random_sample > eps_threshold: # Model estimation
            with torch.no_grad():
                # Add batch dimension if needed
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:  # Random Exploration
            return torch.tensor([[random.randrange(self.action_size)]], 
                            device=self.device, dtype=torch.long)
                
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(self.device).bool()

        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + (self.gamma * next_q_values)

        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))
    
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def store_experience(self, state, action, reward, next_state, done):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = action.to(self.device)
        
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        self.memory.push(state, action, reward, next_state, done)
    
    def print_model_status(self):
        print("Steps processed: " + str(self.steps_processed))
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_processed / self.epsilon_decay)
        print("Eps_threshold: " + str(eps_threshold))
    
    def save_model(self, filepath="dqn_model.pth"):
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'steps_processed': self.steps_processed,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay
        }

        torch.save(checkpoint, filepath)
        print("Saved model.")

    def load_model(self, filepath="dqn_model.pth"):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.steps_processed = checkpoint.get('steps_processed', 0)  
        self.epsilon_start = checkpoint.get('epsilon_start', self.epsilon_start)
        self.epsilon_end = checkpoint.get('epsilon_end', self.epsilon_end)
        self.epsilon_decay = checkpoint.get('epsilon_decay', self.epsilon_decay)
        print("Loaded model.")
