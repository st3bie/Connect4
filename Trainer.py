import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm

from engine.Model import DQNAgent
from game.Connect4Env import Connect4Env

class Trainer:
    def __init__(self, env, agent, device="cpu"):
        self.env = env
        self.agent = agent
        self.device = device

    def train(self):
        for episode in tqdm(range(1, self.num_episodes + 1)):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done = self.env.drop(self.agent.current_player, action)
                total_reward += reward

                self.agent.store_transition(state, action, reward, next_state, done)

                state = next_state

                self.agent.optimize_model()

                if done:
                    break

            if episode % self.agent.target_update == 0:
                self.agent.update_target_network()
        print("Training Complete!")

    def self_train(self, num_agents, num_episodes_per_agent):
        for _ in tqdm(range(num_agents)):
            adv_agent = DQNAgent(device=self.device)
            total_agent_win = 0
            total_agent_reward = 0
            total_adv_agent_win = 0
            total_adv_agent_reward = 0

            for episode in tqdm(range(1, num_episodes_per_agent + 1)):
                state = self.env.reset()
                done = False
                turn = 1

                while not done:
                    if turn % 2 == 1:
                        current_player = 1
                        state = self.env.get_state(current_player)
                        action = self.agent.select_action(state)

                        last_state1 = state
                        last_action1 = action

                        next_state, reward, done = self.env.drop(current_player, action)

                        self.agent.store_transition(state, action, reward, next_state, float(done))

                        if done:
                            if self.env.check_win(1):
                                agent_reward = 10.0
                                adv_agent_reward = -10.0
                                total_agent_win += 1

                                if last_state2 is not None and last_action2 is not None:
                                    adv_agent.store_transition(last_state2, last_action2, adv_agent_reward, self.env.get_state(-1), float(done))

                            elif reward == -20.0:

                                agent_reward = -20.0
                                adv_agent_reward = 0.0

                            else:
                                agent_reward = 0.0
                                adv_agent_reward = 0.0

                                if last_state2 is not None and last_action2 is not None:
                                    adv_agent.store_transition(last_state2, last_action2, adv_agent_reward, self.env.get_state(-1), float(done))
                        else:
                            agent_reward = 0.0
                            adv_agent_reward = 0.0
                        
                    else:
                        current_player = -1
                        state = self.env.get_state(current_player)
                        action = adv_agent.select_action(state)

                        last_state2 = state
                        last_action2 = action

                        next_state, reward, done = self.env.drop(current_player, action)
                        adv_agent.store_transition(state, action, reward, next_state, float(done))
                        
                        if done:
                            if self.env.check_win(-1):
                                adv_agent_reward = 10.0
                                agent_reward = -10.0
                                total_adv_agent_win += 1

                                if last_state1 is not None and last_action1 is not None:
                                    self.agent.store_transition(last_state1, last_action1, agent_reward, self.env.get_state(1), float(done))
                            
                            elif reward == -20.0:
                                adv_agent_reward = -20.0
                                agent_reward = 0.0
                            else:
                                adv_agent_reward = 0.0
                                agent_reward = 0.0

                                if last_state1 is not None and last_action1 is not None:
                                    self.agent.store_transition(last_state1, last_action1, agent_reward, self.env.get_state(1), float(done))
                        else:
                            adv_agent_reward = 0.0
                            agent_reward = 0.0
                    
                    total_agent_reward += agent_reward
                    total_adv_agent_reward += adv_agent_reward
                    turn += 1

                self.agent.optimize()
                adv_agent.optimize()

                if (episode + 1)% 5000 == 0:
                    tqdm.write("Model1 win rate: " + str(round(total_agent_win/5000, 3)))
                    tqdm.write("Model1 avg reward: " + str(round(total_agent_reward/5000, 3)))
                    tqdm.write("Model2 win rate: " + str(round(total_adv_agent_win/5000, 3)))
                    tqdm.write("Model2 avg reward: " + str(round(total_adv_agent_reward/5000, 3)))
                    total_agent_win = 0
                    total_agent_reward = 0
                    total_adv_agent_win = 0
                    total_adv_agent_reward = 0

                    self.agent.save_model("model1.pth")
                    tqdm.write("Saved model")
    
    def human_vs_ai(self, human_player=1):
        ai_player = -human_player

        state = self.env.reset()
        self.env.render_terminal()
        done = False
        turn = 1

        while not done:
            if turn%2 == 1:

                valid_move = False
                while not valid_move:
                    try:
                        action = int(input(f"Your turn! Choose a column (0-{self.env.cols - 1}): "))
                        if self.env.is_valid_col(action):
                            valid_move = True
                        else:
                            print("Invalid move! Column is full or out of range. Try again.")
                    except ValueError:
                        print("Invalid input! Please enter an integer.")
                
                # Execute the human's move
                next_state, reward, done = self.env.drop(human_player, action)
                self.env.render_terminal()

                if done:
                    if reward == 10.0:
                        print("Congratulations! You win!")
                    elif reward == -20.0:
                        print("Invalid move! You lose!")
                    else:
                        print("It's a tie!")
                    break

            else:
                print("AI's turn...")
                action = self.agent.select_action(state)
                print(f"AI chooses column {action}")
                next_state, reward, done = self.env.drop(ai_player, action)
                self.env.render_terminal()

                if done:
                    if reward == 10.0:
                        print("AI wins! Better luck next time.")
                    elif reward == -20.0:
                        print("AI made an invalid move! You win!")
                    else:
                        print("It's a tie!")
                    break

            state = next_state
            turn += 1