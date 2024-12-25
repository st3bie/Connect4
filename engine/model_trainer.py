"""
This module is for training the model.
"""
import pygame
from pygame import QUIT, MOUSEBUTTONDOWN
from tqdm import tqdm

from engine.model import DQNAgent
from game.renderer import GameRenderer
from game.game_config import SQUARE_SIZE

class Trainer:
    """
    Class for training the model, including self-play and human input training.
    """
    def __init__(self, env, agent, device="cpu"):
        self.env = env
        self.agent = agent
        self.device = device

    def self_play(self, num_agents, num_episodes_per_agent):
        """
        This method allows for self-play which initializes an adversary model
        with random presets to train against the main model.
        """
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
                                    adv_agent.store_transition(last_state2,
                                                               last_action2,
                                                               adv_agent_reward,
                                                               self.env.get_state(-1),
                                                               float(done))

                            elif reward == -20.0:
                                agent_reward = -20.0
                                adv_agent_reward = 0.0

                            else:
                                agent_reward = 0.0
                                adv_agent_reward = 0.0

                                if last_state2 is not None and last_action2 is not None:
                                    adv_agent.store_transition(last_state2,
                                                               last_action2,
                                                               adv_agent_reward,
                                                               self.env.get_state(-1),
                                                               float(done))
                        else:
                            agent_reward = 0.0
                            adv_agent_reward = 0.0

                    else:
                        current_player = 2
                        state = self.env.get_state(current_player)
                        action = adv_agent.select_action(state)

                        last_state2 = state
                        last_action2 = action

                        next_state, reward, done = self.env.drop(current_player, action)
                        adv_agent.store_transition(state, action, reward, next_state, float(done))

                        if done:
                            if self.env.check_win(current_player):
                                adv_agent_reward = 10.0
                                agent_reward = -10.0
                                total_adv_agent_win += 1

                                if last_state1 is not None and last_action1 is not None:
                                    self.agent.store_transition(last_state1,
                                                                last_action1,
                                                                agent_reward,
                                                                self.env.get_state(1),
                                                                float(done))

                            elif reward == -20.0:
                                adv_agent_reward = -20.0
                                agent_reward = 0.0
                            else:
                                adv_agent_reward = 0.0
                                agent_reward = 0.0

                                if last_state1 is not None and last_action1 is not None:
                                    self.agent.store_transition(last_state1,
                                                                last_action1,
                                                                agent_reward,
                                                                self.env.get_state(1),
                                                                float(done))
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

        self.agent.save_model("model1.pth")
        tqdm.write("Training Complete")

    def human_vs_ai(self, player_turn=1):
        """
        Train the model against a human player using a graphical interface.
        """
        if player_turn == 1:
            ai_turn = 0
        else:
            ai_turn = 1

        renderer = GameRenderer()

        state = self.env.reset()
        done = False
        turn = 1
        last_state_ai = None
        last_action_ai = None

        p1_board, p2_board = self.env.get_board()
        renderer.draw_board(p1_board, p2_board)

        while not done:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return
                if turn % 2 == player_turn and event.type == MOUSEBUTTONDOWN:
                    x = event.pos[0]
                    column = x // SQUARE_SIZE

                    if self.env.is_valid_col(column):
                        next_state, reward, done = self.env.drop(player_turn, column)

                        p1_board, p2_board = self.env.get_board()
                        renderer.draw_board(p1_board, p2_board)

                        state = next_state
                        if done:
                            break
                        turn += 1

                elif turn % 2 == 0:
                    last_state_ai = state
                    action = self.agent.select_action(state)
                    last_action_ai = action

                    next_state, reward, done = self.env.drop(ai_turn, action)

                    p1_board, p2_board = self.env.get_board()
                    renderer.draw_board(p1_board, p2_board)

                    self.agent.store_transition(
                        last_state_ai,
                        last_action_ai,
                        reward,
                        next_state,
                        float(done),
                    )

                    state = next_state
                    if done:
                        break
                    turn += 1
            pygame.time.wait(100)

        if last_state_ai is not None and last_action_ai is not None:
            if self.env.check_win(player_turn):
                reward = -10
                print("Player Wins")
            if self.env.check_win(ai_turn):
                reward = 10
                print("AI Wins")
            else:
                reward = 0

            self.agent.store_transition(
                last_state_ai,
                last_action_ai,
                reward,
                self.env.get_state(ai_turn),
                float(done),
            )
            self.agent.optimize()
            self.agent.save_model("model1.pth")
        pygame.time.wait(3000)
        pygame.quit()
