import pygame
import sys
import torch
import os

from game.Board import GameBoard
from game.Renderer import GameRenderer
from game.GameConfig import *
from engine.DQN import DQN_AI

class Connect4:
    def __init__(self):
        # Pygame Initialization
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.renderer = GameRenderer(self.screen)

        # Board Initialization
        self.board = GameBoard()

        # Field Initialization
        self.game_over: bool = False
        self.turn: int = 0

        self.ai = DQN_AI()

        # Load pre-trained model if available
        if os.path.exists("dqn_model.pth"):
            self.ai.load_model("dqn_model.pth")
            print("Model loaded successfully.")
        else:
            print("Model not found.")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                posx = event.pos[0]
                col = posx // SQUARE_SIZE
                if col is None:
                    continue

                row = self.board.get_next_row(col)

                print(row, col)

                if self.board.check_valid_position(row, col):
                    piece = self.turn + 1
                    self.board.drop_piece(row, col, piece)

                    print(self.board.grid)

                    if self.board.check_winner(piece):
                        print(f"Player {piece} wins!")
                        self.game_over = True

                    self.turn = (self.turn + 1) % 2

    def run(self):
        while not self.game_over:
            if self.turn == 0:
                self.handle_events()
                self.renderer.draw_board(self.board.grid)
            else:
                self.ai_move()
                self.turn = (self.turn + 1) % 2

            self.renderer.draw_board(self.board.grid)
            pygame.display.update()
            
        self.ai.save_model("dqn_model.pth")
        pygame.time.wait(3000)
    
    def ai_move(self):
        state = self.get_state()

        valid_move = False
        while not valid_move:
            action = self.ai.select_action(state)
            col = action.item()
            row = self.board.get_next_row(col)
            
            if row is not None and self.board.check_valid_position(row, col):
                valid_move = True
            else:
                # Penalize invalid moves
                reward = -5.0
                next_state = state
                self.ai.store_experience(state, action, reward, next_state, True)
                self.ai.train()

        self.board.drop_piece(row, col, self.turn%2 + 1)
        reward = self.get_reward(self.turn%2 + 1)

        if self.board.check_winner(self.turn%2 + 1):
            self.game_over = True
        elif self.board.is_board_full():
            self.game_over = True
            reward = 0.0
        
        next_state = state
        done = self.game_over
        self.ai.store_experience(state, action, reward, next_state, done)
        self.ai.train()
    
    def train(self, episodes: int, visualize: bool):
        target_update_interval = 1000

        for episode in range(episodes):
            self.board = GameBoard()
            self.game_over = False
            self.turn = 0

            #pygame.time.wait(2000)

            while not self.game_over and not self.board.is_board_full():
                state = self.get_state()
                
                valid_move = False
                while not valid_move:
                    action = self.ai.select_action(state)
                    col = action.item()
                    row = self.board.get_next_row(col)
                    
                    if row is not None and self.board.check_valid_position(row, col):
                        valid_move = True
                    else:
                        # Penalize invalid moves
                        reward = -5.0
                        next_state = state
                        self.ai.store_experience(state, action, reward, next_state, True)
                        self.ai.train()

                self.board.drop_piece(row, col, self.turn%2 + 1)
                
                reward = self.get_reward(self.turn%2 + 1)
                next_state = self.get_state()
                
                if self.board.check_winner(self.turn%2 + 1):
                    self.game_over = True
                elif self.board.is_board_full():
                    self.game_over = True
                    reward = 0.0  # Draw

                done = self.game_over
                self.ai.store_experience(state, action, reward, next_state, done)
                self.ai.train()

                self.turn += 1

                if done or episode % target_update_interval == 0:
                    self.ai.update_target_network()

                if visualize:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            sys.exit()
                    self.renderer.draw_board(self.board.grid)
                    pygame.display.update()

            print(f"Episode {episode + 1}/{episodes} completed.")
            self.ai.print_model_status()

            if episode % 100 == 0:
                print(f"Saving model at episode {episode + 1}")
                self.ai.save_model("dqn_model.pth")
    
        self.ai.save_model("dqn_model.pth")
    
    def get_state(self):
        return torch.FloatTensor(self.board.grid.flatten()).to(self.ai.device)
    
    def get_reward(self, piece):
        if self.board.check_winner(piece):
            return 1.0
        elif self.game_over:
            return -1.0
        return 0.0
