import pygame
import sys
from game.board import GameBoard
from game.renderer import GameRenderer
from game.game_config import *

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
            self.handle_events()
            self.renderer.draw_board(self.board.grid)

            if self.game_over:
                pygame.time.wait(3000)
