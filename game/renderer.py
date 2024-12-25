"""
Module for rendering the board.
"""
import pygame
import pygame.gfxdraw
from game.game_config import *

class GameRenderer:
    """
    Renders the board using pygame.
    """
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    def draw_board(self, p1_board, p2_board):
        """
        Draw the board in pygame window based on given states
        """
        for c in range(COLUMNS):
            for r in range(ROWS):
                pygame.gfxdraw.box(
                    self.screen,
                    (c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    FRAME_COLOR
                )
                pygame.gfxdraw.filled_circle(
                    self.screen,
                    int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                    int(r * SQUARE_SIZE + SQUARE_SIZE / 2),
                    RADIUS,
                    BOARD_COLOR
                )

        for c in range(COLUMNS):
            for r in range(ROWS):
                if p1_board[r][c] == 1:
                    pygame.gfxdraw.filled_circle(
                        self.screen,
                        int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                        WINDOW_HEIGHT - int(r * SQUARE_SIZE + SQUARE_SIZE / 2),
                        RADIUS,
                        PLAYER1_COLOR
                    )
                elif p2_board[r][c] == 1:
                    pygame.gfxdraw.filled_circle(
                        self.screen,
                        int(c * SQUARE_SIZE + SQUARE_SIZE / 2),
                        WINDOW_HEIGHT - int(r * SQUARE_SIZE + SQUARE_SIZE / 2),
                        RADIUS,
                        PLAYER2_COLOR
                    )
        pygame.display.update()
