import pygame
from game.GameConfig import *

class GameRenderer:
    def __init__(self, screen):
        self.screen = screen

    def draw_board(self, board_grid):
        for c in range(COLUMNS):
            for r in range(ROWS):
                pygame.draw.rect(
                    self.screen,
                    FRAME_COLOR,
                    (c * SQUARE_SIZE, r * SQUARE_SIZE + SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                )
                pygame.draw.circle(
                    self.screen,
                    BOARD_COLOR,
                    (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), int(r * SQUARE_SIZE + SQUARE_SIZE + SQUARE_SIZE / 2)),
                    RADIUS,
                )

        for c in range(COLUMNS):
            for r in range(ROWS):
                if board_grid[r][c] == 1:
                    pygame.draw.circle(
                        self.screen,
                        PLAYER1_COLOR,
                        (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), WINDOW_HEIGHT - int(r * SQUARE_SIZE + SQUARE_SIZE / 2)),
                        RADIUS,
                    )
                elif board_grid[r][c] == 2:
                    pygame.draw.circle(
                        self.screen,
                        PLAYER2_COLOR,
                        (int(c * SQUARE_SIZE + SQUARE_SIZE / 2), WINDOW_HEIGHT - int(r * SQUARE_SIZE + SQUARE_SIZE / 2)),
                        RADIUS,
                    )

        pygame.display.update()

    def draw_hover_piece(self, col: int, turn: int):
        if turn%2 == 1:
            hover_color = PLAYER1_COLOR
        else:
            hover_color = PLAYER2_COLOR

        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.circle(
            surface,
            hover_color,
            (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
            RADIUS,
        )
        self.screen.blit(surface, (col * SQUARE_SIZE, 0))