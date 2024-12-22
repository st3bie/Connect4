import numpy as np

from game.renderer import GameRenderer
from game.GameConfig import *
from engine.ModelConfig import *

class Connect4:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=np.float32)
        self.current_player = 1
        return self.get_state(1)

    def drop(self, player, action):
        if not self.is_valid_action(action):
            return self.get_state(player), -20, True

        row = self.get_next_open_row(action)
        self.board[row, action] = player

        if self.check_win(player):

            return self.get_state(player), 10, True
        elif np.all(self.board != 0):
            return self.get_state(player), 0, True

        self.current_player *= -1
        return self.get_state(player), 0, False

    def is_valid_action(self, col):
        if col < 0 or col >= self.cols: return False
        return self.board[0, col] == 0

    def get_next_open_row(self, col):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        return None

    def check_win(self, player):
        for r in range(self.rows):
            for c in range(self.cols - (3)):
                if np.all(self.board[r, c:c+4] == player):
                    return True

        for c in range(self.cols):
            for r in range(self.rows - (3)):
                if np.all(self.board[r:r+4, c] == player):
                    return True

        for r in range(self.rows - (3)):
            for c in range(self.cols - (3)):
                if all(self.board[r+i, c+i] == player for i in range(4)):
                    return True
                
        for r in range(3, self.rows):
            for c in range(self.cols - (3)):
                if all(self.board[r-i, c+i] == player for i in range(4)):
                    return True
        return False

    def get_state(self, player):
        if player == 1:
            p1 = (self.board == 1).astype(np.float32)
            p2  = (self.board == -1).astype(np.float32)
        else:
            flipped = self.board * -1
            p2 = (flipped == 1).astype(np.float32)
            p1  = (flipped == -1).astype(np.float32)

        return np.stack([p2, p1], axis=0)
    
    def render_board(self):
        for r in range(self.rows):
            print("|", end="")
            for c in range(self.cols):
                if self.board[r, c] == 1.0:
                    print(" X |", end="")
                elif self.board[r, c] == -1.0:
                    print(" O |", end="")
                else:
                    print("   |", end="")
            print()
        print("  " + "   ".join(str(c) for c in range(self.cols)))
        print()
