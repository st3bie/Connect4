import numpy as np

from game.TrainingConfig import *
from game.GameConfig import ROWS, COLUMNS

class Connect4Env:
    def __init__(self):
        self.rows = ROWS
        self.cols = COLUMNS

        self.top_mask = [1 << (col * 7 + self.rows - 1) for col in range(self.cols)]
        self.bottom_mask = [1 << (col * 7) for col in range(self.cols)]
        self.full_mask = sum(self.top_mask)
        self.directions = [1, 7, 6, 8]  # Win checking shift values
        self.reset()

    def reset(self):
        self.p1_pos = 0
        self.p2_pos = 0
        return self.get_state(1)
    
    def drop(self, player, col):
        reward = 0
        done = False

        if not self.is_valid_col(col):
            reward = INVALID_REWARD
            return self.get_state(player), reward, done
        
        pos_mask = self.get_next_row_bit(col)

        if player == 1:
            self.p1_pos |= pos_mask
        else:
            self.p2_pos |= pos_mask
        
        if self.check_win(player):
            reward = WIN_REWARD
            done = True
        elif self.check_tie():
            reward = TIE_REWARD
            done = True

        return self.get_state(player), reward, done

    def is_valid_col(self, col):
        if col < 0 or col >= self.cols:
            return False
        return (self.p1_pos | self.p2_pos) & self.top_mask[col] == 0

    def get_next_row_bit(self, col):
        for row in range(self.rows):
            bit = self.bottom_mask[col] << row
            if (self.p1_pos | self.p2_pos) & bit == 0:
                return bit
        return -1

    def check_win(self, player):
        if player == 1:
            board = self.p1_pos
        else:
            board = self.p2_pos
        
        for direction in self.directions:
            m = board & (board >> direction)
            if (m & (m >> (2 * direction))) != 0:
                return True
        
        return False
    
    def check_tie(self):
        return ((self.p1_pos | self.p2_pos) & self.full_mask) == self.full_mask
        
    def get_state(self, player):
        p1 = np.zeros((self.rows, self.cols), np.float32)
        p2 = np.zeros((self.rows, self.cols), np.float32)

        for col in range(self.cols):
            for row in range(self.rows):
                bit_pos = 1 << (col * 7 + row)
                if self.p1_pos & bit_pos:
                    p1[row][col] = 1
                
                if self.p2_pos & bit_pos:
                    p2[row][col] = 1
        
        if player == 1:
            return np.stack([p1, p2], axis=0)
        else:
            return np.stack([p2, p1], axis=0)

    def render_terminal(self):
        print("\nCurrent Board:")
        for row in reversed(range(self.rows)):
            print("|", end="")  # Start of the row
            for col in range(self.cols):
                bit = 1 << (col * 7 + row)  # Calculate the bitmask for the current cell
                if self.p1_pos & bit:
                    print(" X |", end="")  # Player 1's piece
                elif self.p2_pos & bit:
                    print(" O |", end="")  # Player 2's piece
                else:
                    print("   |", end="")  # Empty cell
            print()  # Newline after each row
        # Print the column numbers below the board
        print("  " + "   ".join(str(c) for c in range(self.cols)))
        print()
