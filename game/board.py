import numpy as np
from game.GameConfig import ROWS, COLUMNS

class GameBoard:
    def __init__(self):
        self.grid = np.zeros((ROWS, COLUMNS), dtype=int)
    
    def drop_piece(self, row: int, col: int, piece: int):
        self.grid[row][col] = piece

    def check_valid_position(self, row: int, col: int) -> bool:
        if (row is None or row >= ROWS or row < 0):
            return False

        if (col is None or col >= COLUMNS or col < 0):
            return False
        
        return self.grid[row][col] == 0
    
    def get_next_row(self, col: int) -> int:
        for r in range(ROWS):
            if self.grid[r][col] == 0:
                return r
    
    def check_winner(self, piece: int) -> bool:
        for c in range(COLUMNS - 3):
            for r in range(ROWS):
                if all(self.grid[r][c+i] == piece for i in range(4)):
                    return True

        for c in range(COLUMNS):
            for r in range(ROWS - 3):
                if all(self.grid[r+i][c] == piece for i in range(4)):
                    return True

        for c in range(COLUMNS - 3):
            for r in range(ROWS - 3):
                if all(self.grid[r+i][c+i] == piece for i in range(4)):
                    return True

        for c in range(COLUMNS - 3):
            for r in range(3, ROWS):
                if all(self.grid[r-i][c+i] == piece for i in range(4)):
                    return True

        return False

    def is_board_full(self) -> bool:
        for r in range(ROWS):
            for c in range(COLUMNS):
                if self.grid[r][c] == 0:
                    return False
                
        return True
