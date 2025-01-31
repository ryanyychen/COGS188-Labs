import pygame
from pygame.surface import Surface
import sys
from typing import List, Tuple

# define constants
BOARD_SIZE = 4
SQUARE_SIZE = 75  # This sets the size of the chessboard squares
WINDOW_SIZE = BOARD_SIZE * SQUARE_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)


def draw_board(screen: Surface, board_size: int = BOARD_SIZE, square_size: int = SQUARE_SIZE):
    """
    Draw the chessboard.

    Args:
        screen: Surface: pygame screen
        board_size (int, optional): board_size in number of box. Defaults to BOARD_SIZE.
        square_size (int, optional): square_size in number of box. Defaults to SQUARE_SIZE.
    """
    for row in range(board_size):
        for col in range(board_size):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, color, (col * square_size, row * square_size, square_size, square_size))


def place_queens(screen: Surface, queen_positions: List[Tuple[int, int]]):
    """
    Place queens on the board based on the given positions.

    Args:
        queen_positions: List of tuples, each tuple is (row, col) for a queens position
    """
    queen_image = pygame.image.load("queen.png")
    queen_image = pygame.transform.scale(queen_image, (SQUARE_SIZE, SQUARE_SIZE))
    for pos in queen_positions:
        screen.blit(queen_image, (pos[1] * SQUARE_SIZE, pos[0] * SQUARE_SIZE))


def is_safe_row_diag(board, row, col) -> bool:
    """
    Check if it's safe to place a queen at board[row][col]
    You should check for conflicts along the row, and both upper and lower diagonals.
    note that you don't need to check the column.

    Think about why you don't need to check the column in combination with the backtracking algorithm.

    Args:
        board: 2D list of int
        row: int
        col: int
    return: bool, True if it's safe to place a queen, False otherwise
    """

    # Check row and col against bounds
    if row < 0 or row >= len(board) or col < 0 or col >= len(board):
        return False

    # Check row by checking every column in row
    for c in range(len(board)):
        try:
            if board[row][c] == 1:
                return False
        except:
            print(f'row: {row}, col: {col}, c: {c}')
            print(f'board size: {len(board)}')
            return False
        
    # Check upper diagonals
    for dec in range(1, row + 1):
        # We know r = row - dec cannot be out of bounds since dec is based on row
        r = row - dec
        leftc = col - dec
        rightc = col + dec

        # Check column still in bounds
        if (leftc >= 0):
            if board[r][leftc] == 1:
                return False
            
        # Check column still in bounds
        if (rightc < len(board)):
            if board[r][rightc] == 1:
                return False
    
    # Check lower diagonals
    for inc in range(1, len(board) - row):
        # We know r = row + inc cannot be out of bounds since inc is based on row
        r = row + inc
        leftc = col - inc
        rightc = col + inc

        # Check column still in bounds
        if (leftc >= 0):
            if board[r][leftc] == 1:
                return False
        
        # Check column still in bounds
        if (rightc < len(board)):
            if board[r][rightc] == 1:
                return False

    return True


def solve_8_queens(board: List[List[int]], col: int) -> bool:
    """
    Solve the 8 queens problem using backtracking.
    The function will edit the board in place, meaning the board will
    be modified to show the solution.

    Args:
        board (List[List[int]]): 2D list of int, representing the chessboard
        col (int): int, the column to place the queen

    Returns:
        bool: whether the solution exists
    """
    # Base case: all queens placed
    if col == len(board):
        return True
    
    # Recursive step:
    for row in range(len(board)):
        # Attempt to place queen in this column by checking every row
        if is_safe_row_diag(board, row, col):
            board[row][col] = 1
            # Attempt to place queen in next column
            if solve_8_queens(board, col + 1):
                return True
            board[row][col] = 0

    return False


def update_board():
    """
    creates the board and returns the list of valid queens positions
    """
    board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    if not solve_8_queens(board, 0):
        return []  # Return empty if no solution exists

    # If solution exists, prepare the list of queen positions
    queen_positions = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 1:
                queen_positions.append((i, j))

    return queen_positions


def main():

    # setup the pygame window
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("8 Queens Puzzle")

    # main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # draw the board and queens
        draw_board(screen)
        queen_positions = update_board()  # Call the student's solve function
        place_queens(screen, queen_positions)
        pygame.display.update()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
