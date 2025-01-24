import unittest
from src.eight_queens import is_safe_row_diag, solve_8_queens


class TestEightQueensImplementation(unittest.TestCase):
    """Test graph implementation"""

    def setUp(self):
        self.board = [[0] * 8 for _ in range(8)]
        self.small_board = [[0] * 4 for _ in range(4)]
        self.no_solution_board = [[0] * 2 for _ in range(2)]

    def test_is_safe_simple(self):
        self.board[0][0] = 1
        self.assertFalse(is_safe_row_diag(self.board, 0, 1), "0, 1 should be unsafe because there is a queen at 0, 0")
        self.assertTrue(
            is_safe_row_diag(self.board, 1, 0),
            "1, 0 should be safe because there is no queen in the same row, column or diagonal",
        )
        self.assertFalse(
            is_safe_row_diag(self.board, 2, 2),
            "2, 2 should be unsafe because there is a queen at 0, 0, which is on the same diagonal",
        )

    def test_is_safe_complex(self):
        self.board[0][0] = 1
        self.board[1][2] = 1
        self.board[2][4] = 1
        self.assertFalse(
            is_safe_row_diag(self.board, 3, 3),
            "3, 3 should be unsafe because there is a queen at 0, 0, which is on the same diagonal",
        )
        self.assertFalse(
            is_safe_row_diag(self.board, 3, 4),
            "3, 4 should be unsafe because there is a queen at 2, 4, which is on the same column",
        )
        self.assertTrue(
            is_safe_row_diag(self.board, 3, 2),
            "3, 2 should be safe, even though there is a queen at 1, 2, which is on the same column, we don't have any queens on the same row or diagonal",
        )
        self.assertTrue(
            is_safe_row_diag(self.board, 3, 0),
            "3, 0 should be safe because there is no queen in the same row, column or diagonal",
        )

    def test_solve_8_queens_simple(self):
        self.assertTrue(solve_8_queens(self.board, 0), "Should be able to solve the 8 queens problem")
        self.assertEqual(sum(row.count(1) for row in self.board), 8, "Should have 8 queens on the board")
        
    def test_solve_8_queens_no_solution(self):
        self.assertFalse(solve_8_queens(self.no_solution_board, 0), "Should not be able to solve the 8 queens problem")
        self.assertEqual(sum(row.count(1) for row in self.no_solution_board), 0, "Should have no queens on the board")
        
    def test_solve_8_queens_complex(self):
        self.assertTrue(solve_8_queens(self.small_board, 0), "Should be able to solve the 4 queens problem")
        self.assertEqual(sum(row.count(1) for row in self.small_board), 4, "Should have 4 queens on the board")
        self.board = [[0] * 8 for _ in range(8)]
        self.board[0][0] = 1
        self.board[2][1] = 1
        self.board[4][2] = 1
        self.assertFalse(solve_8_queens(self.board, 0), "Should not be able to solve the 8 queens problem from this position")
        self.board = [[0] * 8 for _ in range(8)]
        self.board[0][0] = 1
        self.board[2][1] = 1
        self.board[4][2] = 1
        self.assertTrue(solve_8_queens(self.board, 4), "Should be able to solve the 8 queens problem from this position")
        
