# L3

**Deadline**: Monday, Jan 27, 2025 at 11:59pm on Gradescope

## Installations

I strongly recommend you work on this assignment on your local machine, rather than on DataHub. This is because we use Pygame, which requires a graphical interface to run. The terminal on DataHub does not let you directly open a window for Pygame and it may not work on Jupyter notebooks either. If there are any technical issues with running the code on your local machine, please reach out to us on EdStem as soon as possible and we can help you troubleshoot.

Before running the script, ensure you have the following modules installed on your system:

* Pygame
* Numpy
* Shapely
* Scipy
* Tqdm
* Argparse

You can install these modules using pip:

```bash
pip install pygame numpy shapely scipy tqdm argparse
```

## Part 1: Constrained-Oriented Games: Solving the 8-Queens Problem (1 point)

In this task, you will develop a solution to the classic 8 Queens puzzle using Python and Pygame. The challenge is to place eight queens on a standard chessboard so that no two queens threaten each other. This means that no two queens can share the same row, column, or diagonal.

### Code Organization

**Main Functions:**
* `draw_board()`: Draws an 8x8 chessboard on the window.
* `place_queens(queen_positions)`: Places images of queens on the board based on their positions.
* `is_safe(board, row, col)`: Determines if it is safe to place a queen at the given position.
* `solve_8_queens(board, col)`: Recursively attempts to place queens on the board using backtracking.
* `update_board()`: Initializes the board and finds a valid arrangement of queens.
* `main()`: Contains the main game loop which handles events and updates the display.

### Assignment Tasks

You need to complete several parts of the given code to make the application work:

1. `is_safe` Function: 
  * Implement the logic to check if placing a queen at (row, col) is safe. You should check for conflicts along the row, and both upper and lower diagonals.
2. `solve_8_queens` Function:
  * Complete the loop to iterate through each row in the current column.
  * Implement the condition to check if placing a queen is safe.
  * Fill in the backtrack step which removes a queen if placing it leads to no solution.

### Expected Outcomes

To test your implementation, run the following command:

```bash
python eight_queens.py
```

Upon successfully running the program, the chessboard will display with eight queens placed such that no queen is under threat from another. This will be visualized in real-time as the program computes the positions. If no solution is found, the board will remain empty. It's pretty easy to see if the solution is correct by observing the board and checking that no two queens are in the same row, column, or diagonal.

For further exploration, you can try modifying the code to solve the N-Queens problem for different board sizes. The 8-Queens problem is a classic example, but the solution can be generalized to any size board.

### How to Run Unit Test

After you finish, you can run the unit test to make sure that your implementation is correct. You can run the unit test via the following command on the `/L3` directory

```py
python -m unittest discover -s tests
```

If you see

```bash
(base) <your-username> L3 % python -m unittest discover -s tests
.....
----------------------------------------------------------------------
Ran 5 tests in 0.000s
```

this indicates that you have passed all the tests.


### Submission Guidelines

Download, and submit the `eight_queens.py` to Gradescope, you should be able to see the autograder result within 2 minutes.