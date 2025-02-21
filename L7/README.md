# Lab 7: Temporal Difference Learning (2 points)

**Deadline**: Tuesday, Feb 25, 2025 at 11:59pm on Gradescope

## Installations

Ensure you have the following modules installed on your system:

* Numpy
* Matplotlib
* Gymnasium
* imageio

You can install these modules using pip:

```bash
pip install numpy matplotlib gymnasium imageio
```

## Part 1: SARSA vs. Q-Learning in the Windy Cliff Environment [1 point]

In this part of the assignment, you will implement Q-learning and SARSA algorithms and apply them to the Windy Cliff World environment. You will also experiment with different hyperparameters and visualize the performance of the algorithms.

### Windy Cliff World Environment

The Windy Cliff World is a grid world environment where an agent must reach a goal while navigating through a grid with varying wind intensities that push the agent in specific directions. The environment has the following characteristics:
- Grid Size: 7 rows x 10 columns
- Start State: (3, 0)
- Goal State: (3, 9)
- Cliff: The cells (3, 1) to (3, 8) are cliff cells. Stepping into a cliff cell results in a high negative reward and resets the agent to the start state.
- Obstacles: There are obstacles at positions (2, 4), (4, 4), (2, 7), and (4, 7). Stepping into an obstacle cell results in a negative reward.
- Wind: Each cell in the grid has a wind strength that pushes the agent up, down, or leaves it in the same row.

**Tasks:**

1. **Implement Q-Learning Algorithm:**
   - Open the `windy_cliff.py` file.
   - Implement the Q-learning algorithm in the `q_learning` function.
   - Initialize the Q-table with zeros.
   - For each episode:
     - Initialize the state.
     - Choose an action using an epsilon-greedy policy.
     - Take the action and observe the next state and reward.
     - Update the Q-value using the Q-learning update rule.
     - Repeat until the episode ends.
   - You have already implemented Q-learning in a previous lab, so you can reuse that code here.

2. **Implement SARSA Algorithm:**
   - Implement the SARSA algorithm in the `sarsa` function.
   - Initialize the Q-table with zeros.
   - For each episode:
     - Initialize the state.
     - Choose an action using an epsilon-greedy policy.
     - For each step in the episode:
       - Take the action and observe the next state and reward.
       - Choose the next action using an epsilon-greedy policy.
       - Update the Q-value using the SARSA update rule.
       - Set the current state to the next state and the current action to the next action.
     - Repeat until the episode ends.

3. **Visualize Policies:**
   - Use the `visualize_policy` function to generate .gif files showing the behavior of the agent.
   - Test the Q-learning implementation by running the `windy_cliff_world.py` file and generating the `q_learning_windy_cliff.gif`.
   - Test the SARSA implementation by uncommenting the relevant code in the `windy_cliff_world.py` file and generating the `sarsa_windy_cliff.gif`.

4. **Run Experiments with Different Hyperparameters:**
   - Experiment with different values of the learning rate (alpha), discount factor (gamma), and exploration rate (epsilon).
   - Generate plots showing the total reward over episodes for different values of alpha and epsilon for both Q-learning and SARSA. There should be one plot for Q-learning and one plot for SARSA (You can have multiple lines on each plot, corresponding to different values of alpha and epsilon.)
   - Use at least two different values for alpha (e.g., 0.1 and 0.5) and two different values for epsilon (e.g., 0.1 and 0.5) for your experiments.

## Part 2: SARSA vs. Q-Learning in the Mountain Car Environment [1 point]

The Mountain Car environment is a classic reinforcement learning problem where an underpowered car must drive up a steep hill to get to a goal. However, there's an additional challenge: the state space is continuous, which means that traditional tabular methods like Q-learning and SARSA cannot be directly applied. To address this issue, we will use **tile coding** to approximate the value function in the Mountain Car environment.

The code for this part of the assignment is provided in the `mountain_car.py` file. You will need to implement the SARSA and Q-learning algorithms using tile coding and apply them to the Mountain Car environment.

#### Tile Coding

Tile coding is a form of feature representation used to approximate the value function in continuous state spaces. The main idea is to divide the state space into a grid of tiles, with each tile representing a small region of the state space. Multiple tilings, each with a slightly different offset, are used to create overlapping tiles that capture more detailed information about the state space.

In the `mountain_car.py` file, the `TileCoder` class is provided to implement tile coding. The class has the following attributes and methods:

- **n_tilings**: The number of tilings to use. Each tiling is offset slightly to provide better coverage of the state space.
- **n_bins**: The number of bins (tiles) in each dimension of the state space.
- **low**: The minimum value of each dimension of the state space.
- **high**: The maximum value of each dimension of the state space.

The state space is scaled and shifted by different offsets to generate multiple overlapping tilings. The state is then represented by the indices of the tiles it falls into across all tilings.

**Tasks:** 

1. **Implement Q-Learning with Tile Coding:**
   - Open the `mountain_car.py` file.
   - Implement the Q-learning algorithm using tile coding in the `q_learning` function.
   - Initialize the Q-table for the tiles.
   - For each episode:
     - Initialize the state.
     - Discretize the state using tile coding.
     - Choose an action using an epsilon-greedy policy.
     - Take the action and observe the next state and reward.
     - Discretize the next state using tile coding.
     - Update the Q-value using the Q-learning update rule.
     - Repeat until the episode ends.

2. **Implement SARSA with Tile Coding:**
   - Implement the SARSA algorithm using tile coding in the `sarsa` function.
   - Initialize the Q-table for the tiles.
   - For each episode:
     - Initialize the state.
     - Discretize the state using tile coding.
     - Choose an action using an epsilon-greedy policy.
     - For each step in the episode:
       - Take the action and observe the next state and reward.
       - Discretize the next state using tile coding.
       - Choose the next action using an epsilon-greedy policy.
       - Update the Q-value using the SARSA update rule.
       - Set the current state to the next state and the current action to the next action.
     - Repeat until the episode ends.

3. **Visualize Policies:**
   - Use the `visualize_policy` function to generate .gif files showing the behavior of the agent.
   - Test the Q-learning implementation by running the `mountain_car.py` file. This will generate a video file named `q_learning_mountain_car-episode-0.mp4` in the `videos` folder found in your current directory.
   - Test the SARSA implementation by uncommenting the relevant code in the `mountain_car.py` file and generating a video file named `sarsa_mountain_car-episode-0.mp4` in the `videos` folder found in your current directory.

## Submission

Once you're done with the assignment, submit the following files to Gradescope:

1. `mountain_car.py`
2. `windy_cliff.py`
3. `q_learning_windy_cliff.gif`
4. `sarsa_windy_cliff.gif`
5. `q_learning_windy_cliff_hyperparameters.png`
6. `sarsa_windy_cliff_hyperparameters.png`
7. `q_learning_mountain_car-episode-0.mp4`
8. `sarsa_mountain_car-episode-0.mp4`

Notes:

* The `q_learning_windy_cliff.gif` and `sarsa_windy_cliff.gif` files should show the agent's behavior in the Windy Cliff World environment using the Q-learning and SARSA algorithms, respectively. Instructions on how to generate these files are provided in the starter code. 
* The `q_learning_windy_cliff_hyperparameters.png` and `sarsa_windy_cliff_hyperparameters.png` files should show the total reward over episodes for different values of alpha and epsilon for Q-learning and SARSA, respectively.
* The `q_learning_mountain_car-episode-0.mp4` and `sarsa_mountain_car-episode-0.mp4` files should show the agent's behavior in the Mountain Car environment using the Q-learning and SARSA algorithms, respectively. Instructions on how to generate these files are provided in the starter code.

Make sure to save your Python files before uploading them. You can submit as many times as you like before the deadline. Only the last submission will be graded.