
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

# Create a custom maze environment

class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 2}

    def __init__(self):
        super(MazeEnv, self).__init__()
        self.size = 5  # Size of the maze (5x5)
        self.state = (0, 0)  # Start at bottom-left corner
        self.goal = (4, 4)  # Goal is at top-right corner
        self.obstacles = [(1, 1), (2, 1), (3, 1), (2, 4), (3, 3), (4, 1)]  # Positions of obstacles
        self.action_space = gym.spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = gym.spaces.Box(low=np.array([0,0]), high=np.array([4,4]), dtype=np.int32)

    def step(self, action):
        # Define action effects
        moves = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
        new_position = (self.state[0] + moves[action][0], self.state[1] + moves[action][1])
        
        # Check for boundaries and obstacles
        if 0 <= new_position[0] < self.size and 0 <= new_position[1] < self.size and new_position not in self.obstacles:
            self.state = new_position
        
        # Reward function
        if self.state == self.goal:
            reward = 1  # Reward for reaching the goal
            done = True
        else:
            reward = -0.01  # Slight negative reward for each step
            done = False
        
        return self.state, reward, done, {}

    def reset(self):
        self.state = (0, 0)  # Reset to start position
        return self.state

    def render(self, mode='human', close=False):
        grid = np.zeros((self.size, self.size))
        for obs in self.obstacles:
            grid[obs] = -1  # Represent obstacles
        grid[self.goal] = 0.5  # Represent goal
        grid[self.state] = 1  # Represent agent
        if mode == 'rgb_array':
            return grid
        elif mode == 'human':
            plt.imshow(grid, cmap='viridis', origin='lower')
            plt.grid('on')
            plt.show()