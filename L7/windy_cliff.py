import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

np.random.seed(0)

class WindyCliffWorld(gym.Env):
    def __init__(self):
        super(WindyCliffWorld, self).__init__()
        
        self.grid_size = (7, 10)
        self.start_state = (3, 0)
        self.goal_state = (3, 9)
        self.cliff = [(3, i) for i in range(1, 9)]
        self.obstacles = [(2, 4), (4, 4), (2, 7), (4, 7)]
        
        self.wind_strength = {
            (i, j): np.random.choice([-1, 0, 1]) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
        }

        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right
        self.observation_space = spaces.Discrete(self.grid_size[0] * self.grid_size[1])
        
        self.state = self.start_state
        
        self.action_effects = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }

    def reset(self):
        self.state = self.start_state
        return self.state_to_index(self.state)
    
    def step(self, action):
        new_state = (self.state[0] + self.action_effects[action][0], self.state[1] + self.action_effects[action][1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        # Apply wind effect
        wind = self.wind_strength[new_state]
        new_state = (new_state[0] + wind, new_state[1])
        new_state = (max(0, min(new_state[0], self.grid_size[0] - 1)), max(0, min(new_state[1], self.grid_size[1] - 1)))

        if new_state in self.cliff:
            reward = -100
            done = True
            new_state = self.start_state
        elif new_state == self.goal_state:
            reward = 10
            done = True
        elif new_state in self.obstacles:
            reward = -10
            done = False
        else:
            reward = -1
            done = False

        self.state = new_state
        return self.state_to_index(new_state), reward, done, {}
    
    def state_to_index(self, state):
        return state[0] * self.grid_size[1] + state[1]
    
    def index_to_state(self, index):
        return (index // self.grid_size[1], index % self.grid_size[1])
    
    def render(self):
        grid = np.zeros(self.grid_size)
        grid[self.state] = 1  # Current position
        for c in self.cliff:
            grid[c] = -1  # Cliff positions
        for o in self.obstacles:
            grid[o] = -0.5  # Obstacle positions
        grid[self.goal_state] = 2  # Goal position
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(grid, cmap='viridis')
        plt.axis('off')
        fig.canvas.draw()
        plt.close(fig)
        image = np.array(fig.canvas.renderer.buffer_rgba())
        return image

# Create and register the environment
env = WindyCliffWorld()

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # TODO: Implement Q-learning algorithm
    
    return q_table

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # TODO: Implement SARSA algorithm
    
    return q_table

def save_gif(frames, path='./', filename='gym_animation.gif'):
    imageio.mimsave(os.path.join(path, filename), frames, duration=0.5)

def visualize_policy(env, q_table, filename='q_learning.gif'):
    state = env.reset()
    frames = []
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, _, done, _ = env.step(action)
        frames.append(env.render())
    
    save_gif(frames, filename=filename)

# Example usage:

# Testing Q-Learning
env = WindyCliffWorld()
q_table = q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
visualize_policy(env, q_table, filename='q_learning_windy_cliff.gif')

# Testing SARSA
# env = WindyCliffWorld()
# q_table = sarsa(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
# visualize_policy(env, q_table, filename='sarsa_windy_cliff.gif')

# TODO: Run experiments with different hyperparameters and visualize the results
# You should generate two plots:
# 1. Total reward over episodes for different α and ε values for Q-learning
# 2. Total reward over episodes for different α and ε values for SARSA
# For each plot, use at least 2 different values for α and 2 different values for ε
