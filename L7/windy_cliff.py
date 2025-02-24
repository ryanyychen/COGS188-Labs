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
    total_rewards = []

    for _ in range(num_episodes):
        curr_state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                # Randomize action
                action = np.random.randint(0, env.action_space.n)
            else:
                # Choose best action
                action = np.argmax(q_table[curr_state])
            
            # Take action
            new_state, reward, done, useless_list = env.step(action)

            # Update q_table
            q_table[curr_state][action] += alpha * (reward + gamma * np.max(q_table[new_state]) - q_table[curr_state][action])

            # Update current state
            curr_state = new_state

        # Append to total rewards
        reward_total = q_table.sum()
        total_rewards.append(reward_total)
            
    return q_table, total_rewards

def sarsa(env, num_episodes, alpha, gamma, epsilon):
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    total_rewards = []

    for _ in range(num_episodes):
        curr_state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                # Randomize action
                action = np.random.randint(0, env.action_space.n)
            else:
                # Choose best action
                action = np.argmax(q_table[curr_state])
            
            # Take action
            new_state, reward, done, useless_list = env.step(action)

            # Update q_table using SARSA update rule
            q_table[curr_state][action] += alpha * (reward + gamma * q_table[new_state][action] - q_table[curr_state][action])

            curr_state = new_state
        
        # Append to total rewards
        reward_total = q_table.sum()
        total_rewards.append(reward_total)
    
    return q_table, total_rewards

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
# env = WindyCliffWorld()
# q_table, total_rewards = q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
# visualize_policy(env, q_table, filename='q_learning_windy_cliff.gif')

# Testing SARSA
# env = WindyCliffWorld()
# q_table, total_rewards = sarsa(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
# visualize_policy(env, q_table, filename='sarsa_windy_cliff.gif')

# Run experiments with different hyperparameters
# env = WindyCliffWorld()
# q_table, total_rewards_11 = sarsa(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1)
# q_table, total_rewards_15 = sarsa(env, num_episodes=500, alpha=0.1, gamma=0.99, epsilon=0.5)
# q_table, total_rewards_55 = sarsa(env, num_episodes=500, alpha=0.5, gamma=0.99, epsilon=0.5)
# q_table, total_rewards_51 = sarsa(env, num_episodes=500, alpha=0.5, gamma=0.99, epsilon=0.1)

# plt.plot(total_rewards_11, label='α=0.1, ε=0.1')
# plt.plot(total_rewards_51, label='α=0.5, ε=0.1')
# plt.plot(total_rewards_15, label='α=0.1, ε=0.5')
# plt.plot(total_rewards_55, label='α=0.5, ε=0.5')
# plt.xlabel('Episode')
# plt.ylabel('Total Reward')
# plt.legend()
# plt.show()