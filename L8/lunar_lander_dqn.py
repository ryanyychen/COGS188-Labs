import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

# Define the Lunar Lander environment with render_mode
env = gym.make('LunarLander-v2', render_mode='rgb_array')

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """Neural Network for approximating Q-values."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # TODO: Define the neural network layers
        # You can keep it fairly simple, e.g., with 3 linear layers with 64 units each
        # and use ReLU activations for the hidden layers.
        # However, make sure that the input size of the first layer matches the state size
        # and the output size of the last layer matches the action size
        # This is because the input to the network will be the state and the output will be the Q-values for each action
    
    def forward(self, state):
        """Build a network that maps state -> action values.
        Params
        ======
            state (torch.Tensor): The state input
        Returns
        =======
            torch.Tensor: The predicted action values
        """
        # TODO: Define the forward pass
        # You're basically just passing the state through the network here (based on the layers you defined in __init__) and returning the output

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): Dimension of each action
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            seed (int): Random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # TODO: Implement this method
        # Use the namedtuple 'Experience' to create an experience tuple and append it to the memory

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # TODO: Complete this method
        # We first use random.sample to sample self.batch_size experiences from self.memory
        # Convert the sampled experiences to tensors and return them as a tuple
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # Similarly, convert the other components of the experiences to tensors
        # the `actions` tensor should be of type long
        actions = ...
        rewards = ...
        next_states = ...
        dones = ...
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DQNAgent:
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        # TODO: Initialize Q-networks (local and target)
        # Hints: Use QNetwork to create both qnetwork_local and qnetwork_target, and move them to device
        # Use optim.Adam to create an optimizer for qnetwork_local



        # Replay memory
        # TODO: Initialize replay memory
        # Hint: Create a ReplayBuffer object with appropriate parameters (action_size, buffer_size, batch_size, seed)


        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # TODO: Save the experience in replay memory
        # Hint: Use the add method of ReplayBuffer



        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, gamma=0.99)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (np.ndarray): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # TODO: Compute and minimize the loss
        # 1. Compute Q targets for current states (s')
        # Hint: Use the target network to get the next action values
        # 2. Compute Q expected for current states (s)
        # Hint: Use the local network to get the current action values
        # 3. Compute the loss between Q expected and Q target
        # 4. Perform a gradient descent step to update the local network

        # TODO: Update target network
        # Hint: Use the soft_update method provided

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

agent = DQNAgent(state_size=8, action_size=4, seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()  # Unpack the state from the reset method
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)  # Unpack the state from the step method
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window)}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window)}')
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window)}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

scores = dqn()

import matplotlib.pyplot as plt

def plot_scores(scores):
    """Plot the scores."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig.savefig('training_scores.png')

plot_scores(scores)

def create_video(agent, env, filename="lunar_lander.mp4"):
    """Create a video of the agent's performance."""
    video_env = gym.wrappers.RecordVideo(env, './video', episode_trigger=lambda x: True, video_length=0)
    state, _ = video_env.reset()
    done = False
    while not done:
        action = agent.act(state, eps=0.0)  # Use greedy policy for evaluation
        state, reward, done, _, _ = video_env.step(action)
    video_env.close()

    # Rename the video file to the desired filename
    video_dir = './video'
    video_file = [f for f in os.listdir(video_dir) if f.endswith('.mp4')][0]
    os.rename(os.path.join(video_dir, video_file), filename)

# Load the trained model and create the video
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
create_video(agent, env)
