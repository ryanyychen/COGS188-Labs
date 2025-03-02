# L8: Neural Networks for Reinforcement Learning (2 points)

## Installations

You will need the following libraries:

* PyGame
* NumPy
* Argparse
* Tqdm
* Pickle
* Swig
* Gymnasium
* PyTorch

The environment used is the LunarLander-v2 environment from Gymnasium, which requires `box2d` as a dependency. This, in turn, requires `swig` to be installed on your system. You can install these using the following command:

```bash
pip install swig
pip install box2d-py
```

**Note:** If you have difficulties with installing PyTorch and/or Swig and box2d, do not worry because I have created a Google Colab notebook that you can use to complete the assignment.

For this lab, we'll be using the LunarLander-v2 environment from Gymnasium. The goal of the agent is to land the lunar module on the landing pad within the environment. The agent receives a reward for moving the lander from the top of the screen to the landing pad and zero speed. The agent also receives a reward for landing the lander safely on the landing pad. The episode ends if the lander crashes or comes to rest, and the environment provides additional rewards for landing within the landing pad and moving closer to the center of the pad.

For more details on the LunarLander-v2 environment, refer to the [Gymnasium documentation](https://www.gymlibrary.dev/environments/box2d/lunar_lander/).

To get an intuitive feel for how this environment works, you can run the following code:

```python
python lunar_lander_keyboard.py
```

The code in that file is very simple and allows you to control the spaceship using the arrow keys on your keyboard. The goal is to land the spaceship on the landing pad (in between the two flags) without crashing. I recommend you try playing around with it for a bit to get a feel for how the environment works, and what the rewards are.

**Warning:** It's pretty hard to land the spaceship using the keyboard, so don't worry if you crash a lot! If you can't get it to land after a few tries, I'd recommend moving on. (You'll probably realize that it's easier to train a neural network to solve this task than to do it yourself!)

In this part, you will implement a Deep Q-Network (DQN) to solve the Lunar Lander environment from the Gymnasium library. You will complete the implementation of a DQN agent, including the neural network, experience replay, and training loop.

The starter code can be found in `lunar_lander_dqn.py`. 

If you are unable to get the installations to work on your local machine, you can use the Google Colab notebook at this link: https://colab.research.google.com/drive/1kztw2-2BCtYQZZW_8Pg804ioFGJlCwlb?usp=sharing

The code is divided into several sections:

* `QNetwork` Class: Define the neural network architecture and forward pass.
* `ReplayBuffer` Class: Implement a fixed-size buffer to store experience tuples and sample batches of experiences for training.
* `DQNAgent` Class: Implement the DQN agent, including methods for interacting with the environment, storing experiences, and learning from them.
* `dqn` Function: Implement the main training loop for the DQN agent.
* Plotting and Video Generation: Plot the training progress and create a video of the trained agent's performance.

### Tasks to Complete

#### `QNetwork` Class

The `QNetwork` class defines the neural network used to approximate the Q-values for each action. You will:

* Define the neural network layers in the `__init__` method.
* Implement the forward pass in the forward method.
  
#### `ReplayBuffer` Class

The `ReplayBuffer` class implements a fixed-size buffer to store experience tuples. You will:

* Implement the `add` method to add new experiences to the memory.
* Implement the `sample` method to randomly sample a batch of experiences from the memory.

#### `DQNAgent` Class

The DQNAgent class interacts with the environment and learns from experiences. You will:

* Initialize Q-networks (local and target) and the replay buffer in the `__init__` method.
* Implement the `step` method to save experiences in the replay buffer and periodically sample a batch to learn from.
* Implement the `act` method to select actions based on an epsilon-greedy policy.
* Implement the `learn` method to update the Q-networks using a batch of experiences.
* Implement the `soft_update` method to update the target network parameters.

#### `dqn` Function

The dqn function implements the main training loop. You will:

* Initialize the environment and agent.
* Implement the training loop to interact with the environment, collect experiences, and update the agent.
* Save the trained model when the environment is solved.

#### Plotting and Video Generation

* Use the `plot_scores` function to visualize the training progress.
* Use the `create_video` function to generate a video of the trained agent's performance.

#### Hyperparameters

You can experiment with different hyperparameters to improve the agent's performance. Some hyperparameters you can tune include:

* Learning rate
* Discount factor (gamma)
* Batch size
* Replay buffer size
* Epsilon start, end, and decay rate

You are also welcome to tweak the architecture of the neural network to see if you can improve the agent's performance.

The `TODO` sections in the code indicate where you need to complete the implementation. Once you have completed the code, run the script using the following command:

```bash
python lunar_lander_dqn.py
```

**Note:** I've gotten the training to work pretty fast using a CPU, so I don't think it's necessary to use a GPU for this assignment. However, if you feel it's too slow, you can try running it on a GPU.

This will train the DQN agent to solve the Lunar Lander environment. The training progress will be displayed, and a video of the trained agent's performance will be saved as `lunar_lander.mp4`.

Furthermore, a plot of the training progress will be displayed, showing the scores obtained during training. The plot should show the scores increasing over time as the agent learns to land the lunar module successfully. You will expect to see some noise in the scores due to the stochastic nature of the environment and the exploration strategy, but that is perfectly normal. The resulting plot will be saved to `training_scores.png`.

When you open the video, you should see the agent successfully landing the lunar module on the landing pad. The agent should land slowly and safely, avoiding crashes and coming to rest within the landing pad.

## Submission Instructions

Submit the following files to Gradescope:

1. `lunar_lander_dqn.py` (or the Jupyter notebook if you used Google Colab)
2. `lunar_lander_keyboard.py`
3. `training_scores.png`
4. `lunar_lander.mp4`
5. `checkpoint.pth`

Make sure to save your Python files before uploading them. You can submit as many times as you like before the deadline. Only the last submission will be graded.