import unittest
import numpy as np
from src.mdp import (
    states,
    actions,
    transition,
    reward,
    always_right_policy,
    my_policy,
    simulate_mdp,
    q_learning,
    new_policy,
)

# Create a dummy environment for testing q_learning.
class DummyActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return 0  # Always return action 0 for simplicity

class DummyMazeEnv:
    def __init__(self, size=3):
        self.size = size
        self.action_space = DummyActionSpace(n=3)
        self.state = (0, 0)
        self.step_count = 0

    def reset(self):
        self.state = (0, 0)
        self.step_count = 0
        return self.state

    def step(self, action):
        # Simple deterministic step: move one step towards the bottom-right.
        self.step_count += 1
        next_state = (
            min(self.state[0] + 1, self.size - 1),
            min(self.state[1] + 1, self.size - 1)
        )
        reward_val = 1
        done = self.step_count >= 1  # End episode after one step
        self.state = next_state
        return next_state, reward_val, done, {}

class TestMDPExtended(unittest.TestCase):

    def test_states_and_actions(self):
        # Test that states and actions are defined correctly.
        self.assertListEqual(list(states), [1, 2, 3, 4, 5])
        self.assertEqual(actions, ['left', 'stay', 'right'])

    def test_transition_right(self):
        self.assertEqual(transition(1, 'right'), 2)
        self.assertEqual(transition(5, 'right'), 5)

    def test_transition_left(self):
        self.assertEqual(transition(5, 'left'), 4)
        self.assertEqual(transition(1, 'left'), 1)

    def test_transition_stay(self):
        for s in range(1, 6):
            self.assertEqual(transition(s, 'stay'), s)

    def test_transition_invalid_action(self):
        # For any invalid action, transition should default to staying.
        for s in range(1, 6):
            self.assertEqual(transition(s, 'jump'), s)

    def test_reward_correct(self):
        self.assertEqual(reward(4, 'right'), 10)
        self.assertEqual(reward(1, 'left'), -1)
        self.assertEqual(reward(3, 'right'), -1)
        self.assertEqual(reward(2, 'stay'), -1)

    def test_reward_invalid_action(self):
        self.assertEqual(reward(4, 'left'), -1)
        self.assertEqual(reward(4, 'jump'), -1)

    def test_always_right_policy(self):
        for s in range(1, 6):
            self.assertEqual(always_right_policy(s), 'right')

    def test_my_policy_valid_action(self):
        # Test that my_policy returns a valid action for each state.
        for s in range(1, 6):
            act = my_policy(s)
            self.assertIn(act, actions)
            
    def test_simulate_mdp_always_right(self):
        # Using always_right_policy should end the simulation on reaching state 5
        state_visits, cum_reward, visited_history, reward_history = simulate_mdp(always_right_policy, initial_state=1, simulation_depth=10)
        self.assertEqual(visited_history[0], 1)
        self.assertEqual(visited_history[-1], 5)
        # Expected visited history: 1 -> 2 -> 3 -> 4 -> 5
        self.assertListEqual(visited_history, [1, 2, 3, 4, 5])
        # Check that rewards sum up as expected: -1 for states 1,2,3 and -1 for state 4 with a bonus +10 when leaving 4.
        self.assertEqual(cum_reward, -1 -1 -1 + 10)

    def test_simulate_mdp_my_policy(self):
        # Test simulate_mdp with my_policy, which is stochastic.
        state_visits, cum_reward, visited_history, reward_history = simulate_mdp(my_policy, initial_state=3, simulation_depth=15)
        self.assertEqual(visited_history[0], 3)
        # The episode will run at most max_steps+1 entries; if terminal is reached it may be shorter.
        self.assertTrue(len(visited_history) <= 16)
        # Check that state_visits is an array of length 5.
        self.assertEqual(len(state_visits), 5)
        
    def test_low_y_policy(self):
        # For a state with y-coordinate < 3 (e.g., [0, 0]),
        # the policy should choose "up" (action 0) with prob ≈ 0.7
        # and "down" (action 1) with prob ≈ 0.3.
        num_trials = 10000
        count_up = 0
        count_down = 0

        for _ in range(num_trials):
            action = new_policy([0, 0])
            if action == 0:  # up
                count_up += 1
            elif action == 1:  # down
                count_down += 1

        p_up = count_up / num_trials
        p_down = count_down / num_trials

        self.assertAlmostEqual(p_up, 0.7, delta=0.05)
        self.assertAlmostEqual(p_down, 0.3, delta=0.05)

    def test_high_y_policy(self):
        # For a state with y-coordinate >= 3 (e.g., [0, 3]),
        # the policy should choose "right" (action 3) with prob ≈ 0.7
        # and "left" (action 2) with prob ≈ 0.3.
        num_trials = 10000
        count_right = 0
        count_left = 0

        for _ in range(num_trials):
            action = new_policy([0, 3])
            if action == 3:  # right
                count_right += 1
            elif action == 2:  # left
                count_left += 1

        p_right = count_right / num_trials
        p_left = count_left / num_trials

        self.assertAlmostEqual(p_right, 0.7, delta=0.05)
        self.assertAlmostEqual(p_left, 0.3, delta=0.05)

    def test_q_learning_shape_and_update(self):
        # Test that q_learning returns a Q-table of the correct shape and that learning occurs.
        env = DummyMazeEnv(size=3)
        q_table = q_learning(env, episodes=10, alpha=0.5, gamma=0.9, epsilon=0.1)
        self.assertIsInstance(q_table, np.ndarray)
        self.assertEqual(q_table.shape, (env.size, env.size, env.action_space.n))
        # For the dummy maze, the starting state (0,0) should have been updated.
        self.assertNotEqual(q_table[0, 0, 0], 0)
        # Using always_right_policy should end the simulation on reaching state 5
        state_visits, cum_reward, visited_history, reward_history = simulate_mdp(always_right_policy, initial_state=1, simulation_depth=10)
        self.assertEqual(visited_history[0], 1)
        self.assertEqual(visited_history[-1], 5)
        # Expected visited history: 1 -> 2 -> 3 -> 4 -> 5
        self.assertListEqual(visited_history, [1, 2, 3, 4, 5])
        # Check that rewards sum up as expected: -1 for states 1,2,3 and -1 for state 4 with a bonus +10 when leaving 4.
        self.assertEqual(cum_reward, -1 -1 -1 + 10)

    def test_simulate_mdp_my_policy(self):
        # Test simulate_mdp with my_policy, which is stochastic.
        state_visits, cum_reward, visited_history, reward_history = simulate_mdp(my_policy, initial_state=3, simulation_depth=15)
        self.assertEqual(visited_history[0], 3)
        # The episode will run at most max_steps+1 entries; if terminal is reached it may be shorter.
        self.assertTrue(len(visited_history) <= 16)
        # Check that state_visits is an array of length 5.
        self.assertEqual(len(state_visits), 5)

    def test_q_learning_shape_and_update(self):
        # Test that q_learning returns a Q-table of the correct shape and that learning occurs.
        env = DummyMazeEnv(size=3)
        q_table = q_learning(env, episodes=10, alpha=0.5, gamma=0.9, epsilon=0.1)
        self.assertIsInstance(q_table, np.ndarray)
        self.assertEqual(q_table.shape, (env.size, env.size, env.action_space.n))
        # For the dummy maze, the starting state (0,0) should have been updated.
        self.assertNotEqual(q_table[0, 0, 0], 0)

if __name__ == '__main__':
    unittest.main()