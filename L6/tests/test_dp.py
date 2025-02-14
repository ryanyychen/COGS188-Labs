import unittest
from src import dp
from src.policy import initialize_value, simple_policy
from src.blackjack import Hand, Card, BlackjackEnv

# A dummy environment to simulate fixed episodes for testing policy_evaluation.
class DummyEnv:
    def reset(self):
        # Each episode always starts at state (10, 10, False)
        self.step_count = 0
        return (10, 10, False)
    
    def step(self, action):
        # Ignore the action and return a predetermined sequence:
        # First call: from (10,10,False) -> (15,10,False) with reward 1, not terminal.
        # Second call: from (15,10,False) -> (15,10,False) with reward 1, terminal.
        if self.step_count == 0:
            self.step_count += 1
            return ((15, 10, False), 1, False)
        else:
            self.step_count += 1
            return ((15, 10, False), 1, True)

class TestPolicyEvaluation(unittest.TestCase):
    def setUp(self):
        # Set up dummy environment and initialize V and policy.
        self.env = DummyEnv()
        # V must contain the states encountered in our dummy episode:
        #   - Starting state: (10, 10, False)
        #   - Next state: (15, 10, False)
        self.V = {(10, 10, False): 0.0, (15, 10, False): 0.0}
        # Simple policy mapping (unused by DummyEnv, but required by the function).
        self.policy = {(10, 10, False): 'hit', (15, 10, False): 'hit'}

    def test_policy_evaluation_returns_correct_values(self):
        # Run policy evaluation on a deterministic dummy environment.
        # Use a small number of episodes for testing.
        new_V = dp.policy_evaluation(self.env, self.V, self.policy, episodes=10, gamma=1.0)
        
        # Explanation of expected values:
        # Each episode generates the same two transitions: 
        #  - For state (10,10,False):
        #      * At first visit (index 0): G_0 = 1 (reward from first transition)
        #      * Then from index 1: G becomes 1+1=2.
        #      * Due to the inner loop, the update happens twice per episode,
        #        so total contribution per episode is 1 + 2 = 3 with count 2.
        #      * Hence, the average return for (10,10,False) is 3/2 = 1.5.
        #
        #  - For state (15,10,False):
        #      * Only occurs at index 1 with reward 1,
        #      * So its average return is 1.
        self.assertAlmostEqual(new_V[(10, 10, False)], 1.5)
        self.assertAlmostEqual(new_V[(15, 10, False)], 1.0)

    def test_policy_evaluation_no_episodes(self):
        # When zero episodes are run, the value function should remain unchanged.
        original_V = self.V.copy()
        new_V = dp.policy_evaluation(self.env, self.V, self.policy, episodes=0, gamma=1.0)
        self.assertEqual(new_V, original_V)

if __name__ == '__main__':
    unittest.main()
