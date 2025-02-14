from tqdm import tqdm
from .blackjack import BlackjackEnv, Hand, Card

ACTIONS = ['hit', 'stick']

def policy_evaluation(env, V, policy, episodes=500000, gamma=1.0):
    """
    Monte Carlo policy evaluation:
    - Generate episodes using the current policy
    - Update state value function as an average return
    """
    # TODO:
    # track number of visits to each state and track sum of returns for each state
    # TODO:
    # Initialize returns_sum and returns_count

    for _ in tqdm(range(episodes), desc="Policy evaluation"):
        ...
        # Generate one episode
        # TODO:...
        # First-visit Monte Carlo: Update returns for the first occurrence of each state
            # Compute return from the first visit onward
                # TODO # Update returns_sum and returns_count

    # Update V(s) as the average return
    ... # TODO
