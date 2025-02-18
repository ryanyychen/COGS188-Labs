from tqdm import tqdm
from .blackjack import BlackjackEnv, Hand, Card

ACTIONS = ['hit', 'stick']

def policy_evaluation(env, V, policy, episodes=500000, gamma=1.0):
    """
    Monte Carlo policy evaluation:
    - Generate episodes using the current policy
    - Update state value function as an average return
    """
    returns_sum = {}
    returns_count = {}

    for _ in tqdm(range(episodes), desc="Policy evaluation"):
        episode = []
        state = env.reset()
        done = False

        # Append to lists so that keys exist
        returns_sum[state] = 0
        returns_count[state] = 0

        while not done:
            action = policy[state]
            next_state, reward, done = env.step(action)
            episode.append((state, reward))
            state = next_state

            # Append to lists so that keys exist
            returns_sum[state] = 0
            returns_count[state] = 0
        
        r = 0

        for t in range(len(episode) - 1, -1, -1):
            state = episode[t][0]
            reward = episode[t][1]
            r = reward + gamma * r

            returns_sum[state] += r
            returns_count[state] += 1
            V[state] = returns_sum[state] / returns_count[state]

            if state == (10, 10, False):
                print(state)
                print(episode)
                print(returns_sum[state])
                print(returns_count[state])
                print(V[state])

    return V
