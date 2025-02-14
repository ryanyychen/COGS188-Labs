import numpy as np

# Constants / ranges
PLAYER_SUM_RANGE = range(4, 22)  # Though typically 12..21 is the main range
DEALER_UPCARD_RANGE = range(1, 11)
USABLE_ACE_RANGE = [False, True]
ACTIONS = [0, 1]  # 0->HIT, 1->STICK

def initialize_value():
    """
    Initializes the value function for the given states.

    Returns:
        dict: A dictionary with state tuples as keys and their values initialized to 0.
    """
    # Initialize Value Function
    V = {}
    for psum in PLAYER_SUM_RANGE:
        for dealer_up in DEALER_UPCARD_RANGE:
            for ace in USABLE_ACE_RANGE:
                # Initialize all state values to zero
                V[(psum, dealer_up, ace)] = 0.0
    return V


def simple_policy():
    """
    Based on the possible states, the policy is to HIT if the player sum is less than 20, else STICK.
    This policy is deterministic.
    
    Returns:
        dict: A dictionary with state tuples as keys and their actions (HIT or STICK).
    """
    # Initialize Policy (e.g., always HIT if < 20, else STICK)
    policy = {}
    for psum in PLAYER_SUM_RANGE:
        for dealer_up in DEALER_UPCARD_RANGE:
            for ace in USABLE_ACE_RANGE:
                if psum < 18:
                    policy[(psum, dealer_up, ace)] = 0  # HIT
                else:
                    policy[(psum, dealer_up, ace)] = 1  # STICK
    return policy