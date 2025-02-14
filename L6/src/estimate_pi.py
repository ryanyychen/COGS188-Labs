import matplotlib.pyplot as plt
import random
import math
from matplotlib.animation import FuncAnimation

def estimate_pi(num_samples: int, step: int = 100):
    """Estimate the value of Pi using Monte Carlo simulation.

    Args:
        num_samples (int): The number of samples to generate for the simulation.
        step (int, optional): The number of samples after which to record data. Defaults to 100.

    Returns:
        list: A list of tuples containing the following data: x_inside, y_inside, x_outside, y_outside, pi_estimate.
    """
    # TODO:
    # 1. Initialize a counter for points inside the circle,
    #    lists for x_inside, y_inside, x_outside, y_outside, and an empty data list.
    # 2. Loop through num_samples iterations:
    #     a. Generate a random point (x, y) with values uniformly drawn from [-1, 1].
    #     b. Check if the point is inside the unit circle (x**2 + y**2 <= 1).
    #     c. If it is, update the counter and append the point to the inside lists; otherwise, append it to the outside lists.
    #     d. After every 'step' samples, compute the current estimate of Pi (4 * inside_count / total_samples)
    #        and append a tuple (x_inside.copy(), y_inside.copy(), x_outside.copy(), y_outside.copy(), pi_estimate) to data.
    # 3. Return the data list when complete.
    pass

def create_animation_pi(num_samples: int, step:int=100, filename: str='monte_carlo_pi.gif'):
    """
    Create an animation of the Monte Carlo simulation to estimate Pi.

    Args:
        num_samples (int): The total number of samples to use for the simulation.
        step (int, optional): The number of samples after which to record data. Defaults to 100.
        filename (str, optional): The filename to save the animation. Defaults to 'monte_carlo_pi.gif'.
    """
    data = estimate_pi(num_samples, step)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    scat_inside = ax.scatter([], [], color='blue', s=1, label='Inside Circle')
    scat_outside = ax.scatter([], [], color='red', s=1, label='Outside Circle')
    title = ax.set_title("")
    ax.legend(loc='upper right')

    def update(frame):
        x_inside, y_inside, x_outside, y_outside, pi_estimate = frame
        scat_inside.set_offsets(list(zip(x_inside, y_inside)))
        scat_outside.set_offsets(list(zip(x_outside, y_outside)))
        title.set_text(f"Estimation of Pi after {len(x_inside) + len(x_outside)} Samples: Pi ≈ {pi_estimate:.4f}")

    animation = FuncAnimation(fig, update, frames=data, interval=50)
    animation.save(filename, writer='imagemagick')
    plt.close(fig)
