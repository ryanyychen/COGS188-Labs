import gymnasium as gym
import pygame
import numpy as np
import time

# Define the Lunar Lander environment
env = gym.make('LunarLander-v2', render_mode='human')

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Lunar Lander")

# Action mapping
action_mapping = {
    pygame.K_LEFT: 1,   # Fire left orientation engine
    pygame.K_RIGHT: 3,  # Fire right orientation engine
    pygame.K_UP: 2,     # Fire main engine
    pygame.K_DOWN: 0    # Do nothing
}

# Set the time interval for automatic "do nothing" action
time_interval = 2.5

def play_lunar_lander():
    state, _ = env.reset()
    done = False
    last_action_time = time.time()
    current_action = action_mapping[pygame.K_DOWN]  # Default to "do nothing"

    while not done:
        current_time = time.time()
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                pygame.quit()
                env.close()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key in action_mapping:
                    current_action = action_mapping[event.key]
                    last_action_time = time.time()
            elif event.type == pygame.KEYUP:
                if event.key in action_mapping and current_action == action_mapping[event.key]:
                    current_action = action_mapping[pygame.K_DOWN]  # Default to "do nothing" when key is released

        # If no action was taken within the time interval, press "DOWN"
        if current_time - last_action_time > time_interval:
            current_action = action_mapping[pygame.K_DOWN]
            last_action_time = current_time

        # Perform the current action
        state, reward, done, _, _ = env.step(current_action)
        print(f"Action: {current_action}, Reward: {reward}")
        env.render()

if __name__ == "__main__":
    play_lunar_lander()
