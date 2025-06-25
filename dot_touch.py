import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import argparse

class DotTouchEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human", screen_size=500, img_size : int = 128):
        super().__init__()
        self.screen_size = screen_size
        self.window = None
        self.clock = None
        self.render_mode = render_mode

        # Environment parameters
        self.radius = 20
        self.goal_radius = 20
        self.threshold = 30.0
        self.max_steps = 200
        self.step_count = 0

        # Action: continuous (dx, dy)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)

        # Observation: agent_pos, goal_pos
        self.img_size = 84  # example size for image obs
        self.observation_space = spaces.Dict({
            "state": spaces.Box(low=0.0, high=screen_size, shape=(4,), dtype=np.float32),
            "image": spaces.Box(low=0, high=255, shape=(self.img_size, self.img_size, 3), dtype=np.uint8),
        })
        

        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.goal_pos = np.zeros(2, dtype=np.float32)

        self.is_intervention = False

        self.reset()

        if self.render_mode == "human":
            print("Make red dot (you) touch the black dot (goal)\nCONTROLS:\nUp/Down/Left/Right: move red dot\nSpacebar: Switch intervention on/off")
            self.render()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.random.uniform(0, self.screen_size, size=2).astype(np.float32)
        self.goal_pos = np.random.uniform(0, self.screen_size, size=2).astype(np.float32)
        self.is_intervention = False
        self.step_count = 0

        state_obs = np.concatenate([self.agent_pos, self.goal_pos])
        image_obs = self.get_image()

        return {"state": state_obs, "image": image_obs}, {}

    def step(self, action):
        self.step_count += 1

        if self.is_intervention:
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            action = np.array([0.0, 0.0], dtype=np.float32)

            if keys[pygame.K_LEFT]: action[0] -= 5.0
            if keys[pygame.K_RIGHT]: action[0] += 5.0
            if keys[pygame.K_UP]: action[1] -= 5.0
            if keys[pygame.K_DOWN]: action[1] += 5.0

        self.agent_pos += action
        self.agent_pos = np.clip(self.agent_pos, 0, self.screen_size)

        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        terminated = dist_to_goal < self.threshold or self.step_count >= self.max_steps
        reward = 1.0 if dist_to_goal < self.threshold else 0.0

        state_obs = np.concatenate([self.agent_pos, self.goal_pos])
        image_obs = self.get_image()

        info = {
            "is_intervention": self.is_intervention,
            "action_intervention": action
        }

        # Handle intervention key toggle
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_intervention = not self.is_intervention  # Toggle intervention

        if self.render_mode == "human":
            self.render()   

        return {"state": state_obs, "image": image_obs}, reward, terminated, False, info

    def render_frame(self):
        if not hasattr(self, 'surface'):
            self.surface = pygame.Surface((self.screen_size, self.screen_size))
        self.surface.fill((255, 255, 255))
        pygame.draw.circle(self.surface, (0, 0, 0), self.goal_pos.astype(int), self.goal_radius)
        pygame.draw.circle(self.surface, (255, 0, 0), self.agent_pos.astype(int), self.radius)
        return self.surface

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.screen_size, self.screen_size))
            self.clock = pygame.time.Clock()

        surface = self.render_frame()
        self.window.blit(surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def get_image(self):
        surface = self.render_frame()
        raw_str = pygame.image.tostring(surface, 'RGB')
        img = np.frombuffer(raw_str, dtype=np.uint8)
        img = img.reshape((self.screen_size, self.screen_size, 3))
        # Resize as before...
        factor = self.screen_size // self.img_size
        img_small = img[::factor, ::factor]
        return img_small


    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None



def main(show_img=False):
    env = DotTouchEnv()
    obs, _ = env.reset()

    if show_img:
        import matplotlib.pyplot as plt
        
        plt.ion()  # Interactive mode on
        fig, ax = plt.subplots()
        img_plot = ax.imshow(obs["image"])
        ax.axis('off')
        plt.show()

    for step in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if show_img:
            img_plot.set_data(obs["image"])
            ax.set_title(f"Step {step}")
            plt.pause(0.001)

        if terminated:
            print("Episode finished with reward:", reward)
            break

    env.close()

    if show_img:
        plt.ioff()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DotTouchEnv with optional image display")
    parser.add_argument('--show-img', action='store_true', help="Display image observations using matplotlib")
    args = parser.parse_args()

    main(show_img=args.show_img)