# register_env.py
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='DotTouch-v0',
    entry_point='dot_touch:DotTouchEnv',  # adjust class/module names as needed
)