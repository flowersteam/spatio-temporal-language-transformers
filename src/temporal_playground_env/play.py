import gym
import numpy as np
import pygame
from pygame.locals import *
import time

from pprint import pprint

ENV_NAME = 'TemporalPlaygroundHuman-v1'

from src.grammar.descriptions import sample_descriptions_from_state  # , get_reward_from_state
# from src.temporal_playground_env.descriptions import generate_all_descriptions

from src.temporal_playground_env.env_params import get_env_params as get_env_params
import src.temporal_playground_env.temporal_logic as tl

"""
Playing script. Control the agent with the arrows, close the gripper with the space bar.
"""

env = gym.make(ENV_NAME, reward_screen=False, viz_data_collection=False)
pygame.init()

env_params = get_env_params


# train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)
# all_descriptions = train_descriptions +  test_descriptions

# Select the goal to generate the scene.
# goal_str = np.random.choice(all_descriptions)

env.reset()
# TODO: Be carefull new grammar is now yielding incompatiblity with previous function: 'grow' (new grammar) was 'Grow' with
# capital G

# Init
env.unwrapped.reset_with_goal("Grasp any cat")
t = 0

past_descr = sample_descriptions_from_state(env.unwrapped)
past_descr_buffer = []
first_grow = False
while True:
    key = False
    # init_render
    # time.sleep(0.2)
    action = np.zeros([3])
    for event in pygame.event.get():
        if hasattr(event, 'key'):
            key = True
            # J1
            if (event.key == K_DOWN):
                action[1] = -1
            elif event.key == K_UP:
                action[1] = 1
            # J2
            elif (event.key == K_LEFT):
                action[0] = -1
            elif event.key == K_RIGHT:
                action[0] = 1
            # J3
            elif event.key == K_SPACE:
                action[2] = 1

            elif event.key == K_q:
                stop = True
            if action.sum() != 0:
                time.sleep(0.05)
                break

    out = env.step(action)

    # Temporal Logic
    new_descr = sample_descriptions_from_state(env.unwrapped)
    past_descr_buffer, new_descr = tl.update_descrs_with_temporal_logic(new_descr, past_descr, past_descr_buffer)
    current_descr = past_descr_buffer + new_descr

    env.render()

    past_descr = new_descr


    if key:
        # Sample descriptions of the current state
        key = False
        print('-----------')
        pprint([ddd for ddd in current_descr if 'grasp' in ddd or 'grow' in ddd])

    # assert that the reward function works, should give positive rewards for descriptions sampled, negative for others.
    # for d in descr:
    #     assert get_reward_from_state(out[0], d, env_params)
    # for d in np.random.choice(list(set(all_descriptions) - set(descr)), size=20):
    #     assert not get_reward_from_state(out[0], d, env_params)
