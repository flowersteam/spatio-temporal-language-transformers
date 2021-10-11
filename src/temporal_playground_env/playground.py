# import gym
# import numpy as np
# import pygame
#
# from gym import spaces
#
# from src.temporal_playground_env.objects import generate_objects
# from src.temporal_playground_env.env_params import get_env_params
#
# class Playground(gym.Env):
#     metadata = {
#         'render.modes': ['human', 'rgb_array'],
#         'video.frames_per_second': 30
#     }
#
#     def __init__(self,
#                  max_timesteps=50,
#                  random_init=False,
#                  human=False,
#                  reward_screen=False,
#                  viz_data_collection=False,
#
#                  agent_step_size=0.15,
#                  agent_initial_pos=(0,0),
#                  agent_initial_pos_range=0.6,
#                  max_nb_objects=3,  # number of objects in the scene
#                  random_nb_obj=False,
#                  admissible_actions=('Move', 'Grasp', 'Grow'),  # which types of actions are admissible
#                  admissible_attributes=('colors', 'categories', 'types'),#, 'relative_sizes', 'shades', 'relative_shades', 'sizes', 'relative_positions'),
#                  # which object attributes
#                  # can be used
#                  min_max_sizes=((0.2, 0.25), (0.25, 0.3)),  # ranges of sizes of objects (small and large ones)
#                  agent_size=0.05,  # size of the agent
#                  epsilon_initial_pos=0.3,  # epsilon to sample initial positions
#                  screen_size=800,  # size of the visualization screen
#                  next_to_epsilon=0.3,  # define the area to qualify an object as 'next to' another.
#                  attribute_combinations=False,
#                  obj_size_update=0.04,
#                  render_mode=False
#                  ):
#
#         self.params = get_env_params(max_nb_objects=max_nb_objects,
#                                      admissible_actions=admissible_actions,
#                                      admissible_attributes=admissible_attributes,
#                                      min_max_sizes=min_max_sizes,
#                                      agent_size=agent_size,
#                                      epsilon_initial_pos=epsilon_initial_pos,
#                                      screen_size=screen_size,
#                                      next_to_epsilon=next_to_epsilon,
#                                      attribute_combinations=attribute_combinations,
#                                      obj_size_update=obj_size_update,
#                                      render_mode=render_mode
#                                      )
#
#     def reset(self):
#         if self.random_nb_obj:
#             self.nb_obj = np.random.randint(2, self.max_nb_objects)
#             self.half_dim_obs = self.nb_obj * self.dim_obj + self.dim_body
#             self.dim_obs = int(2 * self.half_dim_obs)
#
#         self.first_action = False
#         self.logits_concat = (0 for _ in range(self.nb_obj))
#         self.SP_feedback = False
#         self.known_goals_update = False
#         return self.reset_scene()
#
#     def reset_scene(self, objects=None):
#
#         self.agent_pos = self.agent_initial_pos
#
#         if self.random_init:
#             self.agent_pos += np.random.uniform(-self.agent_initial_pos_range, self.agent_initial_pos_range, 2)
#             self.gripper_state = np.random.choice([-1, 1])
#         else:
#             self.gripper_state = -1
#
#         self.objects = self.sample_objects(objects)
#
#         # Print objects
#         self.object_grasped = False
#         for obj in self.objects:
#             self.object_grasped = obj.update_state(self.agent_pos,
#                                                    self.gripper_state > 0,
#                                                    self.objects,
#                                                    self.object_grasped,
#                                                    np.zeros([self.dim_act]))
#
#
#         # construct vector of observations
#         self.observation = np.zeros(self.dim_obs)
#         self.observation[:self.half_dim_obs] = self.observe()
#         self.initial_observation = self.observation[:self.half_dim_obs].copy()
#         self.env_step = 0
#         self.done = False
#         return self.observation.copy()
#
#     def sample_objects(self, objects_to_add):
#         object_descr = objects_to_add if objects_to_add is not None else []
#         while len(object_descr) < self.nb_obj:
#             object = dict()
#             for k in self.adm_abs_attributes:
#                 object[k] = np.random.choice(self.attributes[k])
#             object_descr.append(object)
#         object_descr = self.complete_and_check_objs(object_descr)
#         objects_ids = [self.get_obj_identifier(o) for o in object_descr]
#         objects = generate_objects(object_descr, self.params)
#         return objects