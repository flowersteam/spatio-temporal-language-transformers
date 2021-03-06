import random
from collections import OrderedDict, deque

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Batch(object):
    def __init__(self, states, state_idx_buffer, batch_size, id2one_hot, proportion, use_flat_buffer=False):
        self.states = states
        self.buffer = state_idx_buffer
        self.id2one_hot = id2one_hot
        self.batch_size = batch_size
        self.memory = 0
        self.no_more_batch = False
        self.proportion = proportion
        if use_flat_buffer:
            self.flat_buffer = self._flatten_buffer()

    def _flatten_buffer(self):
        flat_buffer = dict(zip(self.buffer.keys(), [[] for _ in range(len(self.buffer))]))
        for ind in self.buffer.keys():
            if self.buffer[ind]['pos_reward']:

                for pos_ind in self.buffer[ind]['pos_reward']:
                    flat_buffer[ind].append((pos_ind, 1))
                for neg_ind in self.buffer[ind]['neg_reward']:
                    flat_buffer[ind].append((neg_ind, 0))

                random.shuffle(flat_buffer[ind])

        return flat_buffer

    def next_batch(self):
        if self.batch_size <= len(self.buffer.keys()):
            indices_instruction = np.random.choice(list(self.buffer.keys()), size=self.batch_size, replace=False)
        else:
            indices_instruction = list(self.buffer.keys())
            np.random.shuffle(indices_instruction)
            indices_instruction += np.random.choice(list(self.buffer.keys()),
                                                    size=self.batch_size - len(self.buffer.keys())).tolist()
        s = []
        inst = []
        r = []
        for i in indices_instruction:
            if self.buffer[i]['pos_reward'] and self.buffer[i]['neg_reward']:
                p = random.random()
                if p < self.proportion:
                    s.append(self.states[random.choice(self.buffer[i]['pos_reward'])])
                    r.append(1)
                else:
                    s.append(self.states[random.choice(self.buffer[i]['neg_reward'])])
                    r.append(0)
                inst.append(self.id2one_hot[i])

        return np.array(s), np.array(inst), np.array(r)

    def next_batch_sampled_from_distribution(self):
        if self.flat_buffer:
            if self.batch_size <= len(self.buffer.keys()):
                indices_instruction = random.sample(list(self.buffer.keys()), self.batch_size)
            else:
                indices_instruction = list(self.buffer.keys())
                random.shuffle(indices_instruction)
                for k in range(self.batch_size - len(self.buffer.keys())):
                    indices_instruction.append(random.choice(list(self.buffer.keys())))
            s = []
            inst = []
            r = []
            for i in indices_instruction:
                state_idx, reward = random.choice(self.flat_buffer[i])
                s.append(self.states[state_idx])
                r.append(reward)
                inst.append(self.id2one_hot[i])

            return np.array(s), np.array(inst), np.array(r)
        else:
            return NameError

    def next_batch_from_known_goal(self, indices_goal):
        if self.batch_size <= len(indices_goal):
            indices_instruction = random.sample(indices_goal, self.batch_size)
        else:
            indices_instruction = indices_goal
            random.shuffle(indices_instruction)
            for k in range(self.batch_size - len(indices_goal)):
                indices_instruction.append(random.choice(indices_goal))
        s = []
        inst = []
        r = []

        for i in indices_instruction:
            if self.buffer[i]['pos_reward']:
                p = random.random()
                if p < 0.5:
                    s.append(self.states[random.choice(self.buffer[i]['pos_reward'])])
                    r.append(1)
                else:
                    if self.buffer[i]['neg_reward']:
                        s.append(self.states[random.choice(self.buffer[i]['neg_reward'])])
                        r.append(0)
                    else:
                        s.append(self.states[random.choice(self.buffer[i]['pos_reward'])])
                        r.append(1)
                inst.append(self.id2one_hot[i])
            else:
                print(i)

        return np.array(s), np.array(inst), np.array(r)


class BatchTransformer(object):
    def __init__(self, states, state_idx_buffer, batch_size, id2one_hot, proportion, use_flat_buffer=False):
        self.states = states
        self.buffer = state_idx_buffer
        self.id2one_hot = id2one_hot
        self.batch_size = batch_size
        self.memory = 0
        self.no_more_batch = False
        self.proportion = proportion
        if use_flat_buffer:
            self.flat_buffer = self._flatten_buffer()

    def _flatten_buffer(self):
        flat_buffer = dict(zip(self.buffer.keys(), [[] for _ in range(len(self.buffer))]))
        for ind in self.buffer.keys():
            if self.buffer[ind]['pos_reward']:

                for pos_ind in self.buffer[ind]['pos_reward']:
                    flat_buffer[ind].append((pos_ind, 1))
                for neg_ind in self.buffer[ind]['neg_reward']:
                    flat_buffer[ind].append((neg_ind, 0))

                random.shuffle(flat_buffer[ind])

        return flat_buffer

    def next_batch(self):
        if self.batch_size <= len(self.buffer.keys()):
            indices_instruction = np.random.choice(list(self.buffer.keys()), size=self.batch_size, replace=False)
        else:
            indices_instruction = list(self.buffer.keys())
            np.random.shuffle(indices_instruction)
            indices_instruction += np.random.choice(list(self.buffer.keys()),
                                                    size=self.batch_size - len(self.buffer.keys())).tolist()
        bodies = []
        objects = []
        inst = []
        r = []
        for i in indices_instruction:
            if self.buffer[i]['pos_reward'] and self.buffer[i]['neg_reward']:
                p = random.random()
                if p < self.proportion:
                    idx_random = random.choice(self.buffer[i]['pos_reward'])
                    bodies.append(self.states[idx_random][0])
                    objects.append(self.states[idx_random][1])
                    r.append(1)
                else:
                    idx_random = random.choice(self.buffer[i]['neg_reward'])
                    bodies.append(self.states[idx_random][0])
                    objects.append(self.states[idx_random][1])
                    r.append(0)
                inst.append(self.id2one_hot[i])

        return np.array(bodies), np.array(objects), np.array(inst), np.array(r)


class BatchLSTM(object):
    def __init__(self, states, state_idx_buffer, batch_size, id2one_hot, proportion, use_flat_buffer=False):
        self.states = states
        self.buffer = state_idx_buffer
        self.id2one_hot = id2one_hot
        self.batch_size = batch_size
        self.memory = 0
        self.no_more_batch = False
        self.proportion = proportion
        if use_flat_buffer:
            self.flat_buffer = self._flatten_buffer()

    def _flatten_buffer(self):
        flat_buffer = dict(zip(self.buffer.keys(), [[] for _ in range(len(self.buffer))]))
        for ind in self.buffer.keys():
            if self.buffer[ind]['pos_reward']:

                for pos_ind in self.buffer[ind]['pos_reward']:
                    flat_buffer[ind].append((pos_ind, 1))
                for neg_ind in self.buffer[ind]['neg_reward']:
                    flat_buffer[ind].append((neg_ind, 0))

                random.shuffle(flat_buffer[ind])

        return flat_buffer

    def next_batch(self):
        if self.batch_size <= len(self.buffer.keys()):
            indices_instruction = np.random.choice(list(self.buffer.keys()), size=self.batch_size, replace=False)
        else:
            indices_instruction = list(self.buffer.keys())
            np.random.shuffle(indices_instruction)
            indices_instruction += np.random.choice(list(self.buffer.keys()),
                                                    size=self.batch_size - len(self.buffer.keys())).tolist()
        states = []
        inst = []
        r = []
        for i in indices_instruction:
            if self.buffer[i]['pos_reward'] and self.buffer[i]['neg_reward']:
                p = random.random()
                if p < self.proportion:
                    idx_random = random.choice(self.buffer[i]['pos_reward'])
                    states.append(self.states[idx_random])

                    r.append(1)
                else:
                    idx_random = random.choice(self.buffer[i]['neg_reward'])
                    states.append(self.state[idx_random])
                    r.append(0)
                inst.append(self.id2one_hot[i])

        return np.array(states), np.array(inst), np.array(r)


def _split_dict(d, n_states):
    split_tresh = int(len(d) - n_states)

    d1 = dict(list(d.items())[:split_tresh])
    d2 = dict(list(d.items())[split_tresh:])
    return d1, d2


class StateIdBuffer(object):
    '''
    This class indexes observations into the dictionnary states.
    It stores the state_ids of positive and negative rewards for each goal_id in a dictionnary of the form
    {goal_id:{pos_reward:[state_id], neg_reward:[state_id]}

    :param max_len: Maximum size of the states memory
    '''

    def __init__(self, max_len):

        self.states = OrderedDict()
        self.state_id_buffer = dict()
        self.max_len = max_len
        self.current_state_id = 0

    def _add(self, state):
        self.states[self.current_state_id] = state
        self.current_state_id += 1

    def _pop(self):
        idx = next(iter(self.states))
        self.states.pop(idx)
        return idx

    def _update_states(self, state):
        self._add(state)

        if len(self.states) > self.max_len:
            return self._pop()
        else:
            return None

    def update(self, state, goal_reached_ids, goal_not_reached_ids):
        """
        Add state to states dict memory and removes the left state of memory (FIFO)
        Add state index to buffer according to goals_reached_ids

        :param state:
        :param goal_reached_ids:
        :return:
        """
        # Adding new Elements in the Buffer
        for goal_id in goal_reached_ids:
            if goal_id not in self.state_id_buffer.keys():
                self.state_id_buffer[goal_id] = dict(pos_reward=deque([self.current_state_id]), neg_reward=deque())
            else:
                self.state_id_buffer[goal_id]['pos_reward'].append(self.current_state_id)

        if goal_not_reached_ids:
            for goal_id in goal_not_reached_ids:
                if goal_id not in self.state_id_buffer.keys():
                    self.state_id_buffer[goal_id] = dict(pos_reward=deque([self.current_state_id]), neg_reward=deque())
                else:
                    self.state_id_buffer[goal_id]['neg_reward'].append(self.current_state_id)

        # Updating the dictionnary of states
        id_state_to_remove = self._update_states(state)

        # Removing the state_id of pos_reward and neg_reward for all the goal_id
        if id_state_to_remove is not None:
            for goal_id in self.state_id_buffer.keys():
                if id_state_to_remove in self.state_id_buffer[goal_id]['pos_reward']:
                    self.state_id_buffer[goal_id]['pos_reward'].popleft()
                if id_state_to_remove in self.state_id_buffer[goal_id]['neg_reward']:
                    self.state_id_buffer[goal_id]['neg_reward'].popleft()

    def split_buffer(self, n_states):
        '''
        Be carefull here the format of the training and testing buffers are differents

        :param proportion_test:
        :return: dict() states_train
                 dict() state_id_buffer_train {goal_id: {pos_reward: [state_ids], neg_reward: [state_ids]}}
                 dict() states_test
                 dict() state_id_buffer_test {goal_id: [(state_id, reward)]}
        '''
        states_train, states_test = _split_dict(self.states, n_states)
        keys_train = states_train.keys()
        keys_test = states_test.keys()
        goal_ids = self.state_id_buffer.keys()
        state_id_buffer_train = OrderedDict(zip(goal_ids, [dict() for _ in range(len(goal_ids))]))
        state_id_buffer_test = OrderedDict()
        for goal_id in goal_ids:
            pos_reward_state_ids = self.state_id_buffer[goal_id]['pos_reward']
            neg_reward_state_ids = self.state_id_buffer[goal_id]['neg_reward']
            state_id_buffer_train[goal_id]['pos_reward'] = [state_id for state_id in pos_reward_state_ids if
                                                            state_id in keys_train]
            state_id_buffer_train[goal_id]['neg_reward'] = [state_id for state_id in neg_reward_state_ids if
                                                            state_id in keys_train]

            state_id_buffer_test[goal_id] = [(state_id, 1) for state_id in pos_reward_state_ids if
                                             state_id in keys_test] + \
                                            [(state_id, 0) for state_id in neg_reward_state_ids if
                                             state_id in keys_test]

        return states_train, state_id_buffer_train, states_test, state_id_buffer_test


