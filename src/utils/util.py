import numpy as np
import random
import torch
import os
import pickle
import json

import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def parse_bool(bool_arg):
    """
    Parse a string representing a boolean.
    :param bool_arg: The string to be parsed
    :return: The corresponding boolean
    """
    if bool_arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif bool_arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean argument expected. Got {bool_arg} instead.')


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)


def find_save_path(dir, trial_id):
    i = 0
    while True:
        save_dir = dir + str(trial_id + i * 100) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
        i += 1
    return save_dir


def json_dump(dict, file):
    with open(file, 'w') as fp:
        json.dump(dict, fp)


def pickle_dump(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
