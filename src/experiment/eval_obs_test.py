import pickle
import json
import os
import os.path as op
import sys
import re
import pathlib
import torch
import logging
import argparse
import pandas as pd
import psutil

sys.path.append('../..')
sys.path.append('../../..')

# add src module to python path
src_path = pathlib.Path(__file__).parent.parent.parent.parent
print(src_path)
sys.path.append(str(src_path))

import src.models.transformer_model as trm
import src.models.lstm_model as lsm
from src.utils.util import pickle_dump, json_dump, parse_bool, setup_logger, set_global_seeds
from src.grammar.descriptions import get_all_descriptions, generate_systematic_split
from src.experiment.utils import evaluate_test_metrics, write_f1_to_df, write_f1_type_to_df, compute_metrics, compute_metric_by_type, \
    append_value_to_dict_list, mean_train_metrics_over_steps, nparams, get_eval_idx, load_model_ckpt, \
    save_model_ckpt

# fix pickle import
from src.utils import nlp_tools

sys.modules['src.train.nlp_tools'] = nlp_tools

print('CUDA is available ? ', torch.cuda.is_available())

# Method to setup several loggers to monitor memory usage
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--dataset_dir', type=str, help='name of dataset folder in the processed directory of data',
        default='../../data/processed/temporal_300k_pruned_random_split/')
    add('--run_dir', type=str, default='path_to_run_dir')
    add('--run_idx', type=int, default='333', help='Trial identifier, name of the saving folder')
    add('--git_commit', type=str, help='Hash of git commit', default='no git commit')
    add('--use_cuda', type=parse_bool, help='Whether to use cuda', default=True)
    add('--size_data_test', type=int, default=5000)
    add('--batch_size', type=int, default=512)
    add('--seed', type=int, default=0)

    # non-used args
    add('--architecture', default='transformer_ut')
    add('--n_steps', type=int, help='Number of training steps', default=150000)
    add('--positive_ratio', type=float, help='Ratio of positive rewards per descriptions', default=0.1)
    add('--evaluate', type=str, help='whether to evaluate or not', default='yes')
    add('--freq_eval', type=int, help='Frequency of evaluation during training', default=50000)
    add('--freq_log_train', type=int, help='Frequency of train metric logging', default=1000)
    add('--num_heads', type=int, default=8)
    add('--layers', type=int, default=4)
    add('--hidden_size', type=int, default=128)
    add('--learning_rate', type=float, default=1e-4)
    add('--dropout', type=float, default=0.1)
    # debug does one forward pass and log mem usage
    add('--mode', default='train', choices=['debug', 'train', 'test'])
    add('--load', default=True, type=parse_bool, help='Whether to reload if existing save '
                                                      'is found, defaults to True')

    # Params parsing
    # params = vars(parser.parse_known_args([
    #     '--run_dir',
    #     '--run_idx',
    #     '--git_commit',
    #     '--use_cuda',
    #     '--size_data_test',
    #     '--batch_size',
    #     '--seed',
    # ]))

    params = vars(parser.parse_args())
    run_idx = params['run_idx']
    # git_commit = params['git_commit']
    use_cuda = params['use_cuda']
    seed = params['seed']
    size_data_test = params['size_data_test']

    set_global_seeds(seed)
    run_dir = params['run_dir']
    save_dir = run_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_filename = op.join(save_dir, 'log_obs_test.log')
    setup_logger('info_logger', log_filename)
    logger = logging.getLogger('info_logger')

    memory_log_filename = op.join(save_dir, 'memory_log_obst_test.log')
    setup_logger('mem_logger', memory_log_filename)
    mem_logger = logging.getLogger('mem_logger')

    gpu_memory_log_filename = op.join(save_dir, 'gpu_memory_log_obs_test.log')
    setup_logger('gpu_mem_logger', gpu_memory_log_filename)
    gpu_mem_logger = logging.getLogger('gpu_mem_logger')
    print(f'gpu filename {gpu_memory_log_filename}')

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # # Data reading

    mem_logger.info('RAM memory % used before data reading: ' + str(psutil.virtual_memory()[2]))

    with open(op.join(params['dataset_dir'], 'descriptions_data.pk'), 'rb') as f:
        descriptions_data = pickle.load(f)

    id2one_hot = descriptions_data['id2one_hot']
    id2description = descriptions_data['id2description']
    vocab = descriptions_data['vocab']
    max_seq_length = descriptions_data['max_seq_length']

    with open(op.join(params['dataset_dir'], 'obs_test_set.pk'), 'rb') as f:
        test_set = pickle.load(f)

    state_idx_buffer_test = test_set['state_idx_buffer']
    states_test = test_set['states']

    all_descr = get_all_descriptions()

    mem_logger.info('RAM memory % used after data reading:' + str(psutil.virtual_memory()[2]))

    # Model init

    body_size = states_test[0][0].shape[2]
    obj_size = states_test[0][1].shape[2]
    voc_size = vocab.size
    seq_length = max_seq_length

    with open(op.join(run_dir, 'params.json'), 'rb') as fp:
        conf = json.load(fp)
    batch_size = conf['batch_size']

    print(f'Architecture is {conf["architecture"]}')
    logger.info(f'Architecture is {conf["architecture"]}')

    if conf['architecture'] == 'transformer_ut':
        model_type = trm.Transformer_UT
        word_aggreg = False
    elif conf['architecture'] == 'transformer_ut_wa':
        model_type = trm.Transformer_UT
        word_aggreg = True
    elif conf['architecture'] == 'transformer_sft':
        model_type = trm.SpatialFirstTransformer
        word_aggreg = False
    elif conf['architecture'] == 'transformer_sft_wa':
        model_type = trm.SpatialFirstTransformer
        word_aggreg = True
    elif conf['architecture'] == 'transformer_tft':
        model_type = trm.TemporalFirstTransformer
        word_aggreg = False
    elif conf['architecture'] == 'transformer_tft_wa':
        model_type = trm.TemporalFirstTransformer
        word_aggreg = True
    elif conf['architecture'] == 'lstm_factored':
        model_type = lsm.FactoredLSTM
        word_aggreg = False
    elif conf['architecture'] == 'lstm_flat':
        model_type = lsm.FlatLSTM
        word_aggreg = False
    else:
        raise NotImplementedError('Chose a valid model')

    reward_func = model_type(
        body_size=body_size,
        obj_size=obj_size,
        voc_size=voc_size,
        seq_length=seq_length,
        hidden_size=conf['hidden_size'],
        num_heads=conf['num_heads'],
        num_layers=conf['layers'],
        device=device,
        word_aggreg=word_aggreg,
    )

    n = nparams(reward_func)
    print(f'Model has {n} parameters')
    logger.info(f'Model has {n} parameters')

    mem_logger.info('RAM memory % used after model init: ' + str(psutil.virtual_memory()[2]))

    if use_cuda:
        reward_func = reward_func.to(device)
    ckpt_dir = op.join(save_dir, 'model_ckpt')


    def get_model_path(ckpt_dir):
        idx = 0
        ckpt_file = None
        for model_file in os.listdir(ckpt_dir):
            idx_tmp = int(re.findall(r'\d+', model_file)[0])
            if idx_tmp > idx:
                idx = idx_tmp
                ckpt_file = model_file
        return op.join(ckpt_dir, ckpt_file)


    model_path = get_model_path(ckpt_dir)
    if model_path is not None:
        reward_func.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError('Could not find model')

    metric_test_df = pd.DataFrame()
    metric_test_by_type_df = pd.DataFrame()

    mem_logger.info('RAM memory % used after batch object creation: ' + str(psutil.virtual_memory()[2]))

    type_train_descr_dict, _, _, _ = load_dumped_split(params['dataset_dir'])
    metric_dict_test = evaluate_test_metrics(reward_func,
                                             test_set['state_idx_buffer'],
                                             test_set['states'],
                                             id2one_hot, size_dataset=size_data_test,
                                             proportion_pos_reward=0.2,
                                             batch_size=batch_size,
                                             logging=logger,
                                             use_cuda=use_cuda,
                                             eval_idx=0)

    metric_dict_test_by_type = compute_metric_by_type(metric_dict_test, test_set['id2description'],
                                                      type_train_descr_dict, logger)

    metric_test_df = write_f1_to_df(metric_test_df, metric_dict_test, test_set['id2description'])
    metric_test_by_type_df = write_f1_type_to_df(metric_test_by_type_df, metric_dict_test_by_type)

    metric_test_df.to_csv(op.join(save_dir, 'f1_test_obs_test.csv'))
    metric_test_by_type_df.to_csv(op.join(save_dir, 'f1_test_by_type_obs_test.csv'))

    stop = 0
