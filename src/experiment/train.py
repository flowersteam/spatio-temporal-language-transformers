import pickle
import json
import os
import os.path as op
import sys
import time
from datetime import datetime as dt

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

from src.experiment.batch import BatchTransformer

from src.utils.util import pickle_dump, json_dump, parse_bool, setup_logger, set_global_seeds
from src.grammar.descriptions import get_all_descriptions, generate_systematic_split
from src.experiment.utils import evaluate_test_metrics, write_f1_to_df, write_f1_type_to_df, compute_metrics, compute_metric_by_type, \
    append_value_to_dict_list, mean_train_metrics_over_steps, nparams, get_eval_idx, load_model_ckpt, \
    save_model_ckpt, load_dumped_split

from src.utils import nlp_tools

sys.modules['src.train.nlp_tools'] = nlp_tools

print('CUDA is available ? ', torch.cuda.is_available())

def datetime_to_formatted_str(datetime):
    return f"{datetime.seconds // 3600}:{datetime.seconds // 60}:" \
           f"{datetime.seconds}"

# Method to setup several loggers to monitor memory usage
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add(
        '--architecture',
        type=str,
        help='Type of architecture to train',
        default='transformer_ut',
        choices=[
            'lstm_factored',
            'lstm_flat',
            'transformer_ut',
            'transformer_ut_wa',
            'transformer_sft',
            'transformer_sft_wa',
            'transformer_tft',
            'transformer_tft_wa',
        ]
    )
    add('--n_steps', type=int, help='Number of training steps', default=150000)
    add('--positive_ratio', type=float, help='Ratio of positive rewards per descriptions', default=0.1)

    add('--dataset_dir', type=str, help='name of dataset folder in the processed directory of data',
        default='../../data/temporal_300k_pruned_random_split/')
    add('--save_dir', type=str, help='directory in which to save the data run')
    add('--run_idx', type=int, default='333', help='Trial identifier, name of the saving folder')
    add('--git_commit', type=str, help='Hash of git commit', default='no git commit')
    add('--evaluate', type=str, help='whether to evaluate or not', default='yes')
    add('--freq_eval', type=int, help='Frequency of evaluation during training', default=100000)
    add('--freq_log_train', type=int, help='Frequency of train metric logging', default=1000)
    add('--use_cuda', type=parse_bool, help='Whether to use cuda', default=True)
    add('--num_heads', type=int, default=8)
    add('--layers', type=int, default=4)
    add('--batch_size', type=int, default=512)
    add('--hidden_size', type=int, default=128)
    add('--learning_rate', type=float, default=1e-4)
    add('--dropout', type=float, default=0.1)
    add('--seed', type=int, default=0)
    # debug does one forward pass and log mem usage
    add('--mode', default='train', choices=['debug', 'train', 'test'])
    add('--size_data_test', type=int, default=8000)
    add('--load', default=True, type=parse_bool, help='Whether to reload if existing save '
                                                      'is found, defaults to True')
    add('--auto_hparams', type=parse_bool, default=False)

    # Params parsing
    params = vars(parser.parse_args())

    # auto hparam fixing
    if params['auto_hparams']:
        if params['architecture'] == 'lstm_factored':
            params['hidden_size'] = 512
            params['layers'] = 4
            # params['num_heads'] =
            params['learning_rate'] = 1e-4
        if params['architecture'] == 'lstm_flat':
            params['hidden_size'] = 512
            params['layers'] = 4
            # params['num_heads'] =
            params['learning_rate'] = 1e-4
        if params['architecture'] == 'transformer_ut':
            params['hidden_size'] = 256
            params['layers'] = 4
            params['num_heads'] = 8
            params['learning_rate'] = 1e-4
        if params['architecture'] == 'transformer_ut':
            params['hidden_size'] = 256
            params['layers'] = 4
            params['num_heads'] = 8
            params['learning_rate'] = 1e-4
        if params['architecture'] == 'transformer_ut_wa':
            params['hidden_size'] = 512
            params['layers'] = 4
            params['num_heads'] = 8
            params['learning_rate'] = 1e-5
        if params['architecture'] == 'transformer_sft':
            params['hidden_size'] = 256
            params['layers'] = 4
            params['num_heads'] = 4
            params['learning_rate'] = 1e-4
        if params['architecture'] == 'transformer_sft_wa':
            params['hidden_size'] = 256
            params['layers'] = 2
            params['num_heads'] = 8
            params['learning_rate'] = 1e-4
        if params['architecture'] == 'transformer_tft':
            params['hidden_size'] = 256
            params['layers'] = 4
            params['num_heads'] = 4
            params['learning_rate'] = 1e-4
        if params['architecture'] == 'transformer_tft_wa':
            params['hidden_size'] = 512
            params['layers'] = 4
            params['num_heads'] = 8
            params['learning_rate'] = 1e-5

    n_steps = params['n_steps']
    positive_ratio = params['positive_ratio']
    run_idx = params['run_idx']
    git_commit = params['git_commit']
    evaluate = params['evaluate']
    freq_eval = params['freq_eval']
    use_cuda = params['use_cuda']
    seed = params['seed']
    size_data_test = params['size_data_test']

    set_global_seeds(seed)

    save_dir = params['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_filename = op.join(save_dir, 'log.log')
    setup_logger('info_logger', log_filename)
    logger = logging.getLogger('info_logger')

    memory_log_filename = op.join(save_dir, 'memory_log.log')
    setup_logger('mem_logger', memory_log_filename)
    mem_logger = logging.getLogger('mem_logger')

    gpu_memory_log_filename = op.join(save_dir, 'gpu_memory_log.log')
    setup_logger('gpu_mem_logger', gpu_memory_log_filename)
    gpu_mem_logger = logging.getLogger('gpu_mem_logger')
    print(f'gpu filename {gpu_memory_log_filename}')

    if params['save_dir'] is None:
        raise ValueError('Please provide a save directory with the save_dir option')

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

    with open(op.join(params['dataset_dir'], 'train_set.pk'), 'rb') as f:
        train_set = pickle.load(f)
    state_idx_buffer = train_set['state_idx_buffer']
    states_train = train_set['states']

    with open(op.join(params['dataset_dir'], 'test_set.pk'), 'rb') as f:
        test_set = pickle.load(f)

    with open(op.join(params['dataset_dir'], 'test_type_dict.json'), 'r') as f:
        test_type_dict = json.load(f)

    all_descr = get_all_descriptions()

    mem_logger.info('RAM memory % used after data reading:' + str(psutil.virtual_memory()[2]))

    json_dump(test_type_dict, op.join(save_dir, 'test_type_dict.json'))
    json_dump(id2description, op.join(save_dir, 'id2description.json'))

    # Model init

    body_size = states_train[0][0].shape[2]
    obj_size = states_train[0][1].shape[2]
    voc_size = vocab.size
    seq_length = max_seq_length

    hidden_size = params['hidden_size']
    num_heads = params['num_heads']
    num_layers = params['layers']

    n_batch = 100
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']

    print(f'Architecture is {params["architecture"]}')
    logger.info(f'Architecture is {params["architecture"]}')

    if params['architecture'] == 'transformer_ut':
        model_type = trm.Transformer_UT
        word_aggreg = False
    elif params['architecture'] == 'transformer_ut_wa':
        model_type = trm.Transformer_UT
        word_aggreg = True
    elif params['architecture'] == 'transformer_sft':
        model_type = trm.SpatialFirstTransformer
        word_aggreg = False
    elif params['architecture'] == 'transformer_sft_wa':
        model_type = trm.SpatialFirstTransformer
        word_aggreg = True
    elif params['architecture'] == 'transformer_tft':
        model_type = trm.TemporalFirstTransformer
        word_aggreg = False
    elif params['architecture'] == 'transformer_tft_wa':
        model_type = trm.TemporalFirstTransformer
        word_aggreg = True
    elif params['architecture'] == 'lstm_factored':
        model_type = lsm.FactoredLSTM
        word_aggreg = False
    elif params['architecture'] == 'lstm_flat':
        model_type = lsm.FlatLSTM
        word_aggreg = False
    else:
        raise NotImplementedError('Chose a valid model')

    reward_func = model_type(
        body_size=body_size,
        obj_size=obj_size,
        voc_size=voc_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=params['dropout'],
        device=device,
        word_aggreg=word_aggreg,
    )

    n = nparams(reward_func)
    print(f'Model has {n} parameters')
    logger.info(f'Model has {n} parameters')

    mem_logger.info('RAM memory % used after model init: ' + str(psutil.virtual_memory()[2]))

    # cuda # TODO check we can remove it, device is given at model creation
    if use_cuda:
        reward_func = reward_func.to(device)

    with open(op.join(save_dir, 'params.json'), 'w') as fp:
        json.dump(params, fp)

    optimizer = torch.optim.Adam(params=reward_func.parameters(), lr=learning_rate)

    # get starting index from model checkpoints
    pathlib.Path(op.join(save_dir, 'model_ckpt')).mkdir(parents=True, exist_ok=True)

    eval_start = 0
    if params['load']:
        idx = get_eval_idx(save_dir)
        if idx == -1:
            eval_start = 0
        else:
            eval_start = idx * freq_eval
            reward_func = load_model_ckpt(reward_func, save_dir, idx)

    csvpath = pathlib.Path(op.join(save_dir, 'f1_test.csv'))
    csvpath_type = pathlib.Path(op.join(save_dir, 'f1_test_by_type.csv'))
    if params['load'] and eval_start != 0:
        if csvpath.exists():
            metric_test_df = pd.read_csv(str(csvpath))
            metric_test_by_type_df = pd.read_csv(str(csvpath_type))
        else:
            metric_test_df = pd.DataFrame()
            metric_test_by_type_df = pd.DataFrame()
    else:
        metric_test_df = pd.DataFrame()
        metric_test_by_type_df = pd.DataFrame()

    reward_func.train()
    batch = BatchTransformer(states_train, state_idx_buffer, batch_size, id2one_hot, positive_ratio)

    mem_logger.info('RAM memory % used after batch object creation: ' + str(psutil.virtual_memory()[2]))

    train_metrics_keys = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    train_metrics_dict_list = {k: [] for k in train_metrics_keys}
    mean_train_metrics_dict_list = {'mean_' + k: [] for k in train_metrics_keys}

    t0 = time.time()
    start_dt = dt.fromtimestamp(t0)

    for i in range(eval_start, n_steps):
        # mean_time = 0
        # t0 = time.time()
        # TODO test this works fine
        batch_b, batch_obj, batch_descr, batch_r = batch.next_batch()

        # TODO include this in next_batch() also add reshape of bathc_descr
        batch_b = torch.tensor(batch_b, dtype=torch.float32)
        batch_obj = torch.tensor(batch_obj, dtype=torch.float32)
        batch_descr = torch.tensor(batch_descr, dtype=torch.float32)
        batch_r = torch.tensor(batch_r, dtype=torch.float32)

        if use_cuda:
            batch_b = batch_b.to(device)
            batch_obj = batch_obj.to(device)
            batch_descr = batch_descr.to(device)
            batch_r = batch_r.to(device)

        batch_pred = reward_func(batch_obj, batch_b, batch_descr).squeeze()
        eps = 1e-7
        loss = torch.mean(
            -torch.log(batch_pred + eps) * batch_r + (1 - batch_r) * (-torch.log(1 - batch_pred + eps)))

        accuracy, precision, recall, f1 = compute_metrics(batch_pred, batch_r)
        tmp_train_metrics = {'loss': loss.item(), 'accuracy': accuracy, 'precision': precision, 'recall': recall,
                             'f1': f1}
        train_metrics_dict_list = append_value_to_dict_list(train_metrics_dict_list, tmp_train_metrics)

        if i % params['freq_log_train'] == 0 and i != 0:

            mean_tmp_train_metrics = mean_train_metrics_over_steps(train_metrics_dict_list)
            mean_train_metrics_dict_list = append_value_to_dict_list(mean_train_metrics_dict_list,
                                                                     mean_tmp_train_metrics)
            logger.info(
                'Step: ' + str(i) + ';  ' + ' '.join([k + ': ' + str(v) for k, v in mean_tmp_train_metrics.items()]))
            mem_logger.info('RAM memory % used at step {}: '.format(i) + str(psutil.virtual_memory()[2]))

            if use_cuda and torch.cuda.is_available():
                gpu_mem_stats = torch.cuda.memory_stats()
                gmem = gpu_mem_stats['active_bytes.all.peak'] / (1024 ** 2)
                gpu_mem_logger.info(f'Peak GPU RAM memory active at step {i}: '
                                    f'{gmem}MiB')

            pickle_dump(mean_train_metrics_dict_list, save_dir + '/loss_log.pk')
            train_metrics_dict_list = {k: [] for k in train_metrics_keys}

        optimizer.zero_grad()
        loss.backward()

        # for test purposes: ram and gpu ram use
        if i == 0 and params['mode'] == 'debug':
            if use_cuda and torch.cuda.is_available():
                with open(op.join(save_dir, 'memory_dump'), 'w') as f:
                    memalloc = torch.cuda.memory_allocated()
                    memalloc_h = memalloc / (1024 ** 2)
                    totalmem = torch.cuda.get_device_properties(0).total_memory
                    totalmem_h = totalmem / (1024 ** 2)
                    prop = memalloc / totalmem

                    f.write('During the forward pass:')
                    f.write(f'Allocated {memalloc_h}MiB ({memalloc} bytes) of '
                            f'{totalmem_h}MiB ({totalmem} bytes); {prop}%')
                    f.write('\n')
                    f.write(torch.cuda.memory_summary(0))
            sys.exit(0)

        optimizer.step()

        if i % freq_eval == 0 and i != eval_start:

            if evaluate == 'yes':

                elapsed_dt = dt.now() - start_dt
                start_dt = dt.now()
                logger.info(f'Time elapsed since last eval: '
                            f'{datetime_to_formatted_str(elapsed_dt)}')


                eval_idx = i // freq_eval
                mem_logger.info('RAM memory % used before evaluation: ' + str(psutil.virtual_memory()[2]))
                metric_dict_test = evaluate_test_metrics(reward_func,
                                                         test_set['state_idx_buffer'],
                                                         test_set['states'],
                                                         id2one_hot, size_dataset=size_data_test,
                                                         proportion_pos_reward=0.2,
                                                         batch_size=batch_size,
                                                         logging=logger,
                                                         use_cuda=use_cuda,
                                                         eval_idx=eval_idx)
                metric_dict_test_by_type = compute_metric_by_type(metric_dict_test, test_set['id2description'],
                                                                  test_type_dict, logger)

                metric_test_df = write_f1_to_df(metric_test_df, metric_dict_test, test_set['id2description'])
                metric_test_by_type_df = write_f1_type_to_df(metric_test_by_type_df, metric_dict_test_by_type)

                metric_test_df.to_csv(op.join(save_dir, 'f1_test.csv'))
                metric_test_by_type_df.to_csv(op.join(save_dir, 'f1_test_by_type.csv'))

                reward_func.train()
                mem_logger.info('RAM memory % used after evaluation: ' + str(psutil.virtual_memory()[2]))

                # save model checkpoint
                save_model_ckpt(reward_func, save_dir, eval_idx)

    i += 1
    if evaluate == 'yes':
        # eval_idx = i // freq_eval

        elapsed_dt = dt.now() - start_dt
        start_dt = dt.now()
        logger.info(f'Time elapsed since last eval: '
                    f'{datetime_to_formatted_str(elapsed_dt)}')

        try:
            eval_idx += 1
        except NameError:
            if params['load']:
                eval_idx = get_eval_idx(save_dir) + 1
            else:
                eval_idx = 1
        mem_logger.info('RAM memory % used before evaluation: ' + str(psutil.virtual_memory()[2]))
        metric_dict_test = evaluate_test_metrics(reward_func,
                                                 test_set['state_idx_buffer'],
                                                 test_set['states'],
                                                 id2one_hot, size_dataset=size_data_test,
                                                 proportion_pos_reward=0.2,
                                                 batch_size=batch_size,
                                                 logging=logger,
                                                 use_cuda=use_cuda,
                                                 eval_idx=eval_idx)
        metric_dict_test_by_type = compute_metric_by_type(metric_dict_test, test_set['id2description'],
                                                          test_type_dict, logger)

        metric_test_df = write_f1_to_df(metric_test_df, metric_dict_test, test_set['id2description'])
        metric_test_by_type_df = write_f1_type_to_df(metric_test_by_type_df, metric_dict_test_by_type)

        metric_test_df.to_csv(op.join(save_dir, 'f1_test.csv'))
        metric_test_by_type_df.to_csv(op.join(save_dir, 'f1_test_by_type.csv'))

        # IMPORTANT Remember to set back model to train mode after evaluation (this was already performed in evaluate_final-state_metri
        reward_func.train()
        mem_logger.info('RAM memory % used after evaluation: ' + str(psutil.virtual_memory()[2]))

        save_model_ckpt(reward_func, save_dir, eval_idx)
