import numpy as np
import torch
import os
import pandas as pd
import pickle
import json
import os.path as op
import re
import pathlib



def nparams(model):
    return sum([p.numel() for p in model.parameters()])


def get_eval_idx(save_dir):
    model_dir = op.join(save_dir, 'model_ckpt')
    filenames = os.listdir(model_dir)
    pattern = r'^ckpt([0-9]+)\.pth$'
    valid = [
        (f, re.search(pattern, f)[1]) for f in filenames \
        if re.search(pattern, f)
    ]
    return max([int(i) for _, i in valid] + [-1])


def load_model_ckpt(model, save_dir, idx=None):
    model_dir = op.join(save_dir, 'model_ckpt')

    if idx is None:
        filenames = os.listdir(model_dir)
        pattern = r'^ckpt([0-9]+)\.pth$'
        valid = [
            (f, re.search(pattern, f)[1]) for f in filenames \
            if re.search(pattern, f)
        ]
        idx = max([int(i) for _, i in valid] + [-1])
        if idx == -1:
            print('No model to load, skipping')
            return model

    print(f'Loading model checkpoint ckpt{idx}.pth')
    load_path = op.join(model_dir, f'ckpt{idx}.pth')
    model.load_state_dict(torch.load(load_path))
    return model


def load_dumped_split(data_dir_path, ):
    with open(op.join(data_dir_path, 'train_type_dict.json'), 'r') as fp:
        train_type_dict = json.load(fp)
    with open(op.join(data_dir_path, 'test_type_dict.json'), 'r') as fp:
        test_type_dict = json.load(fp)
    with open(op.join(data_dir_path, 'train_descriptions.json'), 'r') as fp:
        train_descr = json.load(fp)
    with open(op.join(data_dir_path, 'test_descriptions.json'), 'r') as fp:
        test_descr = json.load(fp)

    return train_type_dict, test_type_dict, train_descr, test_descr


def save_model_ckpt(model, save_dir, idx=None):
    model_dir = op.join(save_dir, 'model_ckpt')

    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    if idx is None:
        filenames = os.listdir(model_dir)
        pattern = r'^ckpt([0-9]+)\.pth$'
        valid = [
            (f, re.search(pattern, f)[1]) for f in filenames \
            if re.search(pattern, f)
        ]
        max_index = max([int(i) for _, i in valid] + [-1])
        idx = max_index + 1

    print(f'Saving model checkpoint ckpt{idx}.pth')
    torch.save(model.state_dict(), op.join(model_dir, f'ckpt{idx}.pth'))


def read_raw_file(raw_data_file):
    with open(raw_data_file, 'rb') as fp:
        raw_data = pickle.load(fp)
    id2description_raw = raw_data['id2description']
    if 'description2id' in raw_data.keys():
        description2id_raw = raw_data['description2id']
    else:
        description2id_raw = None
    obs_raw = raw_data['obs']
    descriptions_ids_raw = raw_data['descriptions_ids']

    if os.path.isfile(raw_data_file[:-3] + '_count.pk'):
        with open(raw_data_file[:-3] + '_count.pk', 'rb') as fp:
            count_dict = pickle.load(fp)
    else:
        count_dict = None
    return obs_raw, descriptions_ids_raw, id2description_raw, description2id_raw, count_dict


def sanity_check_descriptions(descriptions_ids_raw, id2description_raw, id2description):
    ids_to_remove = [id for id in id2description_raw.keys() if id not in id2description.keys()]

    descriptions_ids_all = []
    for d_ids in descriptions_ids_raw:
        new_d_ids = []
        for d in d_ids:
            new_d_ids.append([id for id in d if id not in ids_to_remove])
        descriptions_ids_all.append(new_d_ids)

    return descriptions_ids_all


def compute_metrics(pred_probas, rewards):
    y_pred = (pred_probas > 0.5).to(torch.float32).squeeze().cpu()
    y_true = rewards.cpu()
    tp = torch.mul(y_true, y_pred).sum().to(torch.float32).cpu()
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32).cpu()
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32).cpu()
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32).cpu()
    accuracy = ((tp + tn) / (tp + tn + fp + fn)).float()
    precision = (tp / (tp + fp)).float()
    recall = (tp / (tp + fn)).float()
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1

def evaluate_test_metrics(model, state_idx_buffer, states, id2one_hot, size_dataset, proportion_pos_reward, batch_size,
                          logging, use_cuda=False, eval_idx=0):
    logging.info('Evaluating {}'.format(str(eval_idx)))
    model.eval()

    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    output_dict = {}

    count = 0
    with torch.no_grad():
        for id in list(state_idx_buffer):
            if count % 100 == 0:
                logging.info('Descr {}/{}'.format(str(count), str(len(state_idx_buffer))))
            bodies = []
            objs = []
            rewards = []
            descrs = []

            if len(state_idx_buffer[id]['pos_reward']) < size_dataset * proportion_pos_reward // 2:
                size_pos = int(len(state_idx_buffer[id]['pos_reward']))
            else:
                size_pos = int(size_dataset * proportion_pos_reward // 2)
            size_neg = int(size_dataset - size_pos)

            for pos_idx in state_idx_buffer[id]['pos_reward'][:size_pos]:
                bodies.append(states[pos_idx][0])
                objs.append(states[pos_idx][1])
                descrs.append(id2one_hot[id])
                rewards.append(1)
            for neg_idx in state_idx_buffer[id]['neg_reward'][:size_neg]:
                bodies.append(states[neg_idx][0])
                objs.append(states[neg_idx][1])
                descrs.append(id2one_hot[id])
                rewards.append(0)

            n_batch = int(size_dataset / batch_size)
            pred_probas = torch.tensor([], dtype=torch.float32).to(device)
            for batch in range(n_batch + 1):
                ind1 = batch * batch_size
                if (batch + 1) * batch_size > size_dataset:
                    ind2 = size_dataset
                else:
                    ind2 = (batch + 1) * batch_size
                bodies_batch = torch.tensor(bodies[ind1:ind2], dtype=torch.float32).to(device)
                objs_batch = torch.tensor(objs[ind1:ind2], dtype=torch.float32).to(device)
                descrs_batch = torch.tensor(descrs[ind1:ind2], dtype=torch.float32).to(device)

                pred_probas_batch = model(objs_batch, bodies_batch, descrs_batch)
                pred_probas = torch.cat([pred_probas, pred_probas_batch])

            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            accuracy, precision, recall, f1 = compute_metrics(pred_probas, rewards)
            output_dict[id] = (accuracy, precision, recall, f1)
            count += 1
    model.train()
    return output_dict


def compute_metric_by_type(metric_dict_test, id2description_test, type_dict, logging):
    description2id = {v: k for k, v in id2description_test.items()}
    output_dict = {}
    for k, v in type_dict.items():
        f1_type_list = []
        for descr in v:
            if descr in description2id.keys():
                f1_type_list.append(metric_dict_test[description2id[descr]][3])
            else:
                logging.info('FLAG description: ' + str(descr) + ' is missing from testing data')
        output_dict[k] = np.mean(np.nan_to_num(f1_type_list))
        logging.info(str(k) + str(output_dict[k]))

    return output_dict


def write_f1_type_to_df(df, metric_dict_by_type):
    dict_f1 = {'f1_{}'.format(t): f1 for t, f1 in metric_dict_by_type.items()}
    new_df = pd.DataFrame(dict_f1, index=[0])
    df = df.append(new_df)

    return df


def write_f1_to_df(df, metric_dict, id2description):
    dict_f1 = {'f1_{}'.format(descr): metric_dict[id][3].tolist() for id, descr in id2description.items() if
               id in metric_dict.keys()}
    f1_mean = np.nanmean([metric_dict[id][3] for id in metric_dict.keys()])
    dict_f1['f1_mean'] = f1_mean
    new_df = pd.DataFrame(dict_f1, index=[0])
    df = df.append(new_df)

    return df


def append_value_to_dict_list(dict_list, tmp_dict):
    for k in dict_list.keys():
        dict_list[k].append(tmp_dict[k])
    return dict_list


def mean_train_metrics_over_steps(train_metrics_dict):
    output_dict = {}
    for k in train_metrics_dict.keys():
        output_dict['mean_' + k] = np.mean(np.nan_to_num(train_metrics_dict[k]))
    return output_dict