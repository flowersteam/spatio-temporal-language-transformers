"""
Generate all the (slurm) runfiles for a grid of hparams.
"""
import os
import os.path as op
import re
import random
import json
import datetime
import pathlib

import torch

from collections import OrderedDict
from functools import reduce
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
add = parser.add_argument

add('--expe-path', help='Directory where the experiment folder will be created')
add('--dsdir', default='Directory to look for the processed dataset')
add(
    '--experiment',
    default='random_split',
    choices=['random_split', 'systematic_split'],
    help='What experiment to perform: random split or systematic split.'
)
add('--conda_env_name', default='py_gpu', help='The name of your conda env on the server')

args = parser.parse_args()
current_path = pathlib.Path(__file__).parent.absolute()
dsdir = args.dsdir

# helper fns

foldl = lambda func, acc, xs: reduce(func, xs, acc)
multiply_list = lambda l1, l2: [e1 + e2 for e1 in l1 for e2 in l2]
multiply_all = lambda l: foldl(multiply_list, [[]], l)
unsqueeze_list = lambda l: [[e] for e in l]

if args.experiment == 'random_split':
    dataset_name = 'temporal_300k_pruned_random_split'
else:
    dataset_name = 'temporal_300k_pruned_systematic_split'

configs = OrderedDict(
    learning_rate=[1e-4],
    seed=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    arch=[
        'lstm_factored',
        'lstm_flat',
        'transformer_ut',
        'transformer_ut_wa',
        'transformer_sft',
        'transformer_sft_wa',
        'transformer_tft',
        'transformer_tft_wa',
    ],
    condition=[0],
    dataset_name=[dataset_name],
    batch_size=[512],
)

# dict with the different model choices
conditions = {
    'lstm_factored': [
        {'hidden_size': 512, 'layers': 4, 'num_heads': 8, 'learning_rate': 0.0001}
    ],
    'lstm_flat': [
        {'hidden_size': 512, 'layers': 4, 'num_heads': 4, 'learning_rate': 0.0001}
    ],
    'transformer_ut': [
        {'hidden_size': 256, 'layers': 4, 'num_heads': 8, 'learning_rate': 0.0001, 'long_train': True}
    ],
    'transformer_ut_wa': [
        {'hidden_size': 512, 'layers': 4, 'num_heads': 8, 'learning_rate': 1e-05, 'long_train': True}
    ],
    'transformer_sft': [
        {'hidden_size': 256, 'layers': 4, 'num_heads': 4, 'learning_rate': 0.0001, 'long_train': True}
    ],
    'transformer_sft_wa': [
        {'hidden_size': 256, 'layers': 2, 'num_heads': 8, 'learning_rate': 0.0001, 'long_train': True}
    ],
    'transformer_tft': [
        {'hidden_size': 256, 'layers': 4, 'num_heads': 4, 'learning_rate': 0.0001, 'long_train': True}
    ],
    'transformer_tft_wa': [
        {'hidden_size': 512, 'layers': 4, 'num_heads': 8, 'learning_rate': 1e-05, 'long_train': True}
    ],
}

mem_str = ''
big_mem_str = "\n#SBATCH -C v100-32g\n"
short_train_str = "#SBATCH --time=20:00:00"
long_train_str = "#SBATCH --qos=qos_gpu-t4\n#SBATCH --time=100:00:00"

run_str = """#!/bin/bash

#SBATCH --mem=24G
{time_string}
#SBATCH -o {save_dir}/run.slurm.out
#SBATCH -e {save_dir}/run.slurm.err{mem_str}

conda deactivate
conda activate {conda_env_name}
    
python -u {current_path}/train.py \\
    --architecture {arch} \\
    --dataset_dir {dsdir} \\
    --run_idx {run_idx} \\
    --num_heads {num_heads} \\
    --layers {layers} \\
    --hidden_size {hidden_size} \\
    --learning_rate {learning_rate} \\
    --seed {seed} \\
    --save_dir {save_path} \\
    --batch_size {batch_size}"""

execute_runfile_str = """#!/bin/bash

regex="run([0-9]+)/run.slurm"

for Script in run*/run.slurm ; do

  [[ $Script =~ $regex ]]
  scriptId="${BASH_REMATCH[1]}"
  if [ -n "$scriptId" ]         ; then
    if  [[ "$scriptId" -gt -1 ]] ; then
      echo "running script $Script"
      sbatch "$Script"
    fi
  fi

done
"""
dt = '_'.join(str(datetime.datetime.now()).split(' '))
path = op.join(args.expe_path, 'expe', dt)
Path(path).mkdir(exist_ok=True, parents=True)

# check dataset dir exists and make its path absolute
dspath = Path(dsdir)
if not dspath.exists():
    raise ValueError("Provided dsdir doesn't exist, please provide an existing path")
dsdir = str(dspath.absolute())

# save execute runfiles bash script
with open(op.join(path, 'execute_runfiles.sh'), 'w') as f:
    f.write(execute_runfile_str)

run_idx = 0
# write all combinations configs in their folder
all_combinations = multiply_all([unsqueeze_list(v) for v in configs.values()])

for combination in all_combinations:

    # get path of the run folder
    run_path = op.join(path, f'run{run_idx}')

    conf = dict(zip(configs.keys(), combination))
    conf['seed'] = random.randint(0, 1e9)
    conf['run_idx'] = run_idx
    conf['save_dir'] = run_path
    conf['current_path'] = str(current_path)
    conf['dsdir'] = op.join(dsdir, conf['dataset_name'])
    conf['save_path'] = run_path
    conf['conda_env_name'] = args.conda_env_name

    # get the model condition and add it to the conf
    condition = conditions[conf['arch']][conf['condition']]
    try:
        condition["big_mem"]
        mem_str = big_mem_str
    except KeyError:
        pass

    try:
        condition["long_train"]
        time_string = long_train_str
    except KeyError:
        time_string = short_train_str

    if 'wa' in conf['arch']:
        if conf['condition'] in [2, 8]:
            conf['learning_rate'] = 1e-5

    conf = {**conf, **condition}

    # make expe dir
    print(f'Writing run config and runfile in {run_path}')
    Path(run_path).mkdir(exist_ok=True, parents=True)
    with open(op.join(run_path, 'run.slurm'), 'w') as f:
        conf = {**conf, **{'mem_str': mem_str, 'time_string': time_string}}
        f.write(run_str.format(**conf))
    # dump config in a json
    with open(op.join(run_path, 'conf.json'), 'w') as f:
        f.write(json.dumps(conf))

    run_idx += 1
