# Grounding Spatio Temporal Language with Transformer

This repository gives all the information required to reproduce the results of the paper Grounding Spatio-Temporal Language with Transformers. To reproduce our results, you will need access to nvidia GPUs with at least 16GB ram. 

## Context


## Getting started

Create a new conda environment and install the dependencies with:
```
conda env create -f environment.yml
```

##  Data

You can download the training and testing data (two splits) using the following link: https://drive.google.com/drive/folders/1FIbiGmmC8Nqvag6etnmUg2389YpMXa5N.

Download the two zipped datasets, place them in a folder of your choice and unzip them.

## How to reproduce our results on cluster

If you have access to a cluster using the slurm workload manager, you can directly reproduce our hyperparameter search and our main experiments using our files. To adapt the slurm files to your setup, please see the runfile templates in `generate_experiments.py` and `generate_experiments_retrain.py`, and add any information relative to your server there.

####  How to reproduce our hyper parameter search on a cluster, using slurm:

Once you downloaded the datasets you can reconduct the hyper parameter search we performed (training all models on 3 seeds for 150 000 steps with 18 different parameter configurations) using the script ```src/experiment/generate_experiments.py```

The command you should use is:
```
python generate_experiments --dsdir 'path_to_dataset_directory' --expe-path 'path_where_you_want_to_save_results_of_search' --conda_env_name 'name of your remote conda env'
```
This will generate N_condition * N_seed folders to train all models and a file ```execute_runfiles.sh``` under expe-path. Simply execute the ```execute_runfiles.sh``` to launch the hyper parameter search; this will launch all the jobs using slurm.

The hyper parameter search is conducted using the dataset: ```temporal_300k_pruned_random_split```


####  How to retrain the best models found during the hyperparamter search, using slurm:

Use the ```generate_experiments_retrain.py``` with the same argument as for the hyper parameter search to retrain best models on 10 seeds. You can perform the training on the random split (as seen in Section 3.1 of the main paper) or you can perform the training on the systematic split (as seen in Section 3.2), respectively by providing the ```--experiment random_split``` or ```--experiment systematic split``` argument.

In each run dir, a ```f1_test.csv``` file is generated to monitor the f1 score of each description. Aggregated metrics for each category of meanings are reported in ```f1_test_bt_type.csv```. Training metrics are logged in ```log.log``` and gpu usage can be monitored in the ```gpu_memory_log.log``` file.


## How to train a single seed of a model locally

Models can also be trained on a single seed using the script ```src/experiment/train.py```

Below is an example to train an Temporal-First-Transformer on the data.
Change your working directory to `src/experiment` and execute:

```
python train.py --architecture transformer_tft --dataset_dir 'path_to_downloaded_data_directory' --save_dir 'path_to_output_directory'
```

To automatically select the hypermarameters used in the paper for each architecture, set the `--auto_hparams` flag to `True`.

## Citation

```
@article{karch2021grounding,
	title={Grounding Spatio-Temporal Language with Transformers},
	author={Karch, Tristan and Teodorescu, Laetitia and Hofmann, Katja and Moulin-Frier, Clément and Oudeyer, Pierre-Yves},
	journal={NeurIPS},
	year={2021}
}
```
