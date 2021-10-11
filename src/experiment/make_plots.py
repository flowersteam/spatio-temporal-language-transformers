import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os
import numpy as np
import itertools
import os.path as op
from src.grammar.descriptions import get_all_descriptions, generate_systematic_split

all_descr = get_all_descriptions()


def make_single_exp_plot(res_folder, meta_category_dict):
    f1_test_by_type = pd.read_csv(op.join(res_folder, 'f1_test_by_type.csv'), index_col=0)
    with open(op.join(res_folder, 'params.json'), 'rb') as fp:
        params = json.load(fp)

    freq_eval = params['freq_eval']
    model = params['architecture']

    types = list(f1_test_by_type.columns)
    step = [freq_eval * (i + 1) for i in range(len(f1_test_by_type[types[0]]))]
    f1_test_by_type['step'] = step

    # Plots by type:
    plt.figure()
    for type in types:
        sns.lineplot(x='step', y=type, data=f1_test_by_type, label=type)
    plt.legend()
    plt.title('{}: f1 per generalization type during learning'.format(model))

    last_values = [f1_test_by_type[cat].values[-1] for cat in types]
    plt.figure()
    sns.barplot(types, last_values)
    plt.title('{}: final f1 per generalization type'.format(model))

    # Plots by meta categories
    plt.figure()
    for k, v in meta_category_dict.items():
        values = []
        for cat in v:
            values.append(f1_test_by_type[cat].values)
        plt.plot(step, np.mean(np.array(values), axis=0), label=k)
    plt.title('{}: f1 per meta categories during learning'.format(model))
    plt.xlabel('Training steps')
    plt.ylabel('f1')
    plt.legend()

    plt.figure()
    last_values = []
    for k, v in meta_category_dict.items():
        values = []
        for cat in v:
            values.append(f1_test_by_type[cat].values)
        last_values.append(np.mean(np.array(values), axis=0)[-1])
    sns.barplot(list(meta_category_dict.keys()), last_values)
    plt.title('{}: final f1 per meta categories'.format(model))


def make_mutliple_exp_plot(res_folder_list):
    big_df_by_type = pd.DataFrame()
    big_df_by_type_final = pd.DataFrame()

    models = []
    for res_folder in res_folder_list:
        res_file = op.join(res_folder, 'f1_test_by_type.csv')
        if os.path.exists(res_file):
            f1_test_by_type = pd.read_csv(res_file, index_col=0)
            with open(op.join(res_folder, 'params.json'), 'rb') as fp:
                params = json.load(fp)

            freq_eval = params['freq_eval']
            model = params['architecture']
            models.append(model)

            categories = list(f1_test_by_type.columns)
            new_categories = []
            for cat in categories:
                if 'base' in cat:
                    cat = '1. Basic'
                if 'spatial' in cat:
                    cat = '2. Spatial'
                if '_temporal' in cat:
                    cat = '3. Temporal'
                if 'spatio' in cat:
                    cat = '4. Spatio-temporal'
                if 'f1_1' in cat:
                    cat = '1. Object-attribute'
                if 'f1_2' in cat:
                    cat = '2. Predicate-object'
                if 'f1_3' in cat:
                    cat = '3. One-to-one relation'
                if 'f1_4' in cat:
                    cat = '4. Past spatial relation'
                if 'f1_5' in cat:
                    cat = '5. Past predicate'
                new_categories.append(cat)

            step = [freq_eval * (i + 1) for i in range(len(f1_test_by_type[categories[0]]))]
            f1_test_by_type['step'] = step
            f1_test_by_type['model'] = [model for _ in range(len(f1_test_by_type[categories[0]]))]

            final_values = [f1_test_by_type[cat].values[-1] for cat in categories]
            dict_final_by_type = {'Type': new_categories, 'Final f1': final_values,
                                  'model': [model for _ in range(len(categories))]}
            new_df_by_type_final = pd.DataFrame(dict_final_by_type)

            big_df_by_type_final = big_df_by_type_final.append(new_df_by_type_final, ignore_index=True)
            big_df_by_type = big_df_by_type.append(f1_test_by_type)

    models = list(set(models))
    types = [col for col in list(big_df_by_type.columns) if 'f1' in col]

    big_df_by_type_final = big_df_by_type_final.sort_values(by='model', ascending='True')
    big_df_by_type_final = big_df_by_type_final.sort_values(by=['model', 'Type'], ascending='True')
    plt.figure(figsize=(20, 8))
    palette = {'lstm_factored': '#DE8526', 'lstm_flat': '#DEB626', 'transformer_ut': '#1C5E7A',
               'transformer_ut_wa': '#1DB4EB', 'transformer_tft': '#49A644', 'transformer_tft_wa': '#27F42E',
               'transformer_sft': '#754F5B',
               'transformer_sft_wa': '#B25C77'}

    plt.rcParams['font.size'] = '18'
    ax = sns.barplot(x='Type', y='Final f1', hue='model', data=big_df_by_type_final, palette=palette, capsize=.025)
    plt.ylim([-0.01, 1.11])
    plt.ylabel('$F_1$ Score')
    if '1. Basic' in new_categories:
        plt.xlabel('Sets of descriptions grouped by concepts', fontweight='bold')
    else:
        plt.xlabel('Types of Systematic Generalization', fontweight='bold')
    h, l = ax.get_legend_handles_labels()
    labels = ['LSTM-FACTORED', 'LSTM-FLAT', 'UT', 'UT-WA', 'SFT', 'SFT-WA',
              'TFT', 'TFT-WA']
    ax.legend(h, labels, title="Models", loc='center', bbox_to_anchor=(0.5, 1), ncol=4)
    plt.subplots_adjust(left=0.045, right=0.99)

    plt.show()

    return big_df_by_type_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('res_dir', type=str, help='path to the folder containing run dirs with results')
    args = parser.parse_args()
    res_folder_list = [op.join(res_root, res_folder) for res_folder in os.listdir(args.res_dir)]

    df = make_mutliple_exp_plot(res_folder_list)
