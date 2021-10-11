import pickle
import src.grammar.grammar_modules as gm
import random
from src.temporal_playground_env.env_params import get_env_params
import json

env_params = get_env_params()
CATEGORIES = env_params['categories']


def sample_descriptions_from_state(env):
    worldstate = gm.WorldState(env.objects, env.agent)
    worldstates_filtered = gm.S.filter(worldstate)
    return gm.ws_list_to_string_list(worldstates_filtered)


def get_all_instant_descriptions(remove_relative_descr=False, remove_grow_furniture=True):
    all_descr = [gm.applied_rules_to_string(rule) for rule in gm.S._all_descr()]

    filter_all_descr = []
    if remove_relative_descr:
        for descr in all_descr:
            if not ('grasp' not in descr and 'shake' not in descr and 'grow' not in descr and (
                    'of' in descr or 'most' in descr)):
                filter_all_descr.append(descr)
        all_descr = filter_all_descr

    remove_descr = []
    if remove_grow_furniture:
        for descr in all_descr:
            d_split = descr.split(' ')
            if (d_split[-1] in CATEGORIES['furniture'] or d_split[
                -1] == 'furniture') and 'grow' in descr and ' of' not in descr:
                remove_descr.append(descr)
            if (d_split[-1] in CATEGORIES['supply'] or d_split[
                -1] == 'supply') and 'grow' in descr and ' of' not in descr:
                remove_descr.append(descr)

    for descr in remove_descr:
        all_descr.remove(descr)
    return all_descr


def add_past_reference(d):
    split_d = d.split(' ')
    new_token_list = []
    for token in split_d:
        if token in ['right', 'left', 'top', 'bottom']:
            new_token_list.append('was')
        new_token_list.append(token)
    return ' '.join(new_token_list)


def get_all_past_descriptions(d_i):
    grow_descr = [d for d in d_i if 'grow' in d]
    grasp_descr = [d for d in d_i if 'grasp' in d]
    shake_descr = [d for d in d_i if 'shake' in d]
    predicate_descr = shake_descr + grow_descr + grasp_descr
    rel_predicate_descr = [d for d in predicate_descr if 'most' in d or ' of' in d]

    past_predicate_descr = ['was ' + d for d in predicate_descr]
    instant_predicate_past_rel_descr = [add_past_reference(d) for d in rel_predicate_descr]
    past_predicate_descr_past_rel = ['was ' + add_past_reference(d) for d in rel_predicate_descr]
    return past_predicate_descr, instant_predicate_past_rel_descr, past_predicate_descr_past_rel


def get_all_descriptions():
    d_i = get_all_instant_descriptions(remove_relative_descr=True)
    d_pp_ir, d_ip_pr, d_pp_pr = get_all_past_descriptions(d_i)
    return d_i + d_pp_ir + d_ip_pr + d_pp_pr


def generate_type_description_dict(all_descr):
    all_descr = get_all_descriptions()
    d_i = get_all_instant_descriptions(remove_relative_descr=True)
    d_pp_ir, d_ip_pr, d_pp_pr = get_all_past_descriptions(d_i)
    type_base = [d for d in d_i if
                 'most' not in d and ' of' not in d and 'grow' not in d and 'shake' not in d
                 and 'right' not in d and 'left' not in d and 'top' not in d and 'bottom' not in d and 'center' not in d]
    type_spatial = [d for d in d_i if
                    ('most' in d or ' of' in d) and 'grow' not in d and 'shake' not in d]
    type_temporal = [d for d in d_i if
                     'most' not in d and ' of' not in d and ('grow' in d or 'shake' in d)] + \
                    [d for d in d_pp_ir if 'most' not in d and ' of' not in d]
    type_spatio_temporal = [d for d in d_i if
                            ('most' in d or ' of' in d) and ('grow' in d or 'shake' in d)] + \
                           [d for d in d_pp_ir + d_ip_pr + d_pp_pr if ('most' in d or ' of' in d)]

    return {'base': type_base, 'spatial': type_spatial, 'temporal': type_temporal,
            'spatio-temporal': type_spatio_temporal}


def generate_random_split(all_descr, ratio=0.15):
    type_description_dict = generate_type_description_dict(all_descr)
    type_description_dict_train = {}
    type_description_dict_test = {}

    all_train_descriptions = []
    all_test_descriptions = []
    for type, descriptions in type_description_dict.items():
        test_descriptions = random.sample(descriptions, int(len(descriptions) * ratio))
        train_descriptions = list(set(descriptions) - set(test_descriptions))

        type_description_dict_test[type] = test_descriptions
        type_description_dict_train[type] = train_descriptions
        all_train_descriptions.extend(train_descriptions)
        all_test_descriptions.extend(test_descriptions)

    return type_description_dict_train, type_description_dict_test, all_train_descriptions, all_test_descriptions


def generate_systematic_split(all_descr):
    d_i = get_all_instant_descriptions(remove_relative_descr=True)
    to_remove = [d for d in d_i if (
            'right' in d or 'left' in d or 'center' in d or 'top' in d or 'bottom' in d) and ' of' not in d and 'most' not in d]
    all_d = [d for d in all_descr if d not in to_remove]

    type_1_remove = [d for d in all_d if 'blue cat' in d or 'red door' in d or 'green algae' in d]
    type_1_test = [d for d in type_1_remove if 'grow green algae' not in d and 'was' not in d]

    any_in = lambda a, b: bool(set(a).intersection(b))
    type_2_remove = [d for d in all_d if
                     any_in(d.split(), CATEGORIES['plant']) and 'grow' in d and 'most' not in d and ' of' not in d]
    type_2_test = [d for d in type_2_remove if 'was' not in d]

    type_3_remove = [d for d in all_d if 'bottom most' in d]
    type_3_test = [d for d in type_3_remove if d.split()[2] != 'was' and d.split()[0] != 'was']

    type_4_remove = [d for d in all_d if 'was left of' in d]
    type_4_test = [d for d in type_4_remove if d.split()[0] != 'was']

    type_5_remove = [d for d in all_d if 'was grasp' in d]
    type_5_test = [d for d in type_5_remove if 'most' not in d and ' of' not in d]

    all_remove = list(set(type_1_remove + type_2_remove + type_3_remove + type_4_remove + type_5_remove))
    all_train_descriptions = list(set(all_d) - set(all_remove))
    type_description_dict_train = {}
    type_description_dict_test = {'1': type_1_test, '2': type_2_test, '3': type_3_test, '4': type_4_test,
                                  '5': type_5_test}

    all_test_descriptions = list(set(type_1_test + type_2_test + type_3_test + type_4_test + type_5_test))
    return type_description_dict_train, type_description_dict_test, all_train_descriptions, all_test_descriptions




if __name__ == "__main__":
    d_i = get_all_instant_descriptions(remove_relative_descr=True)

    d_pp_ir, d_ip_pr, d_pp_pr = get_all_past_descriptions(
        d_i)  # past predicate instant rel, instant predicate past rel, past rel past ref

    all_descr = get_all_descriptions()

    type_description_dict_train, type_description_dict_test, all_train_descriptions, all_test_descriptions = generate_systematic_split(
        all_descr)
    generate_type_description_dict(all_descr)
    # with open('../data/raw/data_jz/train_type_dict.json', 'w') as fp:
    #     json.dump(type_description_dict_train, fp)
    # with open('../data/raw/data_jz/test_type_dict.json', 'w') as fp:
    #     json.dump(type_description_dict_test, fp)
    # with open('../data/raw/data_jz/all_train_descriptions.json', 'w') as fp:
    #     json.dump(all_train_descriptions, fp)
    # with open('../data/raw/data_jz/all_test_descriptions.json', 'w')as fp:
    #     json.dump(all_test_descriptions, fp)

    stop = 0
