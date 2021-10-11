from src.temporal_playground_env.env_params import get_env_params

env_params = get_env_params()
CATEGORIES = env_params['categories']


def extract_predicate_descr(new_descr):
    '''

    :param new_descr: list of instantatenous descriptions fulfilled in the scene
    :type new_descr: list
    :return: list of descriptions with a predicate in it
    :rtype: list
    '''
    predicate_descr = []
    for d in new_descr:
        if 'grow' in d or 'grasp' in d or 'shake' in d:
            predicate_descr.append(d)
    return predicate_descr


def extract_predicate_target(predicate_descr):
    '''
    Extract the object being targeted by a predicate

    :param predicate_descr: list of predicate descriptions
    :type predicate_descr: list
    :return: dictionary {predicate: list of object being targeted by the predicate}
    :rtype: dict
    '''
    objects_dict = {'grow': [], 'grasp': [], 'shake': []}
    for d in predicate_descr:
        if ' of' not in d and 'thing' not in d and 'animal' not in d and 'plant' not in d and 'supply' not in d:
            if 'grasp' in d:
                objects_dict['grasp'].append(d.split(' ')[-1])
            elif 'grow' in d:
                objects_dict['grow'].append(d.split(' ')[-1])
            elif 'shake' in d:
                objects_dict['shake'].append(d.split(' ')[-1])
    for k, v in objects_dict.items():
        objects_dict[k] = list(set(v))
    return objects_dict


def update_reference_to_object(new_descr, past_descr_buffer):
    '''
    Update the reference of an object being targeted by a predicate with all its previous relative positions

    :param new_descr: list of new descriptions
    :type new_descr: list
    :param past_descr_buffer: list of all descriptions in the past
    :type past_descr_buffer: list
    :return: list of predicate descritpions updated with past references
    :rtype: list
    '''
    updated_descr_with_past_reference = []
    predicate_descr = extract_predicate_descr(new_descr)

    objects_dict = extract_predicate_target(predicate_descr)

    for k, v in objects_dict.items():
        for o in v:
            for d in past_descr_buffer:
                d_split = d.split(' ')
                if o == d_split[1] and ('of' in d_split):
                    del d_split[1]
                    past_ref = ' '.join(d_split)
                    updated_descr_with_past_reference.append(k + ' thing ' + past_ref)
                if o == d_split[1] and ('most' in d_split):
                    del d_split[1]
                    past_ref = ' '.join(d_split)
                    updated_descr_with_past_reference.append(k + ' ' + past_ref + ' thing')

    updated_descr_with_past_reference_catego = []
    for d in updated_descr_with_past_reference:
        d_split = d.split(' ')
        for k in CATEGORIES.keys():
            if d_split[-1] in CATEGORIES[k]:
                # Category living_thing is now called "living thing"
                if k == 'living_thing':
                    k = 'living thing'
                new_d = d_split[:-1] + [k]
                updated_descr_with_past_reference_catego.append(' '.join(new_d))
        updated_descr_with_past_reference_catego.append(' '.join(d_split[:-1]+['thing']))

    return list(set(updated_descr_with_past_reference)) + list(set(updated_descr_with_past_reference_catego))


def detect_configuration_changes(descr_present, descr_past):
    '''
    Detect changes of configuration in spatial relations or location of objects and retruns the corresponding updated
     configuration descriptions (with past)

    :param descr_present: list of instantaneous descriptions
    :type descr_present: list
    :param descr_past: list of instantaneous descriptions at the previous time step (It is not the past buffer !!!)
    :type descr_past: list
    :return: list of configurations updated with past
    :rtype: list
    '''
    changed_descr = []
    for d in descr_past:
        if d not in descr_present and (
                'grow' not in d and 'shake' not in d and 'grasp' not in d and d.split(' ')[-1] != 'thing'):
            changed_descr.append('was ' + d)

    return changed_descr


def create_predicate_dict(new_descr):
    '''
    Create a dictionary {predicate: list of all descriptions fulfilling the predicate}
    :param new_descr: list of instantaneous valid descriptions
    :type new_descr: list
    :return: dictionary {predicate: list of all descriptions fulfilling the predicate}
    :rtype: dict
    '''
    out_dict = {'grow': [], 'grasp': [], 'shake': []}

    for d in new_descr:
        if 'grow' in d:
            out_dict['grow'].append(d)
        elif 'grasp' in d:
            out_dict['grasp'].append(d)
        elif 'shake' in d:
            out_dict['shake'].append(d)

    return out_dict


def detect_consumed_predicate(descr_present, descr_past):
    '''
    Detect if a predicate is no longer being consumed and update the description with the past accordingly

    :param descr_present: list of instantaneous descriptions
    :type descr_present: list
    :param descr_past: list of instantenous descriptions at preivous time step
    :type descr_past: list
    :return: updated list of descriptions with predicate consumed (added 'was')
    :rtype: list
    '''
    changed_descr = []
    predicate_dict = create_predicate_dict(descr_present)
    if not predicate_dict['grasp']:
        for d in descr_past:
            if d not in descr_present:
                if 'grasp' in d:
                    changed_descr.append('was ' + d)
    if not predicate_dict['grow']:
        for d in descr_past:
            if d not in descr_present:
                if 'grow' in d:
                    changed_descr.append('was ' + d)
    if not predicate_dict['shake']:
        for d in descr_past:
            if d not in descr_present:
                if 'shake' in d:
                    changed_descr.append('was ' + d)

    return changed_descr


def update_past_descr_buffer(past_descr_buffer, new_past_descr):
    '''
    Check if new past descr are only in past descr_buffer, if not add them to the buffer

    '''

    for d in new_past_descr:
        if d not in past_descr_buffer:
            past_descr_buffer.append(d)
    # make sure that if a predicate descriptions contained in the buffer is consumed
    # every reference to this predicate is consumed in memory
    to_remove = []
    for d_past in past_descr_buffer:
        if 'was ' + d_past in new_past_descr:
            to_remove.append(d_past)
    for elem in to_remove:
        past_descr_buffer.remove(elem)
    return past_descr_buffer


def update_descrs_with_temporal_logic(new_descr, past_descr, past_descr_buffer):
    updated_descr_with_past_reference = update_reference_to_object(new_descr, past_descr_buffer)
    new_descr = new_descr + updated_descr_with_past_reference
    past_descr = past_descr + updated_descr_with_past_reference  # must also update past reference with updated descr before checking if predicate is consumed
    changed_descr_predicat = detect_consumed_predicate(descr_present=new_descr, descr_past=past_descr)
    changed_descr_dyn = detect_configuration_changes(descr_present=new_descr, descr_past=past_descr)
    past_descr_buffer = update_past_descr_buffer(past_descr_buffer,
                                                 changed_descr_dyn + changed_descr_predicat + updated_descr_with_past_reference)

    return past_descr_buffer, new_descr
