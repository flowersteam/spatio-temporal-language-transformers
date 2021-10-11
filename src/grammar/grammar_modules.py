"""
This file implements a GrammarModule class that lists possible productions.
It also lists all the possible GMs of Temporal Playground.
"""
import functools as fct
import time
import random
import numpy as np
import torch

import gym
import src.temporal_playground_env as plg

from pprint import pprint
from copy import deepcopy
from gym.spaces import Box

from src.temporal_playground_env.objects import obj_type_to_obj as plg_objects
# from src.playground_env.objects import obj_type_to_obj as plg_objects

# TODO ruleadd split
# TODO negative examples gen
# TODO and and or rules possibly

# TODO add boxes for relational concepts
# TODO add incompatible concepts
# TODO add past tense
# TODO move away from linear comparison of sentence application for applying past tense
# TODO add encapsulating GM object
# TODO expand grammar by adding more GMs

# helper functions

mapl = lambda fn, xs: list(map(fn, xs))
and_binary_fn = lambda x, y: x and y
or_binary_fn = lambda x, y: x or y
add_binary_fn = lambda x, y: x + y
mul_binary_fn = lambda x, y: x * y
foldl = lambda func, acc, xs: fct.reduce(func, xs, acc)
compose = lambda f1, f2: lambda x: f1(f2(x))

# list operations
add_all_list = lambda l: foldl(add_binary_fn, [], l) # nary concat op
mult_lists = lambda l1, l2: [(e1, e2) for e1 in l1 for e2 in l2] # binary list mult op
and_reduce = lambda l: foldl(and_binary_fn, True, l)
or_reduce = lambda l: foldl(or_binary_fn, False, l)

# operations on string lists
def str_add(s1, s2):
    if not s1:
        return s2
    if not s2:
        return s1
    return ' '.join([s1, s2])
mult_slist = lambda l1, l2: [str_add(e1, e2) for e1 in l1 for e2 in l2]
mult_all_slist = lambda l: foldl(mult_slist, [''], l)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# helper classes

class Agent():

    def __init__(self, env):
        self.env = env
        self.name = 'you'

    @property
    def position(self):
        return np.array(self.env.agent_pos, dtype=np.float)

    @property
    def grasped(self):
        return False

    # hacky fix for saying self color is not represented
    @property
    def rgb_code(self):
        return np.array([-1., -1., -1.], dtype=np.float)

    def __repr__(self):
        msg = '\n\nOBJ AGENT'
        return msg

class WorldState():

    def __init__(self, objects, agent=None, string='', *args, **kwargs):
    # def __init__(self, env, string='', *args, **kwargs):
    #     objects = deepcopy(env.objects)
    #     agent = deepcopy(env.agent)

        self.objects = list(objects)
        if agent is not None:
            self.objects += [agent]
        # convert attrs to the right np format
        # self.init_object_attrs()

        self.string = string
        self.applied_rules = []
        # self.agent = agent

        self.timeline = (0, 0)

    def init_object_attrs(self):
        for obj in self.objects:
            obj.pos = obj.pos.expand_dim(0)
            obj.size = np.array(obj.size)
            obj.is_grasped = np.array(obj.is_grasped)
            obj.rgb_code = obj.rgb_code.expand_dim(0)

    def filter(self, fn):
        objects = list(filter(fn, self.objects))
        ws = WorldState(objects)
        ws.applied_rules = list(self.applied_rules) # copy applied rules
        return ws

    def add_str(self, s):
        objects = list(self.objects)
        return WorldState(objects, string=str_add(self.string, s))

    def time_filter(self, boolean, s=''): # TODO: better formalism, this is still sketchy
        if boolean:
            # TODO: fuse given timeline with current worldstate
            return WorldState(self.objects, string=str_add(self.string, s))
        else:
            return WorldState([])

    def contains(self, object):
        for obj in self.objects:
            if object == obj: # use 'is' instead of '==' ?
                return True
        return False

    def is_empty(self):
        return self.objects == []

    def is_unambiguous(self):
        return len(self.objects) == 1

    def is_ambiguous(self):
        return len(self.objects) > 1

    def add(self, object):
        if not self.contains(object):
            self.objects += [object]

    def split_n(self, n):
        return [self] + [deepcopy(self) for _ in range(n)]

    # for making temporal traces in the main worldstate
    # def add_timestep(self, objects, agent):


    def __repr__(self): # TODO: better repr, with timeline
        return str(self.objects) + f'\n\n{self.string}'

EMPTY_WORLDSTATE = WorldState([]) # empty worldstate, neutral element for union

def remove_empty_worldstates(worldstates):
    return [ws for ws in worldstates if not ws.is_empty()]

# locs, colors, sizes

MIN = -np.inf
MAX = np.inf
LEFT_LIM = -0.1
RIGHT_LIM = 0.1
TOP_LIM = 0.1
BOT_LIM = -0.1

class Location():

    def __init__(self, box, name):
        self.box = box
        self.name = name

    def contains(self, object):
        return self.box.contains(object.position)

# location concepts
LEFT = Location(
    Box(low=np.array([MIN, MIN]), high=np.array([LEFT_LIM, MAX]), dtype=np.float32),
    name='left'
)
RIGHT = Location(
    Box(low=np.array([RIGHT_LIM, MIN]), high=np.array([MAX, MAX]), dtype=np.float32),
    name='right'
)
TOP = Location(
    Box(low=np.array([MIN, TOP_LIM]), high=np.array([MAX, MAX]), dtype=np.float32),
    name='top'
)
BOTTOM = Location(
    Box(low=np.array([MIN, MIN]), high=np.array([MAX, BOT_LIM]), dtype=np.float32),
    name='bottom'
)
CENTER = Location(
    Box(low=np.array([LEFT_LIM, BOT_LIM]), high=np.array([RIGHT_LIM, TOP_LIM]), dtype=np.float32),
    name='center'
)

class Color():

    def __init__(self, box, name):
        self.box = box # can have several boxes in a list
        self.name = name

    def contains(self, object):
        if isinstance(self.box, list):
            return or_reduce([b.contains(object.rgb_code) for b in self.box])
        else:
            return self.box.contains(object.rgb_code)

# color concepts
# LIGHT_BLUE = Color(
#     Box(low=np.array([0.3, 0.7, 0.9]), high=np.array([0.5, 0.8, 1.]), dtype=np.float32),
#     name='light blue'
# )
# DARK_BLUE = Color(
#     Box(low=np.array([0.0, 0., 0.8]), high=np.array([0.2, 0.2, 0.9]), dtype=np.float32),
#     name='dark blue'
# )
# LIGHT_GREEN = Color(
#     Box(low=np.array([0.4, 0.8, 0.4]), high=np.array([0.6, 1, 0.5]), dtype=np.float32),
#     name='light green'
# )
# DARK_GREEN = Color(
#     Box(low=np.array([0., 0.4, 0.]), high=np.array([0.1, 0.6, 0.1]), dtype=np.float32),
#     name='dark green'
# )
# LIGHT_RED = Color(
#     Box(low=np.array([0.9, 0.4, 0.35]), high=np.array([1, 0.6, 0.65]), dtype=np.float32),
#     name='light red'
# )
# DARK_RED = Color(
#     Box(low=np.array([0.5, 0., 0.]), high=np.array([0.7, 0.1, 0.1]), dtype=np.float32),
#     name='dark red'
# )
BLUE = Color(
    [Box(low=np.array([0.3, 0.7, 0.9]), high=np.array([0.5, 0.8, 1.]), dtype=np.float32),
     Box(low=np.array([0., 0., 0.8]), high=np.array([0.2, 0.2, 0.9]), dtype=np.float32)],
    name='blue'
)
GREEN = Color(
    [Box(low=np.array([0.4, 0.8, 0.4]), high=np.array([0.6, 1, 0.5]), dtype=np.float32),
     Box(low=np.array([0., 0.4, 0.]), high=np.array([0.1, 0.6, 0.1]), dtype=np.float32)],
    name='green'
)
RED = Color(
    [Box(low=np.array([0.9, 0.4, 0.35]), high=np.array([1, 0.6, 0.65]), dtype=np.float32),
     Box(low=np.array([0.5, 0., 0.]), high=np.array([0.7, 0.1, 0.1]), dtype=np.float32)],
    name='red'
)
DARK = Color(
    Box(low=np.array([0., 0., 0.]), high=np.array([0.3, 0.3, 0.3]), dtype=np.float32),
    name='dark'
)

# relations

class BinarySpatialRelation():

    def __init__(self, truth_fn, name, mode='initial'): # other mode: current
        self.truth_fn = truth_fn
        self.name = name

    # TODO: for multiple referents, the first one is used, change this ("left of the red dogs")
    def relates(self, referents, target):
        return self.truth_fn(target.position, referents[0].position)

LEFT_OF = BinarySpatialRelation(
    truth_fn=lambda pos1, pos2: pos1[0] < pos2[0],
    name='left of'
)
RIGHT_OF = BinarySpatialRelation(
    truth_fn=lambda pos1, pos2: pos1[0] > pos2[0],
    name='right of'
)
TOP_OF = BinarySpatialRelation(
    truth_fn=lambda pos1, pos2: pos1[1] > pos2[1],
    name='top of'
)
BOTTOM_OF = BinarySpatialRelation(
    truth_fn=lambda pos1, pos2: pos1[1] < pos2[1],
    name='bottom of'
)
# TODO FAR_FROM

# temporal binary relations

class BinaryTemporalRelation():

    def __init__(self, truth_fn, name):
        self.truth_fn = truth_fn
        self.name = name

    def relates(self, referent_ws, target_ws): # inputs are worldstates
        return self.truth_fn(target_ws.timestamp, referent_ws.timestamp)

THEN = BinaryTemporalRelation(
    truth_fn=lambda time1, time2: time1 < time2,
    name='then'
)
AFTER = BinaryTemporalRelation(
    truth_fn=lambda time1, time2: time1 > time2,
    name='after'
)

# absolute relations

class AbsoluteSpatialRelation():

    def __init__(self, truth_fn, name):
        self.truth_fn = truth_fn
        self.name = name

    def relates(self, referents, target): # referents are all other objects
        referent_pos = mapl(lambda o: o.position, referents)
        map_relation_others = mapl(fct.partial(self.truth_fn, target.position), referent_pos)
        return foldl(and_binary_fn, True, map_relation_others)

LEFTMOST = AbsoluteSpatialRelation(
    truth_fn=lambda pos1, pos2: pos1[0] <= pos2[0],
    name='left most'
)
RIGHTMOST = AbsoluteSpatialRelation(
    truth_fn=lambda pos1, pos2: pos1[0] >= pos2[0],
    name='right most'
)
TOPMOST = AbsoluteSpatialRelation(
    truth_fn=lambda pos1, pos2: pos1[1] >= pos2[1],
    name='top most'
)
BOTTOMMOST = AbsoluteSpatialRelation(
    truth_fn=lambda pos1, pos2: pos1[1] <= pos2[1],
    name='bottom most'
)

### filtering functions

def identity_filter(worldstate):
    return worldstate.filter(lambda o: True)

# filter all 'obj' instances
def get_object_filter(obj):
    def object_filter(worldstate):
        is_instance = lambda o: isinstance(o, obj)
        return worldstate.filter(is_instance) # for now adding strings is done in the filter fn
    return object_filter, obj.name

# works with Box-based concepts, such as color and location
def get_contains_filter(concept):
    def contains_filter(worldstate):
        is_contained = lambda o: concept.contains(o)
        return worldstate.filter(is_contained)
    return contains_filter, concept.name

def get_abs_rel_filter(relation):
    def abs_rel_filter(worldstate):
        relates = lambda o: relation.relates(worldstate.objects, o)
        return worldstate.filter(relates)
    return abs_rel_filter, relation.name

def grasped_filter(worldstate):
    is_grasped = lambda o: o.grasp
    return worldstate.filter(is_grasped)

def get_predicate_filter(predname): # predname :: str; name of predicate, eg 'grasp'
    def predicate_filter(worldstate):
        def predicate_is_verified(o):
            try:
                return o.__dict__[predname]
            except KeyError:
                # print('invalid predname:',predname)
                return False
        return worldstate.filter(predicate_is_verified)
    return predicate_filter, predname

def get_size_change_filter(size_change):
    def size_change_filter(worldstate):
        is_grown = lambda o: size_change.applies(o)
        return worldstate.filter(is_grown)
    return size_change_filter, size_change.name

### compositional filtering functions

# These functions take as input 2 worldstates and return a third filtered
# one.

# returns a worldstate with objects present in both input worldstates
def intersection(wss):
    ws1, ws2 = wss # hacky fix
    ws2_contains = lambda obj: ws2.contains(obj)
    return ws1.filter(ws2_contains), ws2.applied_rules

# returns a worldstate with objects present in any input worldstates
# def union(wss):
#     ws1, ws2 = wss # hacky fix
#     for obj in ws2.objects:
#         ws1.add(obj)
#     return ws1
#
# def union_all(wss):
#     return foldl(union, EMPTY_WORLDSTATE, wss)

# relation: a relation is a function :: ref, target -> bool
# referents are on the left by convention in the relation_filter fn
def get_object_relation_filter(relation):
    def object_relation_filter(wss):
        ws1, ws2 = wss # hacky fix
        if not ws2.objects: # empty worldstate
            return ws2, []    # return empty # TODO rule application order is not preserved here
        relates = fct.partial(relation.relates, ws2.objects)
        return ws1.filter(relates), ws2.applied_rules
    return object_relation_filter, relation.name

def get_temporal_relation_filter(relation):
    def temporal_relation_filter(wss):
        ws1, ws2 = wss
        relates = relation.relates(ws2, ws1)
        return ws1.time_filter(relates), ws2.applied_rules
    return temporal_relation_filter, relation.name

# Rule tree base classes

class Token():
    def __init__(self, name, is_terminal=False, concept=None):
        self.name = name
        self.is_terminal = is_terminal
        self.concept = concept # storage for concept

        # completed by rule declarations
        self.outgoing_rules = []
        self.incoming_rule = None # for now only one incoming rule

    def mutate(self, applied_rules, rule_to_mutate):
        """
        Given a list of applied rules and a rule to mutate, returns a new
        valid list of applied rules. After checking this list corresponds
        to a false description, one can use it as a negative example.
        """
        if self.is_terminal:
            return []
        else:
            if not applied_rules:
               raise ValueError('Invalid applied rules')

            applied_rule = applied_rules.pop(0)
            if applied_rule == rule_to_mutate:
                applied_rule.mutate(applied_rules, rule_to_mutate)
                rules = self.sample_random()
                return rules
            else:
                rules = applied_rule.mutate(applied_rules, rule_to_mutate)
                rules = [applied_rule] + rules
                return rules

    def apply(self, worldstate, applied_rules):
        # careful, this consumes the list of applied rules
        if self.is_terminal:
            return worldstate
        else:
            if not applied_rules:
                raise ValueError('Invalid applied rules')

            applied_rule = applied_rules.pop(0)
            if applied_rule in self.outgoing_rules:
                ws = applied_rule.apply(worldstate, applied_rules)
                ws.applied_rules = [applied_rule] + ws.applied_rules
                return ws
            else:
                raise ValueError('Invalid applied rules')

    def sample_random(self):
        if self.is_terminal:
            return []
        else:
            rule = np.random.choice(self.outgoing_rules)
            return [rule] + rule.sample_random()

    def filter(self, worldstate, mode='train'):
        if self.is_terminal:
            return [worldstate]
        else:
            # worldstates = add_all_list([r.filter(worldstate) for r in self.outgoing_rules])
            worldstates = []
            for r in self.outgoing_rules:
                wss = r.filter(worldstate, mode=mode)
                worldstates += wss
            return worldstates

    def _all_descr(self):
        if self.is_terminal:
            return [[]]
        else:
            return add_all_list([r._all_descr() for r in self.outgoing_rules])
            # all_applied_rule_list = []
            # for r in self.outgoing_rules:
            #     all_applied_rule_list.append(r._all_descr())
            # return all_applied_rule_list

            # worldstates = []
            # for r in self.outgoing_rules:
            #     wss = r._all_descr(worldstate)
            #     worldstates += wss
            # return worldstates

    def _count(self):
        if self.is_terminal:
            return 1
        else:
            return sum([r._count() for r in self.outgoing_rules])

    def __repr__(self):
        return f'Token : {self.name}'


class Rule():
    def __init__(self, name, source_token, dest_tokens,
                 filter_name_couple=None, concept=None, filter_ambiguous=False,
                 filter_unambiguous=False):

        self.name = name
        self.source_token = source_token
        self.dest_tokens = dest_tokens
        self.concept = concept

        self.source_token.outgoing_rules += [self]
        for token in self.dest_tokens:
            token.incoming_rule = self

        assert len(self.dest_tokens) in [1, 2] # for now only unary and binary

        # filter function
        if filter_name_couple is None:
            if len(self.dest_tokens) == 2:
                # default filter fn for 2 dest tokens is intersection
                self.filter_fn = intersection
                self.string = ''
            elif len(self.dest_tokens) == 1:
                # for one dest token the default is identity
                self.filter_fn = identity_filter
                self.string = ''
        else:
            filter_fn, name = filter_name_couple
            self.filter_fn = filter_fn
            self.string = name

        # for generating splits
        self.forbidden = False
        self.forbidden_subrule_combinations = []

        # advanced filtering for relevance models
        self.filter_ambiguous = filter_ambiguous
        self.filter_unambiguous = filter_unambiguous

    def is_terminal_rule(self):
        return foldl(or_binary_fn, False, [o.is_terminal for o in self.dest_tokens])

    def reset(self):
        self.forbidden = False
        self.forbidden_subrule_combinations = []

    def forbid(self):
        self.forbidden = True

    def sample_forbidden_subrule_combinations(self, prop, **options):

        assert prop >= 0 and prop < 1

        # fills the forbidden_subrule_combination list with stuff
        if len(self.dest_tokens) == 1:
            return # for now only combinatorial generalization on binary rules
        if self.is_terminal_rule():
            return # terminal binary rules not yet supported

        # total possible combinations
        str_condition = lambda r: bool(r.string) # only list rules with nonempty string
        rule_dict_str0 = get_rule_dict(self.dest_tokens[0], str_condition)
        rule_dict_str1 = get_rule_dict(self.dest_tokens[1], str_condition)
        N = len(rule_dict_str0)
        M = len(rule_dict_str1)

        n = int(prop * N * M)

        # TODO check this
        l0 = list(rule_dict_str0.values())
        l1 = list(rule_dict_str1.values())
        combinations = mult_lists(l0, l1)
        self.forbidden_subrule_combinations = random.sample(combinations, n)
        # # l2 = ...
        # for i in range(n):
        #     # couple = sample_sans_remise(combinations)
        #     # self.forbidden_subrule_combinations.append(couple)
        #     first_subrule = np.random.choice(list(rule_dict_str0.values()))
        #     second_subrule = np.random.choice(list(rule_dict_str1.values()))
        #     self.forbidden_subrule_combinations.append(
        #         (first_subrule, second_subrule)
        #     )

    def authorized(self, applied_rules1, applied_rules2, mode):
        """
        For now we check the whole of the applied rules for unauthorized
        combinations.
        Another alternative would be to look only at the last applied rule.
        """
        # TODO what to do when list of forbidden combinations is nonempty and
        #  and self.forbidden is true ?
        if self.forbidden:
            if mode == 'train':
                return False
            elif mode == 'test':
                return True
            else:
                raise ValueError(f'mode must be "train" or "test", not {mode}')

        # if self.is_terminal_rule():
        #     return True
        if len(self.dest_tokens) == 1:
            return True

        if not self.forbidden_subrule_combinations:
            return True
            # TODO make the following work
            # if mode == 'train':
            #     return True
            # # in this case we look if any of the applied subrules have test-train
            # # splits
            # subrules_are_split = or_reduce(
            #     [bool(ar.forbidden_subrule_combinations) for ar in applied_rules1]
            # )
            # subrules_are_split = subrules_are_split or or_reduce(
            #     [bool(ar.forbidden_subrule_combinations) for ar in applied_rules2]
            # )
            # return subrules_are_split

        forbidden_list = [f1 in applied_rules1 and f2 in applied_rules2 for f1, f2 in \
                          self.forbidden_subrule_combinations]
        # forbidden = foldl(or_binary_fn, False, forbidden_list)
        forbidden = or_reduce(forbidden_list)

        if mode == 'train':
            return not forbidden
        elif mode == 'test':
            return forbidden
        else:
            raise ValueError(f'mode must be "train" or "test", not {mode}')

    def apply(self, worldstate, applied_rules):
        # given a list of applied rules and a worldstate, checks if the associated
        # sentence is true or false

        if len(self.dest_tokens) == 1:
            ws = self.dest_tokens[0].apply(worldstate, applied_rules)
            ws = self.filter_fn(ws)
            return ws

        else:
            ws1 = self.dest_tokens[0].apply(worldstate, applied_rules)
            ws2 = self.dest_tokens[1].apply(worldstate, applied_rules)
            ws, ar = self.filter_fn((ws1, ws2))
            ws.applied_rules += ar
            return ws

    def mutate(self, applied_rules, rule_to_mutate):
        if len(self.dest_tokens) == 1:
            ar = self.dest_tokens[0].mutate(applied_rules, rule_to_mutate)
            return ar
        else:
            ar1 = self.dest_tokens[0].mutate(applied_rules, rule_to_mutate)
            ar2 = self.dest_tokens[1].mutate(applied_rules, rule_to_mutate)
            return ar1 + ar2

    def sample_random(self):
        if len(self.dest_tokens) == 1:
            return self.dest_tokens[0].sample_random()
        else:
            return self.dest_tokens[0].sample_random() \
                 + self.dest_tokens[1].sample_random()

    def filter(self, worldstate, mode='train'):

        if len(self.dest_tokens) == 1:
            worldstates = mapl(
                self.filter_fn,
                self.dest_tokens[0].filter(worldstate, mode=mode)
            )
            wss = []
            for ws in worldstates:
                if ws.is_empty():
                    continue
                elif not self.authorized([], [], mode):
                    continue
                elif self.filter_ambiguous and ws.is_ambiguous():
                    continue
                elif self.filter_unambiguous and ws.is_unambiguous():
                    continue
                else:
                    ws.applied_rules = [self] + ws.applied_rules
                    wss.append(ws)
            return wss

        elif len(self.dest_tokens) == 2:
            # all possible combinations
            worldstate_couples = mult_lists(
                self.dest_tokens[0].filter(worldstate, mode=mode),
                self.dest_tokens[1].filter(worldstate, mode=mode)
            )
            results = mapl(self.filter_fn, worldstate_couples)
            # Add names of applied rules and remove empty worldstates
            # Also removes forbidden combinations
            wss = []
            for ws, applied_rules in results:

                if ws.is_empty():
                    continue
                elif not self.authorized(ws.applied_rules, applied_rules, mode):
                    continue
                # TODO relevance model here ?
                else:
                    ws.applied_rules = [self] + ws.applied_rules + applied_rules
                    wss.append(ws)
            return wss

    def _all_descr(self):

        if len(self.dest_tokens) == 1:
            all_applied_rules_0 = self.dest_tokens[0]._all_descr()
            all_applied_rules = []
            for applied_rules in all_applied_rules_0:
                all_applied_rules.append([self] + applied_rules)
            return all_applied_rules

        elif len(self.dest_tokens) == 2:
            all_applied_rules_0 = self.dest_tokens[0]._all_descr()
            all_applied_rules_1 = self.dest_tokens[1]._all_descr()
            all_applied_rules = []
            for ar0 in all_applied_rules_0:
                for ar1 in all_applied_rules_1:
                    all_applied_rules.append([self] + ar0 + ar1)
            return all_applied_rules

        else:
            raise ValueError(f'Rule with invalid number of dest_tokens ({len(self.dest_tokens)})')


    def _count(self):
        return foldl(mul_binary_fn, 1, [t._count() for t in self.dest_tokens])

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'{self.name}'

# tokens

S = Token(name='S')
P = Token(name='P')
P_T = Token(name='P_T')

pred = Token(name='pred') # for all actions
# go = Token(name='go', is_terminal=True)
grow = Token(name='grow', is_terminal=True)
grasp = Token(name='grasp', is_terminal=True)
shake = Token(name='shake', is_terminal=True)

thing_A = Token(name='thing_A')
thing_B = Token(name='thing_B')

loc = Token(name='loc')
attr = Token(name='attr')
subj = Token(name='subj')

# terminals
locs = [
    Token(name='left', is_terminal=True, concept=LEFT),
    Token(name='right', is_terminal=True, concept=RIGHT),
    Token(name='top', is_terminal=True, concept=TOP),
    Token(name='bottom', is_terminal=True, concept=BOTTOM),
    Token(name='center', is_terminal=True, concept=CENTER),
]
things = [Token(name=obj_name, is_terminal=True, concept=obj_class) \
          for obj_name, obj_class in plg_objects.items()]
you = Token(name='you', is_terminal=True, concept=plg.objects.Agent)
attrs = [
    Token(name='blue', is_terminal=True, concept=BLUE),
    Token(name='green', is_terminal=True, concept=GREEN),
    Token(name='red', is_terminal=True, concept=RED),
    # Token(name='dark', is_terminal=True, concept=DARK),
]

### Base GM

# rules
rules = []
rules.append(Rule(
    name='S -> P',
    source_token=S,
    dest_tokens=[P]
))
rules.append(Rule(
    name='P -> P_T',
    source_token=P,
    dest_tokens=[P_T]
))
rules.append(Rule(
    name='P_T -> subj loc',
    source_token=P_T,
    dest_tokens=[subj, loc],
))
rules.append(Rule(
    name='P_T -> pred thing_A',
    source_token=P_T,
    dest_tokens=[pred, thing_A],
))
# rules.append(Rule(  # for debug
#     name='P_T -> attr thing_B',
#     source_token=P_T,
#     dest_tokens=[attr, thing_B],
# ))
rules.append(Rule(
    name='thing_A -> thing_B',
    source_token=thing_A,
    dest_tokens=[thing_B]
))
rules.append(Rule(
    name='thing_A -> attr thing_B',
    source_token=thing_A,
    dest_tokens=[attr, thing_B],
))
# rules involving terminals
rules += [
    Rule(
        name='pred -> grasp',
        source_token=pred,
        dest_tokens=[grasp],
        filter_name_couple=get_predicate_filter('grasp')
    )
]

rules += [
    Rule(
        name='pred -> grow',
        source_token=pred,
        dest_tokens=[grow],
        filter_name_couple=get_predicate_filter('grow')
    )
]
rules += [
    Rule(
        name='pred -> shake',
        source_token=pred,
        dest_tokens=[shake],
        filter_name_couple=get_predicate_filter('shake')
    )
]
rules += [
    Rule(
        name=f'thing_B -> {t.name}',
        source_token=thing_B,
        dest_tokens=[t],
        filter_name_couple=get_object_filter(t.concept)) \
    for t in things
]
rules += [
    Rule(
        name='subj -> you',
        source_token=subj,
        dest_tokens=[you],
        filter_name_couple=get_object_filter(you.concept)
    )
]
rules += [
    Rule(
        name='subj -> thing_B',
        source_token=subj,
        dest_tokens=[thing_B],
    )
]
rules += [
    Rule(
        name=f'loc -> {l.name}',
        source_token=loc,
        dest_tokens=[l],
        filter_name_couple=get_contains_filter(l.concept)) \
    for l in locs
]
rules += [
    Rule(
        name=f'attr -> {a.name}',
        source_token=attr,
        dest_tokens=[a],
        filter_name_couple=get_contains_filter(a.concept)) \
    for a in attrs
]

### Spatial relation GM

# tokens

rel = Token(name='rel')
referent = Token(name='referent')
thing_C = Token(name='thing_C') # only points to Thing

spatial_rel = Token(name='spatial_rel')
left_of = Token(name='left of', is_terminal=True, concept=LEFT_OF)
right_of = Token(name='right of', is_terminal=True, concept=RIGHT_OF)
top_of = Token(name='top of', is_terminal=True, concept=TOP_OF)
bottom_of = Token(name='bottom of', is_terminal=True, concept=BOTTOM_OF)

abs_spatial_rel = Token(name='abs_spatial_rel')
leftmost = Token(name='left most', is_terminal=True, concept=LEFTMOST)
rightmost = Token(name='right most', is_terminal=True, concept=RIGHTMOST)
topmost = Token(name='top most', is_terminal=True, concept=TOPMOST)
bottommost = Token(name='bottom most', is_terminal=True, concept=BOTTOMMOST)

# rules

rules_spatial_rel = []

# Spatial relations subj (description of relative positions of objects)
#
rel_descr = Token(name='rel_descr')
rules += [
    Rule(
        name='loc -> rel_descr}',
        source_token=loc,
        dest_tokens=[rel_descr])
]
rules += [
    Rule(
        name='rel_descr -> referent',
        source_token=rel_descr,
        dest_tokens=[referent]
    )
]
abs_rel_descr = Token(name='abs_rel_descr')
rules += [
    Rule(
        name='P_T -> thing_B abs_rel_descr',
        source_token=P_T,
        dest_tokens=[thing_B, abs_rel_descr]
    )
]

rules += [
    Rule(
        name='abs_rel_descr -> absrel',
        source_token=abs_rel_descr,
        dest_tokens=[abs_spatial_rel]
    )
]

# comment this rule to detach spatial relations
rules += [
    Rule(
        name='thing_A -> rel',
        source_token=thing_A,
        dest_tokens=[rel]
    )
]
rules_spatial_rel += [
    Rule(
        name='rel -> thing_C referent',
        source_token=rel,
        dest_tokens=[thing_C, referent],
    )
]
rules_spatial_rel += [
    Rule(
        name='thing_C -> thing',
        source_token=thing_C,
        dest_tokens=[things[32]],
        # dest_tokens=[thing_B],
        # filter_unambiguous=True,
        filter_name_couple=get_object_filter(things[32].concept)
    )
]
rules_spatial_rel += [
    Rule(
        name=f'referent -> {spatial_rel.name} thing_B',
        source_token=referent,
        dest_tokens=[spatial_rel, thing_B],
        filter_name_couple=get_object_relation_filter(spatial_rel.concept)) \
    for spatial_rel in [left_of, right_of, top_of, bottom_of]
]
# comment this rule to detach absolute spatial relations
rules_spatial_rel += [
    Rule(
        name='thing_A -> abs_spatial_rel thing_C',
        source_token=thing_A,
        dest_tokens=[abs_spatial_rel, thing_C]
    )
]
rules_spatial_rel += [
    Rule(
        name=f'abs_spatial_rel -> {absrel.name}',
        source_token=abs_spatial_rel,
        dest_tokens=[absrel],
        filter_name_couple=get_abs_rel_filter(absrel.concept)
    ) \
    for absrel in [leftmost, rightmost, topmost, bottommost]
]
# absolute spatial relations

# temporal relation GM

# temp_rel = Token(name='temp_rel')
# P_T_target = Token(name='P_T_target')
# then = Token(name='then', is_terminal=True, concept=THEN)
# after = Token(name='after', is_terminal=True, concept=AFTER)
#
# # rules +=[
# #     Rule(
# #         name='S -> temp_rel',
# #         source_token=S,
# #         dest_tokens=[temp_rel]
# #     )
# # ]
# rules += [
#     Rule(
#         name='temp_rel -> P_T, P_T_target',
#         source_token=temp_rel,
#         dest_tokens=[P_T, P_T_target]
#     )
# ]
# rules += [
#     Rule(
#         name=f'P_T_target -> {trel.name} P_T',
#         source_token=temp_rel,
#         dest_tokens=[trel, P_T],
#         filter_name_couple=get_object_relation_filter(trel.concept)
#     ) for trel in [then, after]
# ]

# get all possible descriptions as strings

def get_all_descr():
    return mapl(applied_rules_to_string, S._all_descr())

def get_rule_list(token): # allows duplicates
    if token.is_terminal:
        return []
    l = []
    for rule in token.outgoing_rules:
        l += [rule]
        for dest_token in rule.dest_tokens:
            l += get_rule_list(dest_token)
    return l

def get_rule_dict(token, condition=lambda x: True):
    d = {}
    if token.is_terminal:
        return d
    for rule in token.outgoing_rules:
        if condition(rule):
            d[rule.name] = rule
        for dest_token in rule.dest_tokens:
            new_d = get_rule_dict(dest_token, condition=condition)
            d = {**d, **new_d}
    return d

# to get all possible sentences
def ws_list_to_string_list(wss):
    sentences = []
    for ws in wss:
        sentence = ''
        for i, rule in enumerate(ws.applied_rules):
            sentence = str_add(sentence, rule.string)
        sentences.append(sentence)
    return sentences

def applied_rules_to_string(applied_rules):
    return foldl(str_add, '', [r.string for r in applied_rules])

def are_unambiguous(wss):
    return [len(ws.objects) == 1 for ws in wss]

def rule_pair_comparison(wss): # input is pair of Worldstates
    ws1, ws2 = wss
    if len(ws1.applied_rules) != len(ws2.applied_rules):
        return []
    else:
        # TODO add the concept of incompatible rules (left of/right of etc)
        indices = []
        for i, (e1, e2) in enumerate(zip(ws1.applied_rules, ws2.applied_rules)):
            if e1 is not e2:
                indices.append(i)
        return indices

def compare_worldstate_lists(wss1, wss2):
    pairs = mult_lists(wss1, wss2)
    indices_list = mapl(rule_pair_comparison, pairs)
    return indices_list

def is_a_thing(token):
    return token in [thing_A, thing_B, thing_C]

# def present_past_comparison(wss_present, wss_past, return_present=False):
#     """
#     Give the present ws list and one from the past, returns the tense-adjusted
#     past sentences, and also present sentences if return_present is True.
#
#     wss_present and wss_past are both lists of Worldstates, returned by the
#     grammar at the present and some past time.
#     """
#     # TODO: incompatible rules to not generate stuff like grow was thing
#
#     sentence_list = [] # all sentences output by the fn\
#
#     if return_present:
#         # add all presently true sentences
#         sentence_list += ws_list_to_string_list(wss_present)
#
#     # for all wss in the past, compare them with all present wss
#     for ws_past in wss_past:
#         is_true_in_present = False
#         sentence_candidates = []
#         for ws_present in wss_present:
#
#             # test for equality of applied lists
#             if ws_present.applied_rules == ws_past.applied_rules:
#                 is_true_in_present = True
#
#             indices = rule_pair_comparison((ws_past, ws_present))
#
#             if not indices:
#                 continue
#
#             if len(indices) == 1:
#                 source_token = ws_past.applied_rules[indices[0]].source_token
#                 # the differing rule must not be a predicate-related rule
#                 if (source_token != pred) and not is_a_thing(source_token):
#                     sentence = ''
#                     for i, rule in enumerate(ws_past.applied_rules):
#                         if i == indices[0]:
#                             string = str_add('was', rule.string)
#                         else:
#                             string = rule.string
#                         sentence = str_add(sentence, string)
#
#                     if sentence not in sentence_candidates:
#                         sentence_candidates.append(sentence)
#                     # break
#
#         if not is_true_in_present:
#             if len(sentence_candidates) == 0:
#                 sentence = ws_list_to_string_list([ws_past])[0]
#                 sentence = str_add('was', sentence)
#                 sentence_list.append(sentence)
#             # TODO doenst work, also lists stupid stuff
#             sentence_list += sentence_candidates
#
#     return sentence_list

### truth functions of 1 trace of applied rules

def is_true(applied_rules, ws):
    ar = list(applied_rules)
    ws1 = S.apply(ws, ar)
    return not ws1.is_empty()

rule_dict = get_rule_dict(S)

if __name__ == '__main__':

    env = gym.make('TemporalPlayground-v1')
    # env_orig = gym.make('PlaygroundNavigation-v1')
    env.objects[0].grasp = True # to generate more interesting sentences
    # env.objects[1].grasped = True # to generate more interesting sentences
    # env.objects[2].grasped = True # to generate more interesting sentences

    # change initial size of one object to make it grow
    # env.objects[1].initial_size -= 0.1

    # change some positions at the beginning to control spatial relation change
    env.objects[0].position = np.array([-0.4, -0.4])
    env.objects[1].position = np.array([-0.2, -0.2])
    # env.objects.pop(-1)



    ws = WorldState(env.objects, env.agent)

    res = thing_B._all_descr()
    print(len(res))

    print(env.objects)

    t = time.time()
    res = S.filter(ws)
    print(f'dt: {time.time() - t}')
    pprint(ws_list_to_string_list(res))

    # test apply fn
    worldstate = res[-1]
    print(ws_list_to_string_list([worldstate]))
    print(worldstate.applied_rules)
    wsres = S.apply(ws, worldstate.applied_rules)

    # test mutate

    ar = res[-2].applied_rules
    # ar = wsres.applied_rules
    arr = list(ar) # we make a copy because rules list get consumed when we mutate them
    rule_to_mutate = np.random.choice(arr)
    arrr = S.mutate(arr, rule_to_mutate)
    print(f'rule <{rule_to_mutate}> will be mutated in the rule_list {ar}')
    print(f'Original : {applied_rules_to_string(ar)}')
    print(f'Mutated : {applied_rules_to_string(arrr)}')
    print(f'Mutated sentence is {is_true(arrr, ws)} in current Worldstate')

    # next ws
    ws2 = deepcopy(ws)
    ws2.objects[0].position = np.array([0.1, 0.1])
    # ws2.objects[0].grasped = False
    res2 = S.filter(ws2)
    pprint(ws_list_to_string_list(res2))

    # test spatial relations

    # rulename = "thing_C -> thing_B"
    rulename = "thing_A -> rel"
    r = rule_dict[rulename]
    wss = r.filter(ws)

    res = rel.filter(ws)
    ws1 = res[-1]
    ws2 = res[0]
    rf = left_of.concept
    lmst = leftmost.concept

    # env_orig.render()
    env.render()

    # abs rules for testing
    leftmost_rule = rules_spatial_rel[-4]
    rightmost_rule = rules_spatial_rel[-3]
    topmost_rule = rules_spatial_rel[-2]
    bottommost_rule = rules_spatial_rel[-1]

    o0 = ws.objects[0]
    o1 = ws.objects[1]
    o2 = ws.objects[2]
    ag = ws.objects[3]

    None

    ### temporal relations # TODO change playground
    # 2 timesteps

    # env.objects[0].position = np.stack(env.objects[0].position, env.objects[0].position, dim=0)
    # env.objects[1].position = np.stack(env.objects[1].position, env.objects[1].position, dim=0)