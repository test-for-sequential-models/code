import unittest

import numpy as np

from .actions import InsertAction
from .actions import InsertActionSet
from .actions import SubActionSet
from .actions import IncrDecrActionSet
from .actions import DelActionSet
from .actions import PreActionSet
from .actions import MaxProbSubActionSet
from .models import PoissonBernoulliModel
from .sequences import Sequence


class TestSequence(unittest.TestCase):
    def test_insert(self):
        x = Sequence([0, 1, 2, 3])
        action = InsertAction(complement=True, at=0, x=5)
        self.assertSequenceEqual(action.apply(x)[0], Sequence([0, 1, 2, 3, 5]))
        action = InsertAction(complement=True, at=1, x=5)
        self.assertSequenceEqual(action.apply(x)[0], Sequence([0, 1, 2, 5, 3]))
        action = InsertAction(complement=False, at=0, x=5)
        self.assertSequenceEqual(action.apply(x)[0], Sequence([5, 0, 1, 2, 3]))
        action = InsertAction(complement=False, at=1, x=5)
        self.assertSequenceEqual(action.apply(x)[0], Sequence([0, 5, 1, 2, 3]))

    def test_insert_set(self):
        x = Sequence([0, 1, 2, 3])
        action_set = InsertActionSet(complement=False, min_ix=1, max_ix=None, xs=[5, 6])
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 5, 1, 2, 3],
                [0, 6, 1, 2, 3],
                [0, 1, 5, 2, 3],
                [0, 1, 6, 2, 3],
                [0, 1, 2, 5, 3],
                [0, 1, 2, 6, 3],
                [0, 1, 2, 3, 5],
                [0, 1, 2, 3, 6],
            ]
        })
        action_set = InsertActionSet(complement=True, min_ix=1, max_ix=2, xs=[5, 6])
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 1, 2, 5, 3],
                [0, 1, 2, 6, 3],
            ]
        })

    def test_sub_set(self):
        x = Sequence([0, 1, 2, 3])
        action_set = SubActionSet(complement=False, min_ix=1, max_ix=None, xs=[5, 6])
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 5, 2, 3],
                [0, 6, 2, 3],
                [0, 1, 5, 3],
                [0, 1, 6, 3],
                [0, 1, 2, 5],
                [0, 1, 2, 6],
            ]
        })
        action_set = SubActionSet(complement=True, min_ix=1, max_ix=2, xs=[5, 6])
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 1, 5, 3],
                [0, 1, 6, 3],
            ]
        })

    def test_incrdecr_set(self):
        x = Sequence([0, 1, 2, 3])
        action_set = IncrDecrActionSet(complement=False, min_ix=1, max_ix=None, the_range=1, xs=[1, 3, 0, 2, 4, 5])
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 1, 2, 0],
                [0, 1, 2, 1],
                [0, 1, 4, 3],
                [0, 1, 0, 3],
                [0, 3, 2, 3],
                [0, 5, 2, 3],
            ]
        })
        action_set = IncrDecrActionSet(complement=True, min_ix=1, max_ix=2, the_range=2, xs=[1, 3, 0, 2, 4, 5])
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 1, 4, 3],
                [0, 1, 0, 3],
                [0, 1, 5, 3],
                [0, 1, 3, 3],
            ]
        })

    def test_del_set(self):
        x = Sequence([0, 1, 2, 3])
        action_set = DelActionSet(complement=False, min_ix=1, max_ix=None)
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 2, 3],
                [0, 2, 3],
                [0, 1, 3],
                [0, 1, 3],
                [0, 1, 2],
                [0, 1, 2],
            ]
        })
        action_set = DelActionSet(complement=True, min_ix=1, max_ix=2)
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 1, 3],
                [0, 1, 3],
            ]
        })

    def test_pre_set(self):
        x = Sequence([0, 1, 2, 3, 4, 5])
        action_set = PreActionSet(complement=False, min_ix=2, max_ix=None)
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3],
                [0, 1, 2],
                [0, 1],
            ]
        })
        action_set = PreActionSet(complement=True, min_ix=2, max_ix=4)
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 1, 2, 3],
                [0, 1, 2],
            ]
        })

    def test_max_prob_sub_set(self):
        x = Sequence([0, 0, 0, 0, 0])
        model = PoissonBernoulliModel(lmbda=len(x), p=0.8)
        action_set = MaxProbSubActionSet(tail=True, model=model, min_num_to_sub=2, max_num_to_sub=None)
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
            ]
        })
        action_set = MaxProbSubActionSet(tail=True, model=model, min_num_to_sub=1, max_num_to_sub=3)
        self.assertSetEqual(set(action_set.apply(x)), {
            Sequence(x) for x in [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [0, 0, 1, 1, 1],
            ]
        })
