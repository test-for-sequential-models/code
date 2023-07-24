'''
We need to be able to do three things with an action g:
- Compute g.x
- Compute p(g.x)/p(x)
- Construct a (possibly infinite) set of actions g with different parameterizations.

The actions I've described in the thesis so far are quite simple in this regard.
For MLE-style actions, this becomes model dependent and seems to create a tight coupling
between the model and the action code.
Maybe best to parameterize actions by the model, and have the relative density
computation done in the action.
That way it's easy to support MLE-style actions, and also do the right thing w.r.t. alphabets.
'''

from abc import ABC
from typing import Optional
from collections import defaultdict

import numpy as np

from .sequences import Sequence
from .models import BasicEditType
from .models import SequenceDist


class Action(ABC):
    def apply(self, sequence: Sequence) -> tuple[Sequence, list[BasicEditType]]:
        raise NotImplementedError()


class ActionSet(ABC):
    """
    A possibly infinite set of actions.
    For a given sequence, the set is required to finite.
    Typically, the set is just a function of sequence length and alphabet.
    """

    def __or__(self, other):
        return UnionOfActionSets(self, other)

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        raise NotImplementedError()


class UnionOfActionSets(ActionSet):
    def __init__(self, *action_sets: ActionSet):
        self._action_sets = action_sets

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        sequences = dict()
        for action_set in self._action_sets:
            sequences |= action_set.apply(sequence)
        return sequences


class InsertAction(Action):
    def __init__(self, complement: bool, at: int, x: int):
        self._complement = complement
        self._at = at
        self._x = x

    def apply(self, sequence: Sequence) -> tuple[Sequence, list[BasicEditType]]:
        if self._complement:
            return sequence.insert_complement(self._at, self._x)
        else:
            return sequence.insert(self._at, self._x)


class InsertActionSet(ActionSet):
    def __init__(self, complement: bool, min_ix: int, max_ix: Optional[int], xs: list[int]):
        self._complement = complement
        self._min_ix = min_ix
        self._max_ix = max_ix
        self._xs = xs

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        sequences = dict()
        max_ix = self._max_ix if self._max_ix is not None else len(sequence) + 1
        max_ix = min(max_ix, len(sequence) + 1)
        for ix in range(self._min_ix, max_ix):
            for x in self._xs:
                action = InsertAction(complement=self._complement, at=ix, x=x)
                sequences |= [action.apply(sequence)]
        return sequences


class SubAction(Action):
    def __init__(self, complement: bool, at: int, x: int):
        self._complement = complement
        self._at = at
        self._x = x

    def apply(self, sequence: Sequence) -> tuple[Sequence, list[BasicEditType]]:
        if self._complement:
            return sequence.sub_complement(self._at, self._x)
        else:
            return sequence.sub(self._at, self._x)


class SubActionSet(ActionSet):
    def __init__(self, complement: bool, min_ix: int, max_ix: Optional[int], xs: list[int]):
        self._complement = complement
        self._min_ix = min_ix
        self._max_ix = max_ix
        self._xs = xs

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        sequences = dict()
        max_ix = self._max_ix if self._max_ix is not None else len(sequence)
        max_ix = min(max_ix, len(sequence))
        for ix in range(self._min_ix, max_ix):
            for x in self._xs:
                action = SubAction(complement=self._complement, at=ix, x=x)
                sequences |= [action.apply(sequence)]
        return sequences


class IncrDecrActionSet(ActionSet):
    def __init__(self, complement: bool, min_ix: int, max_ix: Optional[int], the_range: int, xs: list[int]):
        self._complement = complement
        self._min_ix = min_ix
        self._max_ix = max_ix
        xs = [x for x in xs]
        assert len(set(xs)) == len(xs), "alphabet elements must be unique"
        assert the_range >= 1, "range must be positive"
        self._map = defaultdict(lambda: set())
        for d in range(1, the_range + 1):
            for i in range(len(xs)):
                j = (i + d) % len(xs)
                self._map[xs[i]].add(xs[j])
                self._map[xs[j]].add(xs[i])

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        sequences = dict()
        max_ix = self._max_ix if self._max_ix is not None else len(sequence)
        max_ix = min(max_ix, len(sequence))
        for ix in range(self._min_ix, max_ix):
            real_ix = len(sequence) - ix - 1 if self._complement else ix
            prev_x = sequence[real_ix]
            for x in self._map[prev_x]:
                action = SubAction(complement=self._complement, at=ix, x=x)
                sequences |= [action.apply(sequence)]
        return sequences


class DelAction(Action):
    def __init__(self, complement: bool, at: int):
        self._complement = complement
        self._at = at

    def apply(self, sequence: Sequence) -> tuple[Sequence, list[BasicEditType]]:
        if self._complement:
            return sequence.del_complement(self._at)
        else:
            return sequence.del_(self._at)


class DelActionSet(ActionSet):
    def __init__(self, complement: bool, min_ix: int, max_ix: Optional[int]):
        self._complement = complement
        self._min_ix = min_ix
        self._max_ix = max_ix

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        sequences = dict()
        max_ix = self._max_ix if self._max_ix is not None else len(sequence)
        max_ix = min(max_ix, len(sequence))
        for ix in range(self._min_ix, max_ix):
            action = DelAction(complement=self._complement, at=ix)
            sequences |= [action.apply(sequence)]
        return sequences


class PreAction(Action):
    def __init__(self, complement: bool, at: int):
        self._complement = complement
        self._at = at

    def apply(self, sequence: Sequence) -> tuple[Sequence, list[BasicEditType]]:
        if self._complement:
            return sequence.pre_complement(self._at)
        else:
            return sequence.pre(self._at)


class PreActionSet(ActionSet):
    def __init__(self, complement: bool, min_ix: int, max_ix: Optional[int]):
        self._complement = complement
        assert min_ix > 0, "pre_0 is either a no-op (complement=False) or gives the empty seq (complement=True)"
        self._min_ix = min_ix
        self._max_ix = max_ix

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        sequences = dict()
        max_ix = self._max_ix if self._max_ix is not None else len(sequence)
        max_ix = min(max_ix, len(sequence))
        for ix in range(self._min_ix, max_ix):
            action = PreAction(complement=self._complement, at=ix)
            sequences |= [action.apply(sequence)]
        return sequences


class MaxProbSubAction(Action):
    def __init__(self, model: SequenceDist, tail: bool, num_to_sub: int):
        if not tail:
            raise ValueError("tail=False not implemented")
        self._num_to_sub = num_to_sub
        self._model = model

    def apply(self, sequence: Sequence) -> tuple[Sequence, list[BasicEditType]]:
        new_seq, all_edits = sequence.pre_complement(self._num_to_sub)
        for _ in range(self._num_to_sub):
            max_likelihood_char, max_log_prob = None, -np.inf
            for char in self._model.alphabet:
                candidate_seq, edits = new_seq.insert_complement(0, char)
                log_p_ratio = self._model.log_p_edits_over_p_noedit(new_seq.to_numpy(), candidate_seq.to_numpy(), edits)
                if log_p_ratio > max_log_prob:
                    max_likelihood_char, max_log_prob = char, log_p_ratio
            new_seq, edits = new_seq.insert_complement(0, max_likelihood_char)
            all_edits += edits
        return new_seq, all_edits


class MaxProbSubActionSet(ActionSet):
    def __init__(self, model: SequenceDist, tail: bool, min_num_to_sub: int, max_num_to_sub: Optional[int]):
        self._model = model
        self._tail = tail
        assert min_num_to_sub > 0, "max_prob_sub_0 is a no-op"
        self._min_num_to_sub = min_num_to_sub
        self._max_num_to_sub = max_num_to_sub

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        sequences = dict()
        max_num = self._max_num_to_sub if self._max_num_to_sub is not None else len(sequence)
        max_num = min(max_num, len(sequence) - 1)
        for num_to_sub in range(self._min_num_to_sub, max_num + 1):
            action = MaxProbSubAction(self._model, self._tail, num_to_sub)
            sequences |= [action.apply(sequence)]
        return sequences


class MinProbSubAction(Action):
    def __init__(self, model: SequenceDist, tail: bool, num_to_sub: int):
        if not tail:
            raise ValueError("tail=False not implemented")
        self._num_to_sub = num_to_sub
        self._model = model

    def apply(self, sequence: Sequence) -> tuple[Sequence, list[BasicEditType]]:
        new_seq, all_edits = sequence.pre_complement(self._num_to_sub)
        for _ in range(self._num_to_sub):
            min_likelihood_char, min_log_prob = None, np.inf
            for char in self._model.alphabet:
                candidate_seq, edits = new_seq.insert_complement(0, char)
                log_p_ratio = self._model.log_p_edits_over_p_noedit(new_seq.to_numpy(), candidate_seq.to_numpy(), edits)
                if log_p_ratio < min_log_prob:
                    min_likelihood_char, min_log_prob = char, log_p_ratio
            new_seq, edits = new_seq.insert_complement(0, min_likelihood_char)
            all_edits += edits
        return new_seq, all_edits


class MinProbSubActionSet(ActionSet):
    def __init__(self, model: SequenceDist, tail: bool, min_num_to_sub: int, max_num_to_sub: Optional[int]):
        self._model = model
        self._tail = tail
        assert min_num_to_sub > 0, "min_prob_sub_0 is a no-op"
        self._min_num_to_sub = min_num_to_sub
        self._max_num_to_sub = max_num_to_sub

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        sequences = dict()
        max_num = self._max_num_to_sub if self._max_num_to_sub is not None else len(sequence)
        max_num = min(max_num, len(sequence) - 1)
        for num_to_sub in range(self._min_num_to_sub, max_num + 1):
            action = MinProbSubAction(self._model, self._tail, num_to_sub)
            sequences |= [action.apply(sequence)]
        return sequences



class Composition(ActionSet):
    def __init__(self, setA: ActionSet, setB: ActionSet):
        self._setA = setA
        self._setB = setB

    def apply(self, sequence: Sequence) -> dict[Sequence, list[BasicEditType]]:
        sequences = dict()
        for alt_seqA, editsA in self._setA.apply(sequence).items():
            for alt_seqB, editsB in self._setB.apply(alt_seqA).items():
                sequences[alt_seqB] = editsA + editsB
        return sequences
