from typing import Optional
from typing import Literal
from typing import Union
import heapq
import warnings

import numpy as np

from .free_vector_space import FreeVector
from .models import SequenceDist, BasicEditType, apply_basic_edit
from .actions import ActionSet
from .actions import MaxProbSubAction
from .sequences import Sequence


class SteinOperator:
    @property
    def params(self):
        warnings.warn("params property of operators (and kernels, and models) is deprecated",
                      category=DeprecationWarning)
        return NotImplementedError()

    def __hash__(self):
        warnings.warn("hash of operators (and kernels, and models) is deprecated", category=DeprecationWarning)
        return hash((self.__class__, self.params))

    def __eq__(self, other):
        warnings.warn("equality of operators (and kernels, and models) is deprecated", category=DeprecationWarning)
        return (self.__class__, self.params) == (other.__class__, other.params)

    def apply_one(self, sequence) -> FreeVector:
        raise NotImplementedError()

    def apply(self, sequences: FreeVector) -> FreeVector:
        result = FreeVector()
        for sequence, seq_weight in sequences.dict_items():
            result += seq_weight * self.apply_one(sequence)
        if np.isnan(list(result.to_dict().values())).any():
            raise RuntimeError(f"stein({sequences})={result} contains NaN weights")
        return result


class GibbsSteinOperator(SteinOperator):
    def __init__(self, model: SequenceDist):
        self._model = model

    @property
    def params(self):
        return (self._model,)

    def apply_one(self, sequence) -> FreeVector:
        l = len(sequence)
        result = -FreeVector(tuple(sequence), np.array(sequence))
        y_sequence = [s for s in sequence]
        alphabet = list(self._model.alphabet)
        for i in range(l):
            weights = np.zeros(len(alphabet))
            for j, alternative in enumerate(alphabet):
                weights[j] = self._model.p_edit_over_p_noedit(sequence, ('replace', i, alternative))
            weights /= weights.sum()
            for j, alternative in enumerate(alphabet):
                y_sequence[i] = alternative
                weight = weights[j]
                # weight = self._model.p_singleton_conditional(np.array(sequence)[None, :], i, alternative)[0]
                if weight > 0:
                    result += FreeVector(tuple(y_sequence), np.array(y_sequence)) * weight / l
                else:
                    print(y_sequence, i)
                    raise ValueError("impossible sequence")
            y_sequence[i] = sequence[i]
        return result


class OldStyleZanellaSteinOperator(SteinOperator):
    def __init__(
            self,
            model: SequenceDist,
            permissible_actions: list[str] = None,
            fixed_neigh_size: Optional[int] = 1,
            num_replaces_by_likelihood: Optional[int] = None,
    ):
        self._model = model
        self._permissible_actions = permissible_actions
        if fixed_neigh_size is False:
            fixed_neigh_size = None
        self._fixed_neigh_size = fixed_neigh_size
        self._num_replaces_by_likelihood = num_replaces_by_likelihood

    @property
    def params(self):
        return (
            self._model,
            tuple(self._permissible_actions),
            self._fixed_neigh_size,
            self._num_replaces_by_likelihood,
        )

    @staticmethod
    def _barker_kappa(t):
        return t / (t + 1)

    def apply_one(self, sequence) -> FreeVector:
        l = len(sequence)
        result = FreeVector()
        h_seq = FreeVector(tuple(sequence), np.array(sequence))
        actions: list[BasicEditType] = []
        if self._fixed_neigh_size is not None:
            ixs = set(range(l - self._fixed_neigh_size, l)) & set(range(l))
        else:
            ixs = range(l)
        alphabet = list(self._model.alphabet)
        for i in ixs:
            for alternative in alphabet:
                actions.append(('replace', i, alternative))
                actions.append(('insert', i + 1, alternative))
            actions.append(('incr', i))
            actions.append(('pop', i))
            actions.append(('drop_tail', i + 1))
        if self._fixed_neigh_size is None or self._fixed_neigh_size > l:
            for alternative in alphabet:
                actions.append(('insert', 0, alternative))
        replaces = []
        h_seq_weight = 0
        for action in actions:
            # TODO enforce symmetry
            if self._permissible_actions is not None and action[0] not in self._permissible_actions:
                continue
            action_for_model = action
            if action[0] == 'incr':
                ix = action[1]
                xi_next = alphabet[(alphabet.index(sequence[ix]) + 1) % len(alphabet)]
                action_for_model = 'replace', ix, xi_next
            p_rel = self._model.p_edit_over_p_noedit(sequence, action_for_model)
            weight = self._barker_kappa(p_rel)
            y_sequence = apply_basic_edit(action_for_model, sequence)
            if weight > 0:
                if action[0] == 'replace' and self._num_replaces_by_likelihood is not None:
                    replaces.append((weight, y_sequence))
                else:
                    tup = tuple(y_sequence)
                    if not tup in result:
                        result += weight * FreeVector(tup, np.array(y_sequence))
                        h_seq_weight -= weight
        if self._num_replaces_by_likelihood is not None:
            for weight, y_sequence in heapq.nlargest(self._num_replaces_by_likelihood, replaces):
                tup = tuple(y_sequence)
                if not tup in result:
                    result += weight * FreeVector(tup, np.array(y_sequence))
                    h_seq_weight -= weight
        result += h_seq_weight * h_seq
        return result


kappa_str = Union[Literal['barker'], Literal['mpf'], Literal['mh']]


def _kappa(kappa: kappa_str, t: float) -> float:
    if kappa == 'barker':
        return t / (t + 1)
    elif kappa == 'mpf':
        return np.sqrt(t)
    elif kappa == 'mh':
        return min(1., t)
    else:
        raise ValueError(f"invalid choice of kappa function: {kappa}")


class ZanellaSteinOperator(SteinOperator):
    def __init__(self, model: SequenceDist, action_set: ActionSet, kappa: kappa_str):
        self._model = model
        self._action_set = action_set
        self._kappa_str = kappa

    def _kappa(self, t):
        return _kappa(self._kappa_str, t)

    def apply_one(self, sequence) -> FreeVector:
        result = FreeVector()
        sequence = Sequence(sequence)
        total_weight = 0
        for alt_sequence, edits in self._action_set.apply(sequence).items():
            log_prob_ratio = self._model.log_p_edits_over_p_noedit(sequence.to_numpy(), alt_sequence.to_numpy(), edits)
            if log_prob_ratio == np.inf:
                weight = 1
            else:
                prob_ratio = np.exp(log_prob_ratio)
                if prob_ratio == np.inf:
                    weight = 1
                else:
                    weight = self._kappa(prob_ratio)
            if np.isnan(weight):
                raise RuntimeError(
                    f"Z-S stein weight is NaN: sequence={sequence}, alt={alt_sequence}, edits={edits}, log_prob_ratio={log_prob_ratio}")
            result += weight * FreeVector(tuple(alt_sequence), alt_sequence.to_numpy())
            total_weight += weight
        result += (-total_weight) * FreeVector(tuple(sequence), sequence.to_numpy())
        return result


class SumOfSteinOperators(SteinOperator):
    def __init__(self, *operators):
        self._operators = operators

    def apply_one(self, sequence) -> FreeVector:
        result = FreeVector()
        for operator in self._operators:
            result += operator.apply_one(sequence)
        return result


class MaxLikelihoodZSOperator(SteinOperator):
    def __init__(self, model: SequenceDist, num_to_sub: int, kappa: kappa_str):
        self._model = model
        self._num_to_sub = num_to_sub
        self._kappa_str = kappa
        self._action = MaxProbSubAction(model=self._model, tail=True, num_to_sub=self._num_to_sub)

    def _kappa(self, t):
        return _kappa(self._kappa_str, t)

    def apply_one(self, sequence) -> FreeVector:
        if len(sequence) < self._num_to_sub + 1:
            return FreeVector()
        prefix = apply_basic_edit(('drop_tail', self._num_to_sub), sequence)
        sequence = Sequence(sequence)
        alt_sequence, _edits = self._action.apply(sequence)
        log_prob_ratio = self._model.log_p_wildcard_suffix_over_p_sequence(prefix, self._num_to_sub)
        vector = FreeVector(tuple(alt_sequence), alt_sequence.to_numpy())
        if sequence == alt_sequence:
            # we are max likelihood - need to point towards wildcard away from us
            vector *= -1
        else:
            # we are not max likelihood - need to point towards max likelihood
            log_prob_ratio *= -1
        weight = self._kappa(np.exp(log_prob_ratio))
        return weight * vector
