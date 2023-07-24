from collections import deque
from typing import Union, Literal, Optional

import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp
import scipy.special as special

from .sequences import Sequence

BasicEditType = Union[
    tuple[Literal["insert"], int, any],
    tuple[Literal["replace"], int, any],
    tuple[Literal["pop"], int],
    tuple[Literal["drop_tail"], int],
]


# TODO remove drop_tail once no longer referenced


class SequenceDist:
    @property
    def params(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash((self.__class__, self.params))

    def __eq__(self, other):
        return (self.__class__, self.params) == (other.__class__, other.params)

    @property
    def alphabet(self):
        raise NotImplementedError()

    def generate_samples(self, rng, N_sequences):
        raise NotImplementedError()

    def p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        return np.exp(self.log_p_edit_over_p_noedit(sequence, edit))

    def log_p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        raise NotImplementedError()

    def log_p_sequence(self, sequence) -> float:
        raise NotImplementedError()

    def log_p_edits_over_p_noedit(self, old_sequence, new_sequence, edits: list[BasicEditType]) -> float:
        edited_seq = old_sequence.copy()
        log_prob_ratio = 0
        for edit in edits:
            log_prob_ratio += self.log_p_edit_over_p_noedit(edited_seq, edit)
            edited_seq = apply_basic_edit(edit, edited_seq)
            if log_prob_ratio == -np.inf and len(edits) == 1:
                return -np.inf
            if log_prob_ratio == np.inf and len(edits) == 1:
                return np.inf
        assert Sequence(edited_seq) == Sequence(new_sequence), "inconsistent list of edits"
        return log_prob_ratio

    def log_p_wildcard_suffix_over_p_sequence(self, sequence, suffix_length: int) -> float:
        raise NotImplementedError()


class PoissonBernoulliModel(SequenceDist):
    def __init__(self, lmbda, p):
        self._lmbda = lmbda
        self._p = p
        self._p_length_cache = {}

    @property
    def params(self):
        return (self._lmbda, self._p)

    @property
    def alphabet(self):
        return [0, 1]

    def generate_samples(self, rng, N_sequences):
        lengths = rng.poisson(self._lmbda, size=N_sequences)
        return [rng.binomial(1, self._p, size=l) for l in lengths]

    def log_p_sequence(self, sequence) -> float:
        log_p_length = stats.poisson(self._lmbda).logpmf(len(sequence))
        count_ones = np.count_nonzero(sequence)
        log_p_ones = count_ones * np.log(self._p)
        log_p_zeros = (len(sequence) - count_ones) * np.log(1 - self._p)
        return log_p_length + log_p_ones + log_p_zeros

    def log_p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        p = self.p_edit_over_p_noedit(sequence, edit)
        if p == 0:
            return -np.inf
        return np.log(p)

    def p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        kind, *args = edit
        if kind == 'insert':
            _, new_xi = args
            if new_xi == 1:
                prob = self._p
            else:
                prob = 1 - self._p
            rel_p_length = self._p_length(len(sequence) + 1) / self._p_length(len(sequence))
            return prob * rel_p_length
        elif kind == 'pop':
            [index] = args
            old_xi = sequence[index]
            if old_xi == 1:
                prob = self._p
            else:
                prob = 1 - self._p
            rel_p_length = self._p_length(len(sequence) - 1) / self._p_length(len(sequence))
            return rel_p_length / prob
        elif kind == 'replace':
            index, alternative_xi = args
            if alternative_xi == 1:
                prob_new = self._p
            else:
                prob_new = 1 - self._p
            old_xi = sequence[index]
            if old_xi == 1:
                prob_old = self._p
            else:
                prob_old = 1 - self._p
            return prob_new / prob_old
        elif kind == 'drop_tail':
            [num_elem] = args
            assert num_elem > 0
            old_xis = sequence[-num_elem:]
            count_ones = np.count_nonzero(old_xis)
            rel_p_length = self._p_length(len(sequence) - num_elem) / self._p_length(len(sequence))
            prob_old = self._p ** count_ones * (1 - self._p) ** (len(old_xis) - count_ones)
            return rel_p_length / prob_old
        else:
            raise ValueError("invalid edit kind")

    def _p_length(self, length):
        if length in self._p_length_cache:
            return self._p_length_cache[length]
        else:
            p = stats.poisson(self._lmbda).pmf(length)
            self._p_length_cache[length] = p
            return p

    def log_p_wildcard_suffix_over_p_sequence(self, sequence, suffix_length: int) -> float:
        return np.log(self._p_length(len(sequence) + suffix_length)) - np.log(self._p_length(len(sequence)))


class MarkovChainModel(SequenceDist):
    def __init__(self, transition_kernel, initial_dist):
        self._initial_dist = initial_dist
        self._transition_kernel = transition_kernel

    @property
    def params(self):
        return (
            tuple(x for x in self._initial_dist.ravel()),
            tuple(x for x in self._transition_kernel.ravel()),
        )

    @property
    def alphabet(self):
        return range(1, len(self._initial_dist))

    def generate_samples(self, rng, N_sequences):
        result = []
        N = len(self._initial_dist)
        for _ in range(N_sequences):
            sequence = []
            init_dist = self._initial_dist[1:]
            elem = rng.choice(N - 1, p=init_dist / init_dist.sum()) + 1  # zero length not allowed
            while elem != 0:
                sequence.append(elem)
                elem = rng.choice(N, p=self._transition_kernel[elem, :])
            result.append(np.array(sequence))
        return result

    def log_p_sequence(self, sequence) -> float:
        if len(sequence) == 0:
            return -np.inf
        x_prev = sequence[0]
        result = np.log(self._initial_dist[x_prev])
        for i in range(1, len(sequence)):
            x_next = sequence[i]
            result += np.log(self._transition_kernel[x_prev, x_next])
            x_prev = x_next
        return result

    def log_p_wildcard_suffix_over_p_sequence(self, sequence, suffix_length: int) -> float:
        if not np.allclose(self._transition_kernel[1:, 0], self._transition_kernel[1, 0]):
            raise NotImplementedError()
        return np.log(1 - self._transition_kernel[1, 0]) * suffix_length

    def log_p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        p = self.p_edit_over_p_noedit(sequence, edit)
        if p == 0:
            return -np.inf
        return np.log(p)

    def p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        kind, *args = edit
        if kind == 'insert':
            ix, new_xi = args
            if ix > len(sequence):
                raise ValueError("insert action ix is greater than sequence length")
            if ix == 0:
                x_next = sequence[ix]
                return (
                        self._initial_dist[new_xi] * self._transition_kernel[new_xi, x_next]
                        / self._initial_dist[x_next]
                )
            elif ix == len(sequence):
                x_prev = sequence[ix - 1]
                return (
                        self._transition_kernel[x_prev, new_xi] * self._transition_kernel[new_xi, 0]
                        / self._transition_kernel[x_prev, 0]
                )
            else:
                x_prev = sequence[ix - 1]
                x_next = sequence[ix]
                return (
                        self._transition_kernel[x_prev, new_xi] * self._transition_kernel[new_xi, x_next]
                        / self._transition_kernel[x_prev, x_next]
                )
        elif kind == 'drop_tail':
            [amount] = args
            if len(sequence) < amount + 1:
                return 0  # zero length sequences not allowed
            if amount < 1:
                raise ValueError("cannot drop_tail zero or less")
            x_last = sequence[-1]
            result = 1 / self._transition_kernel[x_last, 0]
            for i in range(1, amount + 1):
                xi = sequence[-(i + 1)]
                result *= 1 / self._transition_kernel[xi, x_last]
                x_last = xi
            result *= self._transition_kernel[x_last, 0]
            return result
        elif kind == 'pop':
            [ix] = args
            if len(sequence) < 2:
                return 0  # zero length sequences not allowed
            if ix >= len(sequence):
                raise ValueError("cannot pop beyond sequence end")
            if ix == 0 and len(sequence) > 1:
                x_drop = sequence[ix]
                x_next = sequence[ix + 1]
                return (
                        self._initial_dist[x_next]
                        / (self._initial_dist[x_drop] * self._transition_kernel[x_drop, x_next])
                )
            elif ix > 0 and ix == len(sequence) - 1 and len(sequence) > 1:
                x_drop = sequence[ix]
                x_prev = sequence[ix - 1]
                return (
                        self._transition_kernel[x_prev, 0]
                        / (self._transition_kernel[x_prev, x_drop] * self._transition_kernel[x_drop, 0])
                )
            elif ix > 0 and ix < len(sequence) - 1 and len(sequence) > 1:
                x_next = sequence[ix + 1]
                x_drop = sequence[ix]
                x_prev = sequence[ix - 1]
                return (
                        self._transition_kernel[x_prev, x_next]
                        / (self._transition_kernel[x_prev, x_drop] * self._transition_kernel[x_drop, x_next])
                )
            else:
                raise ValueError("should be impossible")
        elif kind == 'replace':
            ix, new_xi = args
            if ix == 0 and len(sequence) == 1:
                x_was = sequence[ix]
                return (
                        self._initial_dist[new_xi] * self._transition_kernel[new_xi, 0]
                        / (self._initial_dist[x_was] * self._transition_kernel[x_was, 0])
                )
            elif ix == 0 and len(sequence) > 1:
                x_was = sequence[ix]
                x_next = sequence[ix + 1]
                return (
                        self._initial_dist[new_xi] * self._transition_kernel[new_xi, x_next]
                        / (self._initial_dist[x_was] * self._transition_kernel[x_was, x_next])
                )
            elif ix > 0 and ix == len(sequence) - 1:
                x_prev = sequence[ix - 1]
                x_was = sequence[ix]
                return (
                        self._transition_kernel[x_prev, new_xi] * self._transition_kernel[new_xi, 0]
                        / (self._transition_kernel[x_prev, x_was] * self._transition_kernel[x_was, 0])
                )
            elif ix > 0 and ix < len(sequence) - 1:
                x_prev = sequence[ix - 1]
                x_was = sequence[ix]
                x_next = sequence[ix + 1]
                return (
                        self._transition_kernel[x_prev, new_xi] * self._transition_kernel[new_xi, x_next]
                        / (self._transition_kernel[x_prev, x_was] * self._transition_kernel[x_was, x_next])
                )
            else:
                raise ValueError("should be impossible")
        else:
            raise ValueError("invalid edit kind")


class NgramModel(SequenceDist):
    def __init__(self, transition_kernel):
        self._transition_kernel = transition_kernel

    @classmethod
    def fit(cls, n_in_ngram: int, n_states: int, sequences: list[Sequence]) -> 'NgramModel':
        transition_kernel = np.zeros([n_states + 1] * (n_in_ngram + 1))
        for sequence in sequences:
            sequence = Sequence(sequence)
            padded = [0] * n_in_ngram + list(sequence.to_numpy()) + [0]
            for i_start in range(len(sequence) + 1):
                subseq = padded[i_start:][:n_in_ngram + 1]
                transition_kernel[tuple(subseq)] += 1
        transition_kernel += np.where(np.sum(transition_kernel, axis=-1, keepdims=True) > 0, 0, 1)
        transition_kernel /= np.sum(transition_kernel, axis=-1, keepdims=True)
        return cls(transition_kernel)

    @property
    def params(self):
        return (
            tuple(x for x in self._transition_kernel.ravel()),
        )

    @property
    def alphabet(self):
        return range(1, self._transition_kernel.shape[0])

    def generate_samples(self, rng, N_sequences):
        result = []
        N_states = self._transition_kernel.shape[0]
        N_in_ngram = len(self._transition_kernel.shape) - 1
        for _ in range(N_sequences):
            sequence = []
            window = deque([0] * N_in_ngram)
            init_dist = self._transition_kernel[tuple(window)][1:]
            elem = rng.choice(N_states - 1, p=init_dist / init_dist.sum()) + 1  # zero length not allowed
            while elem != 0:
                sequence.append(elem)
                window.append(elem)
                window.popleft()
                elem = rng.choice(N_states, p=self._transition_kernel[tuple(window)])
            result.append(np.array(sequence))
        return result

    def log_p_sequence(self, sequence) -> float:
        if len(sequence) == 0:
            return -np.inf
        N_in_ngram = len(self._transition_kernel.shape) - 1
        window = deque([0] * N_in_ngram)
        result = 0
        for i in range(len(sequence) + 1):
            x_next = sequence[i] if i < len(sequence) else 0
            result += np.log(self._transition_kernel[tuple(window)][x_next])
            window.append(x_next)
            window.popleft()
        return result

    def log_p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        p = self.p_edit_over_p_noedit(sequence, edit)
        if p == 0:
            return -np.inf
        return np.log(p)

    def log_p_wildcard_suffix_over_p_sequence(self, sequence, suffix_length: int) -> float:
        N_in_ngram = len(self._transition_kernel.shape) - 1
        p_stop = self._transition_kernel[tuple([1] * N_in_ngram)][0]
        if not np.allclose(self._transition_kernel[..., 0], p_stop):
            raise NotImplementedError()
        return np.log(1 - p_stop) * suffix_length

    def p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        N_states = self._transition_kernel.shape[0]
        N_in_ngram = len(self._transition_kernel.shape) - 1
        kind, *args = edit
        ix = args[0]
        zero_pad = [0] * max(0, N_in_ngram - ix)
        x_prev = list(sequence[max(0, ix - N_in_ngram):ix])
        x_cur_and_next = list(sequence[ix:ix + N_in_ngram + 1])
        if len(x_cur_and_next) < N_in_ngram + 1:
            x_cur_and_next.append(0)
        padded_current_values = zero_pad + x_prev + x_cur_and_next
        if kind == 'insert':
            _, new_xi = args
            if ix > len(sequence):
                raise ValueError("cannot insert after end of sequence")
            padded_new_values = zero_pad + x_prev + [new_xi] + x_cur_and_next
        elif kind == 'pop':
            if len(sequence) == 1:
                return 0  # zero length not allowed
            if ix >= len(sequence):
                raise ValueError("cannot pop after end of sequence")
            padded_new_values = zero_pad + x_prev + x_cur_and_next[1:]
        elif kind == 'replace':
            _, new_xi = args
            if ix >= len(sequence):
                raise ValueError("cannot replace after end of sequence")
            padded_new_values = zero_pad + x_prev + [new_xi] + x_cur_and_next[1:]
        elif kind == 'drop_tail':
            if ix == len(sequence):
                return 0  # zero length not allowed
            if ix > len(sequence):
                raise ValueError("cannot drop more than the entire sequence")
            end_incl_tail = list(sequence[max(-(ix + N_in_ngram), -len(sequence)):])
            zero_pad = [0] * max(0, (ix + N_in_ngram) - len(sequence))
            padded_current_values = zero_pad + end_incl_tail + [0]
            padded_new_values = zero_pad + end_incl_tail[:-ix] + [0]
        else:
            raise ValueError("invalid edit kind")
        result = 1
        range_current = len(padded_current_values) - N_in_ngram
        range_new = len(padded_new_values) - N_in_ngram
        range_both = min(range_current, range_new)
        for i in range(range_both):
            result *= (
                    self._transition_kernel[tuple(padded_new_values[i:][:N_in_ngram + 1])]
                    / self._transition_kernel[tuple(padded_current_values[i:][:N_in_ngram + 1])]
            )
        for i in range(range_both, range_new):
            result *= self._transition_kernel[tuple(padded_new_values[i:][:N_in_ngram + 1])]
        for i in range(range_both, range_current):
            result /= self._transition_kernel[tuple(padded_current_values[i:][:N_in_ngram + 1])]
        return result

    def interpolate(self, other: 'NgramModel', my_weight: float, other_weight: float):
        assert self._transition_kernel.shape == other._transition_kernel.shape
        transitions = self._transition_kernel * my_weight + other._transition_kernel * other_weight
        transitions /= my_weight + other_weight
        return NgramModel(transitions)


class BetaBinomialModel(SequenceDist):
    def __init__(self, a: float, b: float, E_length: float):
        self._E_length = E_length
        self._dist_length = stats.poisson(E_length - 1)
        self._a = a
        self._b = b

    @property
    def alphabet(self):
        return [0, 1]

    def generate_samples(self, rng, N_sequences):
        sequences = []
        for _ in range(N_sequences):
            length = rng.poisson(self._E_length - 1) + 1
            p_event = rng.beta(self._a, self._b)
            sequence = rng.binomial(1, p_event, size=length)
            sequences.append(sequence)
        return sequences

    def log_p_sequence(self, sequence) -> float:
        length = len(sequence)
        if length == 0:
            # don't allow empty sequences
            return -np.inf
        n_events = np.sum(sequence)
        return self._log_p_sequence(length, n_events)

    def _log_p_sequence(self, length: int, n_events: int) -> float:
        return (
                self._dist_length.logpmf(length - 1)
                + stats.betabinom.logpmf(n_events, length, self._a, self._b)
                - np.log(special.binom(length, n_events))
        )

    def log_p_wildcard_suffix_over_p_sequence(self, sequence, suffix_length: int) -> float:
        length = len(sequence)
        return self._dist_length.logpmf(length + suffix_length - 1) - self._dist_length.logpmf(length - 1)

    def log_p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        length = len(sequence)
        n_events = np.sum(sequence)
        baseline = self._log_p_sequence(length, n_events)
        kind, *args = edit
        ix = args[0]
        if kind == 'insert':
            _, new_xi = args
            if ix > len(sequence):
                raise ValueError("cannot insert after end of sequence")
            return self._log_p_sequence(length + 1, n_events + new_xi) - baseline
        elif kind == 'pop':
            if len(sequence) == 1:
                return -np.inf  # zero length not allowed
            if ix >= len(sequence):
                raise ValueError("cannot pop after end of sequence")
            old_xi = sequence[ix]
            return self._log_p_sequence(length - 1, n_events - old_xi) - baseline
        elif kind == 'replace':
            _, new_xi = args
            if ix >= len(sequence):
                raise ValueError("cannot replace after end of sequence")
            old_xi = sequence[ix]
            return self._log_p_sequence(length, n_events + new_xi - old_xi) - baseline
        elif kind == 'drop_tail':
            if ix == len(sequence):
                return -np.inf  # zero length not allowed
            if ix > len(sequence):
                raise ValueError("cannot drop more than the entire sequence")
            n_events_dropped = np.sum(sequence[-ix:])
            return self._log_p_sequence(length - ix, n_events - n_events_dropped) - baseline
        else:
            raise ValueError("invalid edit kind")


class GriddedPointProcess(SequenceDist):
    def __init__(self, max_num_events: int, E_length: float):
        self._E_length = E_length
        self._dist_length = stats.poisson(E_length)
        self._max_num_events = max_num_events
        self._dist_events = stats.randint(0, self._max_num_events)

    @property
    def alphabet(self):
        return range(self._max_num_events + 1)

    def generate_samples(self, rng, N_sequences):
        sequences = []
        for _ in range(N_sequences):
            length = rng.poisson(self._E_length)
            num_events = rng.integers(self._max_num_events + 1)
            sequence = rng.multinomial(num_events, [1 / length] * length)
            sequences.append(sequence)
        return sequences

    def log_p_sequence(self, sequence) -> float:
        length = len(sequence)
        log_p_length = self._dist_length.logpmf(length)
        n_events = np.sum(sequence)
        if n_events > self._max_num_events:
            return -np.inf
        log_p_num_events = self._dist_events.logpmf(n_events)
        if n_events > 0:
            log_p_location = stats.multinomial(n=n_events, p=[1 / length] * length).logpmf(sequence)
        else:
            log_p_location = 0
        assert not np.isnan(log_p_length)
        assert not np.isnan(log_p_num_events)
        assert not np.isnan(log_p_location)
        return log_p_length + log_p_num_events + log_p_location

    def log_p_wildcard_suffix_over_p_sequence(self, sequence, suffix_length: int) -> float:
        length = len(sequence)
        n_events = np.sum(sequence)
        if n_events > self._max_num_events:
            return -np.inf
        log_p_rel_len = self._dist_length.logpmf(length + suffix_length) - self._dist_length.logpmf(length)
        max_num_to_add = self._max_num_events - n_events
        logs_of_p_rel_num_events = []
        # p_wildcard_suffix = int_{n_added, suffix} P(len=.) P(N(prefix)=.|len) P(suffix|N(prefix)=.,len)
        for n_added in range(0, max_num_to_add + 1):
            # p(prefix) cancels out in the rel
            # p(length) cancels out because it's uniform
            # so we just need p(N(prefix) | N(total))
            binom = stats.binom.logpmf(n_added, n_added + n_events, suffix_length / (suffix_length + length))
            logs_of_p_rel_num_events.append(binom)
        log_p_rel_num_events = logsumexp(logs_of_p_rel_num_events)
        return log_p_rel_len + log_p_rel_num_events

    def log_p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        alt_seq = apply_basic_edit(edit, sequence)
        return self.log_p_sequence(alt_seq) - self.log_p_sequence(sequence)


class CTMCAdjustedModel(SequenceDist):
    def __init__(self, underlying_model: SequenceDist, action_set: 'actions.ActionSet',
                 kappa: 'stein_operators.kappa_str', T: float):
        self._underlying_model = underlying_model
        self._action_set = action_set
        from .stein_operators import _kappa
        self._kappa = lambda t: _kappa(kappa, t)
        self._T = T

    def generate_samples(self, rng, N_sequences):
        underlying_samples = self._underlying_model.generate_samples(rng, N_sequences)
        samples = []
        for sample in underlying_samples:
            sample = Sequence(sample)
            cuml_time = 0
            num_jumps = 0
            while cuml_time < self._T:
                weights = {}
                neighbourhood = self._action_set.apply(sample)
                for alt_sequence, edits in neighbourhood.items():
                    log_prob_ratio = self._underlying_model.log_p_edits_over_p_noedit(sample.to_numpy(),
                                                                                      alt_sequence.to_numpy(),
                                                                                      edits)
                    if log_prob_ratio == np.inf:
                        weight = 1
                    else:
                        weight = self._kappa(np.exp(log_prob_ratio))
                    if np.isnan(weight):
                        raise RuntimeError(
                            f"Z-S stein weight is NaN: sequence={sample}, alt={alt_sequence}, edits={edits}, log_prob_ratio={log_prob_ratio}")
                    weights[alt_sequence] = weight
                alt_seq = list(neighbourhood.keys())
                weights = np.array([weights[k] for k in alt_seq])
                rate = rng.exponential(scale=1 / weights.sum())
                cuml_time += rate
                if cuml_time < self._T:
                    num_jumps += 1
                    p = weights / weights.sum()
                    sample = rng.choice(alt_seq, p=p)
                    sample = Sequence(sample)
            samples.append(sample)
        return samples


all_basic_edit_kinds = [
    'insert', 'replace', 'pop', 'drop_tail',
]


def random_basic_edit(rng: np.random.Generator, sequence: Sequence, model: SequenceDist,
                      desired_kind: Optional[str]) -> Optional[BasicEditType]:
    if desired_kind is None:
        kind = rng.choice(all_basic_edit_kinds)
    else:
        kind = desired_kind
    if kind == 'insert':
        ix = rng.integers(len(sequence) + 1)
        char = rng.choice(model.alphabet)
        return kind, ix, char
    elif kind == 'replace':
        ix = rng.integers(len(sequence))
        char = rng.choice(model.alphabet)
        return kind, ix, char
    elif kind == 'pop':
        ix = rng.integers(len(sequence))
        return kind, ix
    elif kind == 'drop_tail':
        if len(sequence) == 1:
            return None
        ix = rng.integers(1, len(sequence))
        return kind, ix
    else:
        raise AssertionError("impossible")


def apply_basic_edit(edit: BasicEditType, sequence):
    kind, *args = edit
    sequence = [s for s in sequence]
    if kind == 'insert':
        index, new_xi = args
        sequence.insert(index, new_xi)
    elif kind == 'pop':
        [index] = args
        sequence.pop(index)
    elif kind == 'replace':
        index, new_xi = args
        sequence[index] = new_xi
    elif kind == 'drop_tail':
        [num_elem] = args
        assert num_elem > 0
        sequence = sequence[:-num_elem]
    else:
        raise ValueError("invalid basic edit kind")
    return sequence


def mc_add_absorption(non_absorbing_kernel, E_length):
    n_states = non_absorbing_kernel.shape[0]
    P_stop = 1 / E_length
    P_continue = 1 - P_stop
    absorption = P_stop * np.ones(n_states)
    absorbed_state = np.zeros(n_states)
    transition_kernel = np.block([
        [1, absorbed_state[None, :]],
        [absorption[:, None], P_continue * non_absorbing_kernel],
    ])
    return transition_kernel


def ng_add_absorption(non_absorbing_kernel, E_length):
    P_stop = 1 / E_length
    P_continue = 1 - P_stop
    absorption = P_stop * np.ones(non_absorbing_kernel.shape[:-1] + (1,))
    transition_kernel = np.concatenate([absorption, P_continue * non_absorbing_kernel], -1)
    return transition_kernel


def ensure_positive_density(original_kernel):
    proposed_kernel = np.maximum(original_kernel, 1e-3)
    return proposed_kernel / proposed_kernel.sum(-1, keepdims=True)


class RandomWalk(MarkovChainModel):
    def __init__(self, *, E_length, n_states):
        kernel = 1 / 2 * (np.roll(np.eye(n_states), -1, 0) + np.roll(np.eye(n_states), 1, 0))
        unif_initial_dist = np.block([0, 1 / n_states * np.ones(n_states)])
        make_kernel = lambda k: mc_add_absorption(ensure_positive_density(k), E_length)
        super().__init__(make_kernel(kernel), unif_initial_dist)


class RandomUniformIID(NgramModel):
    def __init__(self, *, E_length, n_states):
        transitions = np.zeros(n_states + 1)
        transitions[1:] = 1 / n_states
        transitions[0] = 1 / E_length
        transitions[1:] *= 1 - transitions[0]
        super().__init__(transitions)


class RandomWalkWithHolding(MarkovChainModel):
    def __init__(self, *, E_length, n_states, holding_probability, n_holding_states):
        assert n_states >= 2
        unif_initial_dist = np.block([0, 1 / n_states * np.ones(n_states)])
        make_kernel = lambda k: mc_add_absorption(ensure_positive_density(k), E_length)
        kernel = 1 / 2 * (np.roll(np.eye(n_states), -1, 0) + np.roll(np.eye(n_states), 1, 0))
        for i in range(min(n_holding_states, n_states)):
            kernel[i, i] += holding_probability
            kernel[i, (i + 1) % n_states] *= 1 - holding_probability
            if n_states > 2:
                kernel[i, i - 1] *= 1 - holding_probability
        super().__init__(make_kernel(kernel), unif_initial_dist)


class RandomWalkWithMemory(NgramModel):
    def __init__(self, *, E_length, n_states, memory):
        transitions = np.zeros((n_states + 1, n_states + 1, n_states))
        transitions[0, 0, :] = 1 / n_states
        random_walk_kernel = 1 / 2 * (np.roll(np.eye(n_states), -1, 0) + np.roll(np.eye(n_states), 1, 0))
        transitions[:, 1:, :] += random_walk_kernel[None, :, :]
        ix = np.arange(n_states)
        ix_p1 = np.roll(ix, -1)
        ix_p2 = np.roll(ix, -2)
        transitions[ix + 1, ix_p1 + 1, ix] = 1 - memory
        transitions[ix + 1, ix_p1 + 1, ix_p2] = memory
        make_kernel = lambda k: ng_add_absorption(ensure_positive_density(k), E_length)
        super().__init__(make_kernel(transitions))


class RandomNgramModel(NgramModel):
    def __init__(self, *, rng, E_length, n_in_ngram, n_states, concentration_param):
        transitions = rng.dirichlet(alpha=[concentration_param] * n_states, size=[n_states + 1] * n_in_ngram)
        make_kernel = lambda k: ng_add_absorption(ensure_positive_density(k), E_length)
        super().__init__(make_kernel(transitions))


class RandomNgramModelWithFixedInitDist(NgramModel):
    def __init__(self, *, rng, E_length, n_in_ngram, n_states, concentration_param, initial_dist):
        assert np.all(initial_dist >= 0), "initial_dist contains negative entries, {initial_dist} should be >= 0"
        assert np.isclose(initial_dist.sum(), 1), "initial_dist does not sum to 1, {initial_dist.sum()} != 1"
        transitions = rng.dirichlet(alpha=[concentration_param] * n_states, size=[n_states + 1] * n_in_ngram)
        transitions[tuple([0] * n_in_ngram), :] = initial_dist
        make_kernel = lambda k: ng_add_absorption(ensure_positive_density(k), E_length)
        super().__init__(make_kernel(transitions))


class SimpleMRFModel(SequenceDist):
    def __init__(
            self, *,
            n_states: int,
            concentration_param: float,
            length_potential: float,
            max_length: int,
    ):
        self._n_states = n_states
        self._concentration_param = concentration_param
        self._max_length = max_length
        C = length_potential
        alpha = concentration_param
        N = n_states
        self._length_potential = C - np.log(np.exp(alpha) + N - 1)
        self._log_partition = self._compute_partition()

    def _compute_partition(self) -> float:
        length_potential = 0
        log_partition_parts = []
        for l in range(1, self._max_length + 1):
            length_potential += self._length_potential
            N = self._n_states
            alpha = self._concentration_param
            log_partition_part = np.log(N)
            log_partition_part += (l - 1) * np.log(np.exp(alpha) + N - 1)
            log_partition_part += length_potential
            log_partition_parts.append(log_partition_part)
        return logsumexp(log_partition_parts)

    def generate_samples(self, rng: np.random.Generator, N_sequences: int):
        sequences = []
        N = self._n_states
        alpha = self._concentration_param
        C = self._length_potential
        length_log_ps = []
        for l in range(1, self._max_length + 1):
            log_p = C * l + np.log(N) + (l - 1) * np.log(np.exp(alpha) + N - 1) - self._log_partition
            length_log_ps.append(log_p)
        assert np.isclose(sum(np.exp(length_log_ps)), 1)
        for i in range(N_sequences):
            l = rng.choice(range(self._max_length), p=np.exp(length_log_ps)) + 1
            sequence = [rng.choice(list(self.alphabet))]
            for i in range(1, l):
                if rng.uniform() < np.exp(alpha) / (np.exp(alpha) + N - 1):
                    sequence.append(sequence[-1])
                else:
                    sequence.append(rng.choice(list(set(self.alphabet) - {sequence[-1]})))
            sequences.append(sequence)
        return sequences

    def _pair_potential(self, sequence, ix: int) -> float:
        if ix < 0 or ix + 1 >= len(sequence):
            raise ValueError()
        if sequence[ix] == sequence[ix + 1]:
            return self._concentration_param
        else:
            return 0

    def log_p_edit_over_p_noedit(self, sequence, edit: BasicEditType) -> float:
        kind, *args = edit
        if kind == 'insert':
            if len(sequence) >= self._max_length:
                return -np.inf
            ix, value = args
            result = self._length_potential
            if ix > 0 and sequence[ix - 1] == value:
                result += self._concentration_param
            if ix < len(sequence) and sequence[ix] == value:
                result += self._concentration_param
            if ix > 0 and ix < len(sequence):
                result -= self._pair_potential(sequence, ix - 1)
            return result
        elif kind == 'replace':
            ix, value = args
            result = 0
            if ix > 0 and sequence[ix - 1] == value:
                result += self._concentration_param
            if ix + 1 < len(sequence) and sequence[ix + 1] == value:
                result += self._concentration_param
            if ix > 0:
                result -= self._pair_potential(sequence, ix - 1)
            if ix + 1 < len(sequence):
                result -= self._pair_potential(sequence, ix)
            return result
        elif kind == 'pop':
            if len(sequence) <= 1:
                return -np.inf
            [ix] = args
            result = -self._length_potential
            if ix > 0:
                result -= self._pair_potential(sequence, ix - 1)
            if ix + 1 < len(sequence):
                result -= self._pair_potential(sequence, ix)
            if ix > 0 and ix + 1 < len(sequence) and sequence[ix - 1] == sequence[ix + 1]:
                result += self._concentration_param
            return result
        elif kind == 'drop_tail':
            [num_elems] = args
            result = -self._length_potential * num_elems
            for i in range(num_elems):
                result -= self._pair_potential(sequence, len(sequence) - i - 2)
            return result
        else:
            raise ValueError()

    def log_p_sequence(self, sequence) -> float:
        potential = 0
        if len(sequence) < 1 or len(sequence) > self._max_length:
            return -np.inf
        for i in range(len(sequence)):
            potential += self._length_potential
            if i + 1 < len(sequence):
                potential += self._pair_potential(sequence, i)
        return potential - self._log_partition

    def log_p_wildcard_suffix_over_p_sequence(self, sequence, suffix_length: int) -> float:
        if len(sequence) + suffix_length > self._max_length:
            return -np.inf
        result = self._length_potential * suffix_length
        result += suffix_length * np.log(np.exp(self._concentration_param) + self._n_states - 1)
        return result

    @property
    def alphabet(self):
        return range(self._n_states)
