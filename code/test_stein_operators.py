import unittest

import numpy as np

from .free_vector_space import FreeVector
from .stein_operators import GibbsSteinOperator
from .stein_operators import OldStyleZanellaSteinOperator
from .models import NgramModel


def list_of_vec_of_single_seq(x):
    return [vec_of_single_seq(x)]


def vec_of_single_seq(x):
    return FreeVector(tuple(x), np.array(x))


def make_ngram_model(*, n_states, E_length):
    def add_absorption(non_absorbing_kernel, E_length):
        P_stop = 1 / E_length
        P_continue = 1 - P_stop
        absorption = P_stop * np.ones(non_absorbing_kernel.shape[:-1] + (1,))
        transition_kernel = np.concatenate([absorption, P_continue * non_absorbing_kernel], -1)
        return transition_kernel

    def ensure_positive_density(original_kernel):
        proposed_kernel = np.maximum(original_kernel, 1e-3)
        return proposed_kernel / proposed_kernel.sum(-1, keepdims=True)

    def construct_model(memory):
        transitions = np.zeros((n_states + 1, n_states + 1, n_states))
        transitions[0, 0, :] = 1 / n_states
        random_walk_kernel = 1 / 2 * (np.roll(np.eye(n_states), -1, 0) + np.roll(np.eye(n_states), 1, 0))
        transitions[:, 1:, :] += random_walk_kernel[None, :, :]
        ix = np.arange(n_states)
        ix_p1 = np.roll(ix, -1)
        ix_p2 = np.roll(ix, -2)
        transitions[ix + 1, ix_p1 + 1, ix] = 1 - memory
        transitions[ix + 1, ix_p1 + 1, ix_p2] = memory
        make_kernel = lambda k: add_absorption(ensure_positive_density(k), E_length)
        return NgramModel(make_kernel(transitions))

    return construct_model(0.2)


class TestGibbsSteinOperator(unittest.TestCase):
    def test_overlap_with_zanella(self):
        entropy = 64858763165975086361888970758625481449
        s = np.random.SeedSequence(entropy)
        rng = np.random.default_rng(s)
        for it in range(10):
            tgt_len = rng.poisson(5) + 1
            n = 2
            x = rng.integers(n, size=tgt_len) + 1
            with self.subTest(iteration=it, len=tgt_len, n_states=n, sequence=x):
                model = make_ngram_model(n_states=n, E_length=tgt_len)
                op_gibbs = GibbsSteinOperator(model)
                op_zanella = OldStyleZanellaSteinOperator(model, ['replace'], None)
                x_gibbs = (op_gibbs.apply_one(x) * tgt_len).to_dict()
                x_zanella = op_zanella.apply_one(x).to_dict()
                # assertDictAlmostEqual does not exist, so we do it ourselves:
                self.assertSetEqual(set(x_gibbs.keys()), set(x_zanella.keys()))
                for key in x_gibbs.keys():
                    self.assertAlmostEqual(x_gibbs[key], x_zanella[key], msg=f"key={key}")

    def test_weights_add_to_zero(self):
        entropy = 85588496074808424438713635500671398109
        s = np.random.SeedSequence(entropy)
        rng = np.random.default_rng(s)
        for it in range(10):
            tgt_len = rng.poisson(5) + 1
            n = rng.poisson(4) + 1
            x = rng.integers(n, size=tgt_len) + 1
            with self.subTest(iteration=it, len=tgt_len, n_states=n, sequence=x):
                model = make_ngram_model(n_states=n, E_length=tgt_len)
                op_gibbs = GibbsSteinOperator(model)
                x_gibbs = op_gibbs.apply_one(x).to_dict()
                self.assertAlmostEqual(0, sum(x_gibbs.values()))
