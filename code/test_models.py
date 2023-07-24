import unittest

import numpy as np
from scipy.special import logsumexp

from .models import PoissonBernoulliModel, all_basic_edit_kinds, random_basic_edit, apply_basic_edit
from .models import MarkovChainModel
from .models import NgramModel
from .models import RandomNgramModel
from .models import GriddedPointProcess
from .models import BetaBinomialModel
from .models import SimpleMRFModel


def make_mc_model():
    def add_absorption(non_absorbing_kernel, E_length):
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

    def ensure_positive_density(original_kernel):
        proposed_kernel = np.maximum(original_kernel, 1e-3)
        return proposed_kernel / proposed_kernel.sum(1, keepdims=True)

    n_states = 11
    E_length = 9
    random_walk_kernel = 1 / 2 * (np.roll(np.eye(n_states), -1, 0) + np.roll(np.eye(n_states), 1, 0))
    unif_initial_dist = np.block([0, 1 / n_states * np.ones(n_states)])
    make_kernel = lambda k: add_absorption(ensure_positive_density(k), E_length)
    random_walk_model = MarkovChainModel(make_kernel(random_walk_kernel), unif_initial_dist)
    return random_walk_model


def make_ngram_model():
    def add_absorption(non_absorbing_kernel, E_length):
        P_stop = 1 / E_length
        P_continue = 1 - P_stop
        absorption = P_stop * np.ones(non_absorbing_kernel.shape[:-1] + (1,))
        transition_kernel = np.concatenate([absorption, P_continue * non_absorbing_kernel], -1)
        return transition_kernel

    def ensure_positive_density(original_kernel):
        proposed_kernel = np.maximum(original_kernel, 1e-3)
        return proposed_kernel / proposed_kernel.sum(-1, keepdims=True)

    n_states = 11
    E_length = 9

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


def get_models_to_test(rng):
    return {
        'PoissonBernoulli': PoissonBernoulliModel(10, 0.2),
        'MarkovChain': make_mc_model(),
        'Ngram': make_ngram_model(),
        'RandomNgram': RandomNgramModel(rng=rng, n_in_ngram=4, n_states=5, E_length=10, concentration_param=1),
        'GriddedPointProcess': GriddedPointProcess(E_length=10, max_num_events=60),
        'BetaBinom': BetaBinomialModel(E_length=2, a=5, b=7),
        'SimpleMRFModel': SimpleMRFModel(
            length_potential=0.13,
            concentration_param=1.1,
            n_states=5,
            max_length=20,
        ),
    }


class TestAbstractModel(unittest.TestCase):
    def test_p_is_less_than_one(self):
        entropy = 62584326178936123534599337828600855733
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        models = get_models_to_test(rng)
        for label, model in models.items():
            with self.subTest(model=label):
                for it in range(10):
                    [x] = model.generate_samples(rng, 1)
                    with self.subTest(iteration=it, sequence=x):
                        log_px = model.log_p_sequence(x)
                        self.assertLessEqual(log_px, 0, msg=f'{log_px=}')

    def test_p_edit_over_noedit(self):
        entropy = 225431498788516615499160643763401960928
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        models = get_models_to_test(rng)
        for label, model in models.items():
            with self.subTest(model=label):
                for kind in all_basic_edit_kinds:
                    with self.subTest(action_kind=kind):
                        for it in range(10):
                            action = None
                            while action is None:
                                [x] = model.generate_samples(rng, 1)
                                action = random_basic_edit(rng, x, model, kind)
                            with self.subTest(iteration=it, sequence=x, action=action):
                                log_px = model.log_p_sequence(x)
                                log_py_over_px = model.log_p_edit_over_p_noedit(x, action)
                                y = apply_basic_edit(action, x)
                                log_py = model.log_p_sequence(y)
                                self.assertAlmostEqual(log_py - log_px, log_py_over_px,
                                                       msg=f'log_py={log_py}, log_px={log_px}')

    def test_p_wildcard(self):
        entropy = 139342445159650696176756268176533250427
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        models = get_models_to_test(rng)
        for label, model in models.items():
            with self.subTest(model=label):
                for it in range(10):
                    [x] = model.generate_samples(rng, 1)
                    self.assertAlmostEqual(model.log_p_wildcard_suffix_over_p_sequence(x, 0), 0)
                    for suffix_length in range(1, 3 + 1):
                        with self.subTest(iteration=it, sequence=x, suffix_length=suffix_length):
                            log_suffix_probs = []
                            for c in model.alphabet:
                                action_xy = 'insert', len(x), c
                                log_py_over_px = model.log_p_edit_over_p_noedit(x, action_xy)
                                y = apply_basic_edit(action_xy, x)
                                log_suffix_over_py = model.log_p_wildcard_suffix_over_p_sequence(
                                    y, suffix_length - 1)
                                log_suffix_over_px = log_py_over_px + log_suffix_over_py
                                log_suffix_probs.append(log_suffix_over_px)
                            log_suffix_overall = model.log_p_wildcard_suffix_over_p_sequence(x, suffix_length)
                            expected = logsumexp(log_suffix_probs)
                            self.assertAlmostEqual(expected, log_suffix_overall)


class TestNgramFit(unittest.TestCase):
    def test_likelihood_increases(self):
        entropy = 277028202278580657669931287321562506251
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        for it in range(20):
            n_in_ngram = rng.integers(0, 4)
            n_states = rng.integers(8) + 1
            E_length = rng.integers(30) + 2
            concentration_param = rng.exponential()
            N_sequences = rng.integers(30) + 1
            with self.subTest(
                    iteration=it,
                    n_in_ngram=n_in_ngram,
                    n_states=n_states,
                    E_length=E_length,
                    concentration_param=concentration_param,
                    N_sequences=N_sequences,
            ):
                model = RandomNgramModel(
                    rng=rng,
                    n_in_ngram=n_in_ngram,
                    n_states=n_states,
                    E_length=E_length,
                    concentration_param=concentration_param,
                )
                sequences = model.generate_samples(rng, N_sequences)
                mle_model = NgramModel.fit(n_in_ngram=n_in_ngram, n_states=n_states, sequences=sequences)
                original_log_p = sum(model.log_p_sequence(s) for s in sequences)
                mle_log_p = sum(mle_model.log_p_sequence(s) for s in sequences)
                self.assertGreaterEqual(mle_log_p, original_log_p, "mle should never decrease likelihood")


class TestWriteSamplestoFile(unittest.TestCase):
    def test_write_samples_to_file(self):
        """
        Not a real test. Just write some samples from each model to a file, for inspection.
        We check this file into the repo.
        Think of it as a golden test.
        """
        entropy = 239648862661155927535044943977345994042
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        models = get_models_to_test(rng)
        lines = []
        for label, model in models.items():
            with self.subTest(model=label):
                lines += ['=' * 40, label, '=' * 40]
                xs = model.generate_samples(rng, 10)
                lines += [str(x) for x in xs]
                lines += ['']
        with open('samples_from_models.txt', 'w') as f:
            f.write('\n'.join(lines))
