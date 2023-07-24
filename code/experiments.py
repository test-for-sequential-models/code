import datetime
from abc import ABC
from typing import Callable, Optional, Any
import warnings
import pickle
from collections import deque
import tqdm.auto as tqdm

import numpy as np

from .utils import Seed, get_rng
from .free_vector_space import FreeVector
from .hypothesis_tests import HypothesisTest
from .models import SequenceDist


class TestWrapper(ABC):
    """
    Allows us to cache the construction of a test,
    which allows us to avoid bootstrapping many times.
    We would probably not want to do this for MMD,
    because the generated samples for that introduce randomness.
    But we might, for Stein tests, because the only source of
    randomness is the bootstrap.
    """

    def construct(self, rng: np.random.Generator) -> tuple[HypothesisTest, datetime.timedelta]:
        raise NotImplementedError()


# class ImmediateTest(TestWrapper):
#     def __init__(self, test: Callable[[], HypothesisTest]):
#         before = datetime.datetime.now()
#         self._test = test()
#         after = datetime.datetime.now()
#         self._construction_time = after - before
#
#     def construct(self) -> tuple[HypothesisTest, datetime.timedelta]:
#         return self._test, self._construction_time

class HeldTest(TestWrapper):
    def __init__(self, test: Callable[[np.random.Generator], HypothesisTest]):
        self._test = test

    def construct(self, rng: np.random.Generator) -> tuple[HypothesisTest, datetime.timedelta]:
        before = datetime.datetime.now()
        test = self._test(rng)
        after = datetime.datetime.now()
        construction_time = after - before
        return test, construction_time


def load_cache(
        *,
        test_run: bool,
        cache_path: Optional[str],
):
    if not cache_path:
        return {}
    if test_run:
        return {}
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except:
        return {}


def save_cache(
        value: Any,
        *,
        test_run: bool,
        cache_path: Optional[str],
):
    if not cache_path:
        return
    if not test_run:
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)


def run_test(
        *,
        seed: Seed,
        test: TestWrapper,
        sample_model: SequenceDist,
        N_sequences: int,
        N_indep_tests: int,
        N_repeats_per_test: int,
        test_run: bool = True,
        cache_path: Optional[str] = None,
):
    if not isinstance(seed, tuple):
        warnings.warn('deprecated')
        seed = seed, ''
    num_rejects = 0
    cache = load_cache(test_run=test_run, cache_path=cache_path)
    test_construction_seconds = 0
    test_decision_seconds = 0
    for test_instance in tqdm.trange(N_indep_tests):
        # todo I could cache the instantiated test, it's just the critical value
        if test_instance not in cache:
            cache[test_instance] = dict(rejects=[], test_decision_seconds=[])
        rejects = cache[test_instance]['rejects']
        each_test_decision_seconds = cache[test_instance]['test_decision_seconds']
        if len(rejects) < N_repeats_per_test:
            test_, construction_time = test.construct(get_rng(seed, test_instance))
            cache[test_instance]['test_construction_seconds'] = construction_time.total_seconds()
        test_construction_seconds += cache[test_instance]['test_construction_seconds']
        rejects_ = deque()
        each_test_decision_seconds_ = deque()
        for test_repeat in range(len(rejects), N_repeats_per_test):
            rng = get_rng(seed, f'{test_instance}/{test_repeat}')
            sequences = sample_model.generate_samples(rng, N_sequences)
            sequences = [FreeVector(tuple(s), s) for s in sequences]
            test_result = test_.does_test_reject(sequences)
            reject = test_result.does_test_reject
            decision_seconds = test_result.time_taken_to_eval.total_seconds()
            each_test_decision_seconds_.append(decision_seconds)
            rejects_.append(1 if reject else 0)
        rejects.extend(rejects_)
        each_test_decision_seconds.extend(each_test_decision_seconds_)
        test_decision_seconds += sum(each_test_decision_seconds)
        num_rejects += sum(rejects[:N_repeats_per_test])
        save_cache(test_run=test_run, cache_path=cache_path, value=cache)
    if test_run:
        # hardcoded timing values for reproducibility during tests
        test_construction_seconds = 17
        test_decision_seconds = 23
    root_seed, seed_key = seed
    return dict(
        rejection_rate=num_rejects / N_indep_tests / N_repeats_per_test,
        avg_test_construction_seconds=test_construction_seconds / N_indep_tests,
        avg_test_decision_seconds=test_decision_seconds / N_indep_tests / N_repeats_per_test,
        avg_total_seconds=(test_construction_seconds + test_decision_seconds / N_repeats_per_test) / N_indep_tests,
        total_seconds=test_construction_seconds + test_decision_seconds,
        root_seed=root_seed,
        seed_key=seed_key,
        N_indep_tests=N_indep_tests,
        N_repeats_per_test=N_repeats_per_test,
    )
