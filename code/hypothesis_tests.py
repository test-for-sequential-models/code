'''
Hypothesis tests are parameterized by:
- kernel
- stein operator for stein tests, MMD sample count for MMD tests
- bootstrap choice
- target model
- desired level
'''

from abc import ABC
from typing import List
from typing import Optional
from typing import Callable
from dataclasses import dataclass
import datetime
from collections import deque

import numpy as np
import numpy.typing as npt
import scipy.stats

from .sequences import Sequence
from .models import SequenceDist
from .stein_operators import SteinOperator
from .kernels import Kernel
from .kernels import SteinKernel
from .free_vector_space import FreeVector


def compute_U_statistics(K: npt.NDArray) -> float:
    n, n2 = K.shape[-2:]
    assert n == n2, "cannot compute U-stat on non-square matrices"
    return (K.sum(axis=(-1, -2)) - K.trace(axis1=-1, axis2=-2)) / n / (n - 1)


def compute_gram_matrix(kernel: Kernel, sequences: list[FreeVector]) -> np.ndarray:
    K = kernel.compute_gram_single(sequences)
    if np.isnan(K).any():
        ix, iy = np.nonzero(np.isnan(K))
        ix, iy = ix[0], iy[0]
        raise RuntimeError(f"K[{ix},{iy}] is NaN, X[{ix}]={sequences[ix]}, X[{iy}]={sequences[iy]}")
    return K


def generate_aux_wild_bootstrap_processes(rng: np.random.Generator, N_samples, N_processes):
    nW = rng.multinomial(N_samples, np.full(N_samples, 1 / N_samples), size=(N_processes,))
    return nW - 1


def compute_wild_bootstrap_kernels(K: np.ndarray, Z: np.ndarray) -> np.ndarray:
    _, N_p, N_q = K.shape
    assert N_p == N_q, "unimplemented"
    return np.einsum('zp,bpq,zq->bzpq', Z, K, Z)
    # return Z[:, :, None] * Z[:, None, :] * K[None, :, :]


@dataclass
class TestResult:
    does_test_reject: bool
    time_taken_to_eval: datetime.timedelta


class HypothesisTest(ABC):
    def __init__(self, *, desired_level: float):
        assert desired_level <= 0.5, "you probably want a small level, such as 0.05"
        self._desired_level = desired_level

    def does_test_reject(self, samples: List[FreeVector]) -> TestResult:
        raise NotImplementedError()


class WithParametricBootstrap(HypothesisTest, ABC):
    def __init__(self, *, rng: np.random.Generator, target_model: SequenceDist, n_bootstrap: int,
                 sample_size: Optional[int] = None, **kwargs):
        self._rng = rng
        self._target_model = target_model
        self._bootstrap_samples = None
        self._bootstrap_sample_size = None
        self._n_bootstrap = n_bootstrap
        super().__init__(**kwargs)
        # TODO make this less brittle - perhaps a "precompute" method or something like it would be better
        # right now it is sensitive to ordering of super init calls
        if sample_size is not None:
            self._bootstrap(sample_size)

    def _compute_test_statistic(self, samples: List[FreeVector]) -> float:
        raise NotImplementedError()

    def _bootstrap(self, sample_size: int):
        self._bootstrap_samples = np.full(self._n_bootstrap, np.nan)
        self._bootstrap_sample_size = sample_size
        for i in range(self._n_bootstrap):
            samples = self._target_model.generate_samples(self._rng, sample_size)
            samples = [FreeVector(tuple(s), s) for s in samples]
            self._bootstrap_samples[i] = self._compute_test_statistic(samples)

    def does_test_reject(self, samples: List[FreeVector]) -> TestResult:
        before = datetime.datetime.now()
        if self._bootstrap_sample_size is None or self._bootstrap_sample_size != len(samples):
            self._bootstrap(len(samples))
        test_statistic = self._compute_test_statistic(samples)
        # TODO unsure whether I need to include the test statistic itself here.
        # it's certainly conservative, but check if it is necessary.
        bootstrap_samples = np.block([self._bootstrap_samples, test_statistic])
        critical_value = np.quantile(bootstrap_samples, 1 - self._desired_level, interpolation='higher')
        reject = critical_value < test_statistic
        after = datetime.datetime.now()
        time_taken_to_eval = after - before
        return TestResult(
            time_taken_to_eval=time_taken_to_eval,
            does_test_reject=reject,
        )


class LikelihoodRatioTest(WithParametricBootstrap):
    def __init__(self, *, target_model: SequenceDist,
                 get_mle_model: Callable[[List[Sequence]], SequenceDist], **kwargs):
        self._target_model = target_model
        self._get_mle_model = get_mle_model
        super().__init__(target_model=target_model, **kwargs)

    def _compute_test_statistic(self, samples: List[FreeVector]) -> float:
        sequences = []
        for sample in samples:
            count = 0
            for phi, weight in sample.dict_items():
                assert np.isclose(weight, 1), "Likelihood ratio test requires 'simple' free vectors, weight must be 1"
                sequences.append(phi)
                count += 1
                assert count == 1, "Likelihood ratio test requires 'simple' free vectors, can't have more than one in it"
        mle_model = self._get_mle_model(sequences)
        target_log_p = sum(self._target_model.log_p_sequence(s) for s in sequences)
        mle_log_p = sum(mle_model.log_p_sequence(s) for s in sequences)
        mle_log_p = max(mle_log_p, target_log_p)
        return -2 * (target_log_p - mle_log_p)


class AsymptoticLikelihoodRatioTest(HypothesisTest):
    def __init__(self, *, target_model: SequenceDist,
                 get_mle_model: Callable[[List[Sequence]], SequenceDist],
                 parametric_family_param_count: int,
                 n_bootstrap: int,
                 sample_size: int,
                 rng: np.random.Generator,
                 **kwargs):
        self._target_model = target_model
        self._get_mle_model = get_mle_model
        self._parametric_family_param_count = parametric_family_param_count
        super().__init__(**kwargs)

    def _compute_test_statistic(self, samples: List[FreeVector]) -> float:
        sequences = []
        for sample in samples:
            count = 0
            for phi, weight in sample.dict_items():
                assert np.isclose(weight, 1), "Likelihood ratio test requires 'simple' free vectors, weight must be 1"
                sequences.append(phi)
                count += 1
                assert count == 1, "Likelihood ratio test requires 'simple' free vectors, can't have more than one in it"
        mle_model = self._get_mle_model(sequences)
        target_log_p = sum(self._target_model.log_p_sequence(s) for s in sequences)
        mle_log_p = sum(mle_model.log_p_sequence(s) for s in sequences)
        mle_log_p = max(mle_log_p, target_log_p)
        return -2 * (target_log_p - mle_log_p)

    def does_test_reject(self, samples: List[FreeVector]) -> TestResult:
        before = datetime.datetime.now()
        test_statistic = self._compute_test_statistic(samples)
        critical_value = scipy.stats.chi2.ppf(1 - self._desired_level, self._parametric_family_param_count)
        reject = critical_value < test_statistic
        after = datetime.datetime.now()
        time_taken_to_eval = after - before
        return TestResult(
            time_taken_to_eval=time_taken_to_eval,
            does_test_reject=reject,
        )


class MMDTestParametricBootstrap(WithParametricBootstrap):
    def __init__(self, *, rng: np.random.Generator, kernel: Kernel, n_mmd_samples: int, target_model: SequenceDist,
                 **kwargs):
        self._mmd_samples = target_model.generate_samples(rng, n_mmd_samples)
        self._mmd_samples = [FreeVector(tuple(s), s) for s in self._mmd_samples]
        self._kernel = kernel
        super().__init__(rng=rng, target_model=target_model, **kwargs)

    def _compute_test_statistic(self, samples: List[FreeVector]) -> float:
        KXX = self._kernel.compute_gram_single(samples)
        KXY = self._kernel.compute_gram(samples, self._mmd_samples)
        KYY = self._kernel.compute_gram_single(self._mmd_samples)
        u_statistic = compute_U_statistics(KXX) + compute_U_statistics(KYY) - 2 * KXY.mean()
        # H = KXX[:, :, None, None] + KYY[None, None, :, :] - KXY[:, None, None, :] - KXY[None, :, :, None]
        # u_statistic_slow = compute_U_statistics(compute_U_statistics(H))
        # assert np.isclose(u_statistic_slow, u_statistic), f"{u_statistic_slow} != {u_statistic}"
        return u_statistic


class SteinTestParametricBootstrap(WithParametricBootstrap):
    def __init__(self, *, kernel: Kernel, stein_op: SteinOperator, **kwargs):
        self._stein_kernel = SteinKernel(kernel, stein_op)
        super().__init__(**kwargs)

    def _compute_test_statistic(self, samples: List[FreeVector]) -> float:
        gram_matrix = compute_gram_matrix(self._stein_kernel, samples)
        u_statistic = compute_U_statistics(gram_matrix)
        return u_statistic


class SteinTestWildBootstrap(HypothesisTest):
    def __init__(self, *, rng: np.random.Generator, n_bootstrap: int, kernel: Kernel, stein_op: SteinOperator,
                 sample_size: Optional[int] = None, target_model: SequenceDist, **kwargs):
        self._rng = rng
        self._n_bootstrap = n_bootstrap
        self._stein_kernel = SteinKernel(kernel, stein_op)
        del sample_size  # ignore
        del target_model  # ignore
        super().__init__(**kwargs)

    def does_test_reject(self, samples: List[FreeVector]) -> TestResult:
        before = datetime.datetime.now()
        gram_matrix = compute_gram_matrix(self._stein_kernel, samples)
        test_statistic = compute_U_statistics(gram_matrix)
        Z = generate_aux_wild_bootstrap_processes(self._rng, len(samples), self._n_bootstrap)
        bootstrap_gram_matrices = compute_wild_bootstrap_kernels(gram_matrix[None, ...], Z)
        bootstrap_samples = compute_U_statistics(bootstrap_gram_matrices)
        critical_value = np.quantile(bootstrap_samples, 1 - self._desired_level, interpolation='higher')
        reject = critical_value < test_statistic
        after = datetime.datetime.now()
        time_taken_to_eval = after - before
        return TestResult(
            time_taken_to_eval=time_taken_to_eval,
            does_test_reject=reject,
        )


class MMDTestWildBootstrap(HypothesisTest):
    def __init__(self, *, rng: np.random.Generator, kernel: Kernel, n_bootstrap: int, n_mmd_samples: int,
                 target_model: SequenceDist,
                 sample_size: Optional[int] = None,
                 **kwargs):
        self._rng = rng
        self._n_bootstrap = n_bootstrap
        self._mmd_samples = target_model.generate_samples(rng, n_mmd_samples)
        self._mmd_samples = [FreeVector(tuple(s), s) for s in self._mmd_samples]
        self._kernel = kernel
        del sample_size
        super().__init__(**kwargs)

    def does_test_reject(self, samples: List[FreeVector]) -> TestResult:
        before = datetime.datetime.now()
        KXX = self._kernel.compute_gram_single(samples)
        KXY = self._kernel.compute_gram(samples, self._mmd_samples)
        KYY = self._kernel.compute_gram_single(self._mmd_samples)
        v_statistic = KXX.mean() + KYY.mean() - 2 * KXY.mean()
        # TODO this uses a lot more memory than it has to. not a big deal, but might be worth optimizing later.
        # H = KXX[:, :, None, None] + KYY[None, None, :, :] - KXY[:, None, None, :] - KXY[None, :, :, None]
        # u_statistic = compute_U_statistics(compute_U_statistics(H))
        ZX = generate_aux_wild_bootstrap_processes(self._rng, len(samples), self._n_bootstrap)
        ZX = ZX - np.sum(ZX, axis=-1, keepdims=True)
        ZY = generate_aux_wild_bootstrap_processes(self._rng, len(self._mmd_samples), self._n_bootstrap)
        ZY = ZY - np.sum(ZY, axis=-1, keepdims=True)
        bs_v_statistics = (
                (ZX[:, :, None] * KXX[None, :, :] * ZX[:, None, :]).mean(axis=(1, 2))
                + (ZY[:, :, None] * KYY[None, :, :] * ZY[:, None, :]).mean(axis=(1, 2))
                - 2 * (ZX[:, :, None] * KXY[None, :, :] * ZY[:, None, :]).mean(axis=(1, 2))
        )
        critical_value = np.quantile(bs_v_statistics, 1 - self._desired_level, interpolation='higher')
        reject = critical_value < v_statistic
        after = datetime.datetime.now()
        time_taken_to_eval = after - before
        return TestResult(
            time_taken_to_eval=time_taken_to_eval,
            does_test_reject=reject,
        )


class AggregatedTestParametricBootstrap(HypothesisTest):
    def __init__(self, *, tests_and_weights: list[tuple[WithParametricBootstrap, float]], target_model: SequenceDist,
                 rng: np.random.Generator,
                 **kwargs):
        sum_of_exp_weight = sum(np.exp(-w) for _, w in tests_and_weights)
        assert np.isclose(sum_of_exp_weight, 1), f"exp(-weight) must sum to 1, sums to {sum_of_exp_weight}"
        n_bootstraps = [t._n_bootstrap for t, _ in tests_and_weights]
        self._n_bootstrap = n_bootstraps[0]
        assert np.allclose(n_bootstraps, self._n_bootstrap), "n_bootstrap must be the same for all tests"
        self._B1 = self._n_bootstrap // 2
        self._B2 = self._n_bootstrap - self._B1
        self._tests_and_weights = tests_and_weights
        self._bootstrap_sample_size = None
        self._target_model = target_model
        self._rng = rng
        super().__init__(**kwargs)

    def does_test_reject(self, samples: List[FreeVector]) -> TestResult:
        before = datetime.datetime.now()
        sample_size = len(samples)
        if self._bootstrap_sample_size is None or self._bootstrap_sample_size != sample_size:
            self._bootstrap_samples = np.full((self._n_bootstrap, len(self._tests_and_weights)), np.nan)
            self._bootstrap_sample_size = sample_size
            for i in range(self._n_bootstrap):
                samples = self._target_model.generate_samples(self._rng, sample_size)
                samples = [FreeVector(tuple(s), s) for s in samples]
                for j, (test, _) in enumerate(self._tests_and_weights):
                    self._bootstrap_samples[i, j] = test._compute_test_statistic(samples)
        u_min = self._desired_level
        u_max = min(np.exp(w) for _, w in self._tests_and_weights)
        test_stats = np.full(len(self._tests_and_weights), np.nan)
        critical_values = np.full(len(self._tests_and_weights), np.nan)
        for i, (test, _) in enumerate(self._tests_and_weights):
            test_stats[i] = test._compute_test_statistic(samples)
        while u_max - u_min > 1e-3 * u_min:
            u = (u_min + u_max) / 2
            rejects = np.repeat(0, self._B2)
            for i, (test, weight) in enumerate(self._tests_and_weights):
                M1 = np.block([self._bootstrap_samples[:self._B1, i], test_stats[i]])
                critical_values[i] = np.quantile(M1, 1 - u * np.exp(-weight), interpolation='higher')
                rejects = np.maximum(rejects, critical_values[i] < self._bootstrap_samples[self._B1:, i])
            P_u = rejects.mean()
            if P_u < self._desired_level:
                u_min = u
            else:
                u_max = u

        reject = np.any(critical_values < test_stats)
        after = datetime.datetime.now()
        time_taken_to_eval = after - before
        return TestResult(
            time_taken_to_eval=time_taken_to_eval,
            does_test_reject=reject,
        )
