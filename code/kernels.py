from itertools import combinations
from collections import defaultdict
import sys
from pathlib import Path
import array
from typing import Optional

import numpy as np
import numpy.typing as npt

from .stein_operators import SteinOperator
from .stein_operators import GibbsSteinOperator
from .free_vector_space import FreeVector
from .models import SequenceDist

collection_of_fv = list[FreeVector]


class Kernel:
    @property
    def params(self):
        return NotImplementedError()

    def __hash__(self):
        return hash((self.__class__, self.params))

    def __eq__(self, other):
        return (self.__class__, self.params) == (other.__class__, other.params)

    def compute_gram(self, X: collection_of_fv, Y: collection_of_fv) -> npt.NDArray:
        raise NotImplementedError()

    def compute_gram_single(self, X: collection_of_fv) -> npt.NDArray:
        return self.compute_gram(X, X)


class SteinKernel(Kernel):
    def __init__(self, underlying_kernel: Kernel, stein_operator: SteinOperator):
        self._underlying_kernel = underlying_kernel
        self._stein_operator = stein_operator

    @property
    def params(self):
        return (self._underlying_kernel, self._stein_operator)

    def compute_gram(self, X: collection_of_fv, Y: collection_of_fv) -> npt.NDArray:
        X = [self._stein_operator.apply(s) for s in X]
        Y = [self._stein_operator.apply(s) for s in Y]
        return self._underlying_kernel.compute_gram(X, Y)

    def compute_gram_single(self, X: collection_of_fv) -> npt.NDArray:
        X = [self._stein_operator.apply(s) for s in X]
        return self._underlying_kernel.compute_gram_single(X)


class ExpDistanceKernel(Kernel):
    def __init__(self, kind, n_states: Optional[int] = None):
        assert kind in {'Hamming', 'Gaussian'}
        self._kind = kind
        self._n_states = n_states
        if kind == 'Gaussian':
            assert n_states is not None

    @property
    def params(self):
        return (self._kind)

    def compute_gram(self, X: collection_of_fv, Y: collection_of_fv) -> npt.NDArray:
        N_p = len(X)
        N_q = len(Y)
        K = np.full([N_p, N_q], 0.)
        X_concat_by_len = defaultdict(lambda: [])
        Y_concat_by_len = defaultdict(lambda: [])
        X_weights_by_len = defaultdict(lambda: [])
        Y_weights_by_len = defaultdict(lambda: [])

        for k in range(N_p):
            for phi_x, w_x in X[k].dict_items():
                X_concat_by_len[len(phi_x)].append(phi_x)
                weights = np.full(N_p, 0.)
                weights[k] = w_x
                X_weights_by_len[len(phi_x)].append(weights)
        for l in range(N_q):
            for phi_y, w_y in Y[l].dict_items():
                Y_concat_by_len[len(phi_y)].append(phi_y)
                weights = np.full(N_q, 0.)
                weights[l] = w_y
                Y_weights_by_len[len(phi_y)].append(weights)

        for lenkey in X_concat_by_len.keys() & Y_concat_by_len.keys():
            X_concat = np.stack(X_concat_by_len[lenkey])
            Y_concat = np.stack(Y_concat_by_len[lenkey])
            X_weights = np.stack(X_weights_by_len[lenkey])
            Y_weights = np.stack(Y_weights_by_len[lenkey])
            if self._kind == 'Hamming':
                dists = np.count_nonzero(X_concat[:, None, :] - Y_concat[None, :, :], -1) * 1.
            elif self._kind == 'Gaussian':
                per_seq_dists = np.abs(X_concat[:, None, :] - Y_concat[None, :, :])
                per_seq_dists = np.minimum(per_seq_dists, self._n_states - per_seq_dists)
                dists = np.sum(per_seq_dists ** 2, -1)
            else:
                raise RuntimeError("should be impossible")
            if lenkey == 0:
                assert (dists == 0).all()
            subK = X_weights.T @ np.exp(-dists / max(lenkey, 1)) @ Y_weights
            K += subK
        return K


class HammingKernel(ExpDistanceKernel):
    def __init__(self):
        super().__init__('Hamming')


class GaussianKernel(ExpDistanceKernel):
    def __init__(self, n_states: int):
        super().__init__('Gaussian', n_states)


class ContiguousSubsequenceKernel(Kernel):
    def __init__(self, min_length: int, max_length: int):
        self._min_length = min_length
        self._max_length = max_length

    @property
    def params(self):
        return (self._min_length, self._max_length)

    def _get_subsequences(self, X: FreeVector) -> FreeVector:
        result = FreeVector()
        for phi, weight in X.dict_items():
            num_lengths = min(self._max_length, len(phi)) - self._min_length + 1
            for length in range(self._min_length, self._min_length + num_lengths):
                num_subseqs = len(phi) - length + 1
                phi_sub_counts = defaultdict(lambda: 0)
                for i in range(0, num_subseqs):
                    j = i + length
                    phi_sub = phi[i:j]
                    phi_sub_counts[tuple(phi_sub)] += 1
                for phi_sub, count in phi_sub_counts.items():
                    normalizer = np.sqrt(num_subseqs * num_lengths * count)
                    result += (weight * count / normalizer) * FreeVector(tuple(phi_sub), np.array(phi_sub))
        return result

    def compute_gram(self, X: collection_of_fv, Y: collection_of_fv) -> npt.NDArray:
        X = [self._get_subsequences(x).to_dict() for x in X]
        Y = [self._get_subsequences(y).to_dict() for y in Y]
        k = np.full((len(X), len(Y)), np.nan)
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                k[i, j] = 0
                for phi in x.keys() & y.keys():
                    k[i, j] += x[phi] * y[phi]
        return k


class PrefixKernel(Kernel):
    def __init__(self, prefix_size: int, underlying_kernel: Kernel):
        self._prefix_size = prefix_size
        self._underlying_kernel = underlying_kernel

    def _get_prefix(self, X: FreeVector) -> FreeVector:
        result = FreeVector()
        for phi, weight in X.dict_items():
            phi_sub = phi[:self._prefix_size]
            result += weight * FreeVector(tuple(phi_sub), np.array(phi_sub))
        return result

    def compute_gram(self, X: collection_of_fv, Y: collection_of_fv) -> npt.NDArray:
        X = [self._get_prefix(s) for s in X]
        Y = [self._get_prefix(s) for s in Y]
        return self._underlying_kernel.compute_gram(X, Y)

    def compute_gram_single(self, X: collection_of_fv) -> npt.NDArray:
        X = [self._get_prefix(s) for s in X]
        return self._underlying_kernel.compute_gram_single(X)


class DiracKernelOfLength(Kernel):
    def compute_gram(self, X: collection_of_fv, Y: collection_of_fv) -> npt.NDArray:
        N_p = len(X)
        N_q = len(Y)
        K = np.full([N_p, N_q], 0.)
        X_weights_by_len = defaultdict(lambda: [])
        Y_weights_by_len = defaultdict(lambda: [])

        for k in range(N_p):
            for phi_x, w_x in X[k].dict_items():
                weights = np.full(N_p, 0.)
                weights[k] = w_x
                X_weights_by_len[len(phi_x)].append(weights)
        for l in range(N_q):
            for phi_y, w_y in Y[l].dict_items():
                weights = np.full(N_q, 0.)
                weights[l] = w_y
                Y_weights_by_len[len(phi_y)].append(weights)

        for lenkey in X_weights_by_len.keys() & Y_weights_by_len.keys():
            X_weights = np.stack(X_weights_by_len[lenkey]).sum(axis=0)
            Y_weights = np.stack(Y_weights_by_len[lenkey]).sum(axis=0)
            subK = X_weights[:, None] * Y_weights[None, :]
            K += subK
        return K


class ProductKernel(Kernel):
    def __init__(self, ks: Kernel, kt: Kernel):
        self._ks = ks
        self._kt = kt

    def compute_gram(self, X: collection_of_fv, Y: collection_of_fv) -> npt.NDArray:
        return self._ks.compute_gram(X, Y) * self._kt.compute_gram(X, Y)

    def compute_gram_single(self, X: collection_of_fv) -> npt.NDArray:
        return self._ks.compute_gram_single(X) * self._kt.compute_gram_single(X)


class FixedLengthKernel(Kernel):
    def __init__(self, length: int, underlying_kernel: Kernel):
        self._length = length
        self._underlying_kernel = underlying_kernel

    def _filter_lengths(self, X: FreeVector) -> FreeVector:
        result = FreeVector()
        for phi, weight in X.dict_items():
            if len(phi) == self._length:
                result += weight * FreeVector(tuple(phi), np.array(phi))
        return result

    def compute_gram(self, X: collection_of_fv, Y: collection_of_fv) -> npt.NDArray:
        X = [self._filter_lengths(s) for s in X]
        Y = [self._filter_lengths(s) for s in Y]
        return self._underlying_kernel.compute_gram(X, Y)

    def compute_gram_single(self, X: collection_of_fv) -> npt.NDArray:
        X = [self._filter_lengths(s) for s in X]
        return self._underlying_kernel.compute_gram_single(X)
