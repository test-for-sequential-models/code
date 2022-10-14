import unittest

import numpy as np

from .free_vector_space import FreeVector
from .kernels import HammingKernel
from .kernels import GaussianKernel
from .kernels import ContiguousSubsequenceKernel


def list_of_vec_of_single_seq(x):
    return [vec_of_single_seq(x)]


def vec_of_single_seq(x):
    return FreeVector(tuple(x), np.array(x))


class TestHammingKernel(unittest.TestCase):
    def setUp(self) -> None:
        self._kernel = HammingKernel()

    def _compute_gram(self, x, y):
        return self._kernel.compute_gram(list_of_vec_of_single_seq(x), list_of_vec_of_single_seq(y)).item()

    def test_orthogonal_sequences(self):
        x = np.array([0, 1, 2])
        y = np.array([0, 1])
        self.assertEqual(0, self._compute_gram(x, y))

    def test_same_seq(self):
        entropy = 209081800568965765537591952918293713068
        s = np.random.SeedSequence(entropy)
        rng = np.random.default_rng(s)
        for it in range(10):
            len = rng.poisson(10)
            n = rng.poisson(10)
            x = rng.integers(n, size=len)
            with self.subTest(iteration=it, len=len, n_states=n, sequence=x):
                self.assertEqual(1, self._compute_gram(x, x))

    def test_overlap(self):
        entropy = 39080500734421418149234660147891040699
        s = np.random.SeedSequence(entropy)
        rng = np.random.default_rng(s)
        for it in range(10):
            len = rng.poisson(10) + 1
            n = rng.poisson(10) + 1
            x = rng.integers(n, size=len)
            num_to_edit = rng.integers(len) + 1
            frac_to_edit = num_to_edit / len
            ixs = rng.choice(range(len), replace=False, size=num_to_edit)
            y = x.copy()
            y[ixs] = (y[ixs] + 1) % n
            with self.subTest(iteration=it, len=len, n_states=n, x=x, y=y, num=num_to_edit):
                self.assertEqual(np.exp(-frac_to_edit), self._compute_gram(x, y))

    def test_kernel_is_bilinear(self):
        entropy = 150006755083541437616483542540651263405
        s = np.random.SeedSequence(entropy)
        rng = np.random.default_rng(s)
        got_nonzero_k_sum = False
        for it in range(10):
            n = rng.poisson(2) + 1
            num_seqs = rng.integers(1, 10)
            seqs = []
            for jt in range(10):
                tgt_len = rng.poisson(4) + 1
                x = rng.integers(n, size=tgt_len)
                seqs.append(vec_of_single_seq(x))
            weights_x = rng.normal(size=num_seqs)
            weights_y = rng.normal(size=num_seqs)
            ixs_x = rng.choice(range(len(seqs)), replace=False, size=num_seqs)
            ixs_y = rng.choice(range(len(seqs)), replace=False, size=num_seqs)
            with self.subTest(iteration=it, n_states=n, num_seqs=num_seqs, weights_x=weights_x, weights_y=weights_y,
                              ixs_x=ixs_x, ixs_y=ixs_y):
                vector_x = FreeVector()
                for weight, ix in zip(weights_x, ixs_x):
                    vector_x += weight * seqs[ix]
                vector_y = FreeVector()
                k_sum = 0
                for weight, ix in zip(weights_y, ixs_y):
                    k = lambda x, y: self._kernel.compute_gram([x], [y]).item()
                    k_sum += weight * k(vector_x, seqs[ix])
                    if k_sum != 0:
                        got_nonzero_k_sum = True
                    vector_y += weight * seqs[ix]
                    self.assertAlmostEqual(k_sum, k(vector_x, vector_y))
        self.assertTrue(got_nonzero_k_sum, msg="we never saw any non-zero inner products, so the test was not useful")


class TestGaussianKernel(unittest.TestCase):
    def _compute_gram(self, kernel, x, y):
        return kernel.compute_gram(list_of_vec_of_single_seq(x), list_of_vec_of_single_seq(y)).item()

    def test_orthogonal_sequences(self):
        x = np.array([0, 1, 2])
        y = np.array([0, 1])
        k = GaussianKernel(10)
        self.assertEqual(0, self._compute_gram(k, x, y))

    def test_same_seq(self):
        entropy = 65698999073849341887030416889180655300
        s = np.random.SeedSequence(entropy)
        rng = np.random.default_rng(s)
        for it in range(10):
            len = rng.poisson(10)
            n = rng.poisson(10)
            self._kernel = GaussianKernel(n)
            x = rng.integers(n, size=len)
            k = GaussianKernel(n)
            with self.subTest(iteration=it, len=len, n_states=n, sequence=x):
                self.assertEqual(1, self._compute_gram(k, x, x))

    def test_overlap(self):
        entropy = 89276675773679826326321211396942561620
        s = np.random.SeedSequence(entropy)
        rng = np.random.default_rng(s)
        for it in range(10):
            len = rng.poisson(10) + 1
            n = rng.poisson(10) + 1
            k = GaussianKernel(n)
            x = rng.integers(n, size=len)
            num_to_edit = rng.integers(len) + 1
            ixs = rng.choice(range(len), replace=False, size=num_to_edit)
            dists = rng.choice(range(n // 2), size=num_to_edit)
            y = x.copy()
            y[ixs] = (y[ixs] + dists) % n
            with self.subTest(iteration=it, len=len, n_states=n, x=x, y=y, num=num_to_edit, ixs=ixs, dists=dists):
                self.assertEqual(np.exp(-(dists ** 2).sum() / len), self._compute_gram(k, x, y))


class TestContiguousSubsequenceKernel(unittest.TestCase):
    def test_expected(self):
        test_cases = [
            ('hallo', 'hello', 3, 3, 1 / 3),
            ('hallo', 'hello', 1, 1, 4 / 5),
            ('xxx', 'xxxxx', 1, 1, 1),
            ('car', 'cat', 1, 3, (2 / 3 + 1 / 2) / 3),
            ('bar', 'cat', 1, 3, 1 / 9),
            ('bar', 'cat', 1, 1, 1 / 3),
        ]
        for x, y, min_length, max_length, expected in test_cases:
            with self.subTest(x=x, y=y, min_length=min_length, max_length=max_length):
                x = [ord(c) for c in x]
                y = [ord(c) for c in y]
                k = ContiguousSubsequenceKernel(min_length=min_length, max_length=max_length)
                kxy = k.compute_gram(list_of_vec_of_single_seq(x), list_of_vec_of_single_seq(y)).item()
                self.assertAlmostEqual(expected, kxy)

    def test_is_normalized(self):
        entropy = 167379574421097818580661278195740560266
        s = np.random.SeedSequence(entropy)
        rng = np.random.default_rng(s)
        seen_dups = False
        for it in range(10):
            n = rng.poisson(20) + 1
            min_length = rng.integers(3) + 1
            max_length = rng.integers(2) + min_length
            tgt_len = rng.poisson(10) + 1
            x = rng.integers(n, size=tgt_len)
            if np.unique(x, return_counts=True)[1].max() > 1:
                seen_dups = True
            with self.subTest(iteration=it, n_states=n, min_length=min_length, max_length=max_length, x=x):
                x = list_of_vec_of_single_seq(x)
                k = ContiguousSubsequenceKernel(min_length=min_length, max_length=max_length)
                kxx = k.compute_gram(x, x).item()
                self.assertAlmostEqual(kxx, 1)
        self.assertTrue(seen_dups,
                        msg="we never saw any sequences containing duplicates, so we can't be sure the normalization works correctly")

    def test_kernel_is_bilinear(self):
        entropy = 177787025549511507917274692306995133105
        s = np.random.SeedSequence(entropy)
        rng = np.random.default_rng(s)
        got_nonzero_k_sum = False
        for it in range(10):
            n = rng.poisson(20) + 1
            num_seqs = rng.integers(1, 10)
            min_length = rng.integers(3) + 1
            max_length = rng.integers(2) + min_length
            seqs = []
            for jt in range(10):
                tgt_len = rng.poisson(4) + 1
                x = rng.integers(n, size=tgt_len)
                seqs.append(vec_of_single_seq(x))
            weights_x = rng.normal(size=num_seqs)
            weights_y = rng.normal(size=num_seqs)
            ixs_x = rng.choice(range(len(seqs)), replace=False, size=num_seqs)
            ixs_y = rng.choice(range(len(seqs)), replace=False, size=num_seqs)
            with self.subTest(iteration=it, n_states=n, num_seqs=num_seqs, weights_x=weights_x, weights_y=weights_y,
                              ixs_x=ixs_x, ixs_y=ixs_y, min_length=min_length, max_length=max_length):
                kernel = ContiguousSubsequenceKernel(min_length=min_length, max_length=max_length)
                vector_x = FreeVector()
                for weight, ix in zip(weights_x, ixs_x):
                    vector_x += weight * seqs[ix]
                vector_y = FreeVector()
                k_sum = 0
                for weight, ix in zip(weights_y, ixs_y):
                    k = lambda x, y: kernel.compute_gram([x], [y]).item()
                    k_sum += weight * k(vector_x, seqs[ix])
                    if k_sum != 0:
                        got_nonzero_k_sum = True
                    vector_y += weight * seqs[ix]
                    self.assertAlmostEqual(k_sum, k(vector_x, vector_y))
        self.assertTrue(got_nonzero_k_sum, msg="we never saw any non-zero inner products, so the test was not useful")
