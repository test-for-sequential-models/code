import unittest

import numpy as np

from .sequences import Sequence


class TestSequence(unittest.TestCase):
    def test_hash_and_eq(self):
        rng = np.random.default_rng(seed=42)
        target_len = rng.integers(30)
        seq = rng.integers(10, size=target_len)
        seq2 = rng.integers(10, size=target_len)
        x = Sequence(seq)
        y = Sequence(seq)
        z = Sequence(seq2)
        self.assertEqual(x, y)
        self.assertEqual(hash(x), hash(y))
        self.assertNotEqual(x, z)
        self.assertNotEqual(hash(x), hash(z))
        self.assertEqual(target_len, len(x))
        self.assertListEqual([xi for xi in x], [xi for xi in seq])

    def test_insert(self):
        x = Sequence([0, 1, 2, 3])
        self.assertEqual(x.insert_complement(0, 5)[0], Sequence([0, 1, 2, 3, 5]))
        self.assertEqual(x.insert(0, 5)[0], Sequence([5, 0, 1, 2, 3]))
        self.assertEqual(x.insert(1, 5)[0], Sequence([0, 5, 1, 2, 3]))
        self.assertEqual(x.insert_complement(0, 5)[0], x.insert(len(x), 5)[0])
        self.assertEqual(x.insert_complement(1, 5)[0], Sequence([0, 1, 2, 5, 3]))

    def test_sub(self):
        x = Sequence([0, 1, 2, 3])
        self.assertEqual(x.sub_complement(0, 5)[0], Sequence([0, 1, 2, 5]))
        self.assertEqual(x.sub(0, 5)[0], Sequence([5, 1, 2, 3]))
        self.assertEqual(x.sub(1, 5)[0], Sequence([0, 5, 2, 3]))
        self.assertEqual(x.sub_complement(0, 5)[0], x.sub(len(x) - 1, 5)[0])
        self.assertEqual(x.sub_complement(1, 5)[0], Sequence([0, 1, 5, 3]))

    def test_del(self):
        x = Sequence([0, 1, 2, 3])
        self.assertEqual(x.del_complement(0)[0], Sequence([0, 1, 2]))
        self.assertEqual(x.del_(0)[0], Sequence([1, 2, 3]))
        self.assertEqual(x.del_(1)[0], Sequence([0, 2, 3]))
        self.assertEqual(x.del_complement(0)[0], x.del_(len(x) - 1)[0])
        self.assertEqual(x.del_complement(1)[0], Sequence([0, 1, 3]))

    def test_pre(self):
        x = Sequence([0, 1, 2, 3])
        self.assertEqual(x.pre_complement(1)[0], Sequence([0, 1, 2]))
        self.assertEqual(x.pre(1)[0], Sequence([0]))
        self.assertEqual(x.pre(2)[0], Sequence([0, 1]))
        self.assertEqual(x.pre_complement(1)[0], x.pre(len(x) - 1)[0])
        self.assertEqual(x.pre_complement(2)[0], Sequence([0, 1]))
