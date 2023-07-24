from contextlib import contextmanager
import datetime
import os
import subprocess
import functools
import hashlib
from collections import defaultdict
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
from distributed import Client

from .sequences import Sequence


@contextmanager
def pltctx(figsize='m', style='ggplot'):
    if figsize == 'xl':
        figsize = (12, 8)
    elif figsize == 'l':
        figsize = (9, 6)
    elif figsize == 'm':
        figsize = (6, 4)
    elif figsize == 's':
        figsize = (3, 2)
    with plt.style.context(style, after_reset=True):
        with plt.rc_context({'figure.figsize': figsize}):
            yield
            plt.show()


@contextmanager
def timer():
    before = datetime.datetime.now()
    print(before)
    yield
    after = datetime.datetime.now()
    print(after)
    print('took', after - before)


_seed_usage_count = defaultdict(lambda: 0)
Seed = tuple[int, str]


def extend_seed(parent: Seed, *keys: Any) -> Seed:
    seed, parent_key = parent
    return seed, f'{parent_key}/{"/".join(str(k) for k in keys)}'


def get_rng(parent: Seed, *keys: Any, nth_usage: int = 1) -> np.random.Generator:
    seed, key = extend_seed(parent, *keys)
    usage_key = seed, key
    _seed_usage_count[usage_key] += 1
    if _seed_usage_count[usage_key] != nth_usage:
        raise RuntimeError(f"incorrect multiple use of rng: {usage_key}")
    m = hashlib.sha256()
    m.update(f'{seed}'.encode('ascii'))
    m.update(key.encode('ascii'))
    digest = int(m.hexdigest(), 16)
    return np.random.default_rng(digest)


def with_spawn(xs, seed_or_sequence):
    warnings.warn('deprecated')
    if not isinstance(seed_or_sequence, np.random.SeedSequence):
        seed_or_sequence = np.random.SeedSequence(seed_or_sequence)
    return zip(xs, seed_or_sequence.spawn(len(xs)))


def sequence_to_seed(sequence: np.random.SeedSequence):
    warnings.warn("deprecated")
    rng = np.random.default_rng(sequence)
    return rng.integers(np.iinfo(np.int64).max)


def get_empirical_model(xs):
    counts = defaultdict(lambda: 0)
    for x in xs:
        counts[Sequence(x)] += 1
    total_count = sum(v for v in counts.values())

    class EmpiricalDist:
        def log_p_sequence(self, sequence) -> float:
            count = counts[Sequence(sequence)]
            return np.log(count) - np.log(total_count)

    return EmpiricalDist()


def slugify_unsafe(s):
    return ''.join(c if c.isalnum() else "_" for c in s)


def slugify(s):
    m = hashlib.sha256()
    m.update(s.encode('ascii'))
    digest = m.hexdigest()
    return ''.join(c if c.isalnum() else "_" for c in s) + '-' + digest[:8]
