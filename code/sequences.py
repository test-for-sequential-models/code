import numpy as np


class Sequence:
    """
    This class exists so that we can use sequences
    as keys in a hash data structure, and perform equality checks.
    """

    def __init__(self, xs):
        self._numpy = np.array(xs)
        self._tup = tuple(xs)
        self._hash = hash(self._tup)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self._tup == other._tup

    def __repr__(self):
        return '<' + ','.join(str(s) for s in self._tup) + '>'

    def __len__(self):
        return len(self._tup)

    def __iter__(self):
        for x in self._tup:
            yield x

    def __getitem__(self, item):
        return self._tup[item]

    def insert(self, at, x):
        assert at >= 0, "use *_complement instead?"
        xs = list(self._tup)
        xs.insert(at, x)
        return Sequence(xs), [('insert', at, x)]

    def insert_complement(self, at, x):
        assert at >= 0, "do not use *_complement here?"
        xs = list(self._tup)
        length = len(xs)
        xs.insert(length - at, x)
        return Sequence(xs), [('insert', length - at, x)]

    def sub(self, at, x):
        assert at >= 0, "use *_complement instead?"
        xs = list(self._tup)
        xs[at] = x
        return Sequence(xs), [('replace', at, x)]

    def sub_complement(self, at, x):
        assert at >= 0, "do not use *_complement here?"
        xs = list(self._tup)
        length = len(xs)
        xs[length - at - 1] = x
        return Sequence(xs), [('replace', length - at - 1, x)]

    def del_(self, at):
        assert at >= 0, "use *_complement instead?"
        xs = list(self._tup)
        del xs[at]
        return Sequence(xs), [('pop', at)]

    def del_complement(self, at):
        assert at >= 0, "do not use *_complement here?"
        xs = list(self._tup)
        length = len(xs)
        del xs[length - at - 1]
        return Sequence(xs), [('pop', length - at - 1)]

    def pre(self, size):
        assert size > 0, "use *_complement instead?"
        xs = list(self._tup)
        length = len(xs)
        return Sequence(xs[:size]), [('drop_tail', length - size)]

    def pre_complement(self, size):
        assert size > 0, "do not use *_complement here?"
        xs = list(self._tup)
        length = len(xs)
        return Sequence(xs[:length - size]), [('drop_tail', size)]

    def to_numpy(self):
        return self._numpy.copy()
