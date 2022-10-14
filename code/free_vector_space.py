from collections import defaultdict
from typing import Iterable

__all__ = [
    'FreeVector',
]


class FreeVector:
    def __init__(self, key=None, obj=None):
        self._contents = defaultdict(lambda: 0)
        self._objects = dict()
        if key is not None:
            self._contents[key] += 1
            if obj is not None:
                self._objects[key] = obj
            else:
                self._objects[key] = key
        self._mult = 1

    def __contains__(self, item):
        return item in self._contents.keys()

    def __repr__(self):
        return repr(dict(self._contents))

    def __mul__(self, other: 'FreeVector') -> 'FreeVector':
        result = FreeVector()
        result._contents = self._contents.copy()
        result._objects = self._objects.copy()
        result._mult = self._mult * other
        return result

    def __neg__(self):
        return -1 * self

    def __rmul__(self, other):
        return self * other

    def __add__(self, other: 'FreeVector') -> 'FreeVector':
        result = FreeVector()
        result._contents = {
            **{
                k:
                    (self._contents[k] * self._mult)
                    + (other._contents[k] * other._mult)
                for k in self._contents.keys() & other._contents.keys()
            },
            **{
                k: self._contents[k] * self._mult
                for k in self._contents.keys() - other._contents.keys()
            },
            **{
                k: other._contents[k] * other._mult
                for k in other._contents.keys() - self._contents.keys()
            }
        }
        result._objects = {
            **self._objects,
            **other._objects,
        }
        return result

    def __iadd__(self, other: 'FreeVector') -> 'FreeVector':
        if self._mult == 0:
            self._contents = other._contents
            self._mult = other._mult
            self._objects = other._objects
        for k, v in other._contents.items():
            v_for_self = v * other._mult / self._mult
            self._contents[k] = self._contents[k] + v_for_self
            self._objects[k] = other._objects[k]
        return self

    def __sub__(self, other):
        return self + (-1 * other)

    def __isub__(self, other):
        return self.__iadd__(-1 * other)

    def __rsub__(self, other):
        return self - other

    def __truediv__(self, other):
        return self * (1 / other)

    def __rtruediv__(self, other):
        return self / other

    def dict_items(self) -> Iterable[tuple[any, float]]:
        for k, v in self._contents.items():
            yield self._objects[k], v * self._mult

    def to_dict(self) -> dict[any, float]:
        return dict((k, v * self._mult) for k, v in self._contents.items())
