from arraypa.core import Backend

import numpy as np  # type: ignore[import]


class _NumpyBackend(Backend):
    ArrayTy = np.ndarray

    def add(self, lhs, rhs):
        return np.add(lhs, rhs)

    def mul(self, lhs, rhs):
        return np.multiply(lhs, rhs)

    def cat(self, lhs, rhs):
        return np.concatenate((lhs, rhs))

    def reshape(self, array, new_shape):
        return np.reshape(array, new_shape)


numpy_backend = _NumpyBackend()
