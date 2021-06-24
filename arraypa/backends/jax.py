from arraypa.core import Backend

import jax.numpy as jnp  # type: ignore[import]


class _JaxBackend(Backend):
    ArrayTy = jnp.ndarray

    def add(self, lhs, rhs):
        return jnp.add(lhs, rhs)

    def mul(self, lhs, rhs):
        return jnp.multiply(lhs, rhs)

    def cat(self, lhs, rhs):
        return jnp.concatenate((lhs, rhs))

    def reshape(self, array, new_shape):
        return jnp.reshape(array, new_shape)


jax_backend = _JaxBackend()
