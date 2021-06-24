from arraypa.backends.jax import jax_backend
from arraypa.core import (
    Array, Ast, BinOp, For, Index, Mul, Plus, Psi, Reduce, Shape, Var)
from arraypa.interpreters.eval import eval_ast
from arraypa.interpreters.check import infer_shape

from functools import partial
from jax import jit, make_jaxpr
import numpy as np  # type: ignore[import]

# === convenience wrappers ===


def Get(array, index):
    return Psi(index, array)


# === examples ===


def example_dot():
    m, n, p = 3, 6, 4
    lhs_shape = (m, n)
    rhs_shape = (n, p)
    lhs, rhs = np.ones(lhs_shape), np.ones(rhs_shape)

    i, j, k = tuple(Var(x) for x in ["i", "j", "k"])

    dot = (
        For(i, 0, m,
            For(k, 0, p, Reduce('+',
                For(j, 0, n,
                    Mul(Get(lhs, (i, j)), Get(rhs, (j, k))))))))

    print('Expected shape:', (m, p))
    print('Inferred shape:', infer_shape(dot))
 
    print('Expected eval:', np.dot(lhs, rhs))

    print('=== NumPy backend ===')
    np_result = eval_ast(dot)
    print('Got:', np_result)
    print('Type:', type(np_result))

    print('=== JAX backend ===')
    jax_f = partial(eval_ast, dot, backend=jax_backend)
    print('Got (eager):', jax_f())
    print('Got (jitted):', jit(jax_f)()) # can even jit it!
    print('Type:', type(jax_f()))

def main():
    example_dot()

# TODO: add test programs somewhere

if __name__ == '__main__':
    main()
