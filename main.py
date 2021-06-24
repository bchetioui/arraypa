from arraypa.backends.jax import jax_backend
from arraypa.core import (
    App, Array, Ast, BinOp, For, Index, Lam, Mul, Plus, Psi, Reduce, Shape, Var)
from arraypa.interpreters.eval import eval_ast
from arraypa.interpreters.check import infer_shape

from functools import partial
from jax import jit, make_jaxpr
import numpy as np  # type: ignore[import]
from typing import List, Tuple

# === convenience wrappers ===


def Get(array, index):
    return Psi(index, array)

'''
def Fun(*args: List[Tuple[Var, Shape]], body: Ast):
    """
    Build nested lambda abstractions from a list of arguments and a function
    body.
    """
    if len(args) == 0:
        raise ValueError("can not create a lambda abstraction without argument")
    fun = body
    for var, shape in args[::-1]:
        fun = Lambda(var, shape, fun)
    return fun
'''

# === examples ===


def example_dot():
    m, n, p = 3, 6, 4
    lhs_shape = (m, n)
    rhs_shape = (n, p)
    lhs, rhs = np.ones(lhs_shape), np.ones(rhs_shape)

    i, j, k = tuple(Var(x) for x in ["i", "j", "k"])

    shaped_A, shaped_B = ((Var("A"), lhs_shape), (Var("B"), rhs_shape))
    A, B = shaped_A[0], shaped_B[0]

    dot = (Lam(shaped_A, Lam(shaped_B,
        For(i, 0, m,
            For(k, 0, p, Reduce('+',
                For(j, 0, n,
                    Mul(Get(A, (i, j)), Get(B, (j, k))))))))))

    dot_app = App(App(dot, lhs), rhs)

    print('Expected shape:', (m, p))
    print('Inferred shape:', infer_shape(dot_app))
 
    print('Expected eval:', np.dot(lhs, rhs))

    print('=== NumPy backend ===')
    np_result = eval_ast(dot_app)
    print('Got:', np_result)
    print('Type:', type(np_result))

    print('=== JAX backend ===')
    jax_f = partial(eval_ast, dot_app, backend=jax_backend)
    print('Got (eager):', jax_f())
    print('Got (jitted):', jit(jax_f)()) # can even jit it!
    print('Type:', type(jax_f()))

def main():
    example_dot()

# TODO: add test programs somewhere

if __name__ == '__main__':
    main()
