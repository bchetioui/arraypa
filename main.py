from arraypa.backends.jax import jax_backend
from arraypa.core import (
    App, Array, Ast, BinOp, For, Index, Lam, Mul, Plus, Psi, Reduce, Shape, Var)
from arraypa.interpreters.eval import eval_ast
from arraypa.interpreters.check import infer_shape
from arraypa.interpreters.depanal import analyze_dependencies

from functools import partial
from itertools import product
from jax import jit, make_jaxpr
import numpy as np  # type: ignore[import]
from typing import List, Tuple

# === convenience wrappers ===


def Get(array, index):
    return Psi(index, array)


# === examples ===

def mk_dot_fn_app(m, n, p, lhs=None, rhs=None):
    """Build a dot function based on shape parameters."""
    lhs_shape = (m, n)
    rhs_shape = (n, p)

    if lhs is None:
        lhs = np.ones((m, n))
    if rhs is None:
        rhs = np.ones((n, p))

    i, j, k = tuple(Var(x) for x in ["i", "j", "k"])

    shaped_A, shaped_B = ((Var("A"), lhs_shape), (Var("B"), rhs_shape))
    A, B = shaped_A[0], shaped_B[0]

    dot = (Lam(shaped_A, Lam(shaped_B,
        For(i, 0, m,
            For(k, 0, p, Reduce('+',
                For(j, 0, n,
                    Mul(Get(A, (i, j)), Get(B, (j, k))))))))))

    dot_app = App(App(dot, lhs), rhs)

    return dot_app, [shaped_A, shaped_B]


def example_dot():
    m, n, p = 3, 6, 4
    lhs, rhs = np.ones((m, n)), np.ones((n, p))
    dot_app, _ = mk_dot_fn_app(m, n, p, lhs, rhs)

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

def depanal_dot():
    m, n, p = 3, 6, 4
    dot_app, shaped_vars = mk_dot_fn_app(m, n, p)
    
    deparray = analyze_dependencies(dot_app)
    
    for ij in product(range(m), range(p)):
        # TODO: make pprinter
        index = tuple(ij)
        print('Dependencies for index', ij)
        for var, shape in shaped_vars:
            name = var.name
            rows = []
            for var_i in range(shape[0]):
                row = ((len(name) + 2) * " "
                       if rows != [] else (len(name) + 1) * " ")
                for var_j in range(shape[1]):
                    if (name, (var_i, var_j)) in deparray.dependencies[ij]:
                        row += "x "
                    else:
                        row += "- "
                rows.append(row)
            print(name + '\n'.join(rows))
            print()
            print('-' * 80)
                        

def main():
    #example_dot()
    depanal_dot()

# TODO: add test programs somewhere

if __name__ == '__main__':
    main()
