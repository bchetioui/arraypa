from core import (
    Array, Ast, BinOp, For, Index, Mul, Plus, Psi, Reduce, Shape, Var)

from interpreter import eval_ast, infer_shape
import numpy as np  # type: ignore[import]
Array.register(np.ndarray)

# === convenience wrappers ===


def Get(array, index):
    return Psi(index, array)


# === examples ===


def example_dot():
    m, n, p = 6, 4, 3
    lhs_shape = (m, n)
    rhs_shape = (n, p)
    lhs, rhs = np.ones(lhs_shape), np.ones(rhs_shape)

    i, j, k = tuple(Var(x) for x in ["i", "j", "k"])

    dot = (
        For(i, 0, m,
            For(k, 0, p, Reduce('+',
                For(j, 0, n,
                    Mul(Get(lhs, (i, j)), Get(rhs, (j, k))))))))

    print(dot)
    print(20 * '=')
    print('Expected shape:', (m, p))
    print('Got:', infer_shape(dot))
    print('Expected eval:', np.dot(lhs, rhs))
    print('Got:', eval_ast(dot))

def main():
    example_dot()


if __name__ == '__main__':
    main()
