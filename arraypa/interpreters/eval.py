from arraypa.backends.numpy import numpy_backend
from arraypa.core import (
    App, Array, Ast, Backend, BinOp, Cat, ExpandDim, For, Index, Lam, Mul, Plus,
    Psi, Reduce, Shape, subst, Var)

from typing import Any, Callable, Dict


AstType = Any

# Default backend for computation.
_backend: Backend
_backend = numpy_backend


def eval_ast(node: Ast, backend=None):
    global _backend

    if backend is not None:
        _backend = backend

    node_type = Array if isinstance(node, _backend.ArrayTy) else type(node)
    if node_type not in eval_rules:
        raise ValueError(f"wrong node type {node}")
    return eval_rules[node_type](node)


eval_rules: Dict[AstType, Callable]
eval_rules = {}


def _array_eval_rule(node: Array) -> Array:
    return node


eval_rules[Array] = _array_eval_rule


def _psi_eval_rule(node: Psi) -> Array:
    array = eval_ast(node.array)
    return array[node.index]


eval_rules[Psi] = _psi_eval_rule


def _elementwise_binop_eval_rule(node: BinOp) -> Array:
    def op(lhs, rhs):
        if isinstance(node, Plus):
            return _backend.add(lhs, rhs)
        if isinstance(node, Mul):
            return _backend.mul(lhs, rhs)
        raise ValueError(f"wrong node type {node}")
    return op(eval_ast(node.lhs), eval_ast(node.rhs))


for op in [Plus, Mul]:
    eval_rules[op] = _elementwise_binop_eval_rule


def _cat_eval_rule(node: Cat):
    lhs, rhs = eval_ast(node.lhs), eval_ast(node.rhs)
    # TODO: make non-numpy specific
    return _backend.cat(lhs, rhs)


eval_rules[Cat] = _cat_eval_rule


def _expand_dim_eval_rule(node: ExpandDim):
    # TODO: make independent from numpy (ctx backend?)
    array, axis = eval_ast(node.array), node.axis
    return _backend.reshape(
        array, array.shape[:axis] + (1,) + array.shape[axis:])


eval_rules[ExpandDim] = _expand_dim_eval_rule


def _var_eval_rule(node: Var):
    raise ValueError(f"variable {node} is unbound")


eval_rules[Var] = _var_eval_rule


def _for_eval_rule(node: For):
    acc = None
    for val in range(node.start, node.stop):
        expr = subst(node.var, val, node.body)
        iteration_result = eval_ast(ExpandDim(0, expr))
        if acc is None:
            acc = iteration_result
        else:
            acc = _backend.cat(acc, iteration_result)

    if acc is None:
        acc = _backend.ArrayTy((0,))

    return acc


eval_rules[For] = _for_eval_rule


def _reduce_eval_rule(node: Reduce):
    acc = op = None

    if node.op == '+':
        acc, op = 0, _backend.add  # TODO: avoid implicit broadcast?
    elif node.op == '*':
        acc, op = 1, _backend.mul  # TODO: avoid implicit broadcast?
    else:
        raise ValueError(f"unknown op {op} in reduce")

    array = eval_ast(node.array)

    for i in range(0, array.shape[0]):
        acc = op(acc, array[i])

    return acc


eval_rules[Reduce] = _reduce_eval_rule


def _lam_eval_rule(node: Lam):
    def _lambda_fn(val):
        var, _ = node.shaped_var
        return eval_ast(subst(var, val, node.body))
    return _lambda_fn


eval_rules[Lam] = _lam_eval_rule


def _app_eval_rule(node: App):
    return eval_ast(node.function)(eval_ast(node.parameter))


eval_rules[App] = _app_eval_rule
