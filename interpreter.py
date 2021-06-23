from core import (
    Array, Ast, BinOp, Cat, Empty, ExpandDim, For, Index, Mul, Plus, Psi,
    Reduce, Shape, subst, Var)

from functools import partial, reduce
import numpy as np  # type: ignore[import]
from typing import Any, Callable, Dict

AstType = Any


def _is_valid_index(idx: Index, shape: Shape) -> bool:
    assert len(idx) <= len(shape)
    assert all(map(lambda i: isinstance(i, int), idx)), (
        "indexes containing opaque variables can not be checked")
    return all(map(lambda i, bound: i >= 0 and i < bound, idx, shape))


def _process_generic(rules: Dict[AstType, Callable], node: Ast):
    node_type = Array if issubclass(type(node), Array) else type(node)
    if node_type not in rules:
        raise ValueError(f"wrong node type {node}")
    return rules[node_type](node)


eval_rules: Dict[AstType, Callable]
eval_rules = {}

consistency_rules: Dict[AstType, Callable]
consistency_rules = {}


def _array_eval_rule(node: Array) -> Array:
    return node


def _array_consistency_rule(node: Array) -> Shape:
    return node.shape


eval_rules[Array] = _array_eval_rule
consistency_rules[Array] = _array_consistency_rule


def _psi_eval_rule(node: Psi) -> Array:
    array = eval_ast(node.array)
    return array[node.index]


def _psi_consistency_rule(node: Psi) -> Shape:
    array_shape = infer_shape(node.array)
    assert _is_valid_index(node.index, array_shape), (
        f"{node.index} is not a valid index into shape {array_shape}")
    return array_shape[len(node.index):]


eval_rules[Psi] = _psi_eval_rule
consistency_rules[Psi] = _psi_consistency_rule


def _elementwise_binop_eval_rule(node: BinOp) -> Array:
    def op(lhs, rhs):
        if isinstance(node, Plus):
            return lhs + rhs
        if isinstance(node, Mul):
            return lhs * rhs
        raise ValueError(f"wrong node type {node}")
    return op(eval_ast(node.lhs), eval_ast(node.rhs))


def _elementwise_binop_consistency_rule(node: BinOp) -> Shape:
    lhs_shape, rhs_shape = infer_shape(node.lhs), infer_shape(node.rhs)
    if lhs_shape != rhs_shape:
        raise ValueError(
            f"expected lhs.shape = rhs.shape but got lhs.shape = {lhs_shape} "
            f"and rhs.shape = {rhs_shape}")
    # TODO: implement broadcast rules?
    return lhs_shape


for op in [Plus, Mul]:
    eval_rules[op] = _elementwise_binop_eval_rule
    consistency_rules[op] = _elementwise_binop_consistency_rule


def _cat_eval_rule(node: Cat):
    lhs, rhs = eval_ast(node.lhs), eval_ast(node.rhs)
    # TODO: make non-numpy specific
    return np.concatenate((lhs, rhs))


def _cat_consistency_rule(node: Cat):
    lhs_shape, rhs_shape = infer_shape(lhs), infer_shape(rhs)
    if lhs_shape[1:] != rhs_shape[1:]:
        raise ValueError(
            "expected shapes that differ only in their first component in "
            "concatenation but got {} and {}".format(lhs_shape, rhs_shape))
    if lhs_shape == () and rhs_shape == ():
        return (2,)
    return (rhs_shape[0] + lhs_shape[0],) + rhs_shape[1:]


eval_rules[Cat] = _cat_eval_rule
consistency_rules[Cat] = _cat_consistency_rule


def _expand_dim_eval_rule(node: ExpandDim):
    # TODO: make independent from numpy (ctx backend?)
    array, axis = eval_ast(node.array), node.axis
    return np.reshape(array, array.shape[:axis] + (1,) + array.shape[axis:])


def _expand_dim_consistency_rule(node: ExpandDim):
    array_shape = infer_shape(node.array)
    assert node.axis >= 0 and node.axis <= len(array_shape), (
        "attempted to expand axis {} but array has {} dimensions".format(
            node.axis, array_shape))
    return array_shape[:node.axis] + (1,) + array_shape[axis:]


eval_rules[ExpandDim] = _expand_dim_eval_rule
consistency_rules[ExpandDim] = _expand_dim_consistency_rule


def _var_rule(node: Var):
    raise ValueError(f"variable {node} is unbound")


eval_rules[Var] = _var_rule
consistency_rules[Var] = _var_rule


def _for_eval_rule(node: For):
    acc = None
    for val in range(node.start, node.stop):
        expr = subst(node.var, val, node.body)
        iteration_result = eval_ast(ExpandDim(0, expr))
        if acc is None:
            acc = iteration_result
        else:
            acc = Cat(acc, iteration_result)

    if acc is None:
        acc = Array((0,))

    return eval_ast(acc)


def _for_consistency_rule(node: For):
    assert node.start <= node.stop, (
        f"upper bound smaller than lower bound in for loop binding {node.var}")
    # For the moment, we check every loop iteration. Because we allow bound
    # variables to represent for loop bounds, some iterations might result in
    # a different shape than others, resulting in a shape we do not know how
    # to handle at the moment with our definition of shape (that uses
    # integers and not parameterized index sets).
    shape = None
    for val in range(node.start, node.stop):
        expr = subst(node.var, val, node.body)
        iteration_shape = infer_shape(expr)
        if shape is None:
            shape = iteration_shape
        elif shape != iteration_shape:
            raise NotImplementedError(
                "TODO: bodies with variable shapes in for loops")
    if shape is None:
        return (0,)
    return (node.stop - node.start,) + shape


eval_rules[For] = _for_eval_rule
consistency_rules[For] = _for_consistency_rule


def _reduce_eval_rule(node: Reduce):
    # TODO: make less hackish
    def add(x, y): return x + y
    def mul(x, y): return x * y
    
    acc = op = None
    
    if node.op == '+':
        acc, op = 0, add # TODO: avoid implicit broadcast
    elif node.op == '*':
        acc, op = 1, mul # TODO: avoid implicit broadcast
    else:
        raise ValueError(f"unknown op {op} in reduce")

    array = eval_ast(node.array)

    for i in range(0, array.shape[0]):
        acc = op(acc, array[i])
    
    return acc


def _reduce_consistency_rule(node: Reduce):
    array_shape = infer_shape(node.array)
    if len(array_shape) == 0:
        raise ValueError(
            "reduce requires its operande to have at least one dimension "
            f"but scalar {node.array} has shape ()")
    return array_shape[1:]


eval_rules[Reduce] = _reduce_eval_rule
consistency_rules[Reduce] = _reduce_consistency_rule

eval_ast = partial(_process_generic, eval_rules)
infer_shape = partial(_process_generic, consistency_rules)
