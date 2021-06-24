from arraypa.core import (
    App, Array, Ast, BinOp, Cat, ExpandDim, For, Index, Lam, Mul, Plus, Psi,
    Reduce, Shape, subst, Var)

from typing import Any, Callable, Dict

AstType = Any


def _is_valid_index(idx: Index, shape: Shape) -> bool:
    assert len(idx) <= len(shape)
    assert all(map(lambda i: isinstance(i, int), idx)), (
        "indexes containing opaque variables can not be checked")
    return all(map(lambda i, bound: i >= 0 and i < bound, idx, shape))


check_rules: Dict[AstType, Callable]
check_rules = {}


def infer_shape(node: Ast):
    if type(node) not in check_rules:
        # We treat shaped nodes that are not part of the rules as backend
        # arrays, and abstractify them.
        if hasattr(node, 'shape'):
            node = Array(node.shape)  # type: ignore
        else:
            raise ValueError(f"wrong node type {node}")
    return check_rules[type(node)](node)


def _array_check_rule(node: Array) -> Shape:
    return node.shape


check_rules[Array] = _array_check_rule


def _psi_check_rule(node: Psi) -> Shape:
    array_shape = infer_shape(node.array)
    assert _is_valid_index(node.index, array_shape), (
        f"{node.index} is not a valid index into shape {array_shape}")
    return array_shape[len(node.index):]


check_rules[Psi] = _psi_check_rule


def _elementwise_binop_check_rule(node: BinOp) -> Shape:
    lhs_shape, rhs_shape = infer_shape(node.lhs), infer_shape(node.rhs)
    if lhs_shape != rhs_shape:
        raise ValueError(
            f"expected lhs.shape = rhs.shape but got lhs.shape = {lhs_shape} "
            f"and rhs.shape = {rhs_shape}")
    # TODO: implement broadcast rules?
    return lhs_shape


for op in [Plus, Mul]:
    check_rules[op] = _elementwise_binop_check_rule


def _cat_check_rule(node: Cat):
    lhs_shape, rhs_shape = infer_shape(node.lhs), infer_shape(node.rhs)
    if lhs_shape[1:] != rhs_shape[1:]:
        raise ValueError(
            "expected shapes that differ only in their first component in "
            "concatenation but got {} and {}".format(lhs_shape, rhs_shape))
    if lhs_shape == () and rhs_shape == ():
        return (2,)
    return (rhs_shape[0] + lhs_shape[0],) + rhs_shape[1:]


check_rules[Cat] = _cat_check_rule


def _expand_dim_check_rule(node: ExpandDim):
    array_shape = infer_shape(node.array)
    assert node.axis >= 0 and node.axis <= len(array_shape), (
        "attempted to expand axis {} but array has {} dimensions".format(
            node.axis, array_shape))
    return array_shape[:node.axis] + (1,) + array_shape[node.axis:]


check_rules[ExpandDim] = _expand_dim_check_rule


def _var_check_rule(node: Var):
    raise ValueError(f"variable {node} is unbound")


check_rules[Var] = _var_check_rule


def _for_check_rule(node: For):
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


check_rules[For] = _for_check_rule


def _reduce_check_rule(node: Reduce):
    array_shape = infer_shape(node.array)
    if len(array_shape) == 0:
        raise ValueError(
            "reduce requires its operande to have at least one dimension "
            f"but scalar {node.array} has shape ()")
    return array_shape[1:]


check_rules[Reduce] = _reduce_check_rule


def _lam_check_rule(node: Lam):
    # TODO: implement function types? But who cares...
    raise ValueError("can not extract a shape for a function type")


check_rules[Lam] = _lam_check_rule


def _app_check_rule(node: App):
    # To do anything, we need to resolve the innermost Application, i.e., the
    # App node such that node.function is a Lambda abstraction (or conversely,
    # the App node whose node.function is *NOT* another Application. Any
    # other type of function node is not applicable, and therefore results in
    # a type error.

    def _apply_innermost(app):
        if isinstance(app.function, Lam):
            parameter_shape = infer_shape(app.parameter)
            var, function_in_shape = app.function.shaped_var
            function_in_shape = app.function.shaped_var[1]
            if function_in_shape != parameter_shape:
                raise ValueError(
                    f"expected shape {function_in_shape} for parameter in "
                    f"function application but got {parameter_shape}")
            return subst(var, Array(function_in_shape), app.function.body)
        elif isinstance(app.function, App):
            return App(_apply_innermost(app.function), app.parameter)
        else:
            raise ValueError("unexpected node in function application")

    return infer_shape(_apply_innermost(node))


check_rules[App] = _app_check_rule
