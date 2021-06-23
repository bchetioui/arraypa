from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Any, Tuple, Union

Index = Tuple[Any, ...]  # Tuple[Union["Var", int], ...]
Shape = Tuple[Any, ...]  # Tuple[Union["Var", int], ...]


class Ast(ABC):
    pass


@dataclass
class Array(Ast):
    """An array with a minimal API."""
    shape: Shape

    def __getitem__(self, index: Index) -> "Array":
        raise NotImplementedError("must override")

    def __mul__(self, other: "Array") -> "Array":
        """Elementwise multiplication."""
        raise NotImplementedError("must override")

    def __add__(self, other: "Array") -> "Array":
        raise NotImplementedError("must override")


class Empty(Array):
    def __init__(self, shape: Shape):
        assert 0 in shape, ( "attempted to initialize an empty array with "
                            f"non empty shape {shape}")
        self.shape = shape


@dataclass
class Psi(Ast):
    """Indexing operation into an array using a tuple of integers."""
    index: Index
    array: Ast

    def __repr__(self):
        formatted_idx = str(self.index).replace(")", "").replace("(", "")
        return f'{self.array}[{formatted_idx}]'


@dataclass
class BinOp(Ast):
    """Convenience parent class for binary operations."""
    lhs: Ast
    rhs: Ast


class Plus(BinOp):
    """Elementwise addition."""
    def __repr__(self):
        return f'{self.lhs} + {self.rhs}'


class Mul(BinOp):
    """Elementwise multiplication."""
    def __repr__(self):
        return f'{self.lhs} * {self.rhs}'


class Cat(BinOp):
    """Concatenation across the first axis."""
    def __repr__(self):
        return f"{self.lhs} <> {self.rhs}"


@dataclass
class ExpandDim(Ast):
    axis: Any # Union["Var", int]
    array: Ast


@dataclass
class Var(Ast):
    """A variable."""
    name: str

    def __repr__(self):
        return self.name


@dataclass
class For(Ast):
    """A for loop that binds a variable."""
    var: Var
    start: int
    stop: int
    body: Ast

    def __repr__(self):
        return (
            f'for {self.var} in range({self.start}, {self.stop}):\n' +
            '\n'.join(4 * ' ' + line for line in f'{self.body}'.splitlines()))


@dataclass
class Reduce(Ast):
    """Reduction operation across the first axis."""
    op: str  # omit the clean op to avoid implementing generic functions
    array: Ast

    def __repr__(self):
        return f'reduce(({self.op}), {self.array})'


def subst(var: Var, val: int, node: Ast):
    """Substitute a variable by a value throughout the AST."""
    _subst = partial(subst, var, val)
    result = node

    def _subst_scalar(sc: Union[int, Var]):
        if isinstance(sc, Var) and sc == var:
            return val
        return sc

    if isinstance(node, BinOp):
        builder = None
        for ty in [Cat, Mul, Plus]:
            if isinstance(node, ty):
                builder = ty
                break
        else:
            raise ValueError(f"unknown binary operation")
        result = builder(_subst(node.lhs), _subst(node.rhs))
    elif isinstance(node, ExpandDim):
        result = ExpandDim(_subst_scalar(node.axis), _subst(node.array))
    elif isinstance(node, For):
        if node.var == var:
            raise ValueError(f"expression binds variable {var} more than once")
        result = For(node.var, _subst_scalar(node.start),
                     _subst_scalar(node.stop), _subst(node.body))
    elif isinstance(node, Psi):
        new_index = tuple(map(_subst_scalar, node.index))
        result = Psi(new_index, _subst(node.array))
    elif isinstance(node, Reduce):
        result = Reduce(node.op, _subst(node.array))
    elif isinstance(node, Var):
        raise NotImplementedError("TODO: replace scalar val by () array")
        #result = val if var == node else node
    return result
