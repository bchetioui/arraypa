from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Tuple, Type, Union

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
    axis: Any  # Union["Var", int]
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


@dataclass
class Lam(Ast):
    """Lambda abstraction with a typed var."""
    shaped_var: Tuple[Var, Shape]
    body: Ast


@dataclass
class App(Ast):
    """Lambda application."""
    function: Ast
    parameter: Ast


def subst(var: Var, val, node: Ast):
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
            raise ValueError(f"unknown binary operation {node}")
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
    elif isinstance(node, Lam):
        if node.shaped_var[0] == var:
            raise ValueError(f"expression binds variable {var} more than once")
        result = Lam(node.shaped_var, _subst(node.body))
    elif isinstance(node, App):
        result = App(_subst(node.function), _subst(node.parameter))
    elif isinstance(node, Var) and node == var:
        result = val
    return result


BackendArray = Any


class Backend(ABC):
    """
    A backend overrides all the operations required to evaluate an Ast.
    """
    ArrayTy: Type[BackendArray]

    @abstractmethod
    def add(self, lhs: BackendArray, rhs: BackendArray):
        ...

    @abstractmethod
    def mul(self, lhs: BackendArray, rhs: BackendArray):
        ...

    @abstractmethod
    def cat(self, lhs: BackendArray, rhs: BackendArray):
        ...

    @abstractmethod
    def reshape(self, array: BackendArray, new_shape: Shape):
        ...
