from arraypa.backends.depanal import DepanalArray, depanal_backend
from arraypa.core import Array, Ast, Backend, Lam, subst
import arraypa.interpreters.eval as eval_interpreter

_captured_backend: Backend
_captured_backend = eval_interpreter._backend


def analyze_dependencies(node: Ast, backend=None):
    """
    Analyzes the dependencies of each output index of a function application
    based on the indices of the inputs.

    At the moment, this should behave well when the analyzed expression
    contains exactly one set of successive application nodes (and therefore
    applies only one function with n arguments).
    """
    # Monkey patch arraypa.interpreters.eval_interpreter.eval_ast
    init_eval_ast = eval_interpreter.eval_ast
    eval_interpreter.eval_ast = _analyze_dependencies
    # Monkey patch arraypa.interpreters.eval_interpreter._backend
    init_backend = eval_interpreter._backend
    eval_interpreter._backend = depanal_backend

    # Dependency analysis logic happens here
    result = _analyze_dependencies(node, backend=backend)

    # Reset arraypa.interpreters.eval_interpreter.eval_ast
    eval_interpreter.eval_ast = init_eval_ast
    # Reset arraypa.interpreters.eval_interpreter._backend
    eval_interpreter._backend = init_backend

    return result


def _analyze_dependencies(node: Ast, backend=None):
    global _captured_backend
    if backend is not None:
        _captured_backend = backend

    node_type = (
        Array if (isinstance(node, DepanalArray) or
                  isinstance(node, _captured_backend.ArrayTy))
        else type(node))
    if node_type not in _depanal_rules:
        raise ValueError(f"wrong node type {node}")
    return _depanal_rules[node_type](node)


_depanal_rules = dict()
for ty, rule in eval_interpreter.eval_rules.items():
    _depanal_rules[ty] = rule


# Overwrite Lam rule to introduce our special depanal arrays.
def _lam_depanal_rule(node: Lam):
    def _lambda_fn(val):
        var, shape = node.shaped_var
        depanal_array = DepanalArray(shape, name=var.name)
        return _analyze_dependencies(subst(var, depanal_array, node.body))
    return _lambda_fn


_depanal_rules[Lam] = _lam_depanal_rule
