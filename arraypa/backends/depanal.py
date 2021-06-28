from arraypa.core import Array, Backend, Index

from collections import defaultdict
from typing import Dict, Optional, Set


class DepanalArray(Array):
    name: Optional[str]
    # dependencies contains, for each index, a set of indices into other
    # arrays.
    dependencies: Dict[Index, Set]

    def __init__(self, shape, name=None, dependencies=None):
        self.shape, self.name = shape, name
        if dependencies is not None:
            self.dependencies = dependencies
        else:
            self.dependencies = defaultdict(set)

    def __getitem__(self, index: Index):
        if isinstance(index, int):
            index = tuple([index])
        if len(index) != len(self.shape):
            raise ValueError("expected total index in indexing op")

        deps_set = self.dependencies[index]
        if self.name is not None:
            deps_set = set([(self.name, index)])

        new_deps = defaultdict(set)
        new_deps[tuple()] = deps_set

        return DepanalArray(tuple(), dependencies=new_deps)

    def binop(self, rhs):
        return DepanalArray(self.shape, dependencies=(
            self._joindependencies(rhs.dependencies)))

    def cat(self, rhs):
        updated_rhs_deps = defaultdict(set)
        for index, deps in rhs.dependencies.items():
            new_index = tuple([self.shape[0] + index[0]]) + index[1:]
            updated_rhs_deps[new_index] = deps

        new_deps = self._joindependencies(updated_rhs_deps)
        new_shape = tuple([self.shape[0] + rhs.shape[0]]) + self.shape[1:]
        return DepanalArray(new_shape, dependencies=new_deps)

    def _joindependencies(self, otherdependencies):
        deps1, deps2 = self.dependencies, otherdependencies
        new_deps = defaultdict(set)
        for index in deps1:
            if index in deps2:
                new_deps[index] = deps1[index].union(deps2[index])
            else:
                new_deps[index] = deps1[index]

        for index in deps2:
            if index in deps1:
                continue  # skip
            else:
                new_deps[index] = deps2[index]

        return new_deps

    def reshape(self, new_shape):
        new_dependencies = defaultdict(set)
        for index, deps in self.dependencies.items():
            linear_index = 0
            stride = 1
            for i in range(len(self.shape) - 1, -1, -1):
                shape_component = self.shape[i]
                linear_index += index[i] * stride
                stride *= shape_component

            new_index = list()
            for i in range(len(new_shape) - 1, -1, -1):
                new_shape_component = new_shape[i]
                new_index.append(linear_index % new_shape_component)
                linear_index = linear_index // new_shape_component

            new_index = tuple(new_index[::-1])
            new_dependencies[new_index] = deps

        return DepanalArray(new_shape, name=self.name,
                            dependencies=new_dependencies)


class _DepanalBackend(Backend):
    ArrayTy = DepanalArray

    def _binop(self, lhs, rhs):
        if not isinstance(lhs, DepanalArray):
            # TODO: make it cleaner? This is a dirty way to get the right
            # op instance, since the result does not matter.
            return rhs #.binop(rhs)
        return lhs.binop(rhs)

    def add(self, lhs, rhs):
        return self._binop(lhs, rhs)

    def mul(self, lhs, rhs):
        return self._binop(lhs, rhs)

    def cat(self, lhs, rhs):
        return lhs.cat(rhs)

    def reshape(self, array, new_shape):
        return array.reshape(new_shape)


depanal_backend = _DepanalBackend()
