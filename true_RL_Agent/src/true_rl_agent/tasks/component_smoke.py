"""Task-owned component smoke fixture contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from true_rl_agent.candidates.components import RewardInputs

Number = int | float


@dataclass(frozen=True)
class SmokeTensor:
    """Small tensor-like value for lightweight reward callable smoke tests."""

    values: tuple[Any, ...]

    @classmethod
    def vector(cls, values: list[Number] | tuple[Number, ...]) -> "SmokeTensor":
        return cls(tuple(float(value) for value in values))

    @classmethod
    def matrix(cls, values: list[list[Number]] | tuple[tuple[Number, ...], ...]) -> "SmokeTensor":
        return cls(tuple(tuple(float(item) for item in row) for row in values))

    @property
    def shape(self) -> tuple[int, ...]:
        if not self.values:
            return (0,)
        first = self.values[0]
        if isinstance(first, tuple):
            return (len(self.values), len(first))
        return (len(self.values),)

    def flatten(self) -> tuple[float, ...]:
        flattened: list[float] = []
        for value in self.values:
            if isinstance(value, tuple):
                flattened.extend(float(item) for item in value)
            else:
                flattened.append(float(value))
        return tuple(flattened)

    def abs(self) -> "SmokeTensor":
        return abs(self)

    def float(self) -> "SmokeTensor":
        return self

    def square(self) -> "SmokeTensor":
        return self._unary(lambda value: value * value)

    def sqrt(self) -> "SmokeTensor":
        return self._unary(math.sqrt)

    def pow(self, exponent: Any) -> "SmokeTensor":
        return self**exponent

    def relu(self) -> "SmokeTensor":
        return self.clamp_min(0.0)

    def clamp_min(self, min: Number) -> "SmokeTensor":
        return self.clamp(min=min)

    def clamp_max(self, max: Number) -> "SmokeTensor":
        return self.clamp(max=max)

    def clamp(self, min: Number | None = None, max: Number | None = None) -> "SmokeTensor":
        def _clamp(value: float) -> float:
            if min is not None and value < min:
                return float(min)
            if max is not None and value > max:
                return float(max)
            return value

        return self._unary(_clamp)

    def sum(self, dim: int | None = None, keepdim: bool = False) -> "SmokeTensor" | float:
        if dim is None:
            return float(sum(self.flatten()))
        if dim in {-1, 0} and len(self.shape) == 1:
            total = float(sum(self.flatten()))
            return SmokeTensor.vector([total]) if keepdim else total
        if dim in {-1, 1} and len(self.shape) == 2:
            values = [sum(row) for row in self.values]
            return _row_vector(values) if keepdim else SmokeTensor.vector(values)
        if dim == 0 and len(self.shape) == 2:
            cols = self.shape[1]
            values = [sum(float(row[col]) for row in self.values) for col in range(cols)]
            return SmokeTensor((tuple(float(item) for item in values),)) if keepdim else SmokeTensor.vector(values)
        raise ValueError("unsupported smoke tensor sum dimension")

    def mean(self, dim: int | None = None, keepdim: bool = False) -> "SmokeTensor" | float:
        if dim is None:
            flat = self.flatten()
            return float(sum(flat) / len(flat)) if flat else 0.0
        summed = self.sum(dim=dim, keepdim=keepdim)
        if isinstance(summed, SmokeTensor):
            denom = self.shape[1] if dim in {-1, 1} and len(self.shape) == 2 else self.shape[0]
            return summed / float(denom)
        return float(summed)

    def norm(self, p: Number = 2, dim: int | None = None, keepdim: bool = False) -> "SmokeTensor" | float:
        def _norm(values: tuple[float, ...]) -> float:
            order = float(p)
            if order <= 0:
                raise ValueError("unsupported smoke tensor norm order")
            if math.isinf(order):
                return max((abs(value) for value in values), default=0.0)
            return sum(abs(value) ** order for value in values) ** (1.0 / order)

        if dim is None:
            return _norm(self.flatten())
        if dim in {-1, 0} and len(self.shape) == 1:
            value = _norm(self.flatten())
            return SmokeTensor.vector([value]) if keepdim else value
        if dim in {-1, 1} and len(self.shape) == 2:
            values = [_norm(tuple(float(item) for item in row)) for row in self.values]
            return _row_vector(values) if keepdim else SmokeTensor.vector(values)
        if dim == 0 and len(self.shape) == 2:
            cols = self.shape[1]
            values = [_norm(tuple(float(row[col]) for row in self.values)) for col in range(cols)]
            return SmokeTensor((tuple(float(item) for item in values),)) if keepdim else SmokeTensor.vector(values)
        raise ValueError("unsupported smoke tensor norm dimension")

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __bool__(self) -> bool:
        raise ValueError("Boolean value of SmokeTensor is ambiguous")

    def __getitem__(self, index: Any) -> Any:
        selectors = _expand_index(index, dimensions=len(self.shape))
        if len(self.shape) == 1:
            selector = selectors[0]
            selected = self.values[selector]
            if isinstance(selector, slice):
                return SmokeTensor(tuple(float(value) for value in selected))
            return float(selected)

        rows, cols = selectors
        if isinstance(rows, int) and isinstance(cols, int):
            return float(self.values[rows][cols])
        if isinstance(rows, int):
            selected = self.values[rows][cols]
            return SmokeTensor(tuple(float(item) for item in selected))
        row_values = self.values[rows]
        if isinstance(cols, int):
            return SmokeTensor.vector([float(row[cols]) for row in row_values])
        selected_rows = [tuple(float(item) for item in row[cols]) for row in row_values]
        return SmokeTensor(tuple(selected_rows))

    def __add__(self, other: Any) -> "SmokeTensor":
        return self._binary(other, lambda left, right: left + right)

    def __radd__(self, other: Any) -> "SmokeTensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "SmokeTensor":
        return self._binary(other, lambda left, right: left - right)

    def __rsub__(self, other: Any) -> "SmokeTensor":
        return self._binary(other, lambda left, right: right - left)

    def __mul__(self, other: Any) -> "SmokeTensor":
        return self._binary(other, lambda left, right: left * right)

    def __rmul__(self, other: Any) -> "SmokeTensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "SmokeTensor":
        return self._binary(other, lambda left, right: left / right)

    def __rtruediv__(self, other: Any) -> "SmokeTensor":
        return self._binary(other, lambda left, right: right / left)

    def __pow__(self, other: Any) -> "SmokeTensor":
        return self._binary(other, lambda left, right: left**right)

    def __lt__(self, other: Any) -> "SmokeTensor":
        return self._compare(other, lambda left, right: left < right)

    def __le__(self, other: Any) -> "SmokeTensor":
        return self._compare(other, lambda left, right: left <= right)

    def __gt__(self, other: Any) -> "SmokeTensor":
        return self._compare(other, lambda left, right: left > right)

    def __ge__(self, other: Any) -> "SmokeTensor":
        return self._compare(other, lambda left, right: left >= right)

    def __neg__(self) -> "SmokeTensor":
        return self._unary(lambda value: -value)

    def __abs__(self) -> "SmokeTensor":
        return self._unary(abs)

    def _unary(self, op: Any) -> "SmokeTensor":
        if len(self.shape) == 2:
            return SmokeTensor(tuple(tuple(float(op(float(item))) for item in row) for row in self.values))
        return SmokeTensor(tuple(float(op(float(item))) for item in self.values))

    def _binary(self, other: Any, op: Any) -> "SmokeTensor":
        if isinstance(other, SmokeTensor):
            if self.shape != other.shape:
                raise ValueError("smoke tensor shape mismatch")
            if len(self.shape) == 2:
                return SmokeTensor(
                    tuple(
                        tuple(float(op(float(left), float(right))) for left, right in zip(row, other_row))
                        for row, other_row in zip(self.values, other.values)
                    )
                )
            return SmokeTensor(
                tuple(float(op(float(left), float(right))) for left, right in zip(self.values, other.values))
            )
        scalar = float(other)
        if len(self.shape) == 2:
            return SmokeTensor(
                tuple(tuple(float(op(float(item), scalar)) for item in row) for row in self.values)
            )
        return SmokeTensor(tuple(float(op(float(item), scalar)) for item in self.values))

    def _compare(self, other: Any, op: Any) -> "SmokeTensor":
        return self._binary(other, lambda left, right: 1.0 if op(left, right) else 0.0)


def _row_vector(values: list[Number] | tuple[Number, ...]) -> SmokeTensor:
    return SmokeTensor(tuple((float(value),) for value in values))


def _expand_index(index: Any, *, dimensions: int) -> tuple[int | slice, ...]:
    """Expand one standard ellipsis across the fixture's supported dimensions."""

    parts = list(index if isinstance(index, tuple) else (index,))
    ellipsis_positions = [position for position, item in enumerate(parts) if item is Ellipsis]
    if len(ellipsis_positions) > 1:
        raise IndexError("a smoke tensor index can contain only one ellipsis")
    if ellipsis_positions:
        position = ellipsis_positions[0]
        expanded_dimensions = dimensions - (len(parts) - 1)
        if expanded_dimensions < 0:
            raise IndexError("too many indices for smoke tensor")
        parts[position : position + 1] = [slice(None)] * expanded_dimensions
    if len(parts) > dimensions:
        raise IndexError("too many indices for smoke tensor")
    parts.extend([slice(None)] * (dimensions - len(parts)))
    if any(not isinstance(item, int | slice) for item in parts):
        raise TypeError("smoke tensor indices must be integers, slices, or one ellipsis")
    return tuple(parts)


@dataclass(frozen=True)
class ComponentSmokeFixture:
    """Task-owned inputs for authoring-time component smoke checks."""

    task_id: str
    reward_inputs: RewardInputs
    expected_reward_output_shape: tuple[int, ...]
    public_summary: Mapping[str, Any] = field(default_factory=dict)


class ComponentSmokeAdapter(Protocol):
    """Optional task-owned smoke fixture provider."""

    task_id: str

    def build_component_smoke_fixture(self) -> ComponentSmokeFixture | None:
        """Return a public-safe fixture, or None to skip smoke for this task."""
        ...


@dataclass(frozen=True)
class NeutralComponentSmokeAdapter:
    """Default no-op smoke adapter for tasks without a fixture contract."""

    task_id: str

    def build_component_smoke_fixture(self) -> None:
        return None
