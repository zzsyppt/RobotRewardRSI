"""Stair-owned source rules for hidden axes and raw physical reward inputs."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Mapping

from true_rl_agent.candidates.component_source_semantics import (
    project_canonical_reward_term_sources,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairCompleteRewardRuntimeContract,
)

STAIR_REWARD_SOURCE_SEMANTICS_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.reward_source_semantics.v2"
)

_RAW_PHYSICAL_UNITS = frozenset({"N", "N*m", "rad/s^2"})

# These axes describe real physical groups, but their positional member mapping is
# intentionally absent from the public ABI. Candidates may aggregate a whole axis;
# they may not invent body/ray semantics for fixed positions within it.
STAIR_OPAQUE_INPUT_TRAILING_AXES: Mapping[str, tuple[int, ...]] = {
    "action_delta": (0,),
    "ankle_joint_pos": (0,),
    "ankle_soft_joint_pos_limits": (0,),
    "arm_joint_pos_error": (0,),
    "finger_joint_pos_error": (0,),
    "height_scan": (0,),
    "hip_knee_joint_acc": (0,),
    "hip_yaw_roll_joint_pos_error": (0,),
    "joint_pos": (0,),
    "joint_pos_error": (0,),
    "joint_vel": (0,),
    "last_action": (0,),
    "locomotion_joint_torque": (0,),
    "prev_action": (0,),
    "torso_joint_pos_error": (0,),
}

_ORDER_INVARIANT_REDUCTIONS = frozenset(
    {
        "all",
        "amax",
        "amin",
        "any",
        "max",
        "mean",
        "min",
        "norm",
        "prod",
        "sum",
    }
)
_ORDER_PRESERVING_METHODS = frozenset(
    {
        "abs",
        "bool",
        "clamp",
        "clamp_max",
        "clamp_min",
        "clip",
        "clone",
        "contiguous",
        "detach",
        "double",
        "exp",
        "expm1",
        "float",
        "isfinite",
        "isinf",
        "isnan",
        "log",
        "log1p",
        "masked_fill",
        "maximum",
        "minimum",
        "nan_to_num",
        "neg",
        "pow",
        "reciprocal",
        "relu",
        "round",
        "sigmoid",
        "sign",
        "square",
        "sqrt",
        "tanh",
        "to",
        "type_as",
        "where",
    }
)
_ORDER_PRESERVING_FUNCTIONS = frozenset({"abs"})
_POSITIONAL_CONSTRUCTOR_METHODS = frozenset(
    {
        "new_empty",
        "new_full",
        "new_ones",
        "new_tensor",
        "new_zeros",
    }
)
_POSITION_SENSITIVE_METHODS = frozenset(
    {
        "argsort",
        "chunk",
        "gather",
        "index_select",
        "kthvalue",
        "narrow",
        "select",
        "sort",
        "split",
        "take",
        "take_along_dim",
        "tensor_split",
        "topk",
        "unbind",
        "unfold",
    }
)
_AXIS_RESHAPING_METHODS = frozenset(
    {
        "as_strided",
        "flatten",
        "movedim",
        "permute",
        "reshape",
        "squeeze",
        "swapaxes",
        "swapdims",
        "transpose",
        "unsqueeze",
        "view",
    }
)
_AXIS_RESHAPING_ATTRIBUTES = frozenset({"T", "mT"})


@dataclass(frozen=True)
class _OpaqueValue:
    field_name: str
    rank: int
    opaque_axes: frozenset[int]


_OpaqueValues = frozenset[_OpaqueValue]


def stair_reward_source_semantics_policy(
    contract: StairCompleteRewardRuntimeContract,
) -> dict[str, Any]:
    """Return the single prompt policy enforced by Stair source admission."""

    _validated_opaque_fields(contract)
    raw_fields = _validated_raw_physical_fields(contract)
    return {
        "schema_version": STAIR_REWARD_SOURCE_SEMANTICS_SCHEMA_VERSION,
        "opaque_axis_contract": {
            "schema_version": STAIR_REWARD_SOURCE_SEMANTICS_SCHEMA_VERSION,
            "opaque_trailing_axis_groups": [
                {
                    "axes": list(axes),
                    "fields": sorted(
                        name
                        for name, field_axes in STAIR_OPAQUE_INPUT_TRAILING_AXES.items()
                        if field_axes == axes
                    ),
                }
                for axes in sorted(set(STAIR_OPAQUE_INPUT_TRAILING_AXES.values()))
            ],
            "usage_rule": (
                "Member mapping is hidden. Only whole-axis order-invariant reductions "
                "are valid; fixed or partial indexing, positional selection, and "
                "pre-reduction reshaping are forbidden. Helpers that consume these "
                "fields must be top-level functions."
            ),
        },
        "raw_physical_magnitude_contract": {
            "fields": sorted(raw_fields),
            "units": sorted(_RAW_PHYSICAL_UNITS),
            "normalization_status": "raw_not_normalized",
            "enforcement": "active_reward_term_final_clip",
            "candidate_usage": (
                "New or modified RewardTerms using these fields require finite "
                "clip_min and clip_max. Callable-only transforms do not replace this "
                "final clip; runtime clips before weight. Exact unchanged reference "
                "mechanisms are exempt only by canonical structure signature. The "
                "declared clip and weight expose the resulting weighted bound."
            ),
        },
    }


def stair_reward_source_semantics_policy_hash(
    contract: StairCompleteRewardRuntimeContract,
    *,
    trusted_reference_source_text: str,
) -> str:
    return canonical_json_hash(
        {
            "public_policy": stair_reward_source_semantics_policy(contract),
            "trusted_raw_reward_term_signatures": (
                _trusted_raw_reward_term_signatures(
                    trusted_reference_source_text,
                    contract=contract,
                )
            ),
            "order_invariant_reductions": sorted(_ORDER_INVARIANT_REDUCTIONS),
            "order_preserving_methods": sorted(_ORDER_PRESERVING_METHODS),
            "order_preserving_functions": sorted(_ORDER_PRESERVING_FUNCTIONS),
            "position_sensitive_methods": sorted(_POSITION_SENSITIVE_METHODS),
            "axis_reshaping_methods": sorted(_AXIS_RESHAPING_METHODS),
            "axis_reshaping_attributes": sorted(_AXIS_RESHAPING_ATTRIBUTES),
            "positional_constructor_methods": sorted(
                _POSITIONAL_CONSTRUCTOR_METHODS
            ),
        }
    )


def audit_stair_reward_source_semantics(
    source_text: str,
    *,
    contract: StairCompleteRewardRuntimeContract,
    trusted_reference_source_text: str,
) -> tuple[str, ...]:
    """Enforce task-owned axis and raw-magnitude rules on active components."""

    if not isinstance(source_text, str):
        raise TypeError("Stair reward source must be text")
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        return (f"stair_source_syntax_error:line_{exc.lineno or 0}",)
    analyzer = _OpaqueAxisAnalyzer(
        tree=tree,
        opaque_fields=_validated_opaque_fields(contract),
    )
    issues = set(analyzer.audit())
    issues.update(
        _raw_physical_reward_issues(
            source_text,
            contract=contract,
            trusted_reference_source_text=trusted_reference_source_text,
        )
    )
    return tuple(sorted(issues))


def _validated_raw_physical_fields(
    contract: StairCompleteRewardRuntimeContract,
) -> frozenset[str]:
    if not isinstance(contract, StairCompleteRewardRuntimeContract):
        raise TypeError("Stair source semantics require the complete reward contract")
    fields = frozenset(
        field.name
        for field in contract.input_fields
        if field.unit in _RAW_PHYSICAL_UNITS
    )
    if not fields:
        raise ValueError("Stair raw physical reward fields are absent")
    return fields


def _maximum_absolute_weighted_term_rate(
    contract: StairCompleteRewardRuntimeContract,
) -> float:
    output = contract.payload.get("output_contract")
    if not isinstance(output, Mapping):
        raise ValueError("Stair reward output contract is absent")
    value = output.get("maximum_absolute_term_rate")
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("Stair weighted reward-term limit is invalid")
    return float(value)


def _trusted_raw_reward_term_signatures(
    source_text: str,
    *,
    contract: StairCompleteRewardRuntimeContract,
) -> dict[str, str]:
    terms = project_canonical_reward_term_sources(source_text)
    if terms is None:
        raise ValueError("trusted Stair reference source is not canonical")
    raw_fields = _validated_raw_physical_fields(contract)
    return {
        term.mechanism_id: term.structure_signature
        for term in terms
        if raw_fields.intersection(term.active_callable_attribute_names)
    }


def _raw_physical_reward_issues(
    source_text: str,
    *,
    contract: StairCompleteRewardRuntimeContract,
    trusted_reference_source_text: str,
) -> tuple[str, ...]:
    if not isinstance(trusted_reference_source_text, str) or not (
        trusted_reference_source_text.strip()
    ):
        raise ValueError("trusted Stair reference source must be non-empty text")
    raw_fields = _validated_raw_physical_fields(contract)
    if not any(
        isinstance(node, ast.Attribute) and node.attr in raw_fields
        for node in ast.walk(ast.parse(source_text))
    ):
        return ()
    terms = project_canonical_reward_term_sources(source_text)
    if terms is None:
        return ("raw_physical_reward_component_projection_unavailable",)
    trusted = _trusted_raw_reward_term_signatures(
        trusted_reference_source_text,
        contract=contract,
    )
    maximum_rate = _maximum_absolute_weighted_term_rate(contract)
    issues: list[str] = []
    for term in terms:
        dependencies = sorted(
            raw_fields.intersection(term.active_callable_attribute_names)
        )
        if not dependencies:
            continue
        if trusted.get(term.mechanism_id) == term.structure_signature:
            continue
        issue_suffix = (
            f"{term.mechanism_id}:fields={','.join(dependencies)}"
        )
        if term.clip_min is None or term.clip_max is None:
            issues.append(
                "raw_physical_reward_requires_finite_two_sided_clip:"
                + issue_suffix
            )
            continue
        if term.clip_min >= term.clip_max:
            issues.append("raw_physical_reward_clip_order_invalid:" + issue_suffix)
            continue
        weighted_bound = abs(term.weight) * max(
            abs(term.clip_min),
            abs(term.clip_max),
        )
        if not math.isfinite(weighted_bound) or weighted_bound > maximum_rate:
            issues.append(
                "raw_physical_reward_weighted_bound_exceeds_output_contract:"
                + issue_suffix
            )
    return tuple(sorted(issues))


def _validated_opaque_fields(
    contract: StairCompleteRewardRuntimeContract,
) -> dict[str, tuple[int, tuple[int, ...]]]:
    if not isinstance(contract, StairCompleteRewardRuntimeContract):
        raise TypeError("Stair source semantics require the complete reward contract")
    shapes = {
        field.name: tuple(field.trailing_shape)
        for field in contract.input_fields
    }
    result: dict[str, tuple[int, tuple[int, ...]]] = {}
    for name, axes in STAIR_OPAQUE_INPUT_TRAILING_AXES.items():
        if name not in shapes:
            raise ValueError(f"opaque Stair reward field is absent: {name}")
        trailing_shape = shapes[name]
        if any(axis < 0 or axis >= len(trailing_shape) for axis in axes):
            raise ValueError(f"opaque Stair reward axes are invalid: {name}")
        result[name] = (1 + len(trailing_shape), tuple(1 + axis for axis in axes))
    return result


class _OpaqueAxisAnalyzer:
    def __init__(
        self,
        *,
        tree: ast.Module,
        opaque_fields: Mapping[str, tuple[int, tuple[int, ...]]],
    ) -> None:
        self._tree = tree
        self._opaque_fields = dict(opaque_fields)
        self._source_opaque_fields = frozenset(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in opaque_fields
        )
        self._functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self._issues: set[str] = set()
        if self._source_opaque_fields:
            top_level_function_ids = {
                id(node)
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.FunctionDef)
                    and id(node) not in top_level_function_ids
                ):
                    for field_name in self._source_opaque_fields:
                        self._issues.add(
                            f"opaque_input_nested_helper:{field_name}:"
                            f"line_{getattr(node, 'lineno', 0)}"
                        )
        self._cache: dict[
            tuple[str, tuple[tuple[str, tuple[_OpaqueValue, ...]], ...]],
            _OpaqueValues,
        ] = {}
        self._active: set[
            tuple[str, tuple[tuple[str, tuple[_OpaqueValue, ...]], ...]]
        ] = set()

    def audit(self) -> tuple[str, ...]:
        for name in self._functions:
            self._analyze_function(name, {})
        return tuple(sorted(self._issues))

    def _analyze_function(
        self,
        name: str,
        arguments: Mapping[str, _OpaqueValues],
    ) -> _OpaqueValues:
        key = (
            name,
            tuple(
                (arg, tuple(sorted(values, key=_opaque_value_key)))
                for arg, values in sorted(arguments.items())
            ),
        )
        if key in self._cache:
            return self._cache[key]
        if key in self._active:
            for values in arguments.values():
                self._record_issue(
                    "recursive_helper",
                    values,
                    self._functions[name],
                )
            return _union_values(*arguments.values())
        self._active.add(key)
        function = self._functions[name]
        env = {
            argument.arg: frozenset(arguments.get(argument.arg, ()))
            for argument in function.args.args
        }
        returned, _ = self._analyze_statements(function.body, env)
        self._active.remove(key)
        self._cache[key] = returned
        return returned

    def _analyze_statements(
        self,
        statements: list[ast.stmt],
        env: dict[str, _OpaqueValues],
    ) -> tuple[_OpaqueValues, dict[str, _OpaqueValues]]:
        returned: _OpaqueValues = frozenset()
        for statement in statements:
            if isinstance(statement, ast.Assign):
                value = self._expression(statement.value, env)
                for target in statement.targets:
                    self._bind_target(target, value, env)
            elif isinstance(statement, ast.AnnAssign):
                value = (
                    frozenset()
                    if statement.value is None
                    else self._expression(statement.value, env)
                )
                self._bind_target(statement.target, value, env)
            elif isinstance(statement, ast.AugAssign):
                value = _union_values(
                    self._expression(statement.target, env),
                    self._expression(statement.value, env),
                )
                self._bind_target(statement.target, value, env)
            elif isinstance(statement, ast.Return):
                if statement.value is not None:
                    returned = _union_values(
                        returned,
                        self._expression(statement.value, env),
                    )
            elif isinstance(statement, ast.Expr):
                self._expression(statement.value, env)
            elif isinstance(statement, ast.If):
                self._expression(statement.test, env)
                body_returned, body_env = self._analyze_statements(
                    statement.body,
                    dict(env),
                )
                else_returned, else_env = self._analyze_statements(
                    statement.orelse,
                    dict(env),
                )
                returned = _union_values(
                    returned,
                    body_returned,
                    else_returned,
                )
                self._merge_env(env, body_env, else_env)
            elif isinstance(statement, ast.For | ast.While):
                if isinstance(statement, ast.For):
                    self._expression(statement.iter, env)
                else:
                    self._expression(statement.test, env)
                body_returned, body_env = self._analyze_statements(
                    statement.body,
                    dict(env),
                )
                else_returned, else_env = self._analyze_statements(
                    statement.orelse,
                    dict(env),
                )
                returned = _union_values(
                    returned,
                    body_returned,
                    else_returned,
                )
                self._merge_env(env, body_env, else_env)
            else:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        self._expression(child, env)
        return returned, env

    def _expression(
        self,
        node: ast.AST,
        env: dict[str, _OpaqueValues],
    ) -> _OpaqueValues:
        if isinstance(node, ast.Name):
            return env.get(node.id, frozenset())
        if isinstance(node, ast.Attribute):
            if node.attr in self._opaque_fields:
                rank, axes = self._opaque_fields[node.attr]
                return frozenset(
                    {_OpaqueValue(node.attr, rank, frozenset(axes))}
                )
            values = self._expression(node.value, env)
            if node.attr in _AXIS_RESHAPING_ATTRIBUTES:
                self._record_issue("axis_reshape", values, node, node.attr)
            return values
        if isinstance(node, ast.Subscript):
            values = self._expression(node.value, env)
            return self._apply_subscript(values, node.slice, node)
        if isinstance(node, ast.Call):
            return self._call(node, env)
        if isinstance(node, ast.NamedExpr):
            values = self._expression(node.value, env)
            self._bind_target(node.target, values, env)
            return values
        if isinstance(node, ast.Lambda):
            return self._expression(node.body, env)
        values: list[_OpaqueValues] = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                values.append(self._expression(child, env))
        return _union_values(*values)

    def _call(
        self,
        node: ast.Call,
        env: dict[str, _OpaqueValues],
    ) -> _OpaqueValues:
        positional = [self._expression(item, env) for item in node.args]
        keyword = {
            item.arg: self._expression(item.value, env)
            for item in node.keywords
            if item.arg is not None
        }
        if isinstance(node.func, ast.Name) and node.func.id in self._functions:
            function = self._functions[node.func.id]
            bound: dict[str, _OpaqueValues] = {}
            parameter_names = [item.arg for item in function.args.args]
            for index, values in enumerate(positional):
                if index < len(parameter_names):
                    bound[parameter_names[index]] = values
            for name, values in keyword.items():
                if name in parameter_names:
                    bound[name] = values
            return self._analyze_function(node.func.id, bound)

        receiver: _OpaqueValues = frozenset()
        method = ""
        if isinstance(node.func, ast.Attribute):
            receiver = self._expression(node.func.value, env)
            method = node.func.attr
        all_values = _union_values(receiver, *positional, *keyword.values())
        if method in _POSITIONAL_CONSTRUCTOR_METHODS and self._source_opaque_fields:
            for field_name in self._source_opaque_fields:
                self._issues.add(
                    f"opaque_input_positional_constructor:{field_name}:{method}:"
                    f"line_{getattr(node, 'lineno', 0)}"
                )
            return all_values
        if method in _POSITION_SENSITIVE_METHODS:
            self._record_issue("position_sensitive", receiver, node, method)
            return receiver
        if method in _AXIS_RESHAPING_METHODS:
            self._record_issue("axis_reshape", receiver, node, method)
            return receiver
        if method in _ORDER_INVARIANT_REDUCTIONS:
            return self._apply_reduction(receiver, node)
        if method in _ORDER_PRESERVING_METHODS:
            return all_values
        if method and receiver:
            self._record_issue("unsupported_method", receiver, node, method)
            return receiver
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _ORDER_PRESERVING_FUNCTIONS
        ):
            return all_values
        if all_values:
            operation = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else "dynamic_call"
            )
            self._record_issue("unsupported_call", all_values, node, operation)
        return all_values

    def _apply_subscript(
        self,
        values: _OpaqueValues,
        raw_slice: ast.AST,
        node: ast.Subscript,
    ) -> _OpaqueValues:
        updated: set[_OpaqueValue] = set()
        for value in values:
            selectors = _expanded_selectors(raw_slice, rank=value.rank)
            if selectors is None:
                self._record_issue("axis_selection", {value}, node)
                updated.add(value)
                continue
            new_axis = 0
            new_opaque: set[int] = set()
            invalid = False
            for old_axis, selector in enumerate(selectors):
                if selector is None:
                    self._record_issue("axis_reshape", {value}, node, "newaxis")
                    invalid = True
                    break
                full_slice = _is_full_slice(selector)
                if old_axis in value.opaque_axes and not full_slice:
                    self._record_issue("axis_selection", {value}, node)
                    invalid = True
                    break
                if full_slice or isinstance(selector, ast.Slice):
                    if old_axis in value.opaque_axes:
                        new_opaque.add(new_axis)
                    new_axis += 1
                elif _is_integer_literal(selector):
                    continue
                else:
                    self._record_issue("axis_selection", {value}, node)
                    invalid = True
                    break
            if invalid:
                updated.add(value)
            elif new_opaque:
                updated.add(
                    _OpaqueValue(
                        value.field_name,
                        new_axis,
                        frozenset(new_opaque),
                    )
                )
        return frozenset(updated)

    def _apply_reduction(
        self,
        values: _OpaqueValues,
        node: ast.Call,
    ) -> _OpaqueValues:
        if not values:
            return values
        dimensions = _literal_reduction_dimensions(node)
        keepdim = _literal_keepdim(node)
        if dimensions is None:
            return frozenset()
        updated: set[_OpaqueValue] = set()
        for value in values:
            normalized = {
                dim if dim >= 0 else value.rank + dim
                for dim in dimensions
            }
            if any(dim < 0 or dim >= value.rank for dim in normalized):
                updated.add(value)
                continue
            remaining_opaque = set(value.opaque_axes) - normalized
            if not remaining_opaque:
                continue
            if keepdim:
                new_axes = remaining_opaque
                new_rank = value.rank
            else:
                removed = sorted(normalized)
                new_axes = {
                    axis - sum(removed_axis < axis for removed_axis in removed)
                    for axis in remaining_opaque
                }
                new_rank = value.rank - len(removed)
            updated.add(
                _OpaqueValue(
                    value.field_name,
                    new_rank,
                    frozenset(new_axes),
                )
            )
        return frozenset(updated)

    def _record_issue(
        self,
        kind: str,
        values: Any,
        node: ast.AST,
        operation: str | None = None,
    ) -> None:
        for value in values:
            suffix = "" if operation is None else f":{operation}"
            self._issues.add(
                f"opaque_input_{kind}:{value.field_name}{suffix}:"
                f"line_{getattr(node, 'lineno', 0)}"
            )

    @staticmethod
    def _bind_target(
        target: ast.AST,
        values: _OpaqueValues,
        env: dict[str, _OpaqueValues],
    ) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = values
        elif isinstance(target, ast.Starred):
            _OpaqueAxisAnalyzer._bind_target(target.value, values, env)
        elif isinstance(target, ast.Tuple | ast.List):
            for item in target.elts:
                _OpaqueAxisAnalyzer._bind_target(item, values, env)

    @staticmethod
    def _merge_env(
        target: dict[str, _OpaqueValues],
        *sources: Mapping[str, _OpaqueValues],
    ) -> None:
        for name in set(target).union(*(source.keys() for source in sources)):
            target[name] = _union_values(
                target.get(name, frozenset()),
                *(source.get(name, frozenset()) for source in sources),
            )


def _expanded_selectors(
    raw_slice: ast.AST,
    *,
    rank: int,
) -> tuple[ast.AST | None, ...] | None:
    selectors = list(raw_slice.elts) if isinstance(raw_slice, ast.Tuple) else [raw_slice]
    ellipsis_positions = [
        index
        for index, selector in enumerate(selectors)
        if isinstance(selector, ast.Constant) and selector.value is Ellipsis
    ]
    if len(ellipsis_positions) > 1:
        return None
    consumed = sum(
        not (isinstance(selector, ast.Constant) and selector.value in {None, Ellipsis})
        for selector in selectors
    )
    if consumed > rank:
        return None
    if ellipsis_positions:
        index = ellipsis_positions[0]
        selectors[index : index + 1] = [ast.Slice()] * (rank - consumed)
    else:
        selectors.extend(ast.Slice() for _ in range(rank - consumed))
    return tuple(
        None
        if isinstance(selector, ast.Constant) and selector.value is None
        else selector
        for selector in selectors
    )


def _literal_reduction_dimensions(node: ast.Call) -> tuple[int, ...] | None:
    raw: ast.AST | None = node.args[0] if node.args else None
    for keyword in node.keywords:
        if keyword.arg in {"dim", "axis"}:
            raw = keyword.value
            break
    if raw is None:
        return None
    if _is_integer_literal(raw):
        value = _integer_literal(raw)
        assert value is not None
        return (value,)
    if isinstance(raw, ast.Tuple | ast.List):
        values = tuple(_integer_literal(item) for item in raw.elts)
        if all(value is not None for value in values):
            return tuple(value for value in values if value is not None)
    return ()


def _literal_keepdim(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg in {"keepdim", "keepdims"}:
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
    return False


def _is_full_slice(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Slice)
        and node.lower is None
        and node.upper is None
        and node.step is None
    )


def _is_integer_literal(node: ast.AST) -> bool:
    return _integer_literal(node) is not None


def _integer_literal(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
        node.value,
        bool,
    ):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        return -int(node.operand.value)
    return None


def _opaque_value_key(value: _OpaqueValue) -> tuple[str, int, tuple[int, ...]]:
    return value.field_name, value.rank, tuple(sorted(value.opaque_axes))


def _union_values(*groups: Any) -> _OpaqueValues:
    result: set[_OpaqueValue] = set()
    for group in groups:
        result.update(group)
    return frozenset(result)


__all__ = [
    "STAIR_OPAQUE_INPUT_TRAILING_AXES",
    "STAIR_REWARD_SOURCE_SEMANTICS_SCHEMA_VERSION",
    "audit_stair_reward_source_semantics",
    "stair_reward_source_semantics_policy",
    "stair_reward_source_semantics_policy_hash",
]
