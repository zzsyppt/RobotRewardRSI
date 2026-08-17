"""Static source semantics for canonical candidate mechanism deltas."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.candidates.authoring_policy import (
    ANTI_DEGENERATE_ASSESSMENT_SCHEMA_VERSION,
    CANDIDATE_MECHANISM_OPERATION_DELTA_SCHEMA_VERSION,
    MAX_CANONICAL_COMPONENT_DELTAS,
    SINGLE_COMPONENT_DELTA_INSTRUCTION,
    format_canonical_mechanism_id,
    parse_canonical_mechanism_id,
    require_mechanism_operation_delta,
    select_authoring_anchor_source,
)
from true_rl_agent.candidates.authoring_public_result import public_authoring_result
from true_rl_agent.candidates.components import DiagnosticTerm, RewardTerm, ScheduleTerm
from true_rl_agent.candidates.package_contract import (
    CandidatePackageContractError,
    open_preflighted_candidate_file,
    preflight_candidate_package,
)
from true_rl_agent.candidates.schema import CandidatePackageRef

ANTI_DEGENERATE_SCHEMA_VERSION = ANTI_DEGENERATE_ASSESSMENT_SCHEMA_VERSION
_NON_MECHANISM_DELTA_CLASSIFICATIONS = frozenset(
    {
        "identical_or_format_only",
        "no_component_mechanism_delta",
        "parameter_only_delta",
        "rationale_only_delta",
        "parameter_and_rationale_only_delta",
    }
)
_STRUCTURAL_CALL_KEYWORDS = frozenset(
    {"axis", "axes", "dim", "dims", "keepdim", "keepdims"}
)
_POSITIONAL_DIMENSION_CALLS = frozenset(
    {
        "amax",
        "amin",
        "argmax",
        "argmin",
        "argsort",
        "flatten",
        "max",
        "mean",
        "min",
        "movedim",
        "moveaxis",
        "norm",
        "prod",
        "select",
        "sort",
        "squeeze",
        "sum",
        "topk",
        "transpose",
        "unsqueeze",
    }
)

_public_result = public_authoring_result


@dataclass(frozen=True)
class CanonicalRewardTermSource:
    """Task-neutral projection of one active, canonical RewardTerm declaration."""

    mechanism_id: str
    structure_signature: str
    callable_name: str
    active_callable_attribute_names: tuple[str, ...]
    weight: float
    clip_min: float | None
    clip_max: float | None


@dataclass(frozen=True)
class _ActiveComponentSource:
    kind: str
    mechanism_id: str
    structure_signature: str
    parameter_normalized_structure_signature: str
    resolved_arguments: Mapping[str, ast.expr]


@dataclass(frozen=True)
class _SingleComponentDeltaSemantics:
    kind: str
    presence: str
    mechanism_id: str
    structure_signature: str
    parameter_normalized_structure_signature: str
    input_fields: tuple[str, ...]


class SingleComponentDeltaComparisonUnavailable(ValueError):
    """Raised when either source cannot yield one attributable component delta."""


_INVALID_LITERAL = object()


def assess_anti_degenerate_candidate(
    candidate: CandidatePackageRef,
    prompt_payload: Mapping[str, Any],
    *,
    evaluated_history_sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assess candidate source against its authoring anchor and evaluated history."""

    anchor_source, anchor_hash, anchor_role = select_authoring_anchor_source(
        prompt_payload
    )
    history_coverage = _history_comparison_coverage(
        evaluated_history_sources,
        candidate_anchor_available=anchor_source is not None,
    )
    candidate_source_record = _read_candidate_components_source(candidate)
    if candidate_source_record is None:
        return _with_history_comparison_coverage(
            {
                "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
                "status": "pass",
                "classification": "unknown",
                "reason": "candidate_source_not_available",
                "compares_to": anchor_hash,
                "comparison_anchor": anchor_role,
                "blocking": False,
            },
            history_coverage,
        )
    candidate_source, candidate_source_sha256 = candidate_source_record
    if anchor_source is None:
        anchor_assessment = _public_result(
            {
                "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
                "status": "pass",
                "classification": "unknown",
                "reason": "authoring_anchor_source_not_inline",
                "compares_to": None,
                "comparison_anchor": anchor_role,
                "blocking": False,
            }
        )
    else:
        anchor_assessment = assess_anti_degenerate_source(
            candidate_source=candidate_source,
            current_best_source=anchor_source,
            current_best_sha256=anchor_hash,
        )
        anchor_assessment["candidate_source_sha256"] = candidate_source_sha256
        anchor_assessment["comparison_anchor"] = anchor_role

    history_assessment = _assess_against_evaluated_history(
        candidate_source=candidate_source,
        candidate_anchor_source=anchor_source,
        evaluated_history_sources=evaluated_history_sources,
    )
    if history_assessment is not None:
        history_assessment["candidate_source_sha256"] = candidate_source_sha256
        return _with_history_comparison_coverage(
            history_assessment,
            history_coverage,
        )

    if anchor_assessment.get("blocking") is True:
        return _with_history_comparison_coverage(
            anchor_assessment,
            history_coverage,
        )

    historical_variant_assessment = _assess_rejected_input_dependency_variants(
        candidate_source=candidate_source,
        anchor_source=anchor_source,
        anchor_hash=anchor_hash,
        evaluated_history_sources=evaluated_history_sources,
    )

    result = dict(anchor_assessment)
    result["candidate_source_sha256"] = candidate_source_sha256
    result["comparison_scope"] = "authoring_anchor_and_evaluated_history"
    result["evaluated_history_source_count"] = len(evaluated_history_sources)
    if historical_variant_assessment is not None:
        result["historical_variant_assessment"] = historical_variant_assessment
    return _with_history_comparison_coverage(result, history_coverage)


def _history_comparison_coverage(
    evaluated_history_sources: Sequence[Mapping[str, Any]],
    *,
    candidate_anchor_available: bool,
) -> dict[str, Any]:
    with_anchor = 0
    without_anchor = 0
    for historical in evaluated_history_sources:
        if not isinstance(historical.get("content"), str):
            continue
        if isinstance(historical.get("anchor_content"), str):
            with_anchor += 1
        else:
            without_anchor += 1
    return {
        "evaluated_history_source_count": len(evaluated_history_sources),
        "history_sources_with_anchor": with_anchor,
        "history_sources_without_anchor": without_anchor,
        "candidate_authoring_anchor_available": candidate_anchor_available,
    }


def _with_history_comparison_coverage(
    result: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, Any]:
    projected = dict(result)
    projected.update(coverage)
    return _public_result(projected)


def assess_anti_degenerate_source(
    *,
    candidate_source: str,
    current_best_source: str,
    current_best_sha256: str | None = None,
) -> dict[str, Any]:
    """Classify obvious no-op, parameter-only, or compound source changes."""

    candidate_digest = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
    current_digest = current_best_sha256 or hashlib.sha256(current_best_source.encode("utf-8")).hexdigest()
    try:
        candidate_tree = ast.parse(candidate_source)
        current_tree = ast.parse(current_best_source)
    except SyntaxError as exc:
        return _public_result(
            {
                "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
                "status": "pass",
                "classification": "unknown",
                "reason": "source_parse_failed",
                "error_type": type(exc).__name__,
                "compares_to": current_digest,
                "candidate_source_sha256": candidate_digest,
                "blocking": False,
            }
        )

    candidate_ast = ast.dump(candidate_tree, include_attributes=False)
    current_ast = ast.dump(current_tree, include_attributes=False)
    if candidate_ast == current_ast:
        return _anti_degenerate_result(
            classification="identical_or_format_only",
            status="fail",
            reason="candidate_source_is_identical_or_format_only",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=True,
            feedback=(
                "Rewrite the candidate mechanism instead of resubmitting the "
                "comparison-anchor source."
            ),
        )

    changed_mechanisms = _changed_component_mechanisms(
        candidate_tree=candidate_tree,
        anchor_tree=current_tree,
    )
    if changed_mechanisms is None:
        return _anti_degenerate_result(
            classification="canonical_mechanism_structure_invalid",
            status="fail",
            reason="canonical_component_mechanisms_cannot_be_identified",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=True,
            feedback=(
                "Use literal unique names for canonical RewardTerm, DiagnosticTerm, "
                "or ScheduleTerm entries so exactly one mechanism delta can be verified."
            ),
            details={"structural_comparison": "unavailable"},
        )
    if changed_mechanisms == ():
        return _anti_degenerate_result(
            classification="no_component_mechanism_delta",
            status="fail",
            reason="candidate_changes_no_canonical_component_mechanism",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=True,
            feedback=(
                "Change exactly one canonical RewardTerm, DiagnosticTerm, or "
                "ScheduleTerm; unrelated or unused source does not affect training."
            ),
            details={
                "changed_mechanism_count": 0,
                "changed_mechanisms": [],
                "structural_comparison": "canonical_component_terms",
            },
        )
    candidate_numeric_ast = ast.dump(
        _normalize_parameter_like_numeric_constants(candidate_tree),
        include_attributes=False,
    )
    current_numeric_ast = ast.dump(
        _normalize_parameter_like_numeric_constants(current_tree),
        include_attributes=False,
    )
    if candidate_numeric_ast == current_numeric_ast:
        if (
            changed_mechanisms is not None
            and len(changed_mechanisms) > MAX_CANONICAL_COMPONENT_DELTAS
        ):
            return _compound_mechanism_result(
                compares_to=current_digest,
                candidate_digest=candidate_digest,
                changed_mechanisms=changed_mechanisms,
            )
        return _anti_degenerate_result(
            classification="parameter_only_delta",
            status="warning",
            reason="candidate_changes_only_numeric_constants",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=False,
            feedback="Numeric-only edits are weak evidence; prefer a code-level reward mechanism change.",
            details={
                "changed_mechanism_count": len(changed_mechanisms),
                "changed_mechanisms": list(changed_mechanisms),
                "structural_comparison": "canonical_component_terms",
            },
        )

    candidate_string_ast = ast.dump(_normalize_string_constants(candidate_tree), include_attributes=False)
    current_string_ast = ast.dump(_normalize_string_constants(current_tree), include_attributes=False)
    if candidate_string_ast == current_string_ast:
        return _anti_degenerate_result(
            classification="rationale_only_delta",
            status="warning",
            reason="candidate_changes_only_strings_or_rationale",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=False,
            feedback="Rationale-only edits do not create a new training mechanism.",
            details={
                "changed_mechanism_count": len(changed_mechanisms),
                "changed_mechanisms": list(changed_mechanisms),
                "structural_comparison": "canonical_component_terms",
            },
        )

    candidate_non_mechanism_ast = ast.dump(
        _normalize_non_mechanism_constants(candidate_tree),
        include_attributes=False,
    )
    current_non_mechanism_ast = ast.dump(
        _normalize_non_mechanism_constants(current_tree),
        include_attributes=False,
    )
    if candidate_non_mechanism_ast == current_non_mechanism_ast:
        return _anti_degenerate_result(
            classification="parameter_and_rationale_only_delta",
            status="warning",
            reason="candidate_changes_only_numeric_and_string_constants",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=False,
            feedback=(
                "Numeric and rationale-only edits do not create a new training mechanism."
            ),
            details={
                "changed_mechanism_count": len(changed_mechanisms),
                "changed_mechanisms": list(changed_mechanisms),
                "structural_comparison": "canonical_component_terms",
            },
        )

    if (
        changed_mechanisms is not None
        and len(changed_mechanisms) > MAX_CANONICAL_COMPONENT_DELTAS
    ):
        return _compound_mechanism_result(
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            changed_mechanisms=changed_mechanisms,
        )
    return _anti_degenerate_result(
        classification="mechanism_delta_present",
        status="pass",
        reason="candidate_source_structure_differs_from_comparison_anchor",
        compares_to=current_digest,
        candidate_digest=candidate_digest,
        blocking=False,
        feedback="Mechanism delta is present; runner-owned eval still decides acceptance.",
        details=(
            {
                "changed_mechanism_count": len(changed_mechanisms),
                "changed_mechanisms": list(changed_mechanisms),
                "structural_comparison": "canonical_component_terms",
            }
            if changed_mechanisms is not None
            else {"structural_comparison": "unavailable"}
        ),
    )


def _candidate_dir(candidate: CandidatePackageRef) -> Path | None:
    if "://" in candidate.path:
        return None
    path = Path(candidate.path)
    if not path.is_dir():
        return None
    return path


def _read_candidate_components_source(
    candidate: CandidatePackageRef,
) -> tuple[str, str] | None:
    candidate_dir = _candidate_dir(candidate)
    if candidate_dir is None:
        return None
    try:
        preflight = preflight_candidate_package(
            candidate_dir,
            sealed_artifact=False,
            enforce_size_limits=True,
        )
        state = preflight.file_state("training_code/components.py")
        with open_preflighted_candidate_file(
            preflight,
            "training_code/components.py",
        ) as source:
            payload = source.read(state.size_bytes + 1)
        if len(payload) != state.size_bytes:
            return None
        return payload.decode("utf-8"), hashlib.sha256(payload).hexdigest()
    except (CandidatePackageContractError, OSError, UnicodeDecodeError):
        return None


_COMPONENT_TERM_FIELDS = {
    "RewardTerm": ("name", "weight", "fn", "clip_min", "clip_max"),
    "DiagnosticTerm": ("name", "fn"),
    "ScheduleTerm": ("name", "key", "value"),
}
_COMPONENT_TERM_CLASSES = {
    "RewardTerm": RewardTerm,
    "DiagnosticTerm": DiagnosticTerm,
    "ScheduleTerm": ScheduleTerm,
}
_TRAINING_COMPONENT_LIST_KINDS = {
    "reward_terms": "RewardTerm",
    "diagnostics": "DiagnosticTerm",
    "schedules": "ScheduleTerm",
}
_TRAINING_COMPONENT_FIELDS = (
    "name",
    "reward_terms",
    "diagnostics",
    "schedules",
    "metadata",
    "rationale",
)


def _changed_component_mechanisms(
    *,
    candidate_tree: ast.Module,
    anchor_tree: ast.Module,
) -> tuple[str, ...] | None:
    candidate = _component_mechanism_signatures(candidate_tree)
    anchor = _component_mechanism_signatures(anchor_tree)
    if candidate is None or anchor is None:
        return None
    changed = [
        key
        for key in candidate.keys() & anchor.keys()
        if candidate[key] != anchor[key]
    ]
    removed = sorted(anchor.keys() - candidate.keys())
    added = sorted(candidate.keys() - anchor.keys())
    if (
        len(removed) == 1
        and len(added) == 1
        and removed[0].partition(":")[0] == added[0].partition(":")[0]
    ):
        removed_identity = parse_canonical_mechanism_id(removed[0])
        added_identity = parse_canonical_mechanism_id(added[0])
        if removed_identity is None or added_identity is None:
            return None
        if anchor[removed[0]] != candidate[added[0]]:
            kind, old_name, _ = removed_identity
            _, new_name, _ = added_identity
            changed.append(format_canonical_mechanism_id(kind, old_name, new_name))
    else:
        changed.extend(removed)
        changed.extend(added)
    return tuple(sorted(changed))


def _component_mechanism_signatures(
    tree: ast.Module,
) -> dict[str, str] | None:
    records = _active_component_sources(tree)
    if records is None:
        return None
    return {
        mechanism_id: record.structure_signature
        for mechanism_id, record in records.items()
    }


def project_canonical_reward_term_sources(
    source_text: str,
) -> tuple[CanonicalRewardTermSource, ...] | None:
    """Project active RewardTerms from the canonical builder without executing code.

    The projection shares the exact active-builder resolver and structure signatures
    used by anti-degenerate authoring checks.  Tasks may consume the neutral facts,
    but task-specific meaning must stay in the task plugin.
    """

    if not isinstance(source_text, str):
        raise TypeError("component source must be text")
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None
    records = _active_component_sources(tree)
    if records is None:
        return None
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    projected: list[CanonicalRewardTermSource] = []
    for mechanism_id, record in sorted(records.items()):
        if record.kind != "RewardTerm":
            continue
        arguments = record.resolved_arguments
        callable_node = arguments.get("fn")
        if not (
            isinstance(callable_node, ast.Name)
            and callable_node.id in functions
        ):
            return None
        weight = _literal_finite_number(arguments.get("weight"))
        clip_min = _literal_optional_finite_number(arguments.get("clip_min"))
        clip_max = _literal_optional_finite_number(arguments.get("clip_max"))
        if weight is None or clip_min is _INVALID_LITERAL or clip_max is _INVALID_LITERAL:
            return None
        projected.append(
            CanonicalRewardTermSource(
                mechanism_id=mechanism_id,
                structure_signature=record.structure_signature,
                callable_name=callable_node.id,
                active_callable_attribute_names=(
                    _active_callable_attribute_names(callable_node, functions)
                ),
                weight=weight,
                clip_min=clip_min,
                clip_max=clip_max,
            )
        )
    return tuple(projected)


def extract_training_components_rationale(source_text: str) -> str | None:
    """Extract the canonical experiment hypothesis without executing candidate code.

    Fresh provider candidates must express their hypothesis in the ``rationale``
    field of the canonical ``TrainingComponents`` returned by
    ``build_training_components``.  Only a statically resolvable string literal is
    accepted so the materialized rationale and authoring checks share one source of
    truth.
    """

    if not isinstance(source_text, str):
        raise TypeError("component source must be text")
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return None
    resolved = _canonical_training_components_builder(tree)
    if resolved is None:
        return None
    builder_call, builder_bindings, _ = resolved
    component_arguments = _component_call_arguments(
        builder_call,
        _TRAINING_COMPONENT_FIELDS,
    )
    raw_rationale = component_arguments.get("rationale")
    if raw_rationale is None:
        return None
    rationale_node = _resolve_builder_value(raw_rationale, builder_bindings)
    try:
        rationale = ast.literal_eval(rationale_node)
    except (ValueError, TypeError):
        return None
    if not isinstance(rationale, str) or not rationale.strip():
        return None
    return rationale


def _canonical_training_components_builder(
    tree: ast.Module,
) -> tuple[ast.Call, dict[str, ast.expr], dict[str, ast.FunctionDef]] | None:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    builder = functions.get("build_training_components")
    if builder is None:
        return None
    resolved_builder = _resolved_builder_return(builder)
    if not isinstance(resolved_builder, ast.Call) or (
        _component_call_name(resolved_builder.func) != "TrainingComponents"
    ):
        return None
    if any(isinstance(value, ast.Starred) for value in resolved_builder.args) or any(
        keyword.arg is None for keyword in resolved_builder.keywords
    ):
        return None
    builder_bindings = _builder_bindings(builder)
    if builder_bindings is None:
        return None
    return resolved_builder, builder_bindings, functions


def _active_component_sources(
    tree: ast.Module,
) -> dict[str, _ActiveComponentSource] | None:
    resolved = _canonical_training_components_builder(tree)
    if resolved is None:
        return None
    resolved_builder, builder_bindings, functions = resolved
    component_arguments = _component_call_arguments(
        resolved_builder,
        _TRAINING_COMPONENT_FIELDS,
    )
    records: dict[str, _ActiveComponentSource] = {}
    for list_field, expected_kind in _TRAINING_COMPONENT_LIST_KINDS.items():
        raw_terms = component_arguments.get(list_field)
        if raw_terms is None:
            continue
        terms = _resolve_builder_value(raw_terms, builder_bindings)
        if not isinstance(terms, ast.List | ast.Tuple):
            return None
        for raw_term in terms.elts:
            node = _resolve_builder_value(raw_term, builder_bindings)
            if not isinstance(node, ast.Call):
                return None
            kind = _component_call_name(node.func)
            if kind != expected_kind:
                return None
            fields = _COMPONENT_TERM_FIELDS[kind]
            if any(isinstance(value, ast.Starred) for value in node.args) or any(
                keyword.arg is None for keyword in node.keywords
            ):
                return None
            arguments = _component_call_arguments(node, fields)
            name_node = arguments.get("name")
            if not (
                isinstance(name_node, ast.Constant)
                and isinstance(name_node.value, str)
                and name_node.value
            ):
                return None
            try:
                key = format_canonical_mechanism_id(kind, name_node.value)
            except ValueError:
                return None
            if key in records:
                return None
            signature_parts: list[tuple[str, str]] = []
            normalized_signature_parts: list[tuple[str, str]] = []
            resolved_arguments: dict[str, ast.expr] = {}
            for field_name in fields:
                if field_name == "name":
                    continue
                value = _resolve_builder_value(
                    arguments.get(field_name),
                    builder_bindings,
                )
                if value is not None:
                    resolved_arguments[field_name] = value
                if field_name == "fn":
                    signature = _callable_structure_signature(value, functions)
                    normalized_signature = _callable_structure_signature(
                        value,
                        functions,
                        normalize_parameters=True,
                    )
                    if signature is None:
                        return None
                    if normalized_signature is None:
                        return None
                else:
                    signature = _component_field_signature(
                        kind,
                        field_name,
                        value,
                        normalize_parameters=False,
                    )
                    normalized_signature = _component_field_signature(
                        kind,
                        field_name,
                        value,
                        normalize_parameters=_component_field_is_parameter(
                            kind,
                            field_name,
                        ),
                    )
                signature_parts.append((field_name, signature))
                normalized_signature_parts.append(
                    (field_name, normalized_signature)
                )
            records[key] = _ActiveComponentSource(
                kind=kind,
                mechanism_id=key,
                structure_signature=json.dumps(
                    signature_parts,
                    separators=(",", ":"),
                ),
                parameter_normalized_structure_signature=json.dumps(
                    normalized_signature_parts,
                    separators=(",", ":"),
                ),
                resolved_arguments=resolved_arguments,
            )
    return records


def _literal_finite_number(value: ast.expr | None) -> float | None:
    if value is None:
        return None
    try:
        literal = ast.literal_eval(value)
    except (ValueError, TypeError):
        return None
    if (
        isinstance(literal, bool)
        or not isinstance(literal, int | float)
        or not math.isfinite(literal)
    ):
        return None
    return float(literal)


def _literal_optional_finite_number(
    value: ast.expr | None,
) -> float | None | object:
    if value is None:
        return None
    try:
        literal = ast.literal_eval(value)
    except (ValueError, TypeError):
        return _INVALID_LITERAL
    if literal is None:
        return None
    if (
        isinstance(literal, bool)
        or not isinstance(literal, int | float)
        or not math.isfinite(literal)
    ):
        return _INVALID_LITERAL
    return float(literal)


def _component_term_default_signature(
    kind: str,
    field_name: str,
    *,
    normalize_parameters: bool = False,
) -> str | None:
    field = _COMPONENT_TERM_CLASSES[kind].__dataclass_fields__[field_name]
    if field.default is MISSING:
        return None
    return _ast_value_signature(
        ast.Constant(value=field.default),
        normalize_parameters=normalize_parameters,
    )


def _component_field_signature(
    kind: str,
    field_name: str,
    value: ast.expr | None,
    *,
    normalize_parameters: bool,
) -> str:
    signature = (
        _component_term_default_signature(
            kind,
            field_name,
            normalize_parameters=normalize_parameters,
        )
        if value is None
        else _ast_value_signature(
            value,
            normalize_parameters=normalize_parameters,
        )
    )
    return (
        signature
        if signature is not None
        else _ast_value_signature(
            value,
            normalize_parameters=normalize_parameters,
        )
    )


def _component_field_is_parameter(kind: str, field_name: str) -> bool:
    if kind == "RewardTerm":
        return field_name in {"weight", "clip_min", "clip_max"}
    if kind == "ScheduleTerm":
        return field_name == "value"
    return False


def _builder_bindings(
    builder: ast.FunctionDef,
) -> dict[str, ast.expr] | None:
    bindings: dict[str, ast.expr] = {}
    return_count = 0
    return_seen = False
    for statement in builder.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if return_seen:
            return None
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or statement.targets[0].id in bindings
            ):
                return None
            bindings[statement.targets[0].id] = statement.value
            continue
        if isinstance(statement, ast.AnnAssign):
            if (
                not isinstance(statement.target, ast.Name)
                or statement.value is None
                or statement.target.id in bindings
            ):
                return None
            bindings[statement.target.id] = statement.value
            continue
        if isinstance(statement, ast.Return):
            return_count += 1
            return_seen = True
            continue
        return None
    return bindings if return_count == 1 else None


def _resolved_builder_return(builder: ast.FunctionDef) -> ast.expr | None:
    bindings = _builder_bindings(builder)
    if bindings is None:
        return None
    return_statement = next(
        (statement for statement in builder.body if isinstance(statement, ast.Return)),
        None,
    )
    if return_statement is None or return_statement.value is None:
        return None
    return _resolve_builder_value(return_statement.value, bindings)


def _resolve_builder_value(
    value: ast.expr,
    bindings: Mapping[str, ast.expr],
    *,
    seen: frozenset[str] = frozenset(),
) -> ast.expr:
    if not isinstance(value, ast.Name) or value.id not in bindings:
        return value
    if value.id in seen:
        return value
    return _resolve_builder_value(
        bindings[value.id],
        bindings,
        seen=seen | {value.id},
    )


def _component_call_name(value: ast.expr) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _component_call_arguments(
    node: ast.Call,
    fields: Sequence[str],
) -> dict[str, ast.expr]:
    values = {
        field_name: value
        for field_name, value in zip(fields, node.args, strict=False)
    }
    for keyword in node.keywords:
        if keyword.arg in fields:
            values[keyword.arg] = keyword.value
    return values


def _callable_structure_signature(
    value: ast.expr | None,
    functions: Mapping[str, ast.FunctionDef],
    *,
    normalize_parameters: bool = False,
) -> str | None:
    if isinstance(value, ast.Name) and value.id in functions:
        return _function_structure_signature(
            value.id,
            functions=functions,
            dependency_path=(),
            root=True,
            normalize_parameters=normalize_parameters,
        )
    if isinstance(value, ast.Lambda):
        return None
    return _ast_value_signature(value)


def _function_structure_signature(
    function_name: str,
    *,
    functions: Mapping[str, ast.FunctionDef],
    dependency_path: tuple[str, ...],
    root: bool,
    normalize_parameters: bool = False,
) -> str | None:
    if function_name in dependency_path:
        return "<recursive>"
    parsed = ast.parse(ast.unparse(functions[function_name]))
    function = parsed.body[0]
    if not isinstance(function, ast.FunctionDef):
        return "<unavailable>"
    original_function_name = function.name
    if root and not _component_callable_signature_is_canonical(function):
        return None
    if not root and not _helper_arguments_are_used(function):
        return None
    if any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda)
        for statement in function.body
        for node in ast.walk(statement)
    ):
        return None
    if not _helper_references_are_direct_calls(
        function,
        functions=functions,
    ):
        return None
    function.name = "__component_callable__" if root else "__local_helper__"
    function.decorator_list = []
    function = _strip_nonruntime_callable_syntax(function)
    function = _alpha_normalize_function_bindings(function)
    dependencies = sorted(
        {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
            and node.func.id != function_name
        }
    )
    next_path = (*dependency_path, function_name)
    dependency_signatures = {
        dependency: _function_structure_signature(
            dependency,
            functions=functions,
            dependency_path=next_path,
            root=False,
            normalize_parameters=normalize_parameters,
        )
        for dependency in dependencies
    }
    if any(signature is None for signature in dependency_signatures.values()):
        return None
    dependency_tokens = {
        dependency: "__local_helper_"
        + hashlib.sha256(signature.encode("utf-8")).hexdigest()
        for dependency, signature in dependency_signatures.items()
        if signature is not None
    }
    dependency_keyword_maps = {
        dependency: _function_argument_name_mapping(functions[dependency])
        for dependency in dependencies
    }
    self_keyword_map = _function_argument_name_mapping(functions[function_name])

    class RenameDependencies(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
            updated = self.generic_visit(node)
            if (
                isinstance(updated, ast.Call)
                and isinstance(updated.func, ast.Name)
            ):
                callable_name = updated.func.id
                if callable_name == original_function_name:
                    for keyword in updated.keywords:
                        if keyword.arg in self_keyword_map:
                            keyword.arg = self_keyword_map[keyword.arg]
                    updated.func.id = "__component_self__"
                    return updated
                if callable_name not in dependency_tokens:
                    return updated
                keyword_map = dependency_keyword_maps[callable_name]
                for keyword in updated.keywords:
                    if keyword.arg in keyword_map:
                        keyword.arg = keyword_map[keyword.arg]
                updated.func.id = dependency_tokens[callable_name]
            return updated

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
            return node

        def visit_Lambda(self, node: ast.Lambda) -> ast.AST:  # noqa: N802
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
            return node

    for index, statement in enumerate(function.body):
        function.body[index] = RenameDependencies().visit(statement)
    ast.fix_missing_locations(function)
    if normalize_parameters:
        normalized = _normalize_parameter_like_numeric_constants(function)
        if not (
            isinstance(normalized, ast.Module)
            and len(normalized.body) == 1
            and isinstance(normalized.body[0], ast.FunctionDef)
        ):
            return None
        function = normalized.body[0]
    return json.dumps(
        {
            "function": ast.dump(function, include_attributes=False),
            "local_helpers": sorted(
                {
                    signature
                    for signature in dependency_signatures.values()
                    if signature is not None
                }
            ),
        },
        separators=(",", ":"),
    )


def _alpha_normalize_function_bindings(
    function: ast.FunctionDef,
) -> ast.FunctionDef:
    """Normalize argument/local identifiers without rewriting global dependencies."""

    mapping = _function_argument_name_mapping(function)
    arguments = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    if function.args.vararg is not None:
        arguments.append(function.args.vararg)
    if function.args.kwarg is not None:
        arguments.append(function.args.kwarg)
    excluded: set[str] = set()
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Global | ast.Nonlocal):
                excluded.update(node.names)

    local_names: list[str] = []

    class CollectLocals(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
            if isinstance(node.ctx, ast.Store) and node.id not in excluded:
                if node.id not in local_names:
                    local_names.append(node.id)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
            if isinstance(node.name, str) and node.name not in excluded:
                if node.name not in local_names:
                    local_names.append(node.name)
            for statement in node.body:
                self.visit(statement)

        def visit_MatchAs(self, node: ast.MatchAs) -> None:  # noqa: N802
            if isinstance(node.name, str) and node.name not in excluded:
                if node.name not in local_names:
                    local_names.append(node.name)
            if node.pattern is not None:
                self.visit(node.pattern)

        def visit_MatchStar(self, node: ast.MatchStar) -> None:  # noqa: N802
            if isinstance(node.name, str) and node.name not in excluded:
                if node.name not in local_names:
                    local_names.append(node.name)

        def visit_MatchMapping(self, node: ast.MatchMapping) -> None:  # noqa: N802
            if isinstance(node.rest, str) and node.rest not in excluded:
                if node.rest not in local_names:
                    local_names.append(node.rest)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            return

        def visit_AsyncFunctionDef(
            self,
            node: ast.AsyncFunctionDef,
        ) -> None:  # noqa: N802
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            return

    collector = CollectLocals()
    for statement in function.body:
        collector.visit(statement)
    for name in local_names:
        if name not in mapping:
            mapping[name] = f"__local_{len(mapping)}__"

    class RenameBindings(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
            renamed = mapping.get(node.id)
            if renamed is None:
                return node
            return ast.copy_location(ast.Name(id=renamed, ctx=node.ctx), node)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:  # noqa: N802
            if isinstance(node.name, str) and node.name in mapping:
                node.name = mapping[node.name]
            return self.generic_visit(node)

        def visit_MatchAs(self, node: ast.MatchAs) -> ast.AST:  # noqa: N802
            if isinstance(node.name, str) and node.name in mapping:
                node.name = mapping[node.name]
            return self.generic_visit(node)

        def visit_MatchStar(self, node: ast.MatchStar) -> ast.AST:  # noqa: N802
            if isinstance(node.name, str) and node.name in mapping:
                node.name = mapping[node.name]
            return self.generic_visit(node)

        def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.AST:  # noqa: N802
            if isinstance(node.rest, str) and node.rest in mapping:
                node.rest = mapping[node.rest]
            return self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
            return node

        def visit_AsyncFunctionDef(
            self,
            node: ast.AsyncFunctionDef,
        ) -> ast.AST:  # noqa: N802
            return node

        def visit_Lambda(self, node: ast.Lambda) -> ast.AST:  # noqa: N802
            return node

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
            return node

    for argument in arguments:
        argument.arg = mapping[argument.arg]
    transformer = RenameBindings()
    for index, statement in enumerate(function.body):
        function.body[index] = transformer.visit(statement)
    return ast.fix_missing_locations(function)


def _component_callable_signature_is_canonical(
    function: ast.FunctionDef,
) -> bool:
    return (
        not function.args.posonlyargs
        and len(function.args.args) == 1
        and not function.args.defaults
        and not function.args.kwonlyargs
        and function.args.vararg is None
        and function.args.kwarg is None
    )


def _helper_arguments_are_used(function: ast.FunctionDef) -> bool:
    argument_names = set(_function_argument_name_mapping(function))
    if not argument_names:
        return True
    loaded_names = {
        node.id
        for statement in function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return argument_names <= loaded_names


def _helper_references_are_direct_calls(
    function: ast.FunctionDef,
    *,
    functions: Mapping[str, ast.FunctionDef],
) -> bool:
    function_names = set(functions)
    parent_by_id: dict[int, ast.AST] = {}
    for parent in ast.walk(function):
        for child in ast.iter_child_nodes(parent):
            parent_by_id[id(child)] = parent
    for node in ast.walk(function):
        if not isinstance(node, ast.Name) or node.id not in function_names:
            continue
        parent = parent_by_id.get(id(node))
        if not (
            isinstance(parent, ast.Call)
            and parent.func is node
        ):
            return False
    return True


def _strip_nonruntime_callable_syntax(
    function: ast.FunctionDef,
) -> ast.FunctionDef:
    """Remove syntax that cannot change a component callable's runtime value."""

    function.returns = None
    function.type_comment = None
    arguments = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    if function.args.vararg is not None:
        arguments.append(function.args.vararg)
    if function.args.kwarg is not None:
        arguments.append(function.args.kwarg)
    for argument in arguments:
        argument.annotation = None
        argument.type_comment = None

    class StripNonruntime(ast.NodeTransformer):
        def _visit_statements(
            self,
            statements: Sequence[ast.stmt],
        ) -> list[ast.stmt]:
            normalized: list[ast.stmt] = []
            for statement in statements:
                updated = self.visit(statement)
                if isinstance(updated, ast.stmt):
                    normalized.append(updated)
                elif isinstance(updated, list):
                    normalized.extend(
                        item for item in updated if isinstance(item, ast.stmt)
                    )
                if normalized and isinstance(
                    normalized[-1],
                    ast.Return | ast.Raise | ast.Break | ast.Continue,
                ):
                    break
            return normalized

        def visit_Pass(self, node: ast.Pass) -> None:  # noqa: N802
            return None

        def visit_Expr(self, node: ast.Expr) -> ast.AST | None:  # noqa: N802
            if isinstance(node.value, ast.Constant):
                return None
            return self.generic_visit(node)

        def visit_AnnAssign(
            self,
            node: ast.AnnAssign,
        ) -> ast.AST | None:  # noqa: N802
            if node.value is None:
                return None
            return ast.copy_location(
                ast.Assign(targets=[node.target], value=self.visit(node.value)),
                node,
            )

        def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt] | None:  # noqa: N802
            literal_truth = _literal_truth_value(node.test)
            if literal_truth is not None:
                selected = node.body if literal_truth else node.orelse
                return self._visit_statements(selected)
            return self.generic_visit(node)

        def visit_IfExp(self, node: ast.IfExp) -> ast.AST:  # noqa: N802
            literal_truth = _literal_truth_value(node.test)
            if literal_truth is not None:
                return self.visit(node.body if literal_truth else node.orelse)
            return self.generic_visit(node)

        def visit_While(
            self,
            node: ast.While,
        ) -> ast.AST | list[ast.stmt] | None:  # noqa: N802
            literal_truth = _literal_truth_value(node.test)
            if literal_truth is False:
                return self._visit_statements(node.orelse)
            return self.generic_visit(node)

        def visit_Assign(self, node: ast.Assign) -> ast.AST:  # noqa: N802
            node.type_comment = None
            return self.generic_visit(node)

        def visit_For(self, node: ast.For) -> ast.AST:  # noqa: N802
            node.type_comment = None
            return self.generic_visit(node)

        def visit_With(self, node: ast.With) -> ast.AST:  # noqa: N802
            node.type_comment = None
            return self.generic_visit(node)

    transformer = StripNonruntime()
    normalized_body = transformer._visit_statements(function.body)
    function.body = normalized_body or [ast.Pass()]
    return ast.fix_missing_locations(function)


def _literal_truth_value(value: ast.expr) -> bool | None:
    try:
        literal = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return None
    return bool(literal)


def _function_argument_name_mapping(
    function: ast.FunctionDef,
) -> dict[str, str]:
    arguments = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    if function.args.vararg is not None:
        arguments.append(function.args.vararg)
    if function.args.kwarg is not None:
        arguments.append(function.args.kwarg)
    return {
        argument.arg: f"__arg_{index}__"
        for index, argument in enumerate(arguments)
    }


def _ast_value_signature(
    value: ast.expr | None,
    *,
    normalize_parameters: bool = False,
) -> str:
    if value is None:
        return "<omitted>"
    projected: ast.AST = value
    if normalize_parameters:
        projected = _normalize_parameter_like_numeric_constants(value)
    return ast.dump(projected, include_attributes=False)


def _normalize_parameter_like_numeric_constants(tree: ast.AST) -> ast.AST:
    """Normalize magnitude literals while retaining index/API configuration."""
    parsed = ast.parse(ast.unparse(tree))

    class Visitor(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:  # noqa: N802
            # Slice/index literals select tensor semantics, not a tunable magnitude.
            node.value = self.visit(node.value)
            return node

        def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
            call_name = _component_call_name(node.func)
            if call_name in _COMPONENT_TERM_FIELDS:
                return self.generic_visit(node)
            node.func = self.visit(node.func)
            node.args = [
                argument
                if (
                    call_name in _POSITIONAL_DIMENSION_CALLS
                    and _is_literal_call_configuration(argument)
                )
                else self.visit(argument)
                for argument in node.args
            ]
            for keyword in node.keywords:
                if keyword.arg in _STRUCTURAL_CALL_KEYWORDS:
                    continue
                keyword.value = self.visit(keyword.value)
            return node

        def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
            if isinstance(node.value, int | float) and not isinstance(node.value, bool):
                return ast.copy_location(ast.Constant(value=0), node)
            return node

    return ast.fix_missing_locations(Visitor().visit(parsed))


def _is_literal_call_configuration(value: ast.expr) -> bool:
    try:
        ast.literal_eval(value)
    except (ValueError, TypeError):
        return False
    return True


def _normalize_string_constants(tree: ast.AST) -> ast.AST:
    parsed = ast.parse(ast.unparse(tree))

    class Visitor(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

    return ast.fix_missing_locations(Visitor().visit(parsed))


def _normalize_non_mechanism_constants(tree: ast.AST) -> ast.AST:
    return _normalize_string_constants(
        _normalize_parameter_like_numeric_constants(tree)
    )


def _assess_against_evaluated_history(
    *,
    candidate_source: str,
    candidate_anchor_source: str | None,
    evaluated_history_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    checked_count = 0
    candidate_delta = (
        _try_single_component_delta_semantics(
            candidate_source=candidate_source,
            anchor_source=candidate_anchor_source,
        )
        if candidate_anchor_source is not None
        else None
    )
    for historical in reversed(tuple(evaluated_history_sources)):
        source = historical.get("content")
        if not isinstance(source, str):
            continue
        checked_count += 1
        content_sha256 = historical.get("content_sha256")
        historical_anchor = historical.get("anchor_content")
        historical_delta = (
            _try_single_component_delta_semantics(
                candidate_source=source,
                anchor_source=historical_anchor,
            )
            if isinstance(historical_anchor, str)
            else None
        )
        delta_relative_comparison = (
            candidate_delta is not None and historical_delta is not None
        )
        if delta_relative_comparison:
            equivalence = _component_delta_equivalence(
                candidate_delta,
                historical_delta,
            )
            assessment_candidate_sha256 = hashlib.sha256(
                candidate_source.encode("utf-8")
            ).hexdigest()
            compares_to = (
                content_sha256 if isinstance(content_sha256, str) else None
            )
        else:
            assessment = assess_anti_degenerate_source(
                candidate_source=candidate_source,
                current_best_source=source,
                current_best_sha256=(
                    content_sha256 if isinstance(content_sha256, str) else None
                ),
            )
            equivalence = assessment.get("classification")
            assessment_candidate_sha256 = assessment.get(
                "candidate_source_sha256"
            )
            compares_to = assessment.get("compares_to")
        if equivalence not in _NON_MECHANISM_DELTA_CLASSIFICATIONS:
            continue
        result = {
            "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
            "status": "fail",
            "classification": "historical_mechanism_repeat",
            "reason": "candidate_repeats_previously_evaluated_mechanism",
            "equivalence_classification": equivalence,
            "comparison_scope": (
                "single_component_deltas_against_own_anchors"
                if delta_relative_comparison
                else "evaluated_history_legacy_full_source"
            ),
            "compares_to": compares_to,
            "candidate_source_sha256": assessment_candidate_sha256,
            "matched_round_id": historical.get("round_id"),
            "matched_candidate_id": historical.get("candidate_id"),
            "evaluated_history_source_count": len(evaluated_history_sources),
            "checked_history_source_count": checked_count,
            "blocking": True,
            "feedback_for_author": (
                "This mechanism was already evaluated. Use its recorded outcome to "
                "introduce a structurally different code-level mechanism."
            ),
        }
        if delta_relative_comparison:
            result.update(
                {
                    "candidate_changed_mechanism": candidate_delta.mechanism_id,
                    "matched_changed_mechanism": historical_delta.mechanism_id,
                    "component_kind": candidate_delta.kind,
                    "shared_input_fields": list(candidate_delta.input_fields),
                }
            )
        return _public_result(result)
    return None


def _assess_rejected_input_dependency_variants(
    *,
    candidate_source: str,
    anchor_source: str | None,
    anchor_hash: str | None,
    evaluated_history_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Record nonblocking evidence for a rejected same-input mechanism variant."""

    if anchor_source is None:
        return None
    candidate_delta = _try_single_component_delta_semantics(
        candidate_source=candidate_source,
        anchor_source=anchor_source,
    )
    if candidate_delta is None or not candidate_delta.input_fields:
        return None
    candidate_kind = candidate_delta.kind
    candidate_fields = candidate_delta.input_fields
    canonical_hypothesis = extract_training_components_rationale(candidate_source)
    rationale = (canonical_hypothesis or "").casefold()
    for historical in reversed(tuple(evaluated_history_sources)):
        if historical.get("accepted") is not False:
            continue
        historical_source = historical.get("content")
        historical_anchor = historical.get("anchor_content")
        if not isinstance(historical_anchor, str) and (
            _bare_sha256(historical.get("parent_program_source_checksum"))
            == _bare_sha256(
                anchor_hash
                or hashlib.sha256(anchor_source.encode("utf-8")).hexdigest()
            )
        ):
            historical_anchor = anchor_source
        if not (
            isinstance(historical_source, str)
            and isinstance(historical_anchor, str)
        ):
            continue
        historical_delta = _try_single_component_delta_semantics(
            candidate_source=historical_source,
            anchor_source=historical_anchor,
        )
        if (
            historical_delta is None
            or historical_delta.kind != candidate_kind
            or historical_delta.input_fields != candidate_fields
        ):
            continue
        historical_candidate_id = historical.get("candidate_id")
        if not isinstance(historical_candidate_id, str) or not historical_candidate_id:
            continue
        canonical_hypothesis_available = canonical_hypothesis is not None
        prior_id_mentioned = historical_candidate_id.casefold() in rationale
        return _public_result(
            {
                "status": "pass",
                "classification": "rejected_input_dependency_variant",
                "comparison_scope": "rejected_history_relative_to_own_anchor",
                "compares_to": historical.get("content_sha256"),
                "candidate_source_sha256": hashlib.sha256(
                    candidate_source.encode("utf-8")
                ).hexdigest(),
                "matched_round_id": historical.get("round_id"),
                "matched_candidate_id": historical_candidate_id,
                "component_kind": candidate_kind,
                "shared_input_fields": list(candidate_fields),
                "evaluated_history_source_count": len(evaluated_history_sources),
                "hypothesis_source": (
                    "training_code/components.py:TrainingComponents.rationale"
                    if canonical_hypothesis_available
                    else "unavailable_legacy_candidate"
                ),
                "canonical_hypothesis_available": canonical_hypothesis_available,
                "literal_candidate_id_mentioned_in_rationale": prior_id_mentioned,
                "blocking": False,
            }
        )
    return None


def _bare_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.removeprefix("sha256:").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        return None
    return normalized


def compare_single_component_deltas(
    *,
    candidate_source: str,
    candidate_anchor_source: str,
    comparison_source: str,
    comparison_anchor_source: str,
) -> str | None:
    """Compare two single-component deltas, each relative to its own anchor.

    Component names are not part of the comparison. Numeric parameters are
    normalized while callable structure, input dependencies, and parameter
    presence remain authoritative. Future sibling-lane exclusion must use this
    primitive instead of defining a second mechanism identity rule.
    """

    candidate = _try_single_component_delta_semantics(
        candidate_source=candidate_source,
        anchor_source=candidate_anchor_source,
    )
    comparison = _try_single_component_delta_semantics(
        candidate_source=comparison_source,
        anchor_source=comparison_anchor_source,
    )
    if candidate is None or comparison is None:
        raise SingleComponentDeltaComparisonUnavailable(
            "single-component delta semantics are unavailable"
        )
    return _component_delta_equivalence(candidate, comparison)


def project_single_component_operation_delta(
    *,
    candidate_source: str,
    anchor_source: str,
) -> dict[str, Any]:
    """Project one operation-aware delta from the canonical mechanism analyzer."""

    try:
        candidate_tree = ast.parse(candidate_source)
        anchor_tree = ast.parse(anchor_source)
    except SyntaxError as exc:
        raise SingleComponentDeltaComparisonUnavailable(
            "single-component delta source cannot be parsed"
        ) from exc
    changed = _changed_component_mechanisms(
        candidate_tree=candidate_tree,
        anchor_tree=anchor_tree,
    )
    if changed is None or len(changed) != 1:
        raise SingleComponentDeltaComparisonUnavailable(
            "operation projection requires exactly one canonical component delta"
        )
    candidate_records = _active_component_sources(candidate_tree)
    anchor_records = _active_component_sources(anchor_tree)
    if candidate_records is None or anchor_records is None:
        raise SingleComponentDeltaComparisonUnavailable(
            "canonical component records are unavailable"
        )
    identity = parse_canonical_mechanism_id(changed[0])
    if identity is None:
        raise SingleComponentDeltaComparisonUnavailable(
            "canonical component delta identity is invalid"
        )
    kind, old_name, renamed_to = identity
    before_id = format_canonical_mechanism_id(kind, old_name)
    after_id = format_canonical_mechanism_id(kind, renamed_to or old_name)
    before_record = anchor_records.get(before_id)
    after_record = candidate_records.get(after_id)
    if renamed_to is not None:
        operation = "rename"
        if before_record is None or after_record is None:
            raise SingleComponentDeltaComparisonUnavailable(
                "renamed component records are unavailable"
            )
    elif before_record is None and after_record is not None:
        operation = "add"
    elif before_record is not None and after_record is None:
        operation = "remove"
    elif before_record is not None and after_record is not None:
        operation = "modify"
    else:
        raise SingleComponentDeltaComparisonUnavailable(
            "component operation cannot be resolved"
        )
    projected = {
        "schema_version": CANDIDATE_MECHANISM_OPERATION_DELTA_SCHEMA_VERSION,
        "operation": operation,
        "before_component": _operation_component_projection(
            mechanism_id=before_id,
            record=before_record,
            tree=anchor_tree,
        ),
        "after_component": _operation_component_projection(
            mechanism_id=after_id,
            record=after_record,
            tree=candidate_tree,
        ),
        "anchor_source_checksum": "sha256:"
        + hashlib.sha256(anchor_source.encode("utf-8")).hexdigest(),
        "candidate_source_checksum": "sha256:"
        + hashlib.sha256(candidate_source.encode("utf-8")).hexdigest(),
    }
    return require_mechanism_operation_delta(projected)


def _operation_component_projection(
    *,
    mechanism_id: str,
    record: _ActiveComponentSource | None,
    tree: ast.Module,
) -> dict[str, Any] | None:
    if record is None:
        return None
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    return {
        "mechanism_id": mechanism_id,
        "structure_hash": hashlib.sha256(
            record.structure_signature.encode("utf-8")
        ).hexdigest(),
        "parameter_normalized_structure_hash": hashlib.sha256(
            record.parameter_normalized_structure_signature.encode("utf-8")
        ).hexdigest(),
        "input_fields": list(
            _callable_input_fields(
                record.resolved_arguments.get("fn"),
                functions,
            )
        ),
    }


def _component_delta_equivalence(
    candidate: _SingleComponentDeltaSemantics,
    comparison: _SingleComponentDeltaSemantics,
) -> str | None:
    if (
        candidate.kind != comparison.kind
        or candidate.presence != comparison.presence
        or candidate.input_fields != comparison.input_fields
    ):
        return None
    if candidate.structure_signature == comparison.structure_signature:
        return "identical_or_format_only"
    if (
        candidate.parameter_normalized_structure_signature
        == comparison.parameter_normalized_structure_signature
    ):
        return "parameter_only_delta"
    return None


def _try_single_component_delta_semantics(
    *,
    candidate_source: str,
    anchor_source: str,
) -> _SingleComponentDeltaSemantics | None:
    try:
        candidate_tree = ast.parse(candidate_source)
        anchor_tree = ast.parse(anchor_source)
    except SyntaxError:
        return None
    changed = _changed_component_mechanisms(
        candidate_tree=candidate_tree,
        anchor_tree=anchor_tree,
    )
    if changed is None or len(changed) != 1:
        return None
    candidate_records = _active_component_sources(candidate_tree)
    anchor_records = _active_component_sources(anchor_tree)
    if candidate_records is None or anchor_records is None:
        return None
    mechanism_id = changed[0]
    identity = parse_canonical_mechanism_id(mechanism_id)
    if identity is None:
        return None
    kind, old_name, renamed_to = identity
    after_id = format_canonical_mechanism_id(kind, renamed_to or old_name)
    before_id = format_canonical_mechanism_id(kind, old_name)
    if after_id in candidate_records:
        record = candidate_records[after_id]
        source_tree = candidate_tree
        presence = "present_after"
    elif before_id in anchor_records:
        record = anchor_records[before_id]
        source_tree = anchor_tree
        presence = "removed"
    else:
        return None
    functions = {
        node.name: node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    return _SingleComponentDeltaSemantics(
        kind=kind,
        presence=presence,
        mechanism_id=mechanism_id,
        structure_signature=record.structure_signature,
        parameter_normalized_structure_signature=(
            record.parameter_normalized_structure_signature
        ),
        input_fields=_callable_input_fields(
            record.resolved_arguments.get("fn"),
            functions,
        ),
    )


def _active_callable_attribute_names(
    value: ast.expr | None,
    functions: Mapping[str, ast.FunctionDef],
) -> tuple[str, ...]:
    if not isinstance(value, ast.Name) or value.id not in functions:
        return ()
    attributes: set[str] = set()
    pending = [value.id]
    visited: set[str] = set()
    while pending:
        function_name = pending.pop()
        if function_name in visited:
            continue
        visited.add(function_name)
        function = functions[function_name]
        attributes.update(
            node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)
        )
        pending.extend(
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
            and node.func.id not in visited
        )
    return tuple(sorted(attributes))


def _callable_input_fields(
    value: ast.expr | None,
    functions: Mapping[str, ast.FunctionDef],
) -> tuple[str, ...]:
    """Retain the narrow legacy input-dependency projection used for history hints."""

    if not isinstance(value, ast.Name) or value.id not in functions:
        return ()
    return tuple(
        sorted(
            _function_input_fields(
                value.id,
                functions=functions,
                dependency_path=(),
            )
        )
    )


def _function_input_fields(
    function_name: str,
    *,
    functions: Mapping[str, ast.FunctionDef],
    dependency_path: tuple[str, ...],
) -> set[str]:
    if function_name in dependency_path:
        return set()
    function = functions[function_name]
    positional = [*function.args.posonlyargs, *function.args.args]
    if not positional:
        return set()
    input_name = positional[0].arg
    fields = {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == input_name
    }
    next_path = (*dependency_path, function_name)
    for node in ast.walk(function):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in functions
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == input_name
        ):
            continue
        fields.update(
            _function_input_fields(
                node.func.id,
                functions=functions,
                dependency_path=next_path,
            )
        )
    return fields


def _anti_degenerate_result(
    *,
    classification: str,
    status: str,
    reason: str,
    compares_to: str,
    candidate_digest: str,
    blocking: bool,
    feedback: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
        "status": status,
        "classification": classification,
        "reason": reason,
        "compares_to": compares_to,
        "candidate_source_sha256": candidate_digest,
        "blocking": blocking,
        "feedback_for_author": feedback,
    }
    if details is not None:
        result.update(details)
    return _public_result(result)


def _compound_mechanism_result(
    *,
    compares_to: str,
    candidate_digest: str,
    changed_mechanisms: Sequence[str],
) -> dict[str, Any]:
    return _anti_degenerate_result(
        classification="compound_mechanism_delta",
        status="fail",
        reason="candidate_changes_multiple_component_mechanisms",
        compares_to=compares_to,
        candidate_digest=candidate_digest,
        blocking=True,
        feedback=SINGLE_COMPONENT_DELTA_INSTRUCTION,
        details={
            "changed_mechanism_count": len(changed_mechanisms),
            "changed_mechanisms": list(changed_mechanisms),
            "structural_comparison": "canonical_component_terms",
        },
    )
