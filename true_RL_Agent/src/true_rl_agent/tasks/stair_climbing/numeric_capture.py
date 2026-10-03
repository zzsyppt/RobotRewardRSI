"""Task-owned, lossless first-episode Stair evidence and offline replay.

The archive contains decoder inputs, not a second set of outcome rules. Padding
after an episode closes is explicitly invalid and carries no observation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.runner.numeric_evidence import (
    DEFAULT_MAX_ARCHIVE_BYTES, NumericEvidenceError, NumericEvidenceWriter,
    iter_numeric_evidence, load_numeric_evidence_manifest,
)
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeTracker
from true_rl_agent.tasks.stair_climbing.legacy_transition_reader import STAIR_CONTACT_HISTORY_TENSOR_FIELDS
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import StairEvaluationSemantics
from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
from true_rl_agent.tasks.stair_climbing.transition_decoder import decode_stair_transition_batch
from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_stair_summary
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile

STAIR_NUMERIC_PROFILE_V1 = "stair.unified_first_episode.v1"
STAIR_NUMERIC_PROFILE = "stair.unified_first_episode.v2"
_DYNAMIC_FIELDS = (
    "base_pos_w", "base_quat_w", "joint_pos", "support_pos_w",
    "support_lin_vel_w", "support_contact_forces_w", "forbidden_contact_forces_w",
    "episode_length", "done_mask",
)
_STATIC_FIELDS = ("terrain_env_origins_w", "terrain_types")
_HEADER_FIELDS = (
    "schema_version", "source_schema_version", "runtime_projection_hash",
    "reward_input_schema_hash_vnext", "support_count", "forbidden_contact_count",
)


class StairNumericCapture:
    def __init__(self, root: Path, *, identity: Mapping[str, Any], config: Any,
                 resolved_cell_manifest: ResolvedCellManifestV1,
                 max_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
                 measurement_contract: Mapping[str, Any] | None = None) -> None:
        if not config.is_unified_protocol:
            raise ValueError("Stair numeric capture requires a unified evaluation protocol")
        self.writer = NumericEvidenceWriter(root, identity, max_bytes)
        self.context = {
            "profile": STAIR_NUMERIC_PROFILE,
            "resolved_cell_manifest": resolved_cell_manifest.to_dict(),
            "training_profile": config.unified_protocol.training_profile.to_dict(),
            "evaluation_suite": config.ordered_evaluation_suite.to_dict(),
            "semantics": config.evaluation_semantics.to_dict(),
            "runtime_projection": stair_runtime_projection_from_manifest(resolved_cell_manifest).to_summary_dict(),
            "coordinates": {"positions": "world_m", "forces": "world_N",
                            "velocities": "world_m_per_s", "base_quat_w": "wxyz",
                            "support_position": "bound_rigid_body_reference_point"},
            "coverage": "every_control_step_of_each_scored_first_episode_including_pre_reset_terminal",
            "invalid_padding": "valid=false; zero-filled numeric values are not observations",
            "media_status": "not_collected",
        }
        # R1a exposes the contract interface only. No runtime defaults invent q
        # coordinates, effective limits, imported meshes or collision evidence.
        if measurement_contract is not None:
            from copy import deepcopy
            from .trajectory_capabilities import CAPTURE_PROFILE, validate_measurement_contract
            contract = deepcopy(dict(measurement_contract))
            validate_measurement_contract(contract, stair_runtime_projection_from_manifest(resolved_cell_manifest))
            self.context.update(profile=CAPTURE_PROFILE, measurement_contract=contract)
        self._seeds: list[dict[str, Any]] = []
        self._step = 0
        self._run_index = 0
        self._static: dict[str, np.ndarray] = {}
        self._history_contract: dict[str, Any] | None = None

    def begin_seed(self, *, run_index: int, runner_seed: int, split: str, step_dt: float) -> None:
        if split != "public_dev":
            raise ValueError("Stair numeric capture currently exposes only public_dev")
        self._run_index, self._step = run_index, 0
        self._static = {}
        self._seeds.append({"run_index": run_index, "runner_seed": runner_seed,
                            "split": split, "step_dt": step_dt})

    def record_chunk(self, raw_steps: Sequence[Mapping[str, Any]], wrapper_done: Sequence[Any],
                     valid_steps: Sequence[Sequence[bool]]) -> None:
        if not raw_steps:
            return
        first = raw_steps[0]
        history_contract = first.get("contact_force_history")
        if not isinstance(history_contract, Mapping):
            raise NumericEvidenceError("Stair numeric v2 requires recorded contact force history")
        if self._history_contract is None:
            self._history_contract = dict(history_contract)
            self.context["contact_force_history"] = {"status": "recorded", **history_contract,
                "array_axes": ["control_step", "environment", "history", "bound_body", "xyz"]}
        if any(value.get("contact_force_history") != self._history_contract for value in raw_steps):
            raise NumericEvidenceError("Stair contact force history clock changed")
        valid = np.asarray(valid_steps, dtype=np.bool_)
        arrays = {name: np.stack([value[name].detach().cpu().numpy() for value in raw_steps])
                  for name in (*_DYNAMIC_FIELDS, *STAIR_CONTACT_HISTORY_TENSOR_FIELDS)}
        for name, count in zip(STAIR_CONTACT_HISTORY_TENSOR_FIELDS,
                               (first["support_count"], first["forbidden_contact_count"]), strict=True):
            if arrays[name].shape != (*valid.shape, history_contract["history_length"], count, 3):
                raise NumericEvidenceError("Stair contact force history dimensions mismatch")
        arrays["wrapper_done_mask"] = np.stack([value.detach().cpu().numpy() for value in wrapper_done])
        term_order = tuple(first["termination_term_order"])
        for index, name in enumerate(term_order):
            arrays[f"termination_{index}"] = np.stack([value["termination_masks"][name].detach().cpu().numpy() for value in raw_steps])
        # Discard reset episodes, while preserving the last real terminal frame.
        for value in arrays.values():
            value[~valid] = 0
        arrays["episode_length"][~valid] = 1
        arrays["base_quat_w"][~valid, 0] = 1
        arrays["valid"] = valid
        static_present = not self._static
        if static_present:
            self._static = {name: first[name].detach().cpu().numpy().copy() for name in _STATIC_FIELDS}
            arrays.update(self._static)
        self.writer.write_chunk(arrays, {
            "run_index": self._run_index, "step_start": self._step + 1,
            "steps": len(raw_steps), "static_present": static_present,
            "transition_header": {**{name: first[name] for name in _HEADER_FIELDS},
                                  "contact_force_history": dict(history_contract)},
            "termination_term_order": list(term_order),
        })
        self._step += len(raw_steps)

    def finish_seed(self, outcomes: Sequence[Any]) -> None:
        self._seeds[-1].update(
            observed_control_steps=self._step,
            episode_steps=[outcome.episode_steps for outcome in outcomes],
            outcomes=[outcome.to_public_dict() for outcome in outcomes],
        )

    def complete(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        if any("outcomes" not in seed for seed in self._seeds):
            raise NumericEvidenceError("Stair numeric seed did not close")
        return self.writer.complete({**self.context, "seeds": self._seeds, "summary": dict(summary)})


def recompute_stair_numeric_evidence(reference: Mapping[str, Any], *, observe_transition=None) -> dict[str, Any]:
    """Recompute outcomes and existing aggregates using only persisted inputs."""
    import torch

    manifest = load_numeric_evidence_manifest(reference)
    context = manifest["metadata"]
    profile = context.get("profile")
    from .trajectory_capabilities import CAPTURE_PROFILE
    if profile not in {STAIR_NUMERIC_PROFILE_V1, STAIR_NUMERIC_PROFILE, CAPTURE_PROFILE}:
        raise NumericEvidenceError("Stair numeric profile mismatch")
    has_history = profile != STAIR_NUMERIC_PROFILE_V1
    history = context.get("contact_force_history") if has_history else {"status": "not_recorded"}
    if has_history and (not isinstance(history, Mapping) or history.get("status") != "recorded"):
        raise NumericEvidenceError("Stair numeric v2 contact history declaration is missing")
    resolved = ResolvedCellManifestV1.from_mapping(context["resolved_cell_manifest"])
    projection = stair_runtime_projection_from_manifest(resolved)
    if profile == CAPTURE_PROFILE:
        from .trajectory_capabilities import validate_measurement_contract
        validate_measurement_contract(context.get("measurement_contract", {}), projection)
    training = StairUnifiedTrainingProfile.from_mapping(context["training_profile"])
    suite = StairUnifiedEvaluationSuite.from_mapping(context["evaluation_suite"], training_profile=training)
    raw_semantics = context["semantics"]
    semantics = StairEvaluationSemantics.from_task_outcomes(
        task_protocol_hash=raw_semantics["task_protocol_hash"], task_id=raw_semantics["task_id"],
        episode_outcomes=raw_semantics["episode_outcomes"],
    )
    if semantics.to_dict() != raw_semantics:
        raise NumericEvidenceError("Stair archived semantics mismatch")
    seeds = context["seeds"]
    per_seed, trackers, outcomes, active = [], [], [], []
    current_run, observed_steps = 0, 0
    static: dict[str, Any] = {}
    cell_assignment = suite.private_cell_assignment("public_dev")
    assignment = tuple(cell.bin_id for cell in cell_assignment)
    for metadata, arrays in iter_numeric_evidence(reference):
        if has_history:
            if metadata["transition_header"].get("contact_force_history") != {
                key: value for key, value in history.items() if key not in {"status", "array_axes"}
            }:
                raise NumericEvidenceError("Stair numeric contact history declaration mismatch")
            for name, count in zip(STAIR_CONTACT_HISTORY_TENSOR_FIELDS,
                                   (projection.support_count, projection.forbidden_contact_count), strict=True):
                if (name not in arrays or arrays[name].shape != (*arrays["valid"].shape,
                        projection.contact_history_length, count, 3)):
                    raise NumericEvidenceError("Stair numeric v2 contact history array is missing or malformed")
        run_index = metadata["run_index"]
        if run_index != current_run:
            if current_run and any(active):
                raise NumericEvidenceError("Stair archive seed ended before first episodes closed")
            if run_index != current_run + 1 or not metadata["static_present"]:
                raise NumericEvidenceError("Stair archive seed order mismatch")
            current_run, observed_steps = run_index, 0
            static = {name: torch.from_numpy(arrays[name]) for name in _STATIC_FIELDS}
            trackers = [StairEpisodeTracker(None, semantics, env_index=index, bin_id=cell.bin_id, evaluation_cell=cell)
                        for index, cell in enumerate(cell_assignment)]
            outcomes, active = [None] * len(trackers), [True] * len(trackers)
            per_seed.append(outcomes)
        if metadata["step_start"] != observed_steps + 1:
            raise NumericEvidenceError("Stair archive control-step gap")
        terms = tuple(metadata["termination_term_order"])
        for step in range(metadata["steps"]):
            if arrays["valid"][step].tolist() != active:
                raise NumericEvidenceError("Stair archive first-episode coverage mismatch")
            raw = {**metadata["transition_header"], **static,
                   **{name: torch.from_numpy(arrays[name][step]) for name in
                      (*_DYNAMIC_FIELDS, *(STAIR_CONTACT_HISTORY_TENSOR_FIELDS if has_history else ()))},
                   "termination_term_order": terms,
                   "termination_masks": {name: torch.from_numpy(arrays[f"termination_{i}"][step]) for i, name in enumerate(terms)}}
            transitions = decode_stair_transition_batch(raw,
                wrapper_done_mask=torch.from_numpy(arrays["wrapper_done_mask"][step]),
                protocol=None, semantics=semantics, expected_bin_assignment=assignment,
                expected_cell_assignment=cell_assignment, runtime_projection=projection)
            for index, transition in enumerate(transitions):
                if active[index]:
                    outcome = trackers[index].observe(transition)
                    if observe_transition is not None:
                        observe_transition(run_index, transition, raw, outcome)
                    if outcome is not None:
                        outcomes[index], active[index] = outcome, False
            observed_steps += 1
        if not any(active):
            expected = seeds[run_index - 1]
            if [outcome.to_public_dict() for outcome in outcomes] != expected["outcomes"]:
                raise NumericEvidenceError("Stair archive reconstructed outcome mismatch")
    if not per_seed or any(active) or len(per_seed) != len(seeds):
        raise NumericEvidenceError("Stair archive incomplete episode coverage")
    summary = build_unified_stair_summary(suite=suite, semantics=semantics, split="public_dev", per_seed_outcomes=per_seed)
    if summary != context["summary"]:
        raise NumericEvidenceError("Stair archive reconstructed summary mismatch")
    return {"summary": summary, "per_seed_outcomes": [[outcome.to_public_dict() for outcome in seed] for seed in per_seed],
            "episode_count": sum(len(seed) for seed in per_seed), "profile": profile,
            "contact_force_history": dict(history)}
