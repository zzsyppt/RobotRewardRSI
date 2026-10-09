"""Exercise the public CLI's real task/cell preparation and frozen inputs."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_generation_session as entrypoint
from tests.fakes.backend_environment import use_test_isaac_lab_execution_environment
from tests.fakes.generation_runner import (
    GPU_UUIDS, _nvidia_smi_name,
)
from tests.unit.test_generation_cost_contract import cell_workloads, fixture_cost, fixture_provider_policy
from true_rl_agent.agent_core import generation_preparation
from true_rl_agent.agent_core.generation_lifecycle import GenerationContinuationPolicy
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from true_rl_agent.composition.defaults import build_application_composition_registry
from true_rl_agent.runner.source_checkout import SourceCheckout


class GenerationEntrypointTests(unittest.TestCase):
    def test_external_author_prepares_two_generations_with_normal_g1_pause_without_credentials(self):
        from dataclasses import replace
        from true_rl_agent.agent_core.external_file_provider import ExternalFileCandidateProvider
        from true_rl_agent.agent_core.generation_runtime import generation_provider
        with tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
            root = Path(directory).resolve()
            registry = build_application_composition_registry()
            provider = registry.get("unitree_stair_climbing")
            cell = "stair.go2_12dof.unified_v3"
            composition = resolve_task_composition(provider=provider,
                values={**provider.cli_defaults(), "cell_variant_id": cell},
                context=TaskCompositionContext(project_root=entrypoint.PROJECT_ROOT, output_dir=root / "cost-source"))
            cost = fixture_cost(composition.task_plugin.phased_evaluation_adapter().generation_workload(
                cell_manifest=composition.resolved_cell_manifest), max_generations=2, imports_reference=True)
            cost = replace(cost, imported_reference_timeout_envelope=(10800, 16))
            (root / "cost.json").write_text(json.dumps(cost.to_dict()), encoding="utf-8")
            policy = {"backend": "external_file", "model": "gpt-6.1-sol", "usage_policy": "unknown_full_reserve",
                      "max_attempts_per_lane": 3, "max_output_tokens": 2000,
                      "input_units_per_attempt_reserve": 10000, "timeout_s": 30}
            (root / "provider.json").write_text(json.dumps(policy), encoding="utf-8")
            argv = ["--task-id", provider.task_id, "--cell-variant-id", cell,
                "--output-dir", str(root / "session"), "--session-id", "external-fixture",
                "--lease-root", str(root / "leases"), "--cost-contract", str(root / "cost.json"),
                "--provider-policy", str(root / "provider.json"), "--machine-id", "fixture",
                "--gpu0-uuid", GPU_UUIDS[0], "--gpu1-uuid", GPU_UUIDS[1], "--expected-source-commit", "a" * 40,
                "--max-allocated-active-gpu-hours", str(cost.max_allocated_active_gpu_hours),
                "--max-simulated-control-steps", str(cost.max_simulated_control_steps),
                "--max-output-bytes", str(cost.max_output_bytes), "--max-generations", "2", "--pause-after-generation", "1",
                "--source-reference-genesis-dir", str(root / "qualified-reference"),
                "--source-reference-session-id", "audit3_go2_full_train"]
            with patch.object(entrypoint, "require_source_checkout", return_value=SourceCheckout("a" * 40, False, ())), \
                    patch.object(entrypoint, "discover_gpu_inventory", return_value=dict(enumerate(GPU_UUIDS))):
                foundation = entrypoint.prepare_from_args(entrypoint.parse_args(argv))
            guarded = generation_provider(foundation, generation_id="external-fixture-generation-001", lane_id="lane-a")
            self.assertIsInstance(guarded.delegate, ExternalFileCandidateProvider)
            self.assertEqual(guarded.external_unknown_usage_model, "gpt-6.1-sol")
            self.assertIsNone(foundation.config.provider_env_file)
            self.assertEqual(foundation.continuation_policy.operator_pause_generation_boundaries, (1,))
            with patch.object(entrypoint, "prepare_from_args", return_value=foundation), \
                    patch.object(entrypoint, "isolate_orchestrator_process_group"), \
                    patch.object(entrypoint, "run_generation_session", return_value={"status": "CPU fixture"}) as run, \
                    patch("true_rl_agent.agent_core.openai_compatible_provider.OpenAICompatibleCandidateProvider.validate_credential",
                          side_effect=AssertionError("external author has no API credential")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entrypoint.main(argv), 0)
                run.assert_called_once_with(foundation)

    def test_fresh_and_resumed_cli_require_credentials_until_the_report_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            env_file = root / "provider.env"
            env_file.write_text("UNUSED=fixture\n", encoding="utf-8")
            foundation = SimpleNamespace(config=SimpleNamespace(output_dir=root, provider_env_file=env_file),
                                         provider_policy=fixture_provider_policy())
            args = SimpleNamespace(prepare_only=False, terminal_audit_only=False, resume_existing=False)
            with patch.object(entrypoint, "parse_args", return_value=args), patch.object(
                entrypoint, "prepare_from_args", return_value=foundation,
            ), patch.object(entrypoint, "isolate_orchestrator_process_group"), patch.object(
                entrypoint, "run_generation_session", return_value={"checks": {"status": "PASS"}},
            ) as run, patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
                for resume in (False, True):
                    args.resume_existing = resume
                    with self.subTest(resume=resume), self.assertRaisesRegex(RuntimeError, "missing provider API key"):
                        entrypoint.main([])
                run.assert_not_called()
                # Offline preparation and terminal replay do not call a provider.
                args.prepare_only = True
                foundation.session_contract = SimpleNamespace(contract_hash="a" * 64)
                self.assertEqual(entrypoint.main([]), 0)
                run.assert_not_called()
                args.prepare_only = False
                args.terminal_audit_only = True
                self.assertEqual(entrypoint.main([]), 0)
                args.terminal_audit_only = False
                (root / entrypoint.REPORT_FILENAME).write_text("{}", encoding="utf-8")
                self.assertEqual(entrypoint.main([]), 0)
                (root / entrypoint.REPORT_FILENAME).unlink()
                run.reset_mock()
                env_file.write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
                self.assertEqual(entrypoint.main([]), 0)
                run.assert_called_once_with(foundation)

    def test_gpu_inventory_rejects_malformed_and_duplicate_rows(self):
        from true_rl_agent.runner.gpu_process_binding import discover_gpu_inventory
        valid = f"0, {GPU_UUIDS[0]}\n1, {GPU_UUIDS[1]}\n"
        invalid = (f"0, {GPU_UUIDS[0]}, extra", f"x, {GPU_UUIDS[0]}", "0, GPU-short",
                   valid + f"0, {GPU_UUIDS[1]}\n", f"0, {GPU_UUIDS[0]}\n1, {GPU_UUIDS[0]}")
        for stdout in invalid:
            with self.subTest(stdout=stdout), patch(
                "true_rl_agent.runner.gpu_process_binding.subprocess.run",
                return_value=SimpleNamespace(stdout=stdout),
            ), self.assertRaisesRegex(ValueError, "invalid GPU inventory|duplicate GPU"):
                discover_gpu_inventory("fixture")
        with patch("true_rl_agent.runner.gpu_process_binding.subprocess.run", return_value=SimpleNamespace(stdout=valid)):
            self.assertEqual(discover_gpu_inventory("fixture"), dict(enumerate(GPU_UUIDS)))

    def test_real_cells_and_generation_counts_prepare_and_resume_through_one_cli(self):
        registry = build_application_composition_registry()
        cases = [(cell, 15) for cell, _ in cell_workloads()]
        cases.append(("push.go2_12dof.recovery_v3", 5))
        for cell, max_generations in cases:
            with self.subTest(cell=cell, max_generations=max_generations), tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
                root = Path(directory).resolve()
                task_id = next(task for task in registry.task_ids() if cell.split(".")[0] in task)
                provider = registry.get(task_id)
                composition = resolve_task_composition(provider=provider,
                    values={**provider.cli_defaults(), "cell_variant_id": cell},
                    context=TaskCompositionContext(project_root=entrypoint.PROJECT_ROOT, output_dir=root / "cost-source"))
                cost = fixture_cost(composition.task_plugin.phased_evaluation_adapter().generation_workload(
                    cell_manifest=composition.resolved_cell_manifest), max_generations=max_generations)
                cost_path = root / "cost.json"
                cost_path.write_text(json.dumps(cost.to_dict()), encoding="utf-8")
                policy_path = root / "provider.json"
                policy_path.write_text(json.dumps(fixture_provider_policy()), encoding="utf-8")
                executable = root / _nvidia_smi_name()
                executable.write_text(
                    ("@echo off\n" if os.name == "nt" else "#!/bin/sh\n")
                    + "".join(f"echo {index}, {uuid}\n" for index, uuid in enumerate(GPU_UUIDS)),
                    encoding="utf-8",
                )
                if os.name != "nt":
                    executable.chmod(0o755)
                argv = ["--task-id", task_id, "--cell-variant-id", cell,
                    "--output-dir", str(root / "session"), "--session-id", "cli-matrix-fixture",
                    "--lease-root", str(root / "leases"), "--provider-env-file", str(root / "provider.env"),
                    "--cost-contract", str(cost_path), "--provider-policy", str(policy_path),
                    "--machine-id", "fixture", "--gpu0-uuid", GPU_UUIDS[0], "--gpu1-uuid", GPU_UUIDS[1],
                    "--expected-source-commit", "a" * 40, "--nvidia-smi-executable", str(executable),
                    "--max-allocated-active-gpu-hours", str(cost.max_allocated_active_gpu_hours),
                    "--max-simulated-control-steps", str(cost.max_simulated_control_steps),
                    "--max-output-bytes", str(cost.max_output_bytes), "--prepare-only"]
                if max_generations != 15:
                    argv += ["--max-generations", str(max_generations)]
                self.assertEqual(entrypoint.parse_args(argv).max_generations, max_generations)
                reused = entrypoint.parse_args(argv + [
                    "--source-reference-training-genesis-dir", str(root / "source"),
                    "--source-reference-training-session-id", "qualified-source",
                ])
                self.assertEqual(reused.source_reference_training_genesis_dir, root / "source")
                self.assertEqual(reused.source_reference_training_session_id, "qualified-source")
                for extra in (["--max-generations", "0"], ["--pause-after-generation", "0"],
                              ["--pause-after-generation", str(max_generations)]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        entrypoint.parse_args(argv + extra)
                for missing in ("--task-id", "--cell-variant-id"):
                    incomplete = argv.copy()
                    index = incomplete.index(missing)
                    del incomplete[index:index + 2]
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        entrypoint.parse_args(incomplete)
                with patch.object(entrypoint, "require_source_checkout",
                                  return_value=SourceCheckout("a" * 40, False, ())):
                    first = entrypoint.prepare_from_args(entrypoint.parse_args(argv))
                    before = {str(p.relative_to(root / "session")): (p.read_bytes(), p.stat().st_mtime_ns)
                              for p in (root / "session").rglob("*") if p.is_file()}
                    resumed = entrypoint.prepare_from_args(entrypoint.parse_args(argv + ["--resume-existing"]))
                    with self.assertRaises(ValueError):
                        entrypoint.prepare_from_args(entrypoint.parse_args(
                            argv + ["--resume-existing", "--max-generations", str(max_generations + 1)]))
                    wrong_gpu = entrypoint.parse_args(argv + ["--resume-existing"])
                    wrong_gpu.gpu0_uuid, wrong_gpu.gpu1_uuid = wrong_gpu.gpu1_uuid, wrong_gpu.gpu0_uuid
                    with self.assertRaisesRegex(ValueError, "GPU indices do not match"):
                        entrypoint.prepare_from_args(wrong_gpu)
                    with patch.object(entrypoint, "discover_gpu_inventory",
                                      return_value={0: GPU_UUIDS[0], 1: GPU_UUIDS[1], 2: "extra-card"}):
                        with self.assertRaisesRegex(ValueError, "exactly host GPU indices 0 and 1"):
                            entrypoint.prepare_from_args(entrypoint.parse_args(argv + ["--resume-existing"]))
                self.assertEqual(first.session_contract, resumed.session_contract)
                self.assertEqual(first.composition.resolved_cell_manifest.cell_variant_id, cell)
                self.assertEqual(first.cost_contract, cost)
                self.assertEqual(first.config.max_generations, max_generations)
                self.assertEqual(first.session_contract.max_generations, max_generations)
                self.assertEqual(first.session_contract.max_trials, 2 * max_generations)
                self.assertEqual(first.session_contract.max_train_jobs, 2 * max_generations)
                self.assertEqual(first.final_confirmation_plan.required_closed_generations, max_generations)
                self.assertEqual(before, {str(p.relative_to(root / "session")): (p.read_bytes(), p.stat().st_mtime_ns)
                                         for p in (root / "session").rglob("*") if p.is_file()})
                self.assertEqual(first.continuation_policy.operator_pause_generation_boundaries,
                                 tuple(range(1, max_generations)))
                legacy_argv = argv.copy()
                legacy_argv[legacy_argv.index("--output-dir") + 1] = str(root / "legacy-session")
                with patch.object(entrypoint, "require_source_checkout",
                                  return_value=SourceCheckout("a" * 40, False, ())):
                    with patch.object(generation_preparation, "GenerationContinuationPolicy",
                        side_effect=lambda **kwargs: GenerationContinuationPolicy(**{
                            **kwargs, "operator_pause_generation_boundaries": tuple(
                                item for item in (3, 6, 9, 12) if item < max_generations)})):
                        legacy = entrypoint.prepare_from_args(entrypoint.parse_args(legacy_argv))
                    legacy_before = {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
                                     for p in (root / "legacy-session").rglob("*") if p.is_file()}
                    legacy_resumed = entrypoint.prepare_from_args(entrypoint.parse_args(
                        legacy_argv + ["--resume-existing", "--pause-after-generation", "3"]))
                    self.assertEqual(legacy_resumed.session_contract, legacy.session_contract)
                    with self.assertRaisesRegex(ValueError, "frozen continuation policy"):
                        entrypoint.prepare_from_args(entrypoint.parse_args(
                            legacy_argv + ["--resume-existing", "--pause-after-generation", "2"]))
                    self.assertEqual(legacy_before,
                        {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
                         for p in (root / "legacy-session").rglob("*") if p.is_file()})
                manifest = root / "session/resolved_cell_manifest.json"
                with patch.object(entrypoint, "require_source_checkout",
                                  return_value=SourceCheckout("a" * 40, False, ())):
                    audit_args = entrypoint.parse_args(argv + ["--resume-existing", "--terminal-audit-only"])
                    contract_path = root / "session/generation_session_contract.json"
                    original = contract_path.read_bytes()
                    contract_path.unlink()
                    with self.assertRaises(FileNotFoundError):
                        entrypoint.prepare_from_args(audit_args)
                    self.assertFalse(contract_path.exists())
                    contract_path.write_bytes(original)
                    manifest.write_bytes(b"{}\n")
                    with self.assertRaisesRegex(
                        ValueError,
                        "existing resolved cell manifest differs|resolved cell manifest fields mismatch",
                    ):
                        entrypoint.prepare_from_args(audit_args)
                    self.assertEqual(manifest.read_bytes(), b"{}\n")
                    manifest.unlink()
                    with self.assertRaises((FileNotFoundError, ValueError)) as missing_manifest:
                        entrypoint.prepare_from_args(audit_args)
                    if isinstance(missing_manifest.exception, ValueError):
                        self.assertIsInstance(missing_manifest.exception.__cause__, FileNotFoundError)
                    self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
