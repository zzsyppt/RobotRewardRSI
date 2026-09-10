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
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from true_rl_agent.composition.defaults import build_application_composition_registry
from true_rl_agent.runner.source_checkout import SourceCheckout


class GenerationEntrypointTests(unittest.TestCase):
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

    def test_three_real_cells_prepare_and_resume_through_one_cli(self):
        registry = build_application_composition_registry()
        for cell, workload in cell_workloads():
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
                root = Path(directory).resolve()
                task_id = next(task for task in registry.task_ids() if cell.split(".")[0] in task)
                provider = registry.get(task_id)
                composition = resolve_task_composition(provider=provider,
                    values={**provider.cli_defaults(), "cell_variant_id": cell},
                    context=TaskCompositionContext(project_root=entrypoint.PROJECT_ROOT, output_dir=root / "cost-source"))
                cost = fixture_cost(composition.task_plugin.phased_evaluation_adapter().generation_workload(
                    cell_manifest=composition.resolved_cell_manifest))
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
                self.assertEqual(before, {str(p.relative_to(root / "session")): (p.read_bytes(), p.stat().st_mtime_ns)
                                         for p in (root / "session").rglob("*") if p.is_file()})
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
                    with self.assertRaisesRegex(ValueError, "existing resolved cell manifest differs"):
                        entrypoint.prepare_from_args(audit_args)
                    self.assertEqual(manifest.read_bytes(), b"{}\n")
                    manifest.unlink()
                    with self.assertRaises(FileNotFoundError):
                        entrypoint.prepare_from_args(audit_args)
                    self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
