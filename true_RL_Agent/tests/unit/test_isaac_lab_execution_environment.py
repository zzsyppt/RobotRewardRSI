from __future__ import annotations

import json
import sys
import unittest
import tempfile
import subprocess
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.backend_environment import (  # noqa: E402
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST,
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
    TEST_ISAAC_LAB_MODULE_SOURCES,
    TEST_ISAAC_LAB_PACKAGE_VERSIONS,
    make_test_isaac_lab_execution_environment_manifest,
)
from true_rl_agent.backends.isaac_lab_execution_environment import (  # noqa: E402
    IsaacLabExecutionEnvironmentManifest,
    build_current_isaac_lab_execution_environment_manifest,
    build_isaac_lab_execution_environment_manifest_from_components,
    verify_current_isaac_lab_execution_environment,
)
from true_rl_agent.runner.source_snapshot import project_source_snapshot
from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter


class IsaacLabExecutionEnvironmentTests(unittest.TestCase):
    def test_physx_logged_failure_cannot_be_published_as_success(self) -> None:
        from true_rl_agent.backends.isaac_lab_runtime import require_physx_success

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "summary.kit.log"
            log.write_text("[Warning] [omni.physx.plugin] diagnostic\n", encoding="utf-8")
            require_physx_success(log)
            log.write_text(
                "[Error] [omni.physx.plugin] PhysX error: The application needs to "
                "increase PxGpuDynamicsMemoryConfig::totalAggregatePairsCapacity to "
                "2106194, otherwise, the simulation will miss interactions\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "totalAggregatePairsCapacity"):
                require_physx_success(log)

    def test_rsl5_identity_binds_both_model_and_distribution_sources(self) -> None:
        versions = {**TEST_ISAAC_LAB_PACKAGE_VERSIONS, "rsl-rl-lib": "5.0.1"}
        sources = {k: v for k, v in TEST_ISAAC_LAB_MODULE_SOURCES.items() if k != "ActorCritic"}
        sources.update(MLPModel=b"class MLPModel: pass\n",
                       GaussianDistribution=b"class GaussianDistribution: pass\n")
        def build():
            return build_isaac_lab_execution_environment_manifest_from_components(
                python_identity={"implementation": "CPython", "version": "3.12.14", "cache_tag": "cpython-312"},
                package_versions=versions, module_sources=sources)
        first = build()
        self.assertEqual(first, IsaacLabExecutionEnvironmentManifest.from_mapping(first.to_dict()))
        self.assertEqual(len(first.modules), 6)
        self.assertNotIn("ActorCritic", [item["logical_name"] for item in first.modules])
        sources["GaussianDistribution"] += b"# changed distribution implementation\n"
        self.assertNotEqual(first.manifest_hash, build().manifest_hash)

    def test_parent_rejects_source_drift_before_resampling_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ("src/true_rl_agent", "scripts", "configs"):
                (root / directory).mkdir(parents=True)
            source = root / "scripts/train.py"
            source.write_bytes(b"VALUE = 1\n")
            manifest = replace(
                make_test_isaac_lab_execution_environment_manifest(),
                manifest_hash=None,
                project_source=project_source_snapshot(root),
            )
            with (
                patch(
                    "true_rl_agent.backends.isaac_lab_rsl_rl.running_project_source_snapshot",
                    side_effect=lambda: project_source_snapshot(root),
                ),
                patch(
                    "true_rl_agent.backends.isaac_lab_rsl_rl.build_current_isaac_lab_execution_environment_manifest",
                    return_value=manifest,
                ) as resolver,
            ):
                backend = IsaacLabRslRlBackendAdapter(
                    SimpleNamespace(isaac_lab_python=sys.executable)
                )
                backend.validate_runtime_session_execution_environment(
                    execution_environment=manifest.to_dict()
                )
                resolver.assert_called_once()
                source.write_bytes(b"VALUE = 2\n")
                with self.assertRaisesRegex(
                    ValueError, "project source changed after backend construction"
                ):
                    backend.validate_runtime_session_execution_environment(
                        execution_environment=manifest.to_dict()
                    )
                resolver.assert_called_once()

    def test_child_admission_binds_parent_source_across_locations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("first", "moved"):
                for directory in ("src/true_rl_agent", "scripts", "configs"):
                    (root / name / directory).mkdir(parents=True)
                (root / name / "src/true_rl_agent/runtime.py").write_bytes(b"VALUE = 1\n")
                (root / name / "configs/physics.json").write_bytes(b'{"control_dt":0.02}\n')
            expected = replace(make_test_isaac_lab_execution_environment_manifest(), manifest_hash=None, project_source=project_source_snapshot(root / "first"))
            self.assertEqual(expected.project_source, project_source_snapshot(root / "moved"))
            self.assertEqual(expected, IsaacLabExecutionEnvironmentManifest.from_mapping(expected.to_dict()))
            worker = '''
import sys
from pathlib import Path
from dataclasses import replace
from tests.fakes.backend_environment import make_test_isaac_lab_execution_environment_manifest
from true_rl_agent.runner.source_snapshot import project_source_snapshot
from true_rl_agent.backends.isaac_lab_execution_environment import verify_current_isaac_lab_execution_environment
verify_current_isaac_lab_execution_environment(sys.argv[2], current_manifest_builder=lambda: replace(make_test_isaac_lab_execution_environment_manifest(), manifest_hash=None, project_source=project_source_snapshot(Path(sys.argv[1]))))
print("job-admitted")
'''
            def admit():
                return subprocess.run([sys.executable, "-c", worker, str(root / "moved"), expected.manifest_hash], cwd=PROJECT_ROOT, capture_output=True, text=True)
            admitted = admit()
            self.assertEqual(admitted.returncode, 0, admitted.stderr)
            (root / "moved/src/true_rl_agent/runtime.py").write_bytes(b"VALUE = 2\n")
            refused = admit()
            self.assertNotEqual(refused.returncode, 0)
            self.assertNotIn("job-admitted", refused.stdout)
            self.assertIn("environment drift", refused.stderr)

    def test_stable_fixture_round_trips_with_canonical_hash(self) -> None:
        first = make_test_isaac_lab_execution_environment_manifest()
        second = IsaacLabExecutionEnvironmentManifest.from_mapping(first.to_dict())

        self.assertEqual(first, second)
        self.assertEqual(
            first.manifest_hash,
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
        )
        self.assertEqual(len(first.modules), 5)
        self.assertEqual(
            [item["logical_name"] for item in first.modules],
            [
                "OnPolicyRunner",
                "PPO",
                "ActorCritic",
                "RolloutStorage",
                "RslRlVecEnvWrapper",
            ],
        )

    def test_only_fixed_packages_and_sources_participate_in_identity(self) -> None:
        payload = json.dumps(
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST,
            sort_keys=True,
        )

        for forbidden in (
            "driver",
            "gpu",
            "hardware",
            "hostname",
            "executable_path",
            "pip_freeze",
        ):
            self.assertNotIn(forbidden, payload.lower())
        self.assertEqual(
            {
                item["distribution"]
                for item in TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST["packages"]
            },
            set(TEST_ISAAC_LAB_PACKAGE_VERSIONS),
        )

    def test_hot_path_source_change_changes_manifest_hash(self) -> None:
        changed_sources = dict(TEST_ISAAC_LAB_MODULE_SOURCES)
        changed_sources["PPO"] += b"# changed\n"
        changed = build_isaac_lab_execution_environment_manifest_from_components(
            python_identity={
                "implementation": "CPython",
                "version": "3.11.15",
                "cache_tag": "cpython-311",
            },
            package_versions=TEST_ISAAC_LAB_PACKAGE_VERSIONS,
            module_sources=changed_sources,
        )

        self.assertNotEqual(
            changed.manifest_hash,
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
        )

    def test_validated_manifest_is_deeply_immutable(self) -> None:
        manifest = make_test_isaac_lab_execution_environment_manifest()

        with self.assertRaises(TypeError):
            manifest.python["version"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            manifest.packages[0]["version"] = "changed"  # type: ignore[index]
        with self.assertRaises(TypeError):
            manifest.modules[0]["source_sha256"] = "0" * 64  # type: ignore[index]

    def test_manifest_rejects_tampered_inner_source_identity(self) -> None:
        tampered = json.loads(json.dumps(TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST))
        tampered["modules"][0]["source_sha256"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            IsaacLabExecutionEnvironmentManifest.from_mapping(tampered)

    def test_component_builder_rejects_broadened_package_snapshot(self) -> None:
        broadened = {**TEST_ISAAC_LAB_PACKAGE_VERSIONS, "numpy": "2.0.0"}

        with self.assertRaisesRegex(ValueError, "package set mismatch"):
            build_isaac_lab_execution_environment_manifest_from_components(
                python_identity={
                    "implementation": "CPython",
                    "version": "3.11.15",
                    "cache_tag": "cpython-311",
                },
                package_versions=broadened,
                module_sources=TEST_ISAAC_LAB_MODULE_SOURCES,
            )

    def test_prelaunch_verification_receipt_is_hash_bound_and_fail_closed(self) -> None:
        receipt = verify_current_isaac_lab_execution_environment(
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
            current_manifest_builder=(
                make_test_isaac_lab_execution_environment_manifest
            ),
        )

        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(
            receipt["actual_manifest_hash"],
            TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
        )
        with self.assertRaisesRegex(ValueError, "execution environment drift"):
            verify_current_isaac_lab_execution_environment(
                "0" * 64,
                current_manifest_builder=(
                    make_test_isaac_lab_execution_environment_manifest
                ),
            )

    def test_production_resolver_has_no_missing_package_fallback(self) -> None:
        with (
            patch(
                "true_rl_agent.backends.isaac_lab_execution_environment."
                "importlib.metadata.version",
                side_effect=PackageNotFoundError("isaaclab"),
            ),
            self.assertRaisesRegex(
                ValueError,
                "backend execution environment package is unavailable: isaaclab",
            ),
        ):
            build_current_isaac_lab_execution_environment_manifest()

if __name__ == "__main__":
    unittest.main()
