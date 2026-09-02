from __future__ import annotations

import json
import sys
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
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


class IsaacLabExecutionEnvironmentTests(unittest.TestCase):
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
