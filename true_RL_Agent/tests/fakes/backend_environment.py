"""Stable backend execution-environment identities for dependency-free tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from true_rl_agent.backends.base import (
    BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION,
)
from true_rl_agent.backends.isaac_lab_execution_environment import (
    IsaacLabExecutionEnvironmentManifest,
    build_isaac_lab_execution_environment_manifest_from_components,
)

TEST_ISAAC_LAB_PACKAGE_VERSIONS = {
    "isaaclab": "0.47.2",
    "isaaclab-rl": "0.4.4",
    "isaaclab-tasks": "0.11.6",
    "isaacsim": "5.1.0.0",
    "isaacsim-core": "5.1.0.0",
    "isaacsim-extscache-physics": "5.1.0.0",
    "rsl-rl-lib": "3.0.1",
    "torch": "2.7.0+cu128",
}

TEST_ISAAC_LAB_MODULE_SOURCES = {
    "OnPolicyRunner": b"class OnPolicyRunner: pass\n",
    "PPO": b"class PPO: pass\n",
    "ActorCritic": b"class ActorCritic: pass\n",
    "RolloutStorage": b"class RolloutStorage: pass\n",
    "RslRlVecEnvWrapper": b"class RslRlVecEnvWrapper: pass\n",
}


def make_test_isaac_lab_execution_environment_manifest() -> (
    IsaacLabExecutionEnvironmentManifest
):
    return build_isaac_lab_execution_environment_manifest_from_components(
        python_identity={
            "implementation": "CPython",
            "version": "3.11.15",
            "cache_tag": "cpython-311",
        },
        package_versions=TEST_ISAAC_LAB_PACKAGE_VERSIONS,
        module_sources=TEST_ISAAC_LAB_MODULE_SOURCES,
    )


TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST = (
    make_test_isaac_lab_execution_environment_manifest().to_dict()
)
TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH = str(
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST["manifest_hash"]
)
TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION = {
    "schema_version": BACKEND_EXECUTION_ENVIRONMENT_VERIFICATION_SCHEMA_VERSION,
    "backend_id": "isaac_lab_rsl_rl",
    "status": "pass",
    "expected_manifest_hash": (TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH),
    "actual_manifest_hash": TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
}


@contextmanager
def use_test_isaac_lab_execution_environment() -> Iterator[None]:
    """Inject the stable manifest only while constructing test adapters."""

    from true_rl_agent.backends.isaac_lab_rsl_rl import (
        IsaacLabRslRlBackendAdapter,
    )

    original_init = IsaacLabRslRlBackendAdapter.__init__

    def init_with_test_environment(
        self,
        config,
        *,
        execution_environment_resolver=(
            make_test_isaac_lab_execution_environment_manifest
        ),
    ) -> None:
        original_init(
            self,
            config,
            execution_environment_resolver=execution_environment_resolver,
        )

    with patch.object(
        IsaacLabRslRlBackendAdapter,
        "__init__",
        init_with_test_environment,
    ):
        yield


__all__ = [
    "TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST",
    "TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH",
    "TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION",
    "TEST_ISAAC_LAB_MODULE_SOURCES",
    "TEST_ISAAC_LAB_PACKAGE_VERSIONS",
    "make_test_isaac_lab_execution_environment_manifest",
    "use_test_isaac_lab_execution_environment",
]
