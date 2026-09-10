"""Re-audit the frozen P4 baseline fixture; never create or resume old work."""

import json
from pathlib import Path
from unittest.mock import patch

from tests.fakes.backend_environment import use_test_isaac_lab_execution_environment
from tests.fakes.generation_runner import GPU_UUIDS, FORMAL_SESSION_ID
from true_rl_agent.tasks.stair_climbing import generation_history
from true_rl_agent.tasks.stair_climbing.generation_session_cost import load_stair_generation_session_cost_contract


def audit_fixture(project: Path, root: Path):
    output = root / "formal-output"
    source_commit = json.loads((output / "generation_source_checkout.json").read_text())["source_commit"]
    cost = load_stair_generation_session_cost_contract(output / "generation_session_cost.json")
    config = generation_history.HistoricalStairGenerationConfig(
        project_root=project, output_dir=output, lease_root=root / "formal-leases",
        session_id=FORMAL_SESSION_ID, expected_source_commit=source_commit,
        machine_id="fixture-machine", gpu0_uuid=GPU_UUIDS[0], gpu1_uuid=GPU_UUIDS[1],
        max_allocated_active_gpu_hours=cost.max_allocated_active_gpu_hours,
        max_simulated_control_steps=cost.max_simulated_control_steps,
        max_output_bytes=cost.max_output_bytes, resume_existing=True,
        terminal_audit_only=True, session_mode="formal_generation_session",
    )
    with use_test_isaac_lab_execution_environment(), patch.object(
        generation_history, "_verify_source_checkout", return_value=source_commit,
    ):
        return generation_history.audit_completed_stair_formal_session(config)
