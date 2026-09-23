from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from true_rl_agent.composition.base import TaskCompositionContext
from true_rl_agent.composition.cell_manifest import materialize_resolved_cell_manifest
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.runner.base import RunnerExecutionError
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.checkpoint_ancestry import (
    build_stair_checkpoint_ancestry_admission,
)
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.unified_protocol import StairUnifiedProtocol


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json"


class StairCheckpointAncestryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = StairClimbingTaskConfig.defaults(PROTOCOL)
        cls.manifest = cls._manifest(cls.config)
        cls.admission = build_stair_checkpoint_ancestry_admission(cls.config, cls.manifest)

    @staticmethod
    def _manifest(config):
        return resolve_stair_cell_manifest(
            cell_variant_id=config.cell_variant_id,
            config=config,
            plugin=StairClimbingTaskPlugin(config),
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest_path = materialize_resolved_cell_manifest(self.manifest, output_dir=self.root)
        self.checkpoint_path = self.root / "fixture.pt"
        # Synthetic bytes exercise preflight only; these are not policy weights.
        self.checkpoint_path.write_bytes(b"ancestry-preflight-unit-fixture")
        self.checksum = "sha256:" + hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
        self.runner = SubprocessRunner(
            allowed_executables=(sys.executable,), cwd=ROOT,
            eval_checkpoint_admission_hook=self.admission,
        )

    def _checkpoint(self, source, *, binding=True, role="eval"):
        return CheckpointRef(
            uri="artifact://test/fixture.pt", checksum=self.checksum, role=role, sealed=True,
            metadata={
                "local_path": str(self.checkpoint_path),
                "resolved_cell_manifest_hash": source.manifest_hash,
                "resolved_cell_identity": source.summary_identity(),
                **({"training_binding_hash": self.admission.training_binding_hash} if binding else {}),
            },
        )

    def _preflight(self, checkpoint, *, training_binding=None, metadata=None, train=False):
        option = "--init-checkpoint" if train else "--checkpoint"
        checksum_option = option + "-sha256"
        self.runner._validate_checkpoint_before_launch(
            checkpoint=checkpoint,
            metadata=metadata or {}, job_id="ancestry-unit-preflight",
            argv=[
                sys.executable, "fixture.py", option, str(self.checkpoint_path),
                checksum_option, self.checksum,
                "--resolved-cell-manifest-path", str(self.manifest_path),
                "--resolved-cell-manifest-hash", self.manifest.manifest_hash,
                "--training-binding-hash", training_binding or self.admission.training_binding_hash,
            ],
            checkpoint_path_option=option, checkpoint_checksum_option=checksum_option,
        )

    def test_real_frozen_chain_and_complete_training_binding_match(self):
        self.assertEqual(
            [(m.cell_variant_id, m.manifest_hash) for m in self.admission.manifests],
            [
                ("stair.go2_12dof.unified_v3", "7e8fa0c45109bbfc0162a90cf68d4186a0a3f96a0df9cce1e9b1b09f50aa6de9"),
                ("stair.go2_12dof.unified_v2", "193fa9038a6f46539fb71a7e73edc44bdc024be2b899785394b8f82113c91f37"),
                ("stair.go2_12dof.unified_v1", "54cd180c9f39f65bffd71482ec8a87cda912d0e378e8b434d0c44c839a3ab1c2"),
            ],
        )
        self.assertEqual(self.admission.training_binding_hash,
                         "c0112a87a7fb5750ef11d55a6bb50cfdeb0be662380c1387b2d8cab8c977de98")

    def test_current_direct_and_transitive_predecessors_pass_without_relabelling(self):
        for manifest in self.admission.manifests:
            with self.subTest(source=manifest.cell_variant_id):
                checkpoint = self._checkpoint(manifest)
                before = checkpoint.to_public_dict()
                self._preflight(checkpoint)
                self.assertEqual(checkpoint.to_public_dict(), before)

    def test_current_checkpoint_keeps_existing_optional_training_metadata(self):
        self._preflight(self._checkpoint(self.manifest, binding=False))

    def test_cross_version_source_training_binding_is_required_and_matches_eval(self):
        checkpoint = self._checkpoint(self.admission.manifests[-1], binding=False)
        with self.assertRaisesRegex(RunnerExecutionError, "training binding is missing or mismatched"):
            self._preflight(checkpoint)
        checkpoint = replace(checkpoint, metadata={**checkpoint.metadata, "training_binding_hash": "a" * 64})
        with self.assertRaisesRegex(RunnerExecutionError, "training binding is missing or mismatched"):
            self._preflight(checkpoint)
        with self.assertRaisesRegex(RunnerExecutionError, "evaluation training binding mismatch"):
            self._preflight(self._checkpoint(self.admission.manifests[-1]), training_binding="b" * 64)

    def test_source_identity_and_checkpoint_bytes_are_still_bound(self):
        checkpoint = self._checkpoint(self.admission.manifests[-1])
        wrong_identity = replace(checkpoint, metadata={**checkpoint.metadata,
                                                       "resolved_cell_identity": self.manifest.summary_identity()})
        with self.assertRaisesRegex(RunnerExecutionError, "source cell identity mismatch"):
            self._preflight(wrong_identity)
        self.checkpoint_path.write_bytes(b"changed")
        with self.assertRaisesRegex(RunnerExecutionError, "checksum"):
            self._preflight(checkpoint)

    def test_metadata_hash_list_cannot_bypass_task_owned_chain(self):
        checkpoint = self._checkpoint(self.manifest)
        checkpoint = replace(checkpoint, metadata={**checkpoint.metadata,
                                                   "resolved_cell_manifest_hash": "a" * 64})
        with self.assertRaisesRegex(RunnerExecutionError, "not a verified Stair ancestor"):
            self._preflight(checkpoint, metadata={"training_predecessor_manifest_hash": "a" * 64})

    def test_transitive_admission_does_not_authorize_training_warm_start(self):
        with self.assertRaisesRegex(RunnerExecutionError, "manifest hash mismatch"):
            self._preflight(self._checkpoint(self.admission.manifests[-1], role="warm_start"), train=True)

    def test_declared_predecessor_resolved_identity_must_be_reproducible(self):
        protocol = StairUnifiedProtocol.load(PROTOCOL)
        for field in ("source_resolved_cell_manifest_hash", "source_resolved_cell_task_protocol_hash"):
            with self.subTest(field=field):
                payload = protocol.to_dict()
                payload["training_lineage"][field] = "a" * 64
                payload["task_protocol_hash"] = replace(protocol, _payload=payload).task_protocol_hash
                payload["protocol_hash"] = replace(protocol, _payload=payload).protocol_hash

                def load(path, **kwargs):
                    return payload if Path(path).resolve() == PROTOCOL else load_strict_json_object(path, **kwargs)

                with patch("true_rl_agent.tasks.stair_climbing.unified_protocol.load_strict_json_object", side_effect=load):
                    config = StairClimbingTaskConfig.defaults(PROTOCOL)
                    manifest = self._manifest(config)
                    with self.assertRaisesRegex(ValueError, "predecessor resolved manifest identity mismatch"):
                        build_stair_checkpoint_ancestry_admission(config, manifest)

    def test_other_robot_is_not_an_ancestor(self):
        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_unified_protocol_v3.json")
        other_manifest = self._manifest(config)
        with self.assertRaisesRegex(RunnerExecutionError, "not a verified Stair ancestor"):
            self._preflight(self._checkpoint(other_manifest))
        admission = build_stair_checkpoint_ancestry_admission(config, other_manifest)
        self.assertEqual(len(admission.manifests), 3)

    def test_composition_installs_the_task_policy(self):
        provider = StairClimbingCompositionProvider()
        composition = provider.resolve(
            values={**provider.cli_defaults(), "cell_variant_id": self.config.cell_variant_id},
            context=TaskCompositionContext(project_root=ROOT, output_dir=self.root / "composition"),
        )
        policy = composition.runner.eval_checkpoint_admission_hook
        self.assertIsNotNone(policy)
        self.assertEqual(policy.manifests[0].manifest_hash, composition.resolved_cell_manifest.manifest_hash)
        self.assertEqual(policy.training_binding_hash, self.admission.training_binding_hash)


if __name__ == "__main__":
    unittest.main()
