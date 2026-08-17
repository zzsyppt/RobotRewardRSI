from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import textwrap
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.provider_contract import (
    candidate_package_files_contract,
)
from true_rl_agent.candidates.artifact import (
    CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
    VerifiedCandidateArtifactRef,
    load_sealed_candidate_artifact,
    seal_candidate_package,
    verify_sealed_candidate_artifact,
)
from true_rl_agent.candidates.authoring_policy import (
    MAX_CANONICAL_COMPONENT_NAME_CHARS,
)
from true_rl_agent.candidates.components import (
    DiagnosticTerm,
    RewardInputs,
    RewardTerm,
    ScheduleTerm,
    TrainingComponents,
)
from true_rl_agent.candidates.loader import (
    load_candidate_manifest,
    load_training_components,
)
from true_rl_agent.candidates.package_contract import (
    CANDIDATE_PACKAGE_CONTRACT_HASH,
    CANDIDATE_PACKAGE_FILE_MAX_BYTES,
    CANDIDATE_PACKAGE_MAX_BYTES,
    PROVIDER_AUTHORED_CANDIDATE_FILES,
    SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES,
    CandidatePackageContractError,
    candidate_package_contract_descriptor,
    open_preflighted_candidate_file,
    preflight_candidate_package,
)
from true_rl_agent.candidates.schema import (
    MAX_CANDIDATE_ID_CHARS,
    CandidateChangeSet,
    CandidatePackageRef,
)
from true_rl_agent.candidates.verifier import (
    STATIC_CANDIDATE_VERIFIER_POLICY_ID,
    StaticCandidateVerifier,
    StaticCandidateVerifierPolicy,
    audit_candidate_python_source,
    seal_and_verify_candidate,
    static_verifier_policy_facts,
    static_verifier_policy_hash,
    static_verifier_prompt_rules,
    static_verifier_source_policy_hash,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.training_programs import resolve_candidate_program_identity


class CandidatePackageTests(unittest.TestCase):
    def _write_candidate(self, root: Path, *, components_source: str | None = None) -> Path:
        candidate = root / "candidate"
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            textwrap.dedent(
                """
                schema_version: true_rl_agent.candidate.v1
                candidate_id: candidate_reward
                entrypoint: training_code.components:build_training_components
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (candidate / "training_code" / "components.py").write_text(
            components_source
            or textwrap.dedent(
                """
                from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


                def upright_bonus(inputs):
                    return inputs.base_height


                def build_training_components():
                    return TrainingComponents(
                        name="candidate_reward",
                        reward_terms=[RewardTerm(name="upright_bonus", weight=1.0, fn=upright_bonus)],
                        rationale="code-level reward callable",
                    )
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (candidate / "rationale.md").write_text(
            "Candidate rationale.\n",
            encoding="utf-8",
        )
        (candidate / "operation_diff.json").write_text("{}\n", encoding="utf-8")
        (candidate / "diagnostics.json").write_text("{}\n", encoding="utf-8")
        return candidate

    def test_candidate_identity_is_path_safe_and_bounded_at_the_shared_contract(self) -> None:
        change_set = CandidateChangeSet(frozenset({"reward_callable"}))
        valid_id = "a" * MAX_CANDIDATE_ID_CHARS

        candidate = CandidatePackageRef(
            candidate_id=valid_id,
            path="candidate",
            change_set=change_set,
        )

        self.assertEqual(candidate.candidate_id, valid_id)
        for invalid_id in (
            "a" * (MAX_CANDIDATE_ID_CHARS + 1),
            "bad/path",
            "Candidate",
            "a.",
            "nul",
            "con.txt",
            "com1.log",
            "lpt9",
        ):
            with self.subTest(candidate_id=invalid_id[:20]):
                with self.assertRaisesRegex(ValueError, "candidate_id"):
                    CandidatePackageRef(
                        candidate_id=invalid_id,
                        path="candidate",
                        change_set=change_set,
                    )

    def test_runner_owned_candidate_artifact_io_failure_is_not_a_denial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            candidate = CandidatePackageRef(
                candidate_id="candidate_reward",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            with patch(
                "true_rl_agent.candidates.verifier.seal_candidate_package",
                side_effect=OSError("artifact filesystem unavailable"),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "artifact filesystem unavailable",
                ):
                    seal_and_verify_candidate(
                        candidate,
                        artifact_root=root / "artifacts",
                        verifier=StaticCandidateVerifier(),
                    )

    def test_static_verifier_manifest_io_failure_is_not_a_denial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))
            with patch(
                "true_rl_agent.candidates.verifier.load_candidate_manifest",
                side_effect=OSError("manifest filesystem unavailable"),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "manifest filesystem unavailable",
                ):
                    StaticCandidateVerifier().verify(candidate_dir)

    def test_loader_reads_manifest_and_controlled_training_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(Path(tmp))

            manifest = load_candidate_manifest(candidate)
            components = load_training_components(candidate)

            self.assertEqual(manifest.candidate_id, "candidate_reward")
            self.assertIsInstance(components, TrainingComponents)
            self.assertIsInstance(components.reward_terms[0], RewardTerm)
            self.assertTrue(callable(components.reward_terms[0].fn))
            self.assertEqual(components.to_change_set().changed_components, frozenset({"reward_callable"}))

    def test_loader_tolerates_utf8_bom_manifest_from_windows_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(Path(tmp))
            manifest_path = candidate / "candidate_manifest.yaml"
            manifest_path.write_text("\ufeff" + manifest_path.read_text(encoding="utf-8"), encoding="utf-8")

            manifest = load_candidate_manifest(candidate)

            self.assertEqual(manifest.schema_version, "true_rl_agent.candidate.v1")
            self.assertEqual(manifest.candidate_id, "candidate_reward")

    def test_training_components_support_only_currently_wired_hooks(self) -> None:
        components = TrainingComponents(
            name="mixed",
            reward_terms=[RewardTerm("reward", 1.0, lambda inputs: 1.0)],
            diagnostics=[DiagnosticTerm("diag", lambda inputs: {"ok": True})],
            schedules=[ScheduleTerm("scale", "reward_weight_scale", 1.0)],
        )

        summary = components.to_public_summary()

        self.assertEqual(summary["effective_hook_counts"]["reward_terms"], 1)
        self.assertEqual(summary["effective_hook_counts"]["diagnostics"], 1)
        self.assertEqual(summary["effective_hook_counts"]["schedules"], 1)
        self.assertNotIn("loss_terms", summary["effective_hook_counts"])
        self.assertNotIn("observation_transforms", summary["effective_hook_counts"])

    def test_static_verifier_preflights_forbidden_files_before_source_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source="import os\n\ndef build_training_components():\n    return None\n",
            )
            (candidate / "repo.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertIn("forbidden_file:repo.patch", result.details["issues"])

    def test_static_verifier_rejects_forbidden_tokens_after_clean_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source="import os\n\ndef build_training_components():\n    return None\n",
            )

            result = StaticCandidateVerifier().verify(candidate)

        self.assertEqual(result.status, "fail")
        self.assertIn(
            "forbidden_token:training_code/components.py:import os",
            result.details["issues"],
        )

    def test_static_verifier_accepts_candidate_package_ref_with_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))
            candidate = CandidatePackageRef(
                candidate_id="candidate_reward",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            result = StaticCandidateVerifier().verify(candidate)

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.reason, "static_candidate_verifier_passed")
        self.assertEqual(
            result.details["candidate_package_contract_hash"],
            CANDIDATE_PACKAGE_CONTRACT_HASH,
        )

    def test_sealed_artifact_binds_the_bytes_verified_by_static_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            provider_candidate = CandidatePackageRef(
                candidate_id="candidate_reward",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            artifact = seal_candidate_package(
                provider_candidate,
                artifact_root=root / "artifacts",
            )

            result = StaticCandidateVerifier().verify(artifact)

            self.assertEqual(result.status, "pass")
            self.assertEqual(result.candidate_artifact_hash, artifact.artifact_hash)
            self.assertEqual(
                result.program_identity.source_checksum,
                artifact.program_source_checksum,
            )
            self.assertEqual(
                result.program_identity.manifest_hash,
                artifact.subject_manifest_hash,
            )

    def test_unified_pipeline_is_the_only_public_verified_capability_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            verification = seal_and_verify_candidate(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=CandidateChangeSet(
                        frozenset({"reward_callable"})
                    ),
                ),
                artifact_root=root / "artifacts",
                verifier=StaticCandidateVerifier(),
            )

            artifact = verification.require_verified()

        self.assertTrue(verification.passed)
        self.assertIsInstance(artifact, VerifiedCandidateArtifactRef)
        self.assertEqual(
            artifact.verification_policy,
            STATIC_CANDIDATE_VERIFIER_POLICY_ID,
        )
        self.assertEqual(
            verification.result.details["verification_policy_hash"],
            static_verifier_policy_hash(),
        )
        with self.assertRaisesRegex(TypeError, "verification pipeline"):
            VerifiedCandidateArtifactRef(
                sealed_artifact=artifact.sealed_artifact,
                verification_policy=artifact.verification_policy,
                verification_hash=artifact.verification_hash,
            )

    def test_unified_pipeline_does_not_mint_capability_for_rejected_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(
                root,
                components_source=(
                    "import os\n\n"
                    "def build_training_components():\n"
                    "    return None\n"
                ),
            )
            verification = seal_and_verify_candidate(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=CandidateChangeSet(
                        frozenset({"reward_callable"})
                    ),
                ),
                artifact_root=root / "artifacts",
                verifier=StaticCandidateVerifier(),
            )
            self.assertIsNotNone(verification.sealed_artifact)
            self.assertEqual(
                verify_sealed_candidate_artifact(verification.sealed_artifact),
                verification.sealed_artifact,
            )

        self.assertFalse(verification.passed)
        self.assertIsNone(verification.verified_artifact)
        self.assertIsNotNone(verification.sealed_artifact)
        self.assertEqual(
            verification.result.reason,
            "static_candidate_verifier_failed",
        )

    def test_provider_file_contract_excludes_runner_owned_artifact_manifest(self) -> None:
        prompt_rules = static_verifier_prompt_rules()

        self.assertEqual(
            prompt_rules["allowed_files"],
            sorted(PROVIDER_AUTHORED_CANDIDATE_FILES),
        )
        self.assertNotIn(
            CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
            prompt_rules["allowed_files"],
        )
        self.assertEqual(
            prompt_rules["file_max_bytes"],
            dict(CANDIDATE_PACKAGE_FILE_MAX_BYTES),
        )
        self.assertEqual(
            prompt_rules["candidate_package_contract_hash"],
            CANDIDATE_PACKAGE_CONTRACT_HASH,
        )
        self.assertEqual(
            prompt_rules["max_total_bytes"],
            sum(prompt_rules["file_max_bytes"].values()),
        )
        self.assertEqual(prompt_rules["max_total_bytes"], CANDIDATE_PACKAGE_MAX_BYTES)

    def test_package_limits_have_an_independent_contract_identity(self) -> None:
        descriptor = candidate_package_contract_descriptor()
        policy = StaticCandidateVerifierPolicy()

        self.assertEqual(
            canonical_json_hash(descriptor),
            CANDIDATE_PACKAGE_CONTRACT_HASH,
        )
        self.assertEqual(
            descriptor["file_max_bytes"],
            dict(CANDIDATE_PACKAGE_FILE_MAX_BYTES),
        )
        self.assertEqual(
            candidate_package_files_contract()["package_contract_hash"],
            CANDIDATE_PACKAGE_CONTRACT_HASH,
        )
        self.assertNotIn(
            "candidate_package_contract",
            policy.executable_policy_facts(),
        )
        self.assertNotIn(
            "candidate_package_contract",
            policy.to_snapshot_payload()["policy_facts"],
        )

    def test_candidate_package_preflight_requires_every_fixed_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative_path in PROVIDER_AUTHORED_CANDIDATE_FILES:
                with self.subTest(relative_path=relative_path):
                    candidate = self._write_candidate(root / relative_path.replace("/", "_"))
                    (candidate / relative_path).unlink()

                    with self.assertRaisesRegex(ValueError, "missing required path"):
                        preflight_candidate_package(candidate)

    def test_candidate_package_preflight_rejects_each_file_over_its_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative_path, maximum in CANDIDATE_PACKAGE_FILE_MAX_BYTES.items():
                with self.subTest(relative_path=relative_path):
                    candidate = self._write_candidate(root / relative_path.replace("/", "_"))
                    (candidate / relative_path).write_bytes(b"x" * (maximum + 1))

                    with self.assertRaisesRegex(ValueError, "exceeds its byte limit"):
                        preflight_candidate_package(candidate)

    def test_candidate_package_preflight_rejects_nonregular_and_symlink_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nonregular = self._write_candidate(root / "nonregular")
            rationale = nonregular / "rationale.md"
            rationale.unlink()
            rationale.mkdir()
            with self.assertRaisesRegex(ValueError, "file must be regular"):
                preflight_candidate_package(nonregular)

            symlinked = self._write_candidate(root / "symlinked")
            symlink_path = symlinked / "rationale.md"
            symlink_path.unlink()
            target = root / "outside.md"
            target.write_text("outside\n", encoding="utf-8")
            try:
                os.symlink(target, symlink_path)
            except OSError as exc:
                self.skipTest(f"file symlinks are unavailable: {exc}")
            with self.assertRaisesRegex(ValueError, "does not allow symlinks"):
                preflight_candidate_package(symlinked)

    def test_invalid_layout_stops_before_ast_or_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            (candidate_dir / "unknown.txt").write_text("unknown\n", encoding="utf-8")
            candidate = CandidatePackageRef(
                candidate_id="candidate_reward",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            with patch(
                "true_rl_agent.candidates.verifier.ast.parse",
                side_effect=AssertionError("AST must not run before preflight"),
            ):
                result = StaticCandidateVerifier().verify(candidate)
            with patch(
                "true_rl_agent.candidates.artifact.copy_preflighted_candidate_package",
                side_effect=AssertionError("copy must not run before preflight"),
            ):
                with self.assertRaisesRegex(ValueError, "forbidden path"):
                    seal_candidate_package(candidate, artifact_root=root / "artifacts")

        self.assertEqual(result.details["issues"], ["forbidden_file:unknown.txt"])

    def test_preflight_rejects_first_unknown_entry_without_buffering_directory(self) -> None:
        class UnknownEntry:
            name = "unknown-000000.txt"

            @staticmethod
            def stat(*, follow_symlinks: bool):
                self.assertFalse(follow_symlinks)
                return os.stat_result((0o100644, 0, 0, 1, 0, 0, 1, 0, 0, 0))

        class Scanner:
            def __init__(self) -> None:
                self.yield_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                self.yield_count += 1
                yield UnknownEntry()
                raise AssertionError("preflight buffered entries after first unknown")

        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(Path(tmp))
            real_scandir = os.scandir
            scanner = Scanner()

            def controlled_scandir(path):
                if Path(path).resolve() == candidate.resolve():
                    return scanner
                return real_scandir(path)

            with patch(
                "true_rl_agent.candidates.package_contract.os.scandir",
                side_effect=controlled_scandir,
            ), self.assertRaisesRegex(ValueError, "forbidden path"):
                preflight_candidate_package(candidate)

        self.assertEqual(scanner.yield_count, 1)

    def test_verifier_snapshots_bound_files_before_manifest_and_worker_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(Path(tmp))
            original = candidate / "candidate_manifest.yaml"
            replacement = Path(tmp) / "replacement.yaml"
            replacement.write_bytes(
                b"x" * (CANDIDATE_PACKAGE_FILE_MAX_BYTES["candidate_manifest.yaml"] + 1)
            )
            real_copy = (
                __import__(
                    "true_rl_agent.candidates.package_contract",
                    fromlist=["copy_preflighted_candidate_package"],
                ).copy_preflighted_candidate_package
            )

            def copy_then_swap(preflight, destination):
                evidence = real_copy(preflight, destination)
                original.unlink()
                replacement.replace(original)
                return evidence

            with patch(
                "true_rl_agent.candidates.verifier.copy_preflighted_candidate_package",
                side_effect=copy_then_swap,
            ):
                result = StaticCandidateVerifier().verify(candidate)

        self.assertEqual(result.status, "fail")
        self.assertEqual(
            result.details["issues"],
            ["file_too_large:candidate_manifest.yaml:2049>2048"],
        )

    def test_sealing_uses_fixed_paths_without_recursive_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            candidate = CandidatePackageRef(
                candidate_id="candidate_reward",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            with patch.object(
                Path,
                "rglob",
                side_effect=AssertionError("candidate packages must not use rglob"),
            ):
                artifact = seal_candidate_package(
                    candidate,
                    artifact_root=root / "artifacts",
                )
                self.assertEqual(
                    verify_sealed_candidate_artifact(artifact),
                    artifact,
                )

    def test_old_sealed_artifact_replay_does_not_apply_new_provider_byte_limits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            change_set = CandidateChangeSet(frozenset({"reward_callable"}))
            artifact = seal_candidate_package(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=change_set,
                ),
                artifact_root=root / "artifacts",
            )
            sealed_preflight = preflight_candidate_package(
                artifact.path,
                sealed_artifact=True,
                enforce_size_limits=False,
            )
            self.assertIsNotNone(sealed_preflight.sealed_manifest)
            self.assertNotIn(
                CANDIDATE_ARTIFACT_MANIFEST_FILENAME,
                {state.relative_path for state in sealed_preflight.files},
            )
            legacy_path = root / "legacy-sealed-artifact"
            shutil.copytree(artifact.path, legacy_path)
            source_path = legacy_path / "training_code" / "components.py"
            source_payload = source_path.read_bytes()
            source_payload += b"#" * (
                CANDIDATE_PACKAGE_FILE_MAX_BYTES["training_code/components.py"]
                + 1
                - len(source_payload)
            )
            source_path.write_bytes(source_payload)
            identity = resolve_candidate_program_identity(
                legacy_path,
                expected_candidate_id="candidate_reward",
                sealed_artifact=True,
                enforce_package_size_limits=False,
            )
            manifest_path = legacy_path / CANDIDATE_ARTIFACT_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in manifest["files"]:
                if item["path"] == "training_code/components.py":
                    item["checksum"] = (
                        "sha256:" + hashlib.sha256(source_payload).hexdigest()
                    )
                    item["size_bytes"] = len(source_payload)
            manifest["subject_manifest_hash"] = identity.manifest_hash
            manifest["program_source_checksum"] = identity.source_checksum
            manifest["program_source_size_bytes"] = identity.source_size_bytes
            manifest["artifact_hash"] = canonical_json_hash(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "artifact_hash"
                }
            )
            manifest_path.write_bytes(
                canonical_json_bytes(manifest, trailing_newline=True)
            )

            replayed = load_sealed_candidate_artifact(
                legacy_path,
                change_set=change_set,
                expected_candidate_id="candidate_reward",
            )
            replay_verification = StaticCandidateVerifier().verify(replayed)

        self.assertNotEqual(replayed.program_source_checksum, artifact.program_source_checksum)
        self.assertGreater(
            replayed.program_source_size_bytes,
            CANDIDATE_PACKAGE_FILE_MAX_BYTES["training_code/components.py"],
        )
        self.assertEqual(replayed.path, str(legacy_path.resolve()))
        self.assertEqual(replay_verification.status, "pass")

    def test_old_sparse_sealed_artifact_replays_only_its_declared_fixed_subset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            change_set = CandidateChangeSet(frozenset({"reward_callable"}))
            artifact = seal_candidate_package(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=change_set,
                ),
                artifact_root=root / "artifacts",
            )
            legacy_path = root / "legacy-sparse-sealed-artifact"
            shutil.copytree(artifact.path, legacy_path)
            for relative in (
                "rationale.md",
                "operation_diff.json",
                "diagnostics.json",
            ):
                (legacy_path / relative).unlink()
            manifest_path = legacy_path / CANDIDATE_ARTIFACT_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"] = [
                item
                for item in manifest["files"]
                if item["path"]
                in {"candidate_manifest.yaml", "training_code/components.py"}
            ]
            manifest["artifact_hash"] = canonical_json_hash(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "artifact_hash"
                }
            )
            manifest_path.write_bytes(
                canonical_json_bytes(manifest, trailing_newline=True)
            )

            replayed = load_sealed_candidate_artifact(
                legacy_path,
                change_set=change_set,
                expected_candidate_id="candidate_reward",
            )
            replay_verification = StaticCandidateVerifier().verify(replayed)

        self.assertEqual(replayed.path, str(legacy_path.resolve()))
        self.assertEqual(replay_verification.status, "pass")

    def test_sealed_artifact_manifest_is_bounded_before_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            change_set = CandidateChangeSet(frozenset({"reward_callable"}))
            artifact = seal_candidate_package(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=change_set,
                ),
                artifact_root=root / "artifacts",
            )
            manifest_path = (
                Path(artifact.path) / CANDIDATE_ARTIFACT_MANIFEST_FILENAME
            )
            manifest_path.write_bytes(
                b"x" * (SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES + 1)
            )

            with patch(
                "true_rl_agent.candidates.artifact.strict_json_object_loads",
                side_effect=AssertionError("oversized manifest must not be parsed"),
            ), self.assertRaises(CandidatePackageContractError) as caught:
                load_sealed_candidate_artifact(
                    artifact.path,
                    change_set=change_set,
                    expected_candidate_id="candidate_reward",
                )

        self.assertEqual(
            caught.exception.issue,
            (
                "file_too_large:.true_rl_agent_candidate_artifact.json:"
                f"{SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES + 1}>"
                f"{SEALED_CANDIDATE_ARTIFACT_MANIFEST_MAX_BYTES}"
            ),
        )

    def test_sealed_artifact_manifest_identity_is_bound_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            change_set = CandidateChangeSet(frozenset({"reward_callable"}))
            artifact = seal_candidate_package(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=change_set,
                ),
                artifact_root=root / "artifacts",
            )
            manifest_path = (
                Path(artifact.path) / CANDIDATE_ARTIFACT_MANIFEST_FILENAME
            )
            replacement = root / "replacement-artifact-manifest.json"
            replacement.write_bytes(manifest_path.read_bytes())
            swapped = False

            def swap_then_open(preflight, relative_path):
                nonlocal swapped
                if (
                    not swapped
                    and relative_path == CANDIDATE_ARTIFACT_MANIFEST_FILENAME
                ):
                    replacement.replace(manifest_path)
                    swapped = True
                return open_preflighted_candidate_file(preflight, relative_path)

            with patch(
                "true_rl_agent.candidates.artifact.open_preflighted_candidate_file",
                side_effect=swap_then_open,
            ), patch(
                "true_rl_agent.candidates.artifact.strict_json_object_loads",
                side_effect=AssertionError("replaced manifest must not be parsed"),
            ), self.assertRaisesRegex(
                CandidatePackageContractError,
                "changed after preflight",
            ):
                load_sealed_candidate_artifact(
                    artifact.path,
                    change_set=change_set,
                    expected_candidate_id="candidate_reward",
                )

        self.assertTrue(swapped)

    def test_provider_created_runner_manifest_is_rejected_before_sealing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            (candidate_dir / CANDIDATE_ARTIFACT_MANIFEST_FILENAME).write_text(
                "{}\n",
                encoding="utf-8",
            )
            candidate = CandidatePackageRef(
                candidate_id="candidate_reward",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(
                    frozenset({"reward_callable"})
                ),
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertIn(
                "forbidden_file:.true_rl_agent_candidate_artifact.json",
                result.details["issues"],
            )
            with self.assertRaisesRegex(ValueError, "reserved artifact manifest"):
                seal_candidate_package(
                    candidate,
                    artifact_root=root / "artifacts",
                )

    def test_sealed_artifact_rejects_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = self._write_candidate(root)
            artifact = seal_candidate_package(
                CandidatePackageRef(
                    candidate_id="candidate_reward",
                    path=str(candidate_dir),
                    change_set=CandidateChangeSet(
                        frozenset({"reward_callable"})
                    ),
                ),
                artifact_root=root / "artifacts",
            )
            source = Path(artifact.path) / "training_code" / "components.py"
            source.write_text("def build_training_components():\n    return None\n")

            with self.assertRaisesRegex(ValueError, "file evidence mismatch"):
                verify_sealed_candidate_artifact(artifact)

    def test_static_verifier_binds_package_ref_id_to_loaded_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))
            candidate = CandidatePackageRef(
                candidate_id="different_candidate",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            result = StaticCandidateVerifier().verify(candidate)

        self.assertEqual(result.status, "fail")
        self.assertIn("manifest_candidate_id_mismatch", result.details["issues"])

    def test_static_verifier_accepts_reward_inputs_type_import_exposed_in_prompt_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import RewardInputs, RewardTerm, TrainingComponents


                    def typed_reward(inputs: RewardInputs):
                        return inputs.base_height


                    def build_training_components():
                        return TrainingComponents(
                            name="typed_reward_candidate",
                            reward_terms=[RewardTerm(name="typed_reward", weight=1.0, fn=typed_reward)],
                        )
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier().verify(candidate_dir)

        self.assertEqual(RewardInputs.field_names()[0], "base_height")
        self.assertEqual(result.status, "pass")
        self.assertEqual(result.details["derived_change_set"], ["reward_callable"])

    def test_common_reward_inputs_are_task_neutral_and_push_task_extends_them(self) -> None:
        from true_rl_agent.tasks.push_recovery.reward_input_contract import (
            push_recovery_reward_input_fields,
        )
        from true_rl_agent.tasks.push_recovery.reward_inputs import (
            PushRecoveryRewardInputs,
        )

        common_fields = RewardInputs.field_names()
        push_fields = PushRecoveryRewardInputs.field_names()
        task_contract_fields = push_recovery_reward_input_fields()

        self.assertNotIn("push_active", common_fields)
        self.assertNotIn("push_vec_public", common_fields)
        self.assertIn("push_active", push_fields)
        self.assertIn("push_vec_public", push_fields)
        for name in common_fields:
            self.assertIn(name, push_fields)
            self.assertIn(name, task_contract_fields)
        self.assertLess(task_contract_fields.index("action_delta"), task_contract_fields.index("push_active"))
        self.assertLess(task_contract_fields.index("push_vec_public"), task_contract_fields.index("progress_fraction"))

    def test_static_verifier_tolerates_bom_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))
            component_path = candidate_dir / "training_code" / "components.py"
            component_path.write_text("\ufeff" + component_path.read_text(encoding="utf-8"), encoding="utf-8")

            result = StaticCandidateVerifier().verify(candidate_dir)

            self.assertEqual(result.status, "pass")

    def test_static_verifier_rejects_runtime_cache_as_unknown_package_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))
            pycache = candidate_dir / "training_code" / "__pycache__"
            pycache.mkdir()
            (pycache / "components.cpython-312.pyc").write_bytes(b"cache")

            result = StaticCandidateVerifier().verify(candidate_dir)

        self.assertEqual(result.status, "fail")
        self.assertEqual(
            result.details["issues"],
            ["forbidden_file:training_code/__pycache__"],
        )

    def test_static_verifier_rejects_entrypoint_escape_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(Path(tmp))
            (candidate / "candidate_manifest.yaml").write_text(
                textwrap.dedent(
                    """
                    schema_version: true_rl_agent.candidate.v1
                    candidate_id: candidate_reward
                    entrypoint: ../secret:build_training_components
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertIn("manifest_load_failed:ValueError", result.details["issues"])

    def test_static_verifier_rejects_dynamic_import_and_top_level_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import TrainingComponents

                    leaked = __import__("os")


                    def build_training_components():
                        return TrainingComponents(name="candidate_reward")
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertIn("forbidden_top_level:training_code/components.py:Assign", result.details["issues"])
            self.assertIn("forbidden_call:training_code/components.py:__import__", result.details["issues"])

    def test_static_verifier_rejects_non_callable_reward_term(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


                    def build_training_components():
                        return TrainingComponents(
                            name="candidate_reward",
                            reward_terms=[RewardTerm(name="bad_reward", weight=1.0, fn=1)],
                        )
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertEqual(result.reason, "training_components_invalid")
            self.assertIn("non_callable_reward_term:bad_reward", result.details["issues"])

    def test_static_verifier_enforces_shared_component_name_bound(self) -> None:
        valid_name = "n" * MAX_CANONICAL_COMPONENT_NAME_CHARS
        invalid_name = valid_name + "n"
        for name, expected_status in (
            (valid_name, "pass"),
            ("界" * 21, "pass"),
            ("界" * 22, "fail"),
            (invalid_name, "fail"),
        ):
            with self.subTest(name_length=len(name)):
                with tempfile.TemporaryDirectory() as tmp:
                    candidate = self._write_candidate(
                        Path(tmp),
                        components_source=textwrap.dedent(
                            f"""
                            from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


                            def reward(inputs):
                                return inputs.base_height


                            def build_training_components():
                                return TrainingComponents(
                                    name="candidate_reward",
                                    reward_terms=[RewardTerm(name={name!r}, weight=1.0, fn=reward)],
                                )
                            """
                        ).strip(),
                    )

                    result = StaticCandidateVerifier().verify(candidate)

                self.assertEqual(result.status, expected_status)
                if expected_status == "fail":
                    self.assertEqual(result.reason, "training_components_invalid")
                    self.assertEqual(
                        result.details["issues"],
                        ["invalid_reward_term_name"],
                    )

    def test_static_verifier_times_out_hanging_component_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import TrainingComponents


                    def build_training_components():
                        while True:
                            pass
                        return TrainingComponents(name="unreachable")
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier(component_load_timeout_s=0.25).verify(candidate)

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.reason, "training_components_load_timeout")
        self.assertIn("component_load_timeout", result.details["issues"])

    def test_static_verifier_policy_binds_instance_execution_timeout(self) -> None:
        short = StaticCandidateVerifier(component_load_timeout_s=0.25)
        long = StaticCandidateVerifier(component_load_timeout_s=120.0)

        self.assertNotEqual(
            short.admission_policy_descriptor().policy_hash,
            long.admission_policy_descriptor().policy_hash,
        )
        self.assertEqual(
            short.admission_policy_descriptor().policy_hash,
            static_verifier_policy_hash(component_load_timeout_s=0.25),
        )
        self.assertEqual(
            static_verifier_source_policy_hash(),
            "310e04b305b2744a1ea62a6d5416e993905bdff401ffd33187fa1028bc402275",
        )
        self.assertEqual(
            static_verifier_policy_hash(),
            "17e48a15617337729f47783f32d7b8940b4b25d06ef8632ee0e486c05e3add74",
        )

    def test_static_verifier_policy_cannot_enable_additional_files(self) -> None:
        policy = StaticCandidateVerifierPolicy()

        with self.assertRaisesRegex(
            ValueError,
            "additional candidate package files are not supported",
        ):
            replace(policy, additional_files_allowed=True)
        self.assertFalse(
            StaticCandidateVerifier(policy=policy).authoring_prompt_rules()[
                "additional_files_allowed"
            ]
        )

    def test_static_verifier_policy_cannot_change_fixed_package_files(self) -> None:
        default_policy = StaticCandidateVerifierPolicy()

        with self.assertRaisesRegex(ValueError, "fixed package contract"):
            replace(default_policy, runner_internal_files=frozenset())
        with self.assertRaisesRegex(ValueError, "fixed package contract"):
            replace(default_policy, provider_authored_files=frozenset())

    def test_static_verifier_policy_and_binding_are_immutable(self) -> None:
        policy = StaticCandidateVerifierPolicy()
        verifier = StaticCandidateVerifier(policy=policy)

        with self.assertRaises(FrozenInstanceError):
            policy.additional_files_allowed = True
        with self.assertRaises(FrozenInstanceError):
            verifier.policy = policy

    def test_static_verifier_policy_is_spawn_serializable(self) -> None:
        policy = StaticCandidateVerifierPolicy()

        restored = pickle.loads(pickle.dumps(policy))

        self.assertEqual(restored, policy)
        self.assertEqual(
            static_verifier_policy_hash(policy=restored),
            static_verifier_policy_hash(policy=policy),
        )
        with self.assertRaises(TypeError):
            restored.allowed_imports["policy_probe"] = frozenset({"PolicyProbe"})

    def test_static_verifier_policy_snapshot_rejects_tampering(self) -> None:
        policy = StaticCandidateVerifierPolicy()
        snapshot = policy.to_snapshot_payload()

        self.assertEqual(
            StaticCandidateVerifierPolicy.from_snapshot_payload(snapshot),
            policy,
        )

        tampered = {
            **snapshot,
            "policy_facts": {
                **snapshot["policy_facts"],
                "additional_files_allowed": True,
            },
        }
        with self.assertRaisesRegex(ValueError, "snapshot is invalid"):
            StaticCandidateVerifierPolicy.from_snapshot_payload(tampered)

    def test_every_static_policy_field_is_bound_by_executable_hash(self) -> None:
        policy = StaticCandidateVerifierPolicy()
        base_hash = static_verifier_policy_hash(policy=policy)

        for policy_field in fields(policy):
            value = getattr(policy, policy_field.name)
            if isinstance(value, bool):
                changed_value = not value
            elif isinstance(value, float):
                changed_value = value + 1.0
            elif isinstance(value, str):
                changed_value = value + "policy_probe"
            elif isinstance(value, tuple):
                changed_value = value + ("policy_probe",)
            elif isinstance(value, frozenset):
                changed_value = value | {"policy_probe"}
            else:
                changed_value = {
                    **dict(value),
                    "policy_probe": frozenset({"PolicyProbe"}),
                }
            with self.subTest(policy_field=policy_field.name):
                if policy_field.name in {
                    "additional_files_allowed",
                    "provider_authored_files",
                    "runner_internal_files",
                }:
                    with self.assertRaises(ValueError):
                        replace(policy, **{policy_field.name: changed_value})
                    continue
                changed_policy = replace(policy, **{policy_field.name: changed_value})
                self.assertNotEqual(
                    static_verifier_policy_hash(policy=changed_policy),
                    base_hash,
                )
                self.assertEqual(
                    StaticCandidateVerifierPolicy.from_snapshot_payload(
                        changed_policy.to_snapshot_payload()
                    ),
                    changed_policy,
                )

    def test_static_worker_startup_budget_is_not_candidate_policy(self) -> None:
        short = StaticCandidateVerifier(worker_startup_timeout_s=0.25)
        long = StaticCandidateVerifier(worker_startup_timeout_s=120.0)

        self.assertEqual(
            short.admission_policy_descriptor(),
            long.admission_policy_descriptor(),
        )

    def test_static_verifier_rejects_non_finite_timeouts(self) -> None:
        for value in (True, float("nan"), float("inf"), 0.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive finite"):
                    StaticCandidateVerifier(component_load_timeout_s=value)

    def test_static_verifier_rejects_unallowlisted_schedule_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                components_source=textwrap.dedent(
                    """
                    from true_rl_agent.candidates.components import ScheduleTerm, TrainingComponents


                    def build_training_components():
                        return TrainingComponents(
                            name="candidate_schedule",
                            schedules=[ScheduleTerm(name="obs_schedule", key="obs_transform", value=1.0)],
                        )
                    """
                ).strip(),
            )

            result = StaticCandidateVerifier().verify(candidate)

            self.assertEqual(result.status, "fail")
            self.assertEqual(result.reason, "training_components_invalid")
            self.assertIn("schedule_key_not_allowlisted:obs_schedule:obs_transform", result.details["issues"])

    def test_static_verifier_rejects_mutation_and_process_state_surfaces(self) -> None:
        cases = {
            "tensor_in_place": "inputs.base_height.zero_()",
            "subscript_write": "inputs.base_height[0] = 0.0",
            "attribute_write": "inputs.base_height.data = inputs.base_height",
            "dynamic_builtins": (
                "def reward(inputs):\n"
                "    return __builtins__['__import__']('os')"
            ),
            "dynamic_reflection": (
                "def reward(inputs):\n"
                "    return getattr(inputs, 'base_height')"
            ),
            "class_mutation": "setattr(TrainingComponents, 'leaked', True)",
            "console_side_effect": "print('candidate side effect')",
            "global_state": "global leaked",
            "nonlocal_state": (
                "def outer():\n"
                "    leaked = 0\n"
                "    def reward(inputs):\n"
                "        nonlocal leaked\n"
                "        return inputs.base_height\n"
                "    return reward\n"
            ),
            "mutable_default": "def helper(values=[]):\n    return values",
        }
        for label, injected in cases.items():
            with self.subTest(label=label):
                source = (
                    "from true_rl_agent.candidates.components import "
                    "TrainingComponents\n\n"
                    f"{injected}\n\n"
                    "def build_training_components():\n"
                    "    return TrainingComponents(name='candidate_reward')\n"
                )
                issues = audit_candidate_python_source(source)
                self.assertTrue(issues, label)

    def test_static_verifier_requires_exact_source_defined_stateless_callables(self) -> None:
        cases = {
            "builtin": (
                "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
                "def build_training_components():\n"
                "    return TrainingComponents(\n"
                "        name='candidate_reward',\n"
                "        reward_terms=[RewardTerm(name='bad', weight=1.0, fn=abs)],\n"
                "    )\n"
            ),
            "closure": (
                "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
                "def build_training_components():\n"
                "    state = iter((0.0, 1.0))\n"
                "    reward = lambda inputs: inputs.base_height + next(state)\n"
                "    return TrainingComponents(\n"
                "        name='candidate_reward',\n"
                "        reward_terms=[RewardTerm(name='bad', weight=1.0, fn=reward)],\n"
                "    )\n"
            ),
        }
        for label, source in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                candidate = self._write_candidate(
                    Path(tmp),
                    components_source=source,
                )
                result = StaticCandidateVerifier().verify(candidate)

                self.assertEqual(result.status, "fail")
                self.assertEqual(result.reason, "training_components_invalid")
                self.assertTrue(
                    any(
                        issue.startswith(
                            (
                                "non_source_defined_reward_term:",
                                "stateful_reward_term_closure:",
                            )
                        )
                        for issue in result.details["issues"]
                    ),
                    result.details,
                )

    def test_static_verifier_rejects_random_api_variants(self) -> None:
        sources = (
            "def reward(inputs):\n    return random.random()\n",
            "def reward(inputs):\n    return np.random.normal()\n",
            "def reward(inputs):\n    return torch.randn(1)\n",
            "def reward(inputs):\n    return inputs.base_height.bernoulli()\n",
        )
        for source in sources:
            with self.subTest(source=source):
                issues = audit_candidate_python_source(source)
                self.assertTrue(
                    any(issue.startswith("forbidden_random_call:") for issue in issues),
                    issues,
                )

    def test_static_verifier_policy_is_exposed_from_one_source(self) -> None:
        prompt_rules = static_verifier_prompt_rules()

        self.assertEqual(
            prompt_rules["verification_policy_id"],
            STATIC_CANDIDATE_VERIFIER_POLICY_ID,
        )
        self.assertEqual(
            prompt_rules["verification_policy_hash"],
            static_verifier_policy_hash(),
        )
        facts = static_verifier_policy_facts()
        self.assertEqual(
            prompt_rules["blocked_ast_nodes"],
            facts["forbidden_ast_nodes"],
        )
        self.assertEqual(
            prompt_rules["blocked_assignment_target_nodes"],
            facts["forbidden_assignment_target_nodes"],
        )
        self.assertEqual(
            prompt_rules["blocked_name_prefixes"],
            facts["forbidden_name_prefixes"],
        )
        self.assertEqual(
            prompt_rules["component_callable_contract"],
            facts["component_callable_contract"],
        )
        self.assertFalse(prompt_rules["function_decorators_allowed"])
        self.assertFalse(prompt_rules["nested_functions_allowed"])
        self.assertFalse(prompt_rules["mutable_or_dynamic_defaults_allowed"])
        nested_policy = replace(
            StaticCandidateVerifierPolicy(),
            nested_functions_allowed=True,
        )
        self.assertNotEqual(
            static_verifier_policy_hash(policy=nested_policy),
            prompt_rules["verification_policy_hash"],
        )
        self.assertIn("bernoulli", prompt_rules["blocked_random_calls"])
        self.assertIn("zero_", audit_candidate_python_source("def f(x):\n    return x.zero_()\n")[0])


if __name__ == "__main__":
    unittest.main()
