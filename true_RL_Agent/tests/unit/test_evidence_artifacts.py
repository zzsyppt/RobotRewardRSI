import json
import unittest

from true_rl_agent.runner.evidence import (
    EvidenceArtifact,
    evidence_artifact_from_json,
    validate_evidence_artifact_manifest,
    validate_evidence_projection_bindings,
)


class EvidenceArtifactTests(unittest.TestCase):
    def test_evidence_artifact_rejects_malformed_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            EvidenceArtifact(
                artifact_id="malformed-checkpoint",
                kind="checkpoint",
                sha256="sha256:not-a-digest",
                byte_size=1,
                visibility="private_only",
                producer="runner_eval",
            )

    def test_json_evidence_artifact_is_hash_only_and_public_safe(self) -> None:
        artifact = evidence_artifact_from_json(
            artifact_id="checkpoint-safety-report",
            kind="checkpoint_safety_report",
            payload={"status": "pass", "checkpoint_hash": "sha256:abc"},
            producer="isaaclab_rsl_rl_eval",
            visibility="public_projection_only",
            public_projection_ref="metrics.checkpoint_safety_report",
        )

        record = artifact.to_public_dict()

        self.assertEqual(record["schema_version"], "true_rl_agent.evidence.artifact.v1")
        self.assertEqual(record["kind"], "checkpoint_safety_report")
        self.assertEqual(record["mime_type"], "application/json")
        self.assertEqual(record["visibility"], "public_projection_only")
        self.assertEqual(record["public_projection_ref"], "metrics.checkpoint_safety_report")
        self.assertTrue(str(record["sha256"]).startswith("sha256:"))
        self.assertGreater(record["byte_size"], 0)
        text = json.dumps(record, sort_keys=True)
        self.assertNotIn("checkpoint_hash\": \"sha256:abc", text)
        self.assertNotIn("path", text.lower())
        self.assertNotIn("://", text)

    def test_multimodal_artifact_manifest_shape_has_no_path_or_raw_payload(self) -> None:
        artifact = EvidenceArtifact(
            artifact_id="round-001-rendered-video",
            kind="rendered_video",
            sha256="sha256:" + "a" * 64,
            byte_size=1024,
            visibility="private_only",
            producer="runner_eval",
            mime_type="video/mp4",
            public_projection_ref="metrics.video_summary",
        )

        manifest = validate_evidence_artifact_manifest([artifact.to_public_dict()])

        self.assertEqual(manifest[0]["kind"], "rendered_video")
        self.assertEqual(manifest[0]["mime_type"], "video/mp4")
        text = json.dumps(manifest, sort_keys=True)
        self.assertNotIn("C:", text)
        self.assertNotIn("/tmp", text)
        self.assertNotIn("raw_payload", text)

    def test_public_projection_manifest_is_bound_to_canonical_payload_bytes(self) -> None:
        payload = {"status": "pass", "checkpoint_hash": "sha256:abc"}
        artifact = evidence_artifact_from_json(
            artifact_id="checkpoint-safety-report",
            kind="checkpoint_safety_report",
            payload=payload,
            producer="runner_eval",
            visibility="public_projection_only",
            public_projection_ref="metrics.checkpoint_safety_report",
        ).to_public_dict()

        validated = validate_evidence_projection_bindings(
            [artifact],
            projections={"metrics.checkpoint_safety_report": payload},
        )

        self.assertEqual(validated, [artifact])
        with self.assertRaisesRegex(ValueError, "projection binding mismatch"):
            validate_evidence_projection_bindings(
                [{**artifact, "byte_size": artifact["byte_size"] + 1}],
                projections={"metrics.checkpoint_safety_report": payload},
            )

    def test_evidence_manifest_rejects_path_url_or_raw_payload_keys(self) -> None:
        base = EvidenceArtifact(
            artifact_id="diagnostic-image",
            kind="diagnostic_image",
            sha256="sha256:" + "b" * 64,
            byte_size=256,
            visibility="private_only",
            producer="runner_eval",
            mime_type="image/png",
        ).to_public_dict()

        for forbidden_key in ("path", "uri", "url", "raw_payload", "data"):
            with self.subTest(forbidden_key=forbidden_key):
                item = dict(base)
                item[forbidden_key] = "C:/secret/artifact.bin"
                with self.assertRaisesRegex(ValueError, "forbidden evidence artifact field"):
                    validate_evidence_artifact_manifest([item])

    def test_evidence_manifest_rejects_missing_required_fields_as_value_error(self) -> None:
        base = EvidenceArtifact(
            artifact_id="diagnostic-image",
            kind="diagnostic_image",
            sha256="sha256:" + "b" * 64,
            byte_size=256,
            visibility="private_only",
            producer="runner_eval",
            mime_type="image/png",
        ).to_public_dict()

        for required_key in ("artifact_id", "kind", "sha256", "byte_size", "visibility", "producer"):
            with self.subTest(required_key=required_key):
                item = dict(base)
                del item[required_key]
                with self.assertRaisesRegex(ValueError, f"evidence artifact {required_key} is required"):
                    validate_evidence_artifact_manifest([item])

    def test_evidence_manifest_rejects_invalid_byte_size_types_as_value_error(self) -> None:
        base = EvidenceArtifact(
            artifact_id="diagnostic-image",
            kind="diagnostic_image",
            sha256="sha256:" + "b" * 64,
            byte_size=256,
            visibility="private_only",
            producer="runner_eval",
            mime_type="image/png",
        ).to_public_dict()

        for invalid_value in (
            None,
            [],
            object(),
            True,
            0,
            -1,
            1.25,
            "256",
            "1.25",
            "",
        ):
            with self.subTest(invalid_value=repr(invalid_value)):
                item = dict(base)
                item["byte_size"] = invalid_value
                with self.assertRaisesRegex(ValueError, "evidence artifact byte_size"):
                    validate_evidence_artifact_manifest([item])

    def test_evidence_artifact_constructor_rejects_invalid_byte_size_types_as_value_error(self) -> None:
        for invalid_value in (None, [], object(), True, 0, -1, 1.25, "256", "1.25", ""):
            with self.subTest(invalid_value=repr(invalid_value)):
                with self.assertRaisesRegex(ValueError, "evidence artifact byte_size"):
                    EvidenceArtifact(
                        artifact_id="diagnostic-image",
                        kind="diagnostic_image",
                        sha256="sha256:" + "b" * 64,
                        byte_size=invalid_value,
                        visibility="private_only",
                        producer="runner_eval",
                        mime_type="image/png",
                    )

    def test_evidence_manifest_rejects_coercible_string_fields(self) -> None:
        base = EvidenceArtifact(
            artifact_id="diagnostic-image",
            kind="diagnostic_image",
            sha256="sha256:" + "b" * 64,
            byte_size=256,
            visibility="private_only",
            producer="runner_eval",
            mime_type="image/png",
            public_projection_ref="metrics.image_summary",
        ).to_public_dict()

        for field_name in (
            "artifact_id",
            "kind",
            "sha256",
            "visibility",
            "producer",
            "mime_type",
            "public_projection_ref",
        ):
            invalid_values = (
                (True, 1, "")
                if field_name == "public_projection_ref"
                else (True, None, 1)
            )
            for invalid_value in invalid_values:
                with self.subTest(
                    field_name=field_name,
                    invalid_value=repr(invalid_value),
                ):
                    item = dict(base)
                    item[field_name] = invalid_value
                    with self.assertRaises(ValueError):
                        validate_evidence_artifact_manifest([item])


if __name__ == "__main__":
    unittest.main()
