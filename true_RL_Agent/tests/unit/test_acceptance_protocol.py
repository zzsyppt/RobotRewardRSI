from __future__ import annotations

import unittest

from true_rl_agent.evaluation.acceptance_protocol import (
    ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
    NOISE_AWARE_DELTA_RULE,
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)


class AcceptanceProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = AcceptanceProtocol(
            protocol_id="test-policy-v1",
            score_source="public_accept.score_stats.mean",
            min_score_count=2,
            absolute_accept_floor=0.25,
            min_delta=0.05,
            public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
            decision_rule=NOISE_AWARE_DELTA_RULE,
        )

    def test_public_mapping_round_trip_preserves_protocol_and_hash(self) -> None:
        payload = self.protocol.to_public_dict()

        restored = AcceptanceProtocol.from_mapping(payload)

        self.assertEqual(restored, self.protocol)
        self.assertEqual(restored.protocol_hash, self.protocol.protocol_hash)
        self.assertEqual(
            payload["schema_version"], ACCEPTANCE_PROTOCOL_SCHEMA_VERSION
        )
        self.assertEqual(
            payload["acceptance_protocol_hash"], self.protocol.protocol_hash
        )

    def test_from_mapping_rejects_non_mapping_inputs_as_value_error(self) -> None:
        for malformed in (None, [], "protocol", 1, True):
            with self.subTest(malformed=malformed):
                with self.assertRaisesRegex(
                    ValueError, "acceptance protocol must be a mapping"
                ):
                    AcceptanceProtocol.from_mapping(malformed)

    def test_from_mapping_rejects_missing_or_wrong_schema(self) -> None:
        for schema in (None, "true_rl_agent.acceptance.protocol.v1"):
            payload = self.protocol.to_public_dict()
            payload["schema_version"] = schema
            with self.subTest(schema=schema):
                with self.assertRaisesRegex(ValueError, "schema_version mismatch"):
                    AcceptanceProtocol.from_mapping(payload)

    def test_from_mapping_rejects_missing_or_malformed_hash(self) -> None:
        malformed_hashes = (None, "", 123, "sha256:forged")
        for declared_hash in malformed_hashes:
            payload = self.protocol.to_public_dict()
            if declared_hash is None:
                payload.pop("acceptance_protocol_hash")
            else:
                payload["acceptance_protocol_hash"] = declared_hash
            with self.subTest(declared_hash=declared_hash):
                with self.assertRaises(ValueError):
                    AcceptanceProtocol.from_mapping(payload)

    def test_from_mapping_hash_binds_all_fixed_contract_fields(self) -> None:
        changes = {
            "protocol_id": "other-policy",
            "score_source": "other.score.mean",
            "min_score_count": 3,
            "absolute_accept_floor": 0.3,
            "min_delta": 0.1,
        }
        for field_name, changed_value in changes.items():
            payload = self.protocol.to_public_dict()
            payload[field_name] = changed_value
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    AcceptanceProtocol.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
