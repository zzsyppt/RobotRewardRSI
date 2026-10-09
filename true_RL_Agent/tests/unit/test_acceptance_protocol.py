from __future__ import annotations

import math
import unittest

from true_rl_agent.evaluation.acceptance_protocol import (
    ACCEPTANCE_DELTA_ACCEPT_REASONS,
    ACCEPTANCE_DELTA_REJECT_REASONS,
    ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
    BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
    BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE,
    GUARDED_PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
    GUARDED_PAIRED_HEADROOM_DELTA_RULE,
    NOISE_AWARE_DELTA_RULE,
    PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
    PAIRED_HEADROOM_DELTA_RULE,
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    PUBLIC_SCORE_BINDING_GUARD_MEAN,
    AcceptanceProtocol,
    acceptance_delta_met,
    acceptance_delta_reason,
    acceptance_threshold,
    guard_regression_threshold,
    threshold_components_match,
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

    def test_threshold_component_matching_is_structurally_strict_and_ulp_tolerant(self) -> None:
        expected = {
            "decision_rule": PAIRED_HEADROOM_DELTA_RULE,
            "threshold": 0.006473060342997093,
            "min_delta": 0.001,
        }
        adjacent_runtime_result = {
            **expected,
            "threshold": 0.006473060342997092,
        }
        self.assertTrue(
            threshold_components_match(adjacent_runtime_result, expected)
        )
        expected_guard = {
            "decision_rule": BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE,
            "threshold": 0.028470123456789,
            "hard_cap": 0.05,
            "base_margin": 0.02,
            "paired_standard_error": 0.005646748971192667,
            "paired_noise_allowance": 0.008470123456789,
        }
        adjacent_guard_result = {
            **expected_guard,
            "paired_noise_allowance": math.nextafter(
                expected_guard["paired_noise_allowance"],
                math.inf,
            ),
        }
        self.assertTrue(
            threshold_components_match(adjacent_guard_result, expected_guard)
        )

        invalid = []
        invalid.append(
            {key: value for key, value in expected.items() if key != "min_delta"}
        )
        invalid.append({**expected, "extra": 0.0})
        invalid.append({**expected, "decision_rule": NOISE_AWARE_DELTA_RULE})
        invalid.append({**expected, "threshold": expected["threshold"] + 1.0e-6})
        invalid.append({**expected, "threshold": True})
        invalid.append({**expected, "threshold": math.nan})
        invalid.append({**expected, "threshold": math.inf})
        for declared in invalid:
            with self.subTest(declared=declared):
                self.assertFalse(threshold_components_match(declared, expected))

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

    def test_from_mapping_rejects_unbound_extra_fields(self) -> None:
        payload = self.protocol.to_public_dict()
        payload["score_ceiling"] = 1.0

        with self.assertRaisesRegex(ValueError, "fields mismatch"):
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

    def test_paired_headroom_protocol_round_trip_and_threshold(self) -> None:
        protocol = AcceptanceProtocol(
            protocol_id="paired-policy-v1",
            score_source="public_accept.score_stats.mean",
            min_score_count=5,
            absolute_accept_floor=0.2,
            min_delta=0.0,
            public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
            decision_rule=PAIRED_HEADROOM_DELTA_RULE,
            score_ceiling=1.0,
            headroom_reduction_fraction=0.2,
            paired_standard_error_multiplier=1.5,
        )
        payload = protocol.to_public_dict()

        self.assertEqual(
            payload["schema_version"],
            PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
        )
        self.assertEqual(AcceptanceProtocol.from_mapping(payload), protocol)

        best = [
            1.0,
            0.9642857142857143,
            0.9821428571428571,
            0.9821428571428571,
            0.9821428571428571,
        ]
        candidate = [1.0, 1.0, 0.9821428571428571, 0.9821428571428571, 1.0]
        result = acceptance_threshold(
            protocol,
            candidate_score=sum(candidate) / len(candidate),
            best_score=sum(best) / len(best),
            candidate_score_samples=candidate,
            best_score_samples=best,
        )

        self.assertAlmostEqual(result.paired_standard_error, 1.0 / 140.0)
        self.assertAlmostEqual(result.paired_noise_delta, 3.0 / 280.0)
        self.assertAlmostEqual(
            result.threshold,
            sum(candidate) / 5 - sum(best) / 5,
        )

    def test_old_protocol_hash_payload_remains_v2(self) -> None:
        payload = self.protocol.to_public_dict()

        self.assertEqual(payload["schema_version"], ACCEPTANCE_PROTOCOL_SCHEMA_VERSION)
        self.assertNotIn("score_ceiling", payload)
        self.assertEqual(AcceptanceProtocol.from_mapping(payload), self.protocol)

    def test_guarded_protocol_round_trip_binds_objective_and_guard(self) -> None:
        protocol = AcceptanceProtocol(
            protocol_id="guarded-policy-v1",
            score_source="acceptance_objective.score_stats.mean",
            min_score_count=5,
            absolute_accept_floor=0.2,
            min_delta=0.0,
            public_score_binding=PUBLIC_SCORE_BINDING_GUARD_MEAN,
            decision_rule=GUARDED_PAIRED_HEADROOM_DELTA_RULE,
            score_ceiling=1.0,
            headroom_reduction_fraction=0.2,
            paired_standard_error_multiplier=1.5,
            guard_score_source="public_accept.score_stats.mean",
            guard_absolute_accept_floor=0.2,
            guard_max_regression=0.0,
        )

        payload = protocol.to_public_dict()

        self.assertEqual(
            payload["schema_version"],
            GUARDED_PAIRED_HEADROOM_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
        )
        self.assertEqual(AcceptanceProtocol.from_mapping(payload), protocol)
        changed = dict(payload)
        changed["guard_max_regression"] = 0.01
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            AcceptanceProtocol.from_mapping(changed)

    def test_bounded_paired_guard_uses_margin_noise_and_hard_cap(self) -> None:
        protocol = AcceptanceProtocol(
            protocol_id="bounded-guard-policy-v1",
            score_source="acceptance_objective.score_stats.mean",
            min_score_count=5,
            absolute_accept_floor=0.2,
            min_delta=0.0,
            public_score_binding=PUBLIC_SCORE_BINDING_GUARD_MEAN,
            decision_rule=BOUNDED_PAIRED_GUARD_HEADROOM_DELTA_RULE,
            score_ceiling=1.0,
            headroom_reduction_fraction=0.2,
            paired_standard_error_multiplier=1.5,
            guard_score_source="public_accept.score_stats.mean",
            guard_absolute_accept_floor=0.2,
            guard_max_regression=0.05,
            guard_regression_margin=0.02,
            max_public_to_sealed_score_drop=0.05,
        )
        payload = protocol.to_public_dict()
        self.assertEqual(
            payload["schema_version"],
            BOUNDED_PAIRED_GUARD_ACCEPTANCE_PROTOCOL_SCHEMA_VERSION,
        )
        self.assertEqual(AcceptanceProtocol.from_mapping(payload), protocol)
        accept_reason = acceptance_delta_reason(protocol, accepted=True)
        reject_reason = acceptance_delta_reason(protocol, accepted=False)
        self.assertEqual(accept_reason, "bounded_paired_guard_headroom_delta")
        self.assertEqual(
            reject_reason,
            "bounded_paired_guard_headroom_delta_not_met",
        )
        self.assertIn(accept_reason, ACCEPTANCE_DELTA_ACCEPT_REASONS)
        self.assertIn(reject_reason, ACCEPTANCE_DELTA_REJECT_REASONS)

        best = [0.9821428571428571, 1.0, 1.0, 0.9821428571428571, 1.0]
        noisy = [0.0, 0.0, 0.0, 0.0, 0.25]
        result = guard_regression_threshold(
            protocol,
            candidate_guard_score=sum(noisy) / len(noisy),
            best_guard_score=sum(best) / len(best),
            candidate_guard_score_samples=noisy,
            best_guard_score_samples=best,
        )
        self.assertEqual(result.threshold, 0.05)
        self.assertGreater(
            result.base_margin + result.paired_noise_allowance,
            result.hard_cap,
        )

    def test_numeric_equality_guard_is_limited_to_the_new_rule(self) -> None:
        adaptive = AcceptanceProtocol(
            protocol_id="paired-policy-v1",
            score_source="public_accept.score_stats.mean",
            min_score_count=2,
            absolute_accept_floor=0.2,
            min_delta=0.0,
            public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
            decision_rule=PAIRED_HEADROOM_DELTA_RULE,
            score_ceiling=1.0,
            headroom_reduction_fraction=0.2,
            paired_standard_error_multiplier=1.5,
        )
        threshold = 0.1
        floating_successor = math.nextafter(threshold, math.inf)

        self.assertFalse(
            acceptance_delta_met(adaptive, floating_successor, threshold)
        )
        self.assertTrue(
            acceptance_delta_met(self.protocol, floating_successor, threshold)
        )


if __name__ == "__main__":
    unittest.main()
