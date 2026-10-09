from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.protocol import (
    EvaluationProtocol,
    validate_evaluation_protocol_mapping,
)


class EvaluationProtocolTests(unittest.TestCase):
    def test_protocol_hash_tracks_only_fixed_eval_contract_fields(self) -> None:
        base = EvaluationProtocol(
            eval_protocol_version="fixed_public_accept_sealed_validation_v1",
            public_seed_count=3,
            public_accept_seed_count=3,
            sealed_validation_seed_count=7,
            eval_num_envs=64,
        )
        changed_public = EvaluationProtocol(
            eval_protocol_version="fixed_public_accept_sealed_validation_v1",
            public_seed_count=4,
            public_accept_seed_count=3,
            sealed_validation_seed_count=7,
            eval_num_envs=64,
        )
        changed_sealed = EvaluationProtocol(
            eval_protocol_version="fixed_public_accept_sealed_validation_v1",
            public_seed_count=3,
            public_accept_seed_count=3,
            sealed_validation_seed_count=8,
            eval_num_envs=64,
        )
        changed_env = EvaluationProtocol(
            eval_protocol_version="fixed_public_accept_sealed_validation_v1",
            public_seed_count=3,
            public_accept_seed_count=3,
            sealed_validation_seed_count=7,
            eval_num_envs=128,
        )
        same_contract_with_candidate_metadata = EvaluationProtocol.from_mapping(
            {
                **base.to_public_dict(),
                "candidate_manifest_hash": "sha256:candidate-controlled",
                "provider_json_hash": "sha256:provider-controlled",
            }
        )

        self.assertEqual(base.protocol_hash, same_contract_with_candidate_metadata.protocol_hash)
        self.assertNotEqual(base.protocol_hash, changed_public.protocol_hash)
        self.assertNotEqual(base.protocol_hash, changed_sealed.protocol_hash)
        self.assertNotEqual(base.protocol_hash, changed_env.protocol_hash)
        self.assertNotIn("trusted_seed_count", base.to_public_dict())

    def test_protocol_rejects_missing_or_non_positive_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "sealed_validation_seed_count must be positive"):
            EvaluationProtocol(
                eval_protocol_version="fixed_public_accept_sealed_validation_v1",
                public_seed_count=3,
                public_accept_seed_count=3,
                sealed_validation_seed_count=0,
                eval_num_envs=64,
            )

    def test_protocol_rejects_coercible_and_forged_public_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "public_seed_count must be positive"):
            EvaluationProtocol(
                eval_protocol_version="fixed_public_accept_sealed_validation_v1",
                public_seed_count=True,
                public_accept_seed_count=3,
                sealed_validation_seed_count=7,
                eval_num_envs=64,
            )

        protocol = EvaluationProtocol(
            eval_protocol_version="fixed_public_accept_sealed_validation_v1",
            public_seed_count=3,
            public_accept_seed_count=3,
            sealed_validation_seed_count=7,
            eval_num_envs=64,
        ).to_public_dict()
        protocol["eval_protocol_hash"] = "sha256:forged"

        validation = validate_evaluation_protocol_mapping(protocol)

        self.assertFalse(validation.valid)
        self.assertIn("eval_protocol_hash_mismatch", validation.reasons)
        with self.assertRaisesRegex(ValueError, "eval_protocol_hash_mismatch"):
            EvaluationProtocol.from_mapping(protocol)

        with self.assertRaisesRegex(ValueError, "eval_protocol_version must be non-empty"):
            EvaluationProtocol(
                eval_protocol_version="",
                public_seed_count=3,
                public_accept_seed_count=3,
                sealed_validation_seed_count=7,
                eval_num_envs=64,
            )


if __name__ == "__main__":
    unittest.main()
