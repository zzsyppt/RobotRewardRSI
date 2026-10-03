"""Cards must preserve meanings, examples, source scope and sentence operands."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from tests.unit import test_stair_trajectory_input_inspection as fixture
from tests.unit.test_stair_trajectory_evidence_build import build as evidence_build
from tests.unit.test_trajectory_measurement import sample_episode
from tests.unit.test_trajectory_cohort import summary
from true_rl_agent.evaluation.trajectory_translation.measurement import measure_episode
from true_rl_agent.evaluation.trajectory_translation.contracts import stable_id
from true_rl_agent.evaluation.trajectory_translation.report import (
    CARD_VERSION, build_card, file_sha256, load_bundle, read_link, render_markdown,
    render_statement, validate_claims, make_claim, resolve_path, cohort_metric_literals,
)
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import iter_stair_episodes

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("build_policy_behavior_card", ROOT / "scripts/build_policy_behavior_card.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def recalculate_claim_id(claim):
    claim["claim_id"] = stable_id("claim", CARD_VERSION, {k: v for k, v in claim.items() if k != "claim_id"})


class CardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture.StairTrajectoryInputInspectionTests.setUpClass()
        cls.fixture = fixture.StairTrajectoryInputInspectionTests()
        cls.fixture.setUp()
        cls.reference = cls.fixture._archive()
        cls.path = cls.fixture.root / "p1a"
        evidence_build.build_evidence(cls.reference, cls.path, "fixture")
        cls.manifest_path = cls.path / "manifest.json"
        cls.bundle = load_bundle(cls.manifest_path, file_sha256(cls.manifest_path),
                                 iter_stair_episodes(cls.reference, source_id="fixture"))
        cls.selection = {"source_manifest_hashes": {"fixture": cls.reference["manifest_sha256"]},
                         "episodes": [{"source_id": "fixture", "run_index": 1, "env_index": 0},
                                      {"source_id": "fixture", "run_index": 1, "env_index": 2}]}
        cls.card, cls.evidence = build_card(cls.bundle, [], cls.selection)
        cls.files = {cls.bundle["manifest"]["evidence"]["sha256"]: cls.bundle["input"]}
        cls.records = {}
        for record in cls.evidence:
            if record["record_type"] == "evidence_link":
                cls.records[record["record_id"]] = read_link(record, cls.files)
            elif "record_id" in record:
                cls.records[record["record_id"]] = record
                if record["record_type"] == "episode_summary":
                    cls.records[record["activity"]["record_id"]] = record["activity"]

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def test_all_sentences_have_resolvable_operands_and_exact_reading_view(self):
        validate_claims(self.card["claims"], self.records)
        md = render_markdown(self.card)
        self.assertEqual(md.count("[claim:"), len(self.card["claims"]))
        for c in self.card["claims"]:
            self.assertIn(c["text"], md)
            self.assertTrue(c["operands"])

    def test_output_is_deterministic(self):
        card, evidence = build_card(self.bundle, [], self.selection)
        self.assertEqual(card, self.card)
        self.assertEqual(evidence, self.evidence)

    def test_policy_main_view_is_small_without_dropping_audit_claims(self):
        ids = self.card["policy_summary_claim_ids"]
        self.assertLessEqual(len(ids), 15)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(set(ids) <= {c["claim_id"] for c in self.card["claims"]})
        md = render_markdown(self.card)
        main, detail = md.split("## 展开审计明细", 1)
        self.assertEqual(main.count("[claim:"), len(ids))
        self.assertEqual(main.count("[claim:") + detail.count("[claim:"), len(self.card["claims"]))
        self.assertIn("最终请求留待P1c", main)
        self.assertIn("<details>", detail)

    def test_frozen_examples_not_silently_omitted_and_omissions_explicit(self):
        included = {e["episode_id"] for e in self.card["selected_examples"]}
        required = {e["episode_id"] for e in self.bundle["summaries"] if e["env_index"] in (0, 2)}
        self.assertTrue(required <= included)
        self.assertEqual(len(included) + len(self.card["omissions"]), 288)
        bad = deepcopy(self.selection)
        bad["episodes"][0]["env_index"] = 999
        with self.assertRaisesRegex(ValueError, "missing"):
            build_card(self.bundle, [], bad)

    def test_no_raw_paths_in_sentences_or_reading_view_and_p1c_explicit(self):
        text = render_markdown(self.card)
        for marker in (str(self.fixture.root), ".npz", "manifest_path", "evidence_path", "D:\\"):
            self.assertNotIn(marker, text)
        self.assertFalse(self.card["author_projection"]["actual_payload_checked"])
        self.assertIn("P1c", text)

    def test_false_numeric_sentence_cannot_pass_with_valid_schema(self):
        claim = deepcopy(next(c for c in self.card["claims"] if c["template_id"] == "cohort_scope"))
        claim["text"] = claim["text"].replace("288", "289", 1)
        recalculate_claim_id(claim)
        with self.assertRaisesRegex(ValueError, "resolved operands"):
            validate_claims([claim], self.records)

    def _generic_claim(self, template, key, scope="all"):
        return deepcopy(next(c for c in self.card["claims"] if c["template_id"] == template
                             and c["scope"] == scope and c["operands"]["metric"]["path"] == ["metrics", key]))

    def _rerender_and_rehash(self, claim, records=None):
        records = self.records if records is None else records
        slots = {**claim["literals"], **{name: resolve_path(records[o["record_id"]], o["path"])
                                       for name, o in claim["operands"].items()}}
        claim["text"] = render_statement(claim["template_id"], slots)
        recalculate_claim_id(claim)

    def test_cohort_four_parent_metric_substitutions_rejected_after_rerender_rehash(self):
        fl, _, rl, rr = self.bundle["binding"]["entities"]["support"]
        cases = [("scalar_distribution", "net_progress_m", "terminal_time_s", "all"),
                 ("presence", rl + ".low_speed_off.long_observed", rr + ".low_speed_off.long_observed", "success"),
                 ("ratio", fl + ".marker", fl + ".history_diagnostic", "all"),
                 ("off_runs", fl + ".marker", rr + ".marker", "all")]
        for template, key, target, scope in cases:
            with self.subTest(template=template):
                claim = self._generic_claim(template, key, scope)
                claim["operands"]["metric"]["path"] = ["metrics", target]
                self._rerender_and_rehash(claim)
                with self.assertRaisesRegex(ValueError, "cohort.*(mismatch|semantics)"):
                    validate_claims([claim], self.records)

    def test_cohort_label_unit_entity_and_combo_literal_mutations_rejected(self):
        fl, _, _, rr = self.bundle["binding"]["entities"]["support"]
        cases = [("scalar_distribution", "net_progress_m", {"label": "有效终点时刻"}),
                 ("scalar_distribution", "net_progress_m", {"unit": "s"}),
                 ("ratio", fl + ".marker", {"label": fl + " history ANY采样占比"}),
                 ("off_runs", fl + ".marker", {"entity": rr}),
                 ("ratio", "combination.0", {"label": "接触组合code=11"})]
        for template, key, literals in cases:
            with self.subTest(literals=literals):
                claim = self._generic_claim(template, key)
                claim["literals"].update(literals)
                self._rerender_and_rehash(claim)
                with self.assertRaisesRegex(ValueError, "label/entity/unit/role mismatch"):
                    validate_claims([claim], self.records)

    def test_cohort_extra_operand_cannot_override_derived_unit(self):
        claim = self._generic_claim("scalar_distribution", "net_progress_m")
        claim["operands"]["unit"] = {"record_id": claim["scope_record_id"], "path": ["metrics", "terminal_time_s", "unit"]}
        self._rerender_and_rehash(claim)
        with self.assertRaisesRegex(ValueError, "operand role mismatch"):
            validate_claims([claim], self.records)

    def test_cohort_metric_unit_must_match_its_descriptor(self):
        claim = self._generic_claim("scalar_distribution", "net_progress_m")
        scope = deepcopy(self.records[claim["scope_record_id"]])
        scope["metrics"]["net_progress_m"]["unit"] = "s"
        records = {**self.records, claim["scope_record_id"]: scope}
        claim["literals"]["unit"] = "s"
        self._rerender_and_rehash(claim, records)
        with self.assertRaisesRegex(ValueError, "unit differs from descriptor"):
            validate_claims([claim], records)

    def test_cohort_legal_metric_switch_derives_new_semantics(self):
        fl, _, rl, rr = self.bundle["binding"]["entities"]["support"]
        cases = [("scalar_distribution", "net_progress_m", "terminal_time_s"),
                 ("presence", rl + ".low_speed_off.long_observed", rr + ".low_speed_off.long_observed"),
                 ("ratio", fl + ".marker", fl + ".norm_share"),
                 ("off_runs", fl + ".marker", rr + ".marker"),
                 ("ratio", "combination.0", "combination.11")]
        for template, key, target in cases:
            with self.subTest(target=target):
                claim = self._generic_claim(template, key)
                claim["operands"]["metric"]["path"] = ["metrics", target]
                claim["literals"] = cohort_metric_literals(template, target, self.records[claim["scope_record_id"]]["metrics"])
                self._rerender_and_rehash(claim)
                validate_claims([claim], self.records)

    def test_wrong_entity_is_rejected_even_after_rehash(self):
        claim = deepcopy(next(c for c in self.card["claims"] if c["template_id"] == "episode_marker"))
        claim["literals"]["entity"] = "wrong-foot"
        value = self.records[claim["operands"]["value"]["record_id"]]["value"]
        claim["text"] = render_statement("episode_marker", {"entity": "wrong-foot", "value": value})
        recalculate_claim_id(claim)
        with self.assertRaisesRegex(ValueError, "entity"):
            validate_claims([claim], self.records)

    def test_same_foot_other_episode_cannot_be_substituted_under_original_scope(self):
        all_claims = [c for c in self.card["claims"] if c["template_id"] == "episode_marker"]
        claim = deepcopy(all_claims[0])
        other = next(c for c in all_claims if c["scope"] != claim["scope"] and c["literals"] == claim["literals"])
        claim["operands"] = deepcopy(other["operands"])
        claim["text"] = other["text"]
        recalculate_claim_id(claim)
        with self.assertRaisesRegex(ValueError, "episode/source scope"):
            validate_claims([claim], self.records)

    def test_force_entity_order_cannot_be_changed_as_a_free_label(self):
        claim = deepcopy(next(c for c in self.card["claims"] if c["template_id"] == "episode_force"))
        record = self.records[claim["operands"]["value"]["record_id"]]
        del claim["operands"]["entities"]
        claim["literals"]["entities"] = list(reversed(record["entity_ids"]))
        claim["text"] = render_statement("episode_force", {"value": record["value"], **claim["literals"]})
        recalculate_claim_id(claim)
        with self.assertRaisesRegex(ValueError, "entity order"):
            validate_claims([claim], self.records)

    def test_missing_attitude_endpoints_are_called_first_last_valid_records(self):
        e = sample_episode(4)
        e.arrays["base_quat_w"][[0, 3]] = np.nan
        e.arrays["base_quat_w"][2] = [0, 1, 0, 0]
        value = next(r["value"] for r in measure_episode(e) if r["record_type"] == "measurement" and r["key"] == "body_up")
        text = render_statement("episode_attitude", {"value": value})
        self.assertIn("首个/末个有效记录值=1/-1（步2/3）", text)
        self.assertIn("姿态未知2条", text)

    def test_current_history_swap_is_not_accepted_by_structural_validation(self):
        claim = deepcopy(next(c for c in self.card["claims"] if c["template_id"] == "episode_history"))
        original = self.records[claim["operands"]["value"]["record_id"]]
        bad_record = deepcopy(original)
        bad_record["key"] = bad_record["key"].replace(".history_diagnostic", ".marker")
        records = {**self.records, original["record_id"]: bad_record}
        with self.assertRaisesRegex(ValueError, "meaning mismatch"):
            validate_claims([claim], records)

    def test_composite_template_operand_roles_cannot_be_swapped(self):
        for template, roles in (("episode_base_runs", ("current", "history")),
                                ("cell_comparison", ("success", "failure"))):
            claim = deepcopy(next(c for c in self.card["claims"] if c["template_id"] == template))
            a, b = roles
            claim["operands"][a], claim["operands"][b] = claim["operands"][b], claim["operands"][a]
            values = {name: resolve_path(self.records[o["record_id"]], o["path"]) for name, o in claim["operands"].items()}
            claim["text"] = render_statement(template, {**values, **claim["literals"]})
            recalculate_claim_id(claim)
            with self.assertRaisesRegex(ValueError, "role"):
                validate_claims([claim], self.records)

    def test_inverted_before_entity_must_match_off_interval(self):
        e = sample_episode(5)
        e.arrays["base_quat_w"][0] = [0, 1, 0, 0]
        e.arrays["support_contact_forces_w"][2:, 0] = 0
        s = summary(e)
        r = next(r for r in measure_episode(e) if r.get("record_type") == "measurement" and r.get("key") == "robot_0.support.FL.low_speed_off")
        records = {s["record_id"]: s, r["record_id"]: r}
        claim = make_claim("inverted_before", "fixture", s["display_name"], {
            "negative_step": (s["record_id"], ["highlights", "first_negative"]),
            "start_step": (r["record_id"], ["value", "longest", "interval", "steps_half_open", 0])},
            {"entity": "robot_0.support.RR"}, records, s["record_id"])
        with self.assertRaisesRegex(ValueError, "entity mismatch"):
            validate_claims([claim], records)

    def test_deleted_operand_and_wrong_byte_offset_fail(self):
        claim = next(c for c in self.card["claims"] if c["template_id"] == "episode_marker")
        record_id = claim["operands"]["value"]["record_id"]
        records = dict(self.records)
        del records[record_id]
        with self.assertRaisesRegex(ValueError, "unresolved"):
            validate_claims([claim], records)
        link = deepcopy(next(e for e in self.evidence if e["record_type"] == "evidence_link" and e["record_id"] == record_id))
        link["byte_offset"] += 1
        with self.assertRaises((ValueError, json.JSONDecodeError)):
            read_link(link, self.files)

    def test_unsupported_template_and_kind_status_rejected(self):
        claim = deepcopy(self.card["claims"][0])
        claim["kind"] = "visual_description"
        recalculate_claim_id(claim)
        with self.assertRaisesRegex(ValueError, "kind/status"):
            validate_claims([claim], self.records)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            render_statement("gait_cause", {})

    def test_causal_or_strong_behavior_labels_are_not_generated(self):
        physical_claims = [c["text"] for c in self.card["claims"] if c["kind"] == "measured_relation"]
        for text in physical_claims:
            for forbidden in ("因而失败", "所以失败", "承重占比为", "一直没有物理接触", "主动拒绝", "齐跳步态"):
                self.assertNotIn(forbidden, text)

    def test_success_and_before_claims_require_actual_premises(self):
        with self.assertRaisesRegex(ValueError, "premises"):
            render_statement("successful_return", {"success": False, "drawdown": {"max_drawdown_m": 1, "return_step": 4}})
        with self.assertRaisesRegex(ValueError, "ordering"):
            render_statement("inverted_before", {"negative_step": 10, "start_step": 8, "entity": "foot"})

    def test_aggregate_is_primary_only_and_cannot_accept_duplicate_source(self):
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            build_card(self.bundle, [self.bundle], self.selection)
        primary = next(e for e in self.evidence if e["record_type"] == "cohort_summary" and e["group"] == "all")
        self.assertEqual(primary["episode_count"], 288)
        self.assertEqual(len(set(primary["member_ids"])), 288)

    def test_all_failure_card_has_no_success_reference_fallback(self):
        bundle = deepcopy(self.bundle)
        for s in bundle["summaries"]:
            s["task"]["outcome"].update(success=False, primary_reason="timeout")
        card, _ = build_card(bundle, [], self.selection)
        self.assertTrue(any("没有成功回合" in c["text"] for c in card["claims"]))
        self.assertFalse(any(c["template_id"] in ("cell_comparison", "successful_return") for c in card["claims"]))

    def test_all_final_pointers_resolve_to_same_record_and_source(self):
        for record in self.evidence:
            if record["record_type"] == "evidence_link":
                resolved = read_link(record, self.files)
                if resolved["record_type"] != "episode":
                    self.assertEqual(resolved["record_id"], record["record_id"])
                    self.assertEqual(resolved["evidence_ref"]["manifest_sha256"], self.reference["manifest_sha256"])

    def test_one_row_terminal_current_and_history_are_separate_sentences(self):
        claims = [c for c in self.card["claims"] if c["scope"] == "fixture.r1.e2"]
        self.assertTrue(any(c["template_id"] == "episode_terminal" for c in claims))
        short = next(c for c in claims if c["template_id"] == "episode_base_runs")
        self.assertIn("单条短事件保留", short["text"])
        self.assertIn("current", short["text"])
        self.assertIn("history", short["text"])

    def test_wrong_manifest_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            load_bundle(self.manifest_path, "0" * 64, [])

    def _mutated_bundle(self, name, mutate):
        path = self.fixture.root / name
        path.mkdir()
        manifest = deepcopy(self.bundle["manifest"])
        lines = (self.path / "evidence.jsonl").read_bytes().splitlines(keepends=True)
        records = [json.loads(line) for line in lines]
        mutate(records, manifest)
        payload = b"".join((json.dumps(r, separators=(",", ":")) + "\n").encode() for r in records)
        (path / "evidence.jsonl").write_bytes(payload)
        manifest["evidence"].update(sha256=file_sha256(path / "evidence.jsonl"), bytes=len(payload))
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return path / "manifest.json"

    def test_header_trial_checkpoint_protocol_permission_mutations_rejected(self):
        for index, field in enumerate(("trial_id", "checkpoint_hash", "evaluation_binding", "acceptance_split")):
            def mutate(records, manifest, field=field):
                records[0]["source"][field] = "sealed_validation" if field == "acceptance_split" else "wrong"
            path = self._mutated_bundle("header" + str(index), mutate)
            with self.assertRaisesRegex(ValueError, "validated raw"):
                load_bundle(path, file_sha256(path), iter_stair_episodes(self.reference, source_id="fixture"))

    def test_missing_measurement_is_not_silently_unprocessed(self):
        path = self._mutated_bundle("missingmeasure", lambda records, manifest: records.remove(next(r for r in records if r.get("key") == "force_summary")))
        with self.assertRaisesRegex(ValueError, "incomplete enabled"):
            load_bundle(path, file_sha256(path), iter_stair_episodes(self.reference, source_id="fixture"))

    def test_full_evidence_hash_checked_not_only_selected_example_lines(self):
        path = self.fixture.root / "brokenhash"
        path.mkdir()
        manifest = deepcopy(self.bundle["manifest"])
        payload = (self.path / "evidence.jsonl").read_bytes()
        (path / "evidence.jsonl").write_bytes(payload + b" ")
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_bundle(path / "manifest.json", file_sha256(path / "manifest.json"),
                        iter_stair_episodes(self.reference, source_id="fixture"))

    def test_four_artifact_build_and_no_overwrite(self):
        selection_path = self.fixture.root / "selection.json"
        selection_path.write_text(json.dumps(self.selection), encoding="utf-8")
        output = self.fixture.root / "card"
        result = builder.build(self.manifest_path, file_sha256(self.manifest_path), [], selection_path,
                               file_sha256(selection_path), output)
        self.assertEqual({p.name for p in output.iterdir()}, {"manifest.json", "evidence.jsonl", "policy_card.json", "policy_card_zh.md"})
        self.assertTrue(result["complete"])
        self.assertGreater(result["cost"]["peak_process_memory_bytes"], 0)
        for name, artifact in result["artifacts"].items():
            self.assertEqual(file_sha256(output / name), artifact["sha256"])
        with self.assertRaises(FileExistsError):
            builder.build(self.manifest_path, file_sha256(self.manifest_path), [], selection_path,
                          file_sha256(selection_path), output)
