#!/bin/bash
PYI=/data/yuyangliu/envs/isaaclab6_validation/bin/python
D=/data/yuyangliu/p1_audit
A=/data/yuyangliu/AGENT/true_RL_Agent
R=/data/yuyangliu/p1_worktree/true_RL_Agent
for v in sparse sparse_notop token; do
  echo "=== stair $v $(date -u +%T)"; $PYI $D/p1_replay_audit.py --root $R --task stair --source $A/outputs/formal_go2_stair_15round_20260901_c6ab8ba --output $D/out/variant_${v}_stair --frames 13,16 --label variant_${v}_stair --context scratch --memory-variant $v 2>&1 | grep '^{"variant'
  echo "=== push $v $(date -u +%T)"; $PYI $D/p1_replay_audit.py --root $R --task push --source $A/PLAN/2new_world/Go2_push/attempt_08/session --output $D/out/variant_${v}_push --frames 6 --label variant_${v}_push --context session --memory-variant $v 2>&1 | grep '^{"variant'
done
echo "=== server tests with PYTHONPATH $(date -u +%T)"
cd $R && PYTHONPATH=$R/src $PYI -m unittest tests.unit.test_runtime_evidence_projection tests.unit.test_feedback_evidence_boundary tests.unit.test_memory_context_phase13 tests.unit.test_prompt_projection_and_mechanism tests.unit.test_acceptance_replay tests.integration.test_fixed_width_generation 2>&1 | tail -3
echo "=== DONE2 $(date -u +%T)"
