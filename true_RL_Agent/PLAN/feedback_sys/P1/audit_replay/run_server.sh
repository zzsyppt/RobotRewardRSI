#!/bin/bash
PYI=/data/yuyangliu/envs/isaaclab6_validation/bin/python
D=/data/yuyangliu/p1_audit
A=/data/yuyangliu/AGENT/true_RL_Agent
mkdir -p $D/out
run() { # variant root
  v=$1; root=$2
  echo "=== $v stair $(date -u +%T)"; $PYI $D/p1_replay_audit.py --root $root --task stair --source $A/outputs/formal_go2_stair_15round_20260901_c6ab8ba --output $D/out/${v}_stair --frames 4,13,16 --label ${v}_stair --context scratch 2>&1 | tail -8
  echo "=== $v push $(date -u +%T)"; $PYI $D/p1_replay_audit.py --root $root --task push --source $A/PLAN/2new_world/Go2_push/attempt_08/session --output $D/out/${v}_push --frames 3,6 --label ${v}_push --context session 2>&1 | tail -8
}
run overlay /data/yuyangliu/p1_worktree/true_RL_Agent
run baseline $A
echo "=== overlay unit tests on server $(date -u +%T)"
cd /data/yuyangliu/p1_worktree/true_RL_Agent && $PYI -m unittest tests.unit.test_runtime_evidence_projection tests.unit.test_feedback_evidence_boundary tests.unit.test_memory_context_phase13 tests.unit.test_prompt_projection_and_mechanism tests.unit.test_acceptance_replay tests.integration.test_fixed_width_generation 2>&1 | tail -4
echo "=== DONE $(date -u +%T)"
