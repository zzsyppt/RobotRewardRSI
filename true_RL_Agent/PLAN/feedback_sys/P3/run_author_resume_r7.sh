#!/usr/bin/env bash
# Continue the original journal after the documented output-accounting amendment.
set -u
stage_root=/data/yuyangliu/AGENT/feedback_p3_20260924
/usr/bin/time -v -o "$stage_root/author_run_r7.time" \
  /data/yuyangliu/envs/isaaclab6_validation/bin/python -u "$stage_root/tools_r7/author_experiment.py" run \
  --root "$stage_root/source_r7" \
  --stage "$stage_root/tools_r7/author_stage.json" \
  --cases "$stage_root/author_case_inputs_r5/case_inputs.json" \
  --output "$stage_root/author_run_r5" \
  --freeze "$stage_root/author_freeze_api_r7.json"
author_status=$?
printf '%s\n' "$author_status" > "$stage_root/author_run_r7.exit"
exit "$author_status"
