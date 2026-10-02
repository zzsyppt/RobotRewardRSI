#!/usr/bin/env bash
# Resume the original 48-episode journal; never start a replacement experiment.
set -u
stage_root=/data/yuyangliu/AGENT/feedback_p3_20260924
/usr/bin/time -v -o "$stage_root/author_run_r6.time" \
  /data/yuyangliu/envs/isaaclab6_validation/bin/python -u "$stage_root/tools/author_experiment.py" run \
  --root "$stage_root/source_r6" \
  --stage "$stage_root/tools/author_stage.json" \
  --cases "$stage_root/author_case_inputs_r5/case_inputs.json" \
  --output "$stage_root/author_run_r5" \
  --freeze "$stage_root/author_freeze_api_r6.json"
author_status=$?
printf '%s\n' "$author_status" > "$stage_root/author_run_r6.exit"
exit "$author_status"
