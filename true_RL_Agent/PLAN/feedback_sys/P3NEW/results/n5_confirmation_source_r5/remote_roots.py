from pathlib import Path
s=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r5_final');e=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20261001_r5');cap=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r5')
assert not s.exists() and not e.exists() and not cap.exists()
s.mkdir();(e/'source_control').mkdir(parents=True)
print('fresh R5 source and control created; capture root absent')
