from pathlib import Path
p=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_postprocess_source_r1_inputs')
assert not p.exists()
assert not p.with_name('n5_postprocess_source_r1').exists()
p.mkdir()
print('exclusive CPU postprocess inputs created')
