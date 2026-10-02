"""Read-only exact-value disambiguation of saved cohort-history candidates."""
from pathlib import Path
import hashlib
import json

BASE = Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_preflight_r3')
history_path = BASE / 'cohort_history_check.json'
history = json.loads(history_path.read_bytes())
tokens = set(history['tokens'])
rows = []
exact = []


def walk(value, path=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk(item, path + '/' + str(key).replace('~', '~0').replace('/', '~1'))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from walk(item, path + '/' + str(i))
    else:
        yield path, value


for candidate in history['matching_runtime_candidates']:
    p = Path(candidate['path'])
    raw = p.read_bytes()
    hits = []
    for pointer, value in walk(json.loads(raw)):
        text = str(value)
        found = [t for t in tokens if t in text]
        if found:
            hit = dict(json_pointer=pointer, value=value, tokens=found, exact_scalar_match=text in tokens)
            hits.append(hit)
            if hit['exact_scalar_match']:
                exact.append(dict(path=str(p), **hit))
    rows.append(dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), hits=hits))

print(json.dumps(dict(history_ref=dict(path=str(history_path), sha256=hashlib.sha256(history_path.read_bytes()).hexdigest()),
    files_checked=history['files_checked'], candidate_files=rows, exact_scalar_matches=exact,
    status='NO_EXACT_TARGET_COHORT_VALUES_IN_SEARCHED_RUNTIME_FILES' if not exact else 'EXACT_MATCHES_REQUIRE_INTERPRETATION',
    boundary='Same explicitly listed history roots and exclusions; no global-server nonuse claim.',
    additional_cache_roots=[dict(path=str(p), exists=p.exists(), resolved=str(p.resolve())) for p in (
        Path.home() / '.cache/ov', Path.home() / '.cache', Path.home() / '.local/share/ov', Path('/data/yuyangliu/cache/xdg/ov'))])))
