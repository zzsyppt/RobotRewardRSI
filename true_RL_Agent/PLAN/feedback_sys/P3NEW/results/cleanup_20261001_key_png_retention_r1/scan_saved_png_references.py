"""Read-only fixed old-Stair PNG references in saved evidence; emits a retention audit."""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

base, manifest = map(Path, sys.argv[1:3])
rows = [r for line in manifest.read_text().splitlines() if (r := json.loads(line)).get('record_type') == 'file_pair'
        and r.get('group') in ('R1_stair', 'dev_stair_dual')]
assert len(rows) == 6802
by_sha = {r['source_sha256']: r for r in rows}
by_name = {PurePosixPath(r['source_path']).name: r for r in rows}
assert len(by_sha) == len(by_name) == len(rows)
roots = {str(PurePosixPath(r[k]).parent) for r in rows for k in ('source_path','export_path')}
patterns = {root: re.compile(re.escape(root)+r'/([^\s"<>]+?\.png)') for root in roots}
png_pattern = re.compile(r'public_dev_seed_\d+(?:_env_\d+)?\.step\d+\.v\d+\.png')
sha_pattern = re.compile(rb'\b[0-9a-f]{64}\b')
data_pattern = re.compile(rb'data:image/[^;," ]+;base64,([A-Za-z0-9+/=]+)')
hits = []
ambiguous = []
scanned = scanned_bytes = image_payloads = 0
skips = set()
errors = []
local_png_matches = []
for cur, dirs, names in os.walk(base):
    dirs[:] = [d for d in dirs if not (d.startswith(('cleanup','storage_duplicate','n5_confirmation_source','n5_postprocess_source'))
               or d in ('.git','.codegraph','__pycache__','src','tests','configs'))]
    for name in names:
        p = Path(cur)/name
        if p.suffix == '.png' and os.name == 'nt':
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            if digest in by_sha:
                local_png_matches.append({'file':str(p),'sha256':digest,'source_path':by_sha[digest]['source_path']})
            continue
        if p.suffix not in ('.json','.jsonl','.md','.txt','.py'):
            continue
        if name.endswith('.snapshots.json') or name == 'recordings.json' or any(x in name.lower() for x in ('credential','secret','provider_config')):
            skips.add(name)
            continue
        try:
            raw = p.read_bytes()
            text = raw.decode('utf-8')
        except (UnicodeDecodeError,OSError) as exc:
            errors.append({'file':str(p),'error':type(exc).__name__})
            continue
        scanned += 1
        scanned_bytes += len(raw)
        reasons = {}
        for root, pattern in patterns.items():
            if root in text:
                for image_name in pattern.findall(text):
                    if image_name in by_name:
                        reasons.setdefault(by_name[image_name]['source_path'],set()).add('explicit_old_source_or_export_path')
                # Saved old-root plus relative source frame names also bind the capture.
                group = 'dev_stair_dual' if 'stair_dual_view_dev_r3' in root else 'R1_stair'
                for image_name in png_pattern.findall(text):
                    if image_name in by_name and by_name[image_name]['group'] == group:
                        reasons.setdefault(by_name[image_name]['source_path'],set()).add('old_root_and_frame_name')
        for digest in set(sha_pattern.findall(raw)):
            key = digest.decode()
            if key in by_sha:
                reasons.setdefault(by_sha[key]['source_path'],set()).add('exact_original_png_sha_in_saved_evidence')
        if 'request' in name or 'receipt' in name:
            for encoded in data_pattern.findall(raw):
                image_payloads += 1
                digest = hashlib.sha256(base64.b64decode(encoded)).hexdigest()
                if digest in by_sha:
                    reasons.setdefault(by_sha[digest]['source_path'],set()).add('exact_original_png_bytes_in_image_payload')
        if reasons:
            hits.append({'file':str(p),'file_sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw),
                         'references':[{'source_path':k,'reasons':sorted(v)} for k,v in sorted(reasons.items())]})
        else:
            names_here=sorted(set(png_pattern.findall(text)).intersection(by_name))
            if names_here:
                ambiguous.append({'file':str(p),'file_sha256':hashlib.sha256(raw).hexdigest(),'image_names':names_here})
print(json.dumps({'scope':str(base),'old_png_members':6802,'scanned_text_files':scanned,'scanned_text_bytes':scanned_bytes,
    'image_payloads_decoded':image_payloads,'excluded_capture_inventory_names':sorted(skips),'errors':errors,
    'hits':hits,'local_exact_png_copies':local_png_matches,'ambiguous_basename_only':ambiguous,
    'limits':'Saved evidence only; generic capture inventory and cleanup manifests excluded. Basename-only matches are not treated as proof of same capture.'}))
