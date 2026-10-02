"""CPU token counting only; PAYLOAD_BASE64 is supplied in-memory by the caller."""
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile
import tokenizers
from tokenizers import Tokenizer

digest=lambda data:hashlib.sha256(data).hexdigest()
compressed=base64.b64decode(PAYLOAD_BASE64)
payload=json.loads(gzip.decompress(compressed))
path=Path('/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930/control/numeric_materials_r1/token_capacity_r1/deepseek_v4_tokenizer.zip')
blob=path.read_bytes()
assert digest(blob)=='e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495'
expected='89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614'
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    matched=[(name,z.read(name)) for name in z.namelist() if name.endswith('tokenizer.json') and digest(z.read(name))==expected]
assert len(matched)==1
member,raw=matched[0];tokenizer=Tokenizer.from_str(raw.decode())
rules=payload['reservation_rules'];reserve=sum(rules[k] for k in ('system_tools_reserved','output_reserved','proxy_margin_reserved'))
assert rules['total_proxy_reserved']==rules['input_token_proxy_limit']+reserve
records=[]
for request in payload['requests']:
    parts=[]
    for part in request['text_blocks']:
        text=part['text'];assert digest(text.encode())==part['text_sha256']
        parts.append({k:v for k,v in part.items() if k!='text'}|{'tokens':len(tokenizer.encode(text,add_special_tokens=False).ids)})
    total=sum(p['tokens'] for p in parts)
    record={k:v for k,v in request.items() if k!='text_blocks'}
    record.update(text_blocks=parts,serialized_text_tokens_deepseek_v4=total,
                  catalog_text_tokens=sum(p['tokens'] for p in parts if p.get('material_role')=='reference_catalog'),
                  conservative_reserved_text_proxy=total+reserve,
                  text_only_proxy_within_input_limit=total<=rules['input_token_proxy_limit'],
                  text_only_reserved_proxy_below_autocompact=total+reserve<rules['auto_compact_limit'],
                  text_only_reserved_proxy_within_context=total+reserve<=rules['context_window'],
                  image_tokens=None,provider_usage=None,actual_sol_tokens=None,capacity_release=False)
    records.append(record)
result={'schema_version':'p3new.reference_blocks_text_capacity.v1','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'STATIC_TEXT_MEASURED_NOT_RELEASED','compressed_payload_sha256':digest(compressed),
        'tokenizer_zip':{'path':str(path),'sha256':digest(blob),'bytes':len(blob)},'tokenizer_member':member,'tokenizer_json_sha256':expected,
        'tokenizers_version':tokenizers.__version__,'reservation_rules':rules,'capture_card_source':payload['capture_card_source'],
        'encoding_receipt_source':payload['encoding_receipt_source'],
        'requests_counted':len(records),'records':records,'max_text_tokens':max(r['serialized_text_tokens_deepseek_v4'] for r in records),
        'max_catalog_text_tokens':max(r['catalog_text_tokens'] for r in records),
        'all_text_only_proxy_comparisons_pass':all(r['text_only_proxy_within_input_limit'] and r['text_only_reserved_proxy_below_autocompact'] and r['text_only_reserved_proxy_within_context'] for r in records),
        'method':'Sum all complete system and user text blocks using cached DeepSeek V4 Tokenizer.encode(add_special_tokens=False). No PNG/base64 is tokenized.',
        'unmeasured':['image tokens','provider chat framing and special tokens','actual SOL tokenizer','agent/tool/runtime peak context','actual provider usage'],
        'source_files_written':0,'tokenizer_downloads':0,'model_calls':0,'gpu_calls':0,'capacity_release':False,
        'limits':['Measured text counts do not release multimodal requests; image token cost remains unmeasured.','SOL figures are a DeepSeek tokenizer proxy, not SOL actual tokens.','No semantic correctness or actual citation compliance is established.']}
print(json.dumps(result,ensure_ascii=False,indent=2))
