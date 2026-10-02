"""Extract all actual prepared text, gzip locally, count with cached remote tokenizer."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

D=Path(__file__).resolve().parent
digest=lambda data:hashlib.sha256(data).hexdigest()
ref=lambda p:{'file':str(p.resolve()),'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size}
for name in ('text_input.json.gz','text_capacity.json','text_capacity.stderr.txt','text_capacity_execution.json'):
    if (D/name).exists():raise RuntimeError('refuse overwrite '+name)
card_path=D.parent/'n5_confirmation_execution_r2/capture_card.json';card=json.loads(card_path.read_bytes())
rules={k:card['materials'][k] for k in ('input_token_proxy_limit','system_tools_reserved','output_reserved','proxy_margin_reserved','total_proxy_reserved')}
rules.update({k:card['model_roles']['gpt6_sol_judge'][k] for k in ('context_window','auto_compact_limit')})
requests=[]
receipt_path=D/'receipt.json';receipt=json.loads(receipt_path.read_bytes())
assert receipt['status']=='PASS_ENGINEERING_ONLY' and receipt['packet_count']==40
expected_requests={(packet['packet_id'],judge):record for packet in receipt['packets'] for judge,record in packet['requests'].items()}
for path in sorted((D/'packets').glob('*/**/prepared/prepare.json')):
    prepared=json.loads(path.read_bytes());body_path=path.with_name('request.http_body.json');body_raw=body_path.read_bytes()
    assert digest(body_raw)==prepared['request_sha256'] and len(body_raw)==prepared['request_bytes']
    expected=expected_requests[(prepared['packet_id'],prepared['judge_id'])]
    assert digest(body_raw)==expected['sha256'] and len(body_raw)==expected['bytes']
    body=json.loads(body_raw);parts=[];images=0
    for mi,message in enumerate(body['messages']):
        blocks=[{'type':'text','text':message['content']}] if isinstance(message['content'],str) else message['content']
        for bi,block in enumerate(blocks):
            if block['type']=='image_url':images+=1;continue
            assert block['type']=='text'
            text=block['text'];label={}
            try:label=json.loads(text)
            except ValueError:pass
            parts.append({'message_index':mi,'block_index':bi,'role':message['role'],'material_role':label.get('role'),
                          'text':text,'text_sha256':digest(text.encode()),'text_utf8_bytes':len(text.encode())})
    assert images==prepared['png_count']
    requests.append({'packet_id':prepared['packet_id'],'judge_id':prepared['judge_id'],'request':ref(body_path),
                     'prepare':ref(path),'image_count':images,'body_output_cap':body['max_tokens'],'text_blocks':parts})
assert len(requests)==80
payload={'capture_card_source':ref(card_path),'encoding_receipt_source':ref(receipt_path),'reservation_rules':rules,'requests':requests}
raw=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode();compressed=gzip.compress(raw,mtime=0)
with (D/'text_input.json.gz').open('xb') as f:f.write(compressed)
template=(D/'count_text_remote.py').read_bytes()
source=b'PAYLOAD_BASE64 = '+repr(base64.b64encode(compressed).decode()).encode()+b'\n'+template
argv=['ssh','humanoid-server','/data/yuyangliu/envs/isaaclab6_validation/bin/python','-B','-']
start=time.time()
with (D/'text_capacity.json').open('xb') as out,(D/'text_capacity.stderr.txt').open('xb') as err:
    process=subprocess.run(argv,input=source,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
execution={'argv':argv,'exit_code':process.returncode,'elapsed_seconds':time.time()-start,'remote_source_writes':0,
           'complete_stdin_script_sha256':digest(source),'template':ref(D/'count_text_remote.py'),'caller':ref(Path(__file__)),
           'text_input_uncompressed_bytes':len(raw),'compressed_text_input':ref(D/'text_input.json.gz'),
           'output':ref(D/'text_capacity.json'),'stderr':ref(D/'text_capacity.stderr.txt')}
with (D/'text_capacity_execution.json').open('x',encoding='utf-8') as f:json.dump(execution,f,indent=2)
print(json.dumps(execution))
if process.returncode:print((D/'text_capacity.stderr.txt').read_text(encoding='utf-8'));raise SystemExit(process.returncode)
result=json.loads((D/'text_capacity.json').read_bytes())
print(json.dumps({k:result[k] for k in ('status','requests_counted','max_text_tokens','max_catalog_text_tokens','all_text_only_proxy_comparisons_pass')}))
