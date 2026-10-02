"""Fixed two-rollout observable-access audit; no image viewing or semantic grading."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import struct

HERE = Path(__file__).resolve().parent
EXPECTED_IDS = {'e1':'01a0f417-53db-7400-a1eb-9f442ba48c65', 'e2':'01a0f41b-2e69-7120-9962-b9a0f489a927'}
EXPECTED_RAW_SHA = {'e1':'cdfe977746b6e6fe6bb7733f002f9134d8b34b948f5ac9295dcbf2ba84d779e2', 'e2':'102d4f6805aa98f970533ba573f9a241a49fd198b40b9fbde3497f2ad5c2175c'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    data = path.read_bytes()
    return {'path':str(path),'sha256':sha(data),'size_bytes':len(data)}


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    original_registration = ref(HERE/'desktop_invocations.json')
    registration = load(HERE/'desktop_invocations.json')
    checks, bindings = [], []
    for alias, uuid in EXPECTED_IDS.items():
        packet = HERE/alias; blind = packet/'blind_inputs'; trace = packet/'desktop_trace'
        manifest_ref = ref(blind/'materials.json'); manifest = load(blind/'materials.json')
        declared = next(x for x in registration['invocations'] if x['alias']==alias)
        assert manifest_ref['sha256']==declared['input_manifest_sha256']
        assert len(manifest['materials'])==13
        for item in manifest['materials']:
            target=(blind/item['path']).resolve(); assert target.is_relative_to(blind.resolve())
            assert ref(target)['sha256']==item['sha256']
        images=[x for x in manifest['materials'] if x['role']=='original_image']; assert len(images)==12
        calls=load(trace/'observable_tool_trace.json'); assert len(calls)==15
        assert all(x['name']=='exec' for x in calls)
        session=load(trace/'observable_session.json'); contexts=session['turn_contexts']
        assert len(session['session_meta'])==len(contexts)==1
        meta=session['session_meta'][0]
        assert meta['id']==uuid and meta['agent_path']==f'/root/r4_push_observer_{alias}'==declared['agent_path']
        assert contexts[0]['model']=='gpt-6.1-sol' and contexts[0]['effort']=='high'
        extraction=load(trace/'extraction_receipt.json'); rollout=Path(extraction['rollout_snapshot']['path'])
        data=rollout.read_bytes(); assert sha(data)==extraction['rollout_snapshot']['sha256']
        outputs=[]; events=[]; image_returns=[]
        # Never export or inspect assistant messages/reasoning or arbitrary session fields.
        for line_number, line in enumerate(data.splitlines(),1):
            record=json.loads(line); kind=record.get('type'); payload=record.get('payload',{})
            if kind=='event_msg' and payload.get('type') in ('task_started','task_complete','turn_aborted'):
                events.append({'timestamp':record.get('timestamp'),'type':payload['type'],'turn_id':payload.get('turn_id')})
            if kind!='response_item' or payload.get('type') not in ('function_call_output','custom_tool_call_output'):
                continue
            assert isinstance(payload['output'],list)
            projected=[]
            for block in payload['output']:
                if block['type']=='input_text':
                    assert set(block)=={'type','text'}
                    projected.append(dict(block))
                elif block['type']=='input_image':
                    assert set(block)=={'type','image_url','detail'}
                    header,encoded=block['image_url'].split(',',1)
                    assert header=='data:image/png;base64'
                    png=base64.b64decode(encoded,validate=True)
                    assert png[:8]==b'\x89PNG\r\n\x1a\n' and png[12:16]==b'IHDR'
                    info={'type':'input_image_metadata','mime_type':'image/png','sha256':sha(png),
                          'size_bytes':len(png),'dimensions_px':list(struct.unpack('>II',png[16:24])),
                          'detail':block['detail'],'image_bytes_omitted_from_export':True}
                    projected.append(info)
                    image_returns.append({'call_id':payload['call_id'],'timestamp':record['timestamp'],**info})
                else:
                    raise ValueError('Unreviewed observable output block type')
            outputs.append({'timestamp':record['timestamp'],'type':payload['type'],'call_id':payload['call_id'],
                            'source_line':line_number,'source_line_sha256':sha(line),'output':projected})
        assert len(outputs)==15 and [x['call_id'] for x in outputs]==[x['call_id'] for x in calls]
        assert [e['type'] for e in events]==['task_started','task_complete']
        assert len(image_returns)==12
        for index,(item,call,returned) in enumerate(zip(images,calls[2:14],image_returns),1):
            text=call.get('input',call.get('arguments',''))
            # Only the three exact native wrappers actually recorded in these two runs.
            if alias == 'e1':
                base_declaration='store("base",'+json.dumps(str(blind.resolve()))+');'
                assert base_declaration in calls[1]['input']
                fragment=json.dumps('\\'+item['path'].replace('/','\\'))
                if index == 1:
                    expected='const p=load("base")+'+fragment+'; const r=await tools.view_image({path:p,detail:"original"}); image(r.image_url,"original");'
                else:
                    expected='const r=await tools.view_image({path:load("base")+'+fragment+',detail:"original"});image(r.image_url,"original");'
            else:
                expected='const r = await tools.view_image({path:'+json.dumps(str((blind/item['path']).resolve()))+',detail:"original"}); image(r.image_url,"original");'
            assert text.strip()==expected
            image_path=(blind/item['path']).resolve()
            assert returned['call_id']==call['call_id'] and returned['sha256']==item['sha256']
            assert returned['dimensions_px']==item['image_size_px']==[1920,1080] and returned['detail']=='original'
            returned.update(view_order=index,path=item['path'],step=item['step'],view_id=item['view_id'],
                            tool_input_sha256=sha(text.encode()),matches_current_original_png=True)
        for call in calls:
            text=call.get('input',call.get('arguments',''))
            assert f'\\\\{"e2" if alias=="e1" else "e1"}\\\\' not in text
        for call in (calls[0],calls[1],calls[-1]):
            assert f'\\\\{alias}\\\\blind_inputs' in call.get('input',call.get('arguments',''))
        final_input=calls[-1].get('input',calls[-1].get('arguments',''))
        if alias == 'e1':
            assert final_input.count('[System.IO.File]::Open(')==2
            assert final_input.count('[System.IO.FileMode]::CreateNew')==2
            assert final_input.count('$rawStream.Write(')==final_input.count('$receiptStream.Write(')==1
            write_limit='Both output files use exclusive CreateNew in their single visible original write; current bytes match the returned original write SHA. No claim about unobservable activity.'
        else:
            assert final_input.count('[System.IO.File]::WriteAllText(')==2
            write_limit='WriteAllText is overwrite-capable. Exactly one visible write per output file and current bytes equal its returned SHA; no claim of exclusive-create protection or inaccessible activity.'
        assert all(not any(marker in c.get('input',c.get('arguments','')) for marker in ('WriteAllText','[System.IO.File]::Open(')) for c in calls[:-1])
        last_text='\n'.join(x['text'] for x in outputs[-1]['output'] if x['type']=='input_text')
        emitted=dict(re.findall(r'(raw_answer\.txt|access_receipt\.json).*?([0-9a-f]{64})',last_text.lower(),re.S))
        current={name:ref(packet/'sol_response'/name) for name in ('raw_answer.txt','access_receipt.json')}
        assert len(emitted)==2 and all(current[n]['sha256']==emitted[n] for n in current)
        assert current['raw_answer.txt']['sha256']==EXPECTED_RAW_SHA[alias]
        access=load(packet/'sol_response/access_receipt.json')
        assert access['materials_sha256']==manifest_ref['sha256']
        expected_reads={'materials.json':manifest_ref['sha256'],**{x['path']:x['sha256'] for x in manifest['materials']}}
        def receipt_relative_path(value):
            assert Path(value).is_absolute(), 'Expected the actual full Windows-path receipt shape'
            return Path(value).resolve().relative_to(blind.resolve()).as_posix()
        assert {receipt_relative_path(x['path']):x['sha256'] for x in access['files_read']}==expected_reads and len(access['files_read'])==14
        order_key='index' if alias=='e1' else 'ordinal'
        assert len(access['images_viewed'])==12
        for index,(got,item) in enumerate(zip(access['images_viewed'],images),1):
            assert receipt_relative_path(got['path'])==item['path']
            assert all(got[k]==item[k] for k in ('sha256','step','view_id'))
            assert got[order_key]==index and got['detail']=='original' and got['original_dimensions_px']==[1920,1080]
        save(trace/'observable_tool_outputs.json',outputs)
        save(trace/'returned_image_metadata.json',image_returns)
        save(trace/'observable_completion.json',events)
        usage=load(trace/'visible_usage.json')['last_visible_info']; compactions=load(trace/'compaction_events.json')
        checks.append({'alias':alias,'rollout_uuid':uuid,'tool_trace':ref(trace/'observable_tool_trace.json'),
                       'tool_outputs':ref(trace/'observable_tool_outputs.json'),'returned_images':ref(trace/'returned_image_metadata.json'),
                       'raw_answer':current['raw_answer.txt'],'access_receipt':current['access_receipt.json'],
                       'input_manifest':manifest_ref,'tool_calls':15,'original_image_calls_in_manifest_order':12,
                       'image_returns_match_original_png_bytes_and_dimensions':True,'read_whitelist_files':14,
                       'model_and_effort':{'model':'gpt-6.1-sol','effort':'high'},'last_visible_usage_info':usage,
                       'compaction_events_observed':compactions['observed_event_count'],
                       'visible_write_calls_per_output_file':1,'current_outputs_match_hashes_returned_at_original_write':True,
                       'manual_access_review':'Reviewed all 15 original tool inputs: initial manifest/hash read, whitelist hash/text read, 12 original image calls, one final command writing only raw_answer/access_receipt and reading whitelist hashes/PNG headers. e1 uses only functions store/load for its own blind-root string. No observable outside-evidence, sibling-answer, repository code, personal memory/history, network or agent access.',
                       'write_limit':write_limit,
                       'original_access_receipt_shape':{'path_kind':'absolute Windows path','image_order_key':order_key,'comparison_only_path_normalization':True,'original_receipt_rewritten':False},
                       'semantic_review_performed':False})
        bindings.append({'alias':alias,'registered_agent_path':declared['agent_path'],'actual_agent_path':meta['agent_path'],
                         'actual_rollout_uuid':uuid,'actual_turn_id':contexts[0]['turn_id'],
                         'session_created_at_utc':meta['timestamp'],'task_started_at_utc':events[0]['timestamp'],
                         'task_completed_at_utc':events[1]['timestamp'],'rollout_snapshot':extraction['rollout_snapshot'],
                         'input_manifest':manifest_ref,'raw_answer':current['raw_answer.txt'],'access_receipt':current['access_receipt.json'],
                         'fork_turns_registered':declared['fork_turns'],'fork_limit':'fork_turns=none is the preserved dispatch registration; trace attests agent identity/configuration, not hidden context isolation.'})
        assert sha(rollout.read_bytes())==extraction['rollout_snapshot']['sha256']
    assert ref(HERE/'desktop_invocations.json')==original_registration
    now=datetime.now(timezone.utc).isoformat()
    save(HERE/'desktop_invocations.trace_binding.json',{'schema_version':'p3new.actual_observer_trace_binding.v1',
         'recorded_at_utc':now,'original_registration':original_registration,'original_registration_unchanged':True,'invocations':bindings})
    save(HERE/'desktop_access_audit.json',{'recorded_at_utc':now,'scope':'R4 Push observable access and transport only',
         'status':'PASS_OBSERVABLE_ACCESS_AND_TRANSPORT_NOT_SEMANTIC','checks':checks,
         'audit_script':ref(Path(__file__)),'reused_stair_auditor':ref(HERE.parent/'stair_observer_inputs/audit_desktop_transport.py'),
         'limits':['Only the two explicitly supplied completed rollout snapshots were audited; no whole-session search.',
                   'No assistant analysis/private reasoning, base instructions, account fields, compaction summaries or full rollout were exported.',
                   'Returned original image bytes prove transport, not image comprehension or semantic accuracy.',
                   'Shared filesystem is not OS isolation; hidden/backend activity and server fingerprint remain unverified.',
                   'Visible usage is cumulative over tool turns, not simultaneous context occupancy or a monetary invoice.',
                   'Zero recognized compaction events does not rule out unknown encodings or hidden truncation.'],
         'desktop_cost_usd':None,'server_fingerprint':None,'new_model_calls':0,'new_api_calls':0,'gpu_runs':0,
         'formal_facts_calls_by_this_audit':0,'formal_report_calls_by_this_audit':0})
    print(json.dumps({'status':'PASS_OBSERVABLE_ACCESS_AND_TRANSPORT_NOT_SEMANTIC','images_verified':24,
                      'audit':ref(HERE/'desktop_access_audit.json'),'trace_binding':ref(HERE/'desktop_invocations.trace_binding.json')}))


if __name__=='__main__':
    main()
