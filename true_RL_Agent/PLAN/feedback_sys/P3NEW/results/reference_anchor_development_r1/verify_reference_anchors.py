"""Offline current-runner engineering check on the original 40 R2 facts packets.

No API calls, review answers, seals, or original files are written. New materials
retain absolute bindings to originals except newly generated protocol documents.
"""
from collections import Counter
from datetime import datetime, timezone
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
HERE=ROOT/'PLAN/feedback_sys/P3NEW'
OUT=Path(__file__).resolve().parent
E=HERE/'results/n5_confirmation_execution_r2'
refs={};checks=[]


def sha(raw):return hashlib.sha256(raw).hexdigest()


def watch(p):
    p=Path(p).resolve();raw=p.read_bytes()
    refs[str(p)]={'file':str(p),'sha256':sha(raw),'bytes':len(raw)}
    return raw


def read(p):return json.loads(watch(p))


def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as stream:stream.write((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode())


def check(name,value,**extra):
    checks.append({'check':name,'status':'PASS' if value else 'FAIL',**extra})
    if not value:raise AssertionError(name)


def direct_pointer(value,path):
    if not path:return value
    if not path.startswith('/'):raise ValueError(path)
    for component in path[1:].split('/'):
        key=component.replace('~1','/').replace('~0','~')
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value


def without_protocol_or_catalog(parts):
    retained=[]
    for block in parts:
        if block['type']=='text' and json.loads(block['text']).get('role') in ('review_protocol','reference_catalog'):continue
        retained.append(block)
    return retained


def main():
    if (OUT/'receipt.json').exists():raise RuntimeError('refuse overwriting completed audit')
    watch(Path(__file__))
    runner_path=HERE/'run_confirmation_review.py';watch(runner_path)
    coverage_path=HERE/'confirmation_review_coverage.py';watch(coverage_path)
    check('current coverage revision',refs[str(coverage_path)]['sha256']=='7548da8dcf1e43fad256ce993923589b90d00bcab8733ef6bc36177b3a28957f')
    spec=importlib.util.spec_from_file_location('current_reference_runner',runner_path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    coverage=runner.coverage
    protocol_path=HERE/'N5_FACT_REVIEW_PROTOCOL_ZH.md';watch(protocol_path)
    current_protocol=runner.packet_protocol(protocol_path)
    manifests=sorted((E/'sol_facts').glob('episode-*/facts.manifest.json'))+sorted((E/'sol_facts').glob('focus-*/facts.manifest.json'))+sorted((E/'sol_facts_media_bundled').glob('media-*/facts.manifest.json'))
    check('40 original manifests',len(manifests)==40)
    packets=[];original_images=[];new_images=[];anchor_n=0
    for ordinal,old_path in enumerate(manifests,1):
        old=read(old_path);pid=old['packet_id'];base=OUT/'packets'/pid
        check(pid+'/legacy contract absent','reference_contract' not in old)
        old_card=read(old_path.parent/'prepared/prepare.json')
        old_body=watch(old_path.parent/'prepared/request.http_body.json')
        # Watch the old instruction and release files as well; no legacy receipts change.
        for name in ('judge_instruction.txt','release.draft.json'):watch(old_path.parent/'prepared'/name)
        built,instruction,card=runner.build(old_path,old_card['judge_id'])
        check(pid+'/legacy HTTP byte equality',built==old_body)
        check(pid+'/legacy recorded request hash',sha(old_body)==old_card['request_sha256'])
        old_request=json.loads(old_body);new=copy.deepcopy(old)
        new['reference_contract']=coverage.ANCHOR_CONTRACT
        bound=[]
        for item in new['materials']:
            p=(old_path.parent/item['file']).resolve();raw=watch(p)
            check(pid+'/'+item['id']+'/input hash',sha(raw)==item['sha256'])
            if item['role']=='review_protocol':
                protocol={**json.loads(raw),**current_protocol,'status':'REFERENCE_ANCHOR_DEVELOPMENT_NOT_FROZEN'}
                p=base/'review_protocol.json';save(p,protocol)
                item['file']=str(p.resolve());item['sha256']=sha(p.read_bytes())
            else:
                item['file']=str(p)
                bound.append({'material_id':item['id'],'role':item['role'],'file':str(p),'sha256':item['sha256'],'bytes':len(raw)})
                if item['role']=='raw_image':original_images.append({'packet_id':pid,'material_id':item['id'],'sha256':sha(raw)})
        new_path=base/'facts.manifest.json';save(new_path,new)
        check(pid+'/all absolute bindings',all(Path(i['file']).is_absolute() for i in new['materials']))
        check(pid+'/material IDs roles order unchanged',[(i['id'],i['role'],i['mime_type']) for i in old['materials']]==[(i['id'],i['role'],i['mime_type']) for i in new['materials']])
        requests={};cards={}
        for judge in ('deepseek','gpt6-sol'):
            prepared=base/judge/'prepared'
            cards[judge]=runner.prepare(new_path,judge,prepared)
            requests[judge]=json.loads((prepared/'request.http_body.json').read_bytes())
        check(pid+'/identical reviewer user blocks',requests['deepseek']['messages'][1]==requests['gpt6-sol']['messages'][1])
        parts=requests['deepseek']['messages'][1]['content']
        check(pid+'/all original nonprotocol blocks bytes and order',runner.encoded(without_protocol_or_catalog(parts))==runner.encoded(without_protocol_or_catalog(old_request['messages'][1]['content'])))
        materials,inventory=coverage.request_materials(requests['deepseek'])
        catalog=coverage.request_reference_catalog(requests['deepseek'])
        check(pid+'/facts has no report inventory',inventory is None)
        check(pid+'/catalog exact shape',set(catalog)=={'schema_version','items'} and catalog['schema_version']==coverage.ANCHOR_CONTRACT)
        ids=[item['anchor_id'] for item in catalog['items']]
        check(pid+'/catalog identities unique sorted',ids==sorted(set(ids)))
        for entry in catalog['items']:
            check(pid+'/'+entry['anchor_id']+'/locator-only shape',set(entry)=={'anchor_id','locator'})
            locator=entry['locator'];material=materials[locator['material_id']]
            check(pid+'/'+entry['anchor_id']+'/pure locator',set(locator)=={'material_id','sha256','json_pointer'} and material['role'] not in ('review_protocol','focus_questions'))
            expected_id='a-'+sha(coverage.canonical(locator))[:16]
            check(pid+'/'+entry['anchor_id']+'/deterministic identity',entry['anchor_id']==expected_id)
            normalized=coverage.normalize_reference({'anchor_id':entry['anchor_id']},materials,catalog)
            check(pid+'/'+entry['anchor_id']+'/exact resolves',normalized==locator and coverage.resolve_reference(normalized,materials)==direct_pointer(material,locator['json_pointer']))
        # Compare image payload bytes to original allowlisted bytes, in displayed order.
        emitted=[base64.b64decode(block['image_url']['url'].split(',',1)[1]) for block in parts if block['type']=='image_url']
        expected_images=[i for i in new['materials'] if i['role']=='raw_image']
        check(pid+'/PNG sequence count',len(emitted)==len(expected_images)==cards['deepseek']['png_count'])
        for raw,item in zip(emitted,expected_images):
            check(pid+'/'+item['id']+'/unchanged PNG',raw==Path(item['file']).read_bytes())
            new_images.append({'packet_id':pid,'material_id':item['id'],'sha256':sha(raw)})
        cat_raw=coverage.canonical(catalog);anchor_n+=len(ids)
        cat_block=next(block for block in parts if block['type']=='text' and json.loads(block['text']).get('role')=='reference_catalog')
        record={'packet_id':pid,'old_manifest':refs[str(old_path.resolve())],'old_prepared_request':refs[str((old_path.parent/'prepared/request.http_body.json').resolve())],
                'old_body_identical':True,'original_material_bindings':bound,'new_manifest':runner.ref(new_path),
                'user_blocks_identical':True,'nonprotocol_blocks_unchanged':True,'png_count':len(emitted),
                'catalog_items':len(ids),'catalog_canonical_sha256':sha(cat_raw),'catalog_canonical_bytes':len(cat_raw),
                'catalog_serialized_block_bytes':len(runner.encoded(cat_block)),
                'requests':{j:{'sha256':cards[j]['request_sha256'],'bytes':cards[j]['request_bytes']} for j in cards}}
        save(base/'comparison.json',record);packets.append(record)
        if ordinal%10==0:print(json.dumps({'completed_packets':ordinal,'png_count_so_far':len(new_images)}),flush=True)
    check('48 original PNGs retained',len(original_images)==len(new_images)==48 and original_images==new_images)
    # Address diagnostics are operator records, never reviewer answer repair.
    diagnostic=read(E/'deepseek_facts_control/actual_first_call/reference_failure_diagnosis.json')
    request=json.loads((OUT/'packets/episode-01/deepseek/prepared/request.http_body.json').read_bytes())
    materials,_=coverage.request_materials(request);catalog=coverage.request_reference_catalog(request)
    conversions=[]
    for row in diagnostic['records']:
        target=row['actual_existing_locator'];pointer=target['json_pointer']
        candidates=[item for item in catalog['items'] if item['locator']['material_id']==target['material_id'] and item['locator']['sha256']==target['sha256'] and (pointer==item['locator']['json_pointer'] or pointer.startswith(item['locator']['json_pointer']+'/'))]
        entry=max(candidates,key=lambda item:len(item['locator']['json_pointer']))
        ref={'anchor_id':entry['anchor_id'],'within_anchor_pointer':pointer[len(entry['locator']['json_pointer']):]}
        normalized=coverage.normalize_reference(ref,materials,catalog)
        check(row['answer_locator']+'/diagnostic target exact representation',normalized==target)
        check(row['answer_locator']+'/same target value',coverage.resolve_reference(normalized,materials)==row['resolved_value'])
        conversions.append({'answer_locator':row['answer_locator'],'original_reference':row['original_reference'],'diagnostic_existing_locator':target,'anchor_reference':ref,'normalized_locator':normalized,'value_sha256':sha(coverage.canonical(row['resolved_value'])),'status':'EXACT_LOCATION_ONLY_NOT_ANSWER_REPAIR'})
    check('all 19 diagnostic targets represented',len(conversions)==19)
    save(OUT/'diagnostic_address_representations.json',{'scope':'Location only. No original answer is edited or accepted; no semantic score assigned.','records':conversions})
    unchanged=all(sha(Path(p).read_bytes())==r['sha256'] for p,r in refs.items());check('all watched inputs unchanged',unchanged)
    outputs=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file():outputs.append({'file':str(p.resolve()),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size})
    maxima={field:max(packets,key=lambda p:p[field]) for field in ('catalog_items','catalog_canonical_bytes','catalog_serialized_block_bytes')}
    body_max=max(({'packet_id':p['packet_id'],'judge_id':j,**r} for p in packets for j,r in p['requests'].items()),key=lambda r:r['bytes'])
    result={'schema_version':'p3new.reference_anchor_development.v1','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_ENGINEERING_ONLY','packet_count':len(packets),'legacy_bodies_byte_identical':len(packets),'png_count':len(new_images),'png_unique_sha256':len({i['sha256'] for i in new_images}),
        'all_reviewers_user_blocks_identical':True,'all_nonprotocol_material_blocks_unchanged':True,'all_catalog_anchors_exactly_resolve':True,
        'catalog_total_items':anchor_n,'catalog_maxima':{field:{'packet_id':p['packet_id'],'value':p[field]} for field,p in maxima.items()},'largest_request':body_max,
        'diagnostic_targets_represented':len(conversions),'input_refs':list(refs.values()),'output_refs':outputs,'packets':packets,'checks':checks,
        'inputs_unchanged':unchanged,'model_calls':0,'gpu_calls':0,'semantic_score_assigned':False,'original_answer_edited':False,'facts_seal_created':False,
        'token_measurement':'NOT_MEASURED_IN_THIS_CPU_CARRIER_CHECK','limits':['Only current facts packets, not report-stage original-seal replay.','Resolvable addresses do not prove semantics, necessity, completeness, or actual model compliance.','The 19 diagnostics are representable locations, never substituted model answers.','New protocol documents and anchor catalogs are development artifacts, not a new frozen confirmation round.']}
    save(OUT/'receipt.json',result)
    print(json.dumps({k:result[k] for k in ('status','packet_count','png_count','catalog_total_items','catalog_maxima','largest_request','diagnostic_targets_represented')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
