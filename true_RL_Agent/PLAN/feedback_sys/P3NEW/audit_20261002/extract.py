import json,sys,io,glob,os,pickle
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
out={}
for d in sorted(glob.glob('prepared/episode-*')):
    p=os.path.join(d,'gpt6.1-sol-a','request.http_body.json')
    body=json.load(open(p,encoding='utf-8'))
    parts=body['messages'][1]['content']
    blocks={}
    q=None
    for part in parts:
        try: o=json.loads(part['text'])
        except Exception: continue
        if o.get('role')=='focus_questions': q=o['data']
        if o.get('role')=='reference_material_block':
            loc=o['locator']
            if loc['material_id'] in('related-raw',) or loc['material_id'].startswith('related'):
                blocks[loc['json_pointer']]=o['value']
    out[os.path.basename(d)]={'q':q,'blocks':blocks}
    b=blocks
    print(os.path.basename(d), q['task'], q['evidence_scope'], json.dumps(b.get('/data/episode/original_terminal_record'),ensure_ascii=False)[:200], len(b))
pickle.dump(out,open(sys.argv[1],'wb'))
