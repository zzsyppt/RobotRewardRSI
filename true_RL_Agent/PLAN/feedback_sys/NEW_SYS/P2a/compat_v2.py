"""Compare every v2 record with base-commit code without another GB artifact."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src'))
BASE='aaf67aa23bd7e4aa8aae7a9ec9539f3aa45c6c3e'


def old_module(relative,name):
    raw=subprocess.check_output(['git','show',BASE+':true_RL_Agent/src/true_rl_agent/'+relative],cwd=ROOT)
    path=ROOT/'outputs/p2a'/('baseline_'+relative.rsplit('/',1)[-1])
    if not path.exists():path.write_bytes(raw)
    assert path.read_bytes()==raw
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module,hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',choices=['k1','k3'],required=True);args=p.parse_args()
    adapter,ah=old_module('tasks/stair_climbing/trajectory_adapter.py','true_rl_agent.tasks.stair_climbing._baseline_adapter')
    measure,mh=old_module('evaluation/trajectory_translation/measurement.py','true_rl_agent.evaluation.trajectory_translation._baseline_measurement')
    report,rh=old_module('evaluation/trajectory_translation/report.py','true_rl_agent.evaluation.trajectory_translation._baseline_report')
    out=ROOT/'outputs/p2a'/args.source
    replay=json.loads((out/'replay.json').read_text());manifest_path=Path(replay['numeric_manifest']);nm=json.loads(manifest_path.read_text())
    started=time.perf_counter();records=episodes=0;digest=hashlib.sha256()
    with manifest_path.with_name('evidence.jsonl').open('rb') as f:
        for e in adapter.iter_stair_episodes(nm['source_manifest_ref'],source_id=nm['source_id']):
            raw=f.readline();header=json.loads(raw)
            assert header['source']==e.source
            assert header['task']==e.task
            digest.update(raw);episodes+=1
            for record in measure.measure_episode(e):
                expected=(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n').encode()
                actual=f.readline()
                assert actual==expected,(args.source,e.run_index,e.env_index,record['key'])
                digest.update(expected);records+=1
        assert f.read()==b''
    assert digest.hexdigest()==nm['evidence']['sha256']
    # Old card builder, same archive: every claim/ID/selection stays unchanged.
    from true_rl_agent.tasks.stair_climbing.trajectory_artifacts import _load
    from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
    bundle=_load(manifest_path,file_sha256(manifest_path))
    card,_=report.build_card(bundle,[],{'episodes':[],'source_manifest_hashes':{}})
    current=json.loads(Path(replay['card_reference']['manifest_path']).with_name('policy_card.json').read_text(encoding='utf-8'))
    assert card==current
    result={'passed':True,'source':args.source,'base_commit':BASE,'episodes':episodes,'record_bytes_compared':records,
        'numeric_sha256':digest.hexdigest(),'all_card_claims_and_ids_equal':len(card['claims']),
        'baseline_code_sha256':{'adapter':ah,'measurement':mh,'report':rh},'wall_seconds':time.perf_counter()-started,
        'storage':'only small old source snapshots; no duplicate numeric evidence'}
    with (ROOT/'outputs/p2a'/(args.source+'_compat.json')).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
