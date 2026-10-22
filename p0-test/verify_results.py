"""Audit saved native-model results; mock API tests are recorded separately."""
import json
from pathlib import Path
import re
import numpy as np

root=Path(__file__).resolve().parent
base_dir=root/'p0-final-matrix'
final_dir=root/'p0-final-adaptive'
base=json.loads((base_dir/'results.json').read_text())
adaptive=json.loads((final_dir/'results.json').read_text())
off={x['case']:x for x in base if x['mode']=='off'}
expected_cases={'warmup','ok','arithmetic','capital','story','chinese','code','sample'}
assert set(off)==expected_cases
assert {x['case'] for x in adaptive}==expected_cases|{'context_tail'}
summary={'matched_cache_slots':19000,'max_context':512,'bs':1,'greedy_tps':{},'seeded_sample_tps':{},
         'target_equivalence':[],'short_answers':{},'limits':['one measured request per workload/mode after warmup',
         'fixed4/fixed5 are archived exploratory runs, not mixed into the final timing table',
         'native INT8 KV on one gfx936 only; no new full-model quality benchmark']}
for mode in ['off','fixed2','fixed3','adaptive']:
    rows=adaptive if mode=='adaptive' else [x for x in base if x['mode']==mode]
    if mode!='off':
        for item in rows:
            if item['case']=='context_tail':continue
            assert item['generated_ids']==off[item['case']]['generated_ids'],(mode,item['case'])
            summary['target_equivalence'].append({'mode':mode,'case':item['case'],'tokens':len(item['generated_ids'])})
        directory=final_dir if mode=='adaptive' else base_dir
        for i in range(8):
            x=np.fromfile(base_dir/f'off.logits.f32.{i}',np.float32)
            y=np.fromfile(directory/f'{mode}.logits.f32.{i}',np.float32)
            assert x.size==248320 and y.size==x.size and np.isfinite(x).all() and np.isfinite(y).all()
            assert np.array_equal(x.view('u4'),y.view('u4')),(mode,i)
    summary['greedy_tps'][mode]={x['case']:x['decode_tps'] for x in rows if x['case'] in ['story','chinese','code']}
    summary['seeded_sample_tps'][mode]=next(x['decode_tps'] for x in rows if x['case']=='sample')
    summary['short_answers'][mode]={x['case']:x['text'] for x in rows if x['case'] in ['ok','arithmetic','capital']}
    assert summary['short_answers'][mode]=={'ok':'OK','arithmetic':'4','capital':'北京'}
counts=[int(x) for x in re.findall(r'^mtp budget: (\d+) observations;',
    (final_dir/'adaptive.stderr.log').read_text(),re.M)]
assert len(counts)==9 and len(set(counts[:4]))==1,'EOS windows were included as useful-token feedback'
tail=next(x for x in adaptive if x['case']=='context_tail')
assert len(tail['prompt_ids'])==501 and len(tail['generated_ids'])==3
assert tail['done'].split()[5]=='length'
protocol=json.loads((root/'p0-protocol/results.json').read_text())
assert protocol['engine_exit']==0
checks={x['case']:x for x in protocol['protocol']}
assert checks['cancel']['result']['done'].split()[5]=='cancel'
assert checks['recovery']['result']['text'].strip()=='OK'
assert checks['seeded_repeat']['results'][0]['ids']==checks['seeded_repeat']['results'][1]['ids']
assert len(checks['one_token_limit']['result']['ids'])==1
assert all(x['exit']==2 for x in protocol['invalid_options'])
front=(root/'p0-front-check.stderr.log').read_text()
front_checks=re.findall(r'mtp front-only check OK: cells (\d+)\.\.(\d+), (\d+) bytes',front)
assert len(front_checks)>=4 and any(int(b)>int(a) for a,b,_ in front_checks)
cap8=json.loads((root/'p0-cap8/results.json').read_text())
assert cap8['window_cap']==8 and cap8['engine_exit']==0 and cap8['generated_ids']==off['story']['generated_ids'][:64]
assert re.search(r'T8=[1-9]\d*@',cap8['budget'])
default=json.loads((root/'p0-wrapper-off.json').read_text())
previous=json.loads((root.parent/'profile_output/vec4-story-abba/0-vec4-story.json').read_text())
assert default['prompt_ids']==previous['prompt_ids'] and default['generated_ids']==previous['generated_ids']
summary.update(first_logits_bit_identical=True,matched_requests=len(summary['target_equivalence']),
    eos_feedback_censored=True,context_tail_passed=True,front_only_kv_checks=front_checks,
    protocol_passed=True,window_cap8_passed=True,default_matches_6ec15a8=True)
(root/'verified-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print('PASS: 24 MTP requests/first-logit arrays match off; EOS feedback, boundary, KV, STOP, RNG, guards and default verified')
for mode,values in summary['greedy_tps'].items():print(mode,{k:round(v,2) for k,v in values.items()})
