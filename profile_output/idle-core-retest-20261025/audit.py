"""Check final scope, real feature activation, fixed resources, QA and scheduling evidence."""
from pathlib import Path
import json,re,hashlib,subprocess
R=Path('/root/private_data/strata-dcu/idle-core-retest-20261025');V=R/'v2'
status=json.loads((V/'status.json').read_text());labels=['idle-bf16','idle-bf16-proxy','idle-selected','idle-selected-proxy']
assert [x['label'] for x in status]==labels and all(x['returncode']==0 for x in status)
assert json.loads((V/'job-status.json').read_text())['returncode']==0
subprocess.run(['python3',str(R/'analyze.py'),str(V)],check=True)
a=json.loads((V/'analysis.json').read_text());report=[]
qa_cases={'ok','arithmetic','multiply','sum','capital','uppercase','sort-json','json','code-function','multi-turn','sample-0','sample-1','one_token','context_tail','code-search','code-intervals','code-encoding','data-transform'}
for label in labels:
 d=V/label;result=a[label];rows=json.loads((d/'results.json').read_text());cmd=json.loads((d/'command.json').read_text());selection=json.loads((d/'idle-core-selection.json').read_text());w=result['weights'];q=result['qa'];protocol=json.loads((d/'protocol.json').read_text())
 assert qa_cases.issubset({x['case'] for x in rows})
 assert q['functional_assertions_passed'] and q['qa_scope']=='full' and q['extended_qa'] and q['engine_rc']==0
 assert protocol['cancel'].split()[5]=='cancel' and protocol['one_token_limit'] and protocol['context_tail']
 assert w['experts']==23500800000 and w['weights']<=w['limit']==34351349760
 assert w['mtp_proxy']==(85043200 if 'proxy' in label else 0)
 assert len(selection['chosen']['cores'])==13 and len(selection['candidates'])==8 and selection['all_quiet']
 for k,g in result['workloads'].items():
  assert len(g['tps'])==3
  assert all(int(x['done'].split()[1])==256 for x in rows if re.fullmatch(k+r'-\d+',x['case']))
 for r in result['requests']:
  if not re.fullmatch(r'(story|chinese|code)-\d+',r['case']):continue
  assert len(r['workers'])==12
  assert int(r['main']['affinity'])==selection['chosen']['cores'][0]
  assert {int(t['affinity']) for t in r['workers']}==set(selection['chosen']['cores'][1:])
 actual=cmd['command'];assert actual[actual.index('--expert-cache')+1]=='8500' and actual[actual.index('--pool-workers')+1]=='12'
 selected='selected' in label
 assert ('dense-q8.gguf' in actual[actual.index('--native')+1])==selected
 assert ('mtp-tuned' in actual[actual.index('--mtp')+1])==selected
 assert cmd['overrides']['STRATA_HC_Q8']==('1' if selected else '0')
 assert cmd['overrides']['STRATA_DRAFT_HEAD_PROXY']==('1' if 'proxy' in label else '0')
 report.append(dict(label=label,passed=True,qa_prompts=18,weights=w['weights'],cores=selection['chosen']['cores'],workloads=result['workloads']))
E=Path('/root/strata-dcu/build-gfx936/strata');sha=hashlib.sha256(E.read_bytes()).hexdigest();assert sha=='3ae138ead7fdbab0324bb17b423022d32c497eb18e3fb5aa82901011b98c9450'
(V/'audit.json').write_text(json.dumps(dict(passed=True,engine_sha256=sha,cases=report),indent=2)+'\n')
print('PASS FINAL AUDIT',flush=True)
