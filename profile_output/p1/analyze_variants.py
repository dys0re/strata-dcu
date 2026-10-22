"""Audit optional fusion experiments, including the shared-stream prerequisite."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
reference={r['case']:r for r in json.loads((root/'a0/results.json').read_text())}
primary=json.loads((root/'primary-results.json').read_text())
summary={}
for name in ['inline-control','fusion-qdedup','fusion-qfuse']:
 exit_code=int((root/(name+'.exit')).read_text())
 cfg=json.loads((root/name/'command.json').read_text());cmd=cfg['command'];env=cfg['overrides']
 assert '--mtp-adaptive' in cmd and env['STRATA_BF16_Q8_COLS']=='4'
 assert not cfg['profile'] and not cfg['dump_logits']
 if name in ['inline-control','fusion-qdedup']:assert env['STRATA_SH_STREAM']=='0'
 else:assert 'STRATA_SH_STREAM' not in env
 assert env['STRATA_VERIFY_QDEDUP']==str(int(name=='fusion-qdedup'))
 assert env['STRATA_QFUSE']==str(int(name=='fusion-qfuse'))
 data_path=root/name/'results.json'
 rows=json.loads(data_path.read_text()) if data_path.exists() else []
 mismatched=[r['case'] for r in rows if r['prompt_ids']!=reference[r['case']]['prompt_ids'] or r['generated_ids']!=reference[r['case']]['generated_ids']]
 good=exit_code==0 and len(rows)==17 and not mismatched
 record={'engine_driver_exit':exit_code,'recorded_requests':len(rows),'mismatched_cases':mismatched,'all_17_requests_match':good,'performance':{}}
 if good:
  p=json.loads((root/name/'protocol.json').read_text())
  assert p['cancel'].split()[5]=='cancel' and p['recovery'].split()[5]=='stop'
  assert p['seed_repeat'] and p['one_token_limit'] and p['context_tail']
  for case in ['story','chinese','code']:
   samples=[r['decode_tps'] for r in rows if r['case'].startswith(case+'-')];assert len(samples)==3
   record['performance'][case]={'median':statistics.median(samples),'min':min(samples),'max':max(samples),'samples':samples}
 summary[name]=record
for case in ['story','chinese','code']:
 if summary['fusion-qdedup']['all_17_requests_match'] and summary['inline-control']['all_17_requests_match']:
  q=summary['fusion-qdedup']['performance'][case];control=summary['inline-control']['performance'][case]
  q['relative_to_same_inline_control_percent']=(q['median']/control['median']-1)*100
 for name,record in summary.items():
  if record['all_17_requests_match']:
   r=record['performance'][case]
   r['relative_to_primary_p1_percent']=(r['median']/primary['performance'][case]['b']['median']-1)*100
(root/'fusion-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
for name,r in summary.items():
 print(name,'exit',r['engine_driver_exit'],'all 17 requests match',r['all_17_requests_match'])
 print({case:round(x['median'],2) for case,x in r['performance'].items()})
 if r['mismatched_cases']:print('MISMATCHES',r['mismatched_cases'])
print('Exploratory variants: one engine startup and three requests per workload; not ABBA replications')
