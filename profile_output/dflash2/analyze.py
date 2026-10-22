"""Validate and summarize the actual DFlash2/Strata runs; no synthetic performance numbers."""
import json,re,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
runs=['graph-main','adaptive-main','mtp-final']
out={'standard':'all GPU model weights <= 50% of physical VRAM','runs':{}}
for name in runs:
 records=json.loads((r/name/'results.json').read_text());usage=json.loads((r/name/'weight-usage.json').read_text())
 assert usage['weights']<=usage['limit']==usage['total']//2
 assert usage['weights']==sum(usage[x] for x in ['pack','dense','head','experts','mtp'])+usage.get('external_draft',0)
 assert 'PASS' in (r/(name+'.log')).read_text()
 events=json.loads((r/name/'memory-after.json').read_text())['memory.events']
 assert all(int(s.split()[1])==0 for s in events.splitlines() if s.split()[0] in ['oom','oom_kill'])
 entry={'weights':usage,'workloads':{}}
 for kind in ['story','chinese','code']:
  rows=[x for x in records if x['case'].startswith(kind+'-')]
  assert len(rows)==3 and all(len(x['generated_ids'])==256 for x in rows)
  vals=[x['decode_tps'] for x in rows]
  entry['workloads'][kind]={'median_tps':statistics.median(vals),'min_tps':min(vals),'max_tps':max(vals),'tps':vals,'requests':len(rows)}
 if name!='mtp-final':
  w=json.loads((r/name/'worker.json').read_text());assert w['worker_error'] is None and w['engine_exit']==0
  times=[x for x in w['timings'] if x.get('graph')]
  entry['worker']={'weight_bytes':w['weight_bytes'],'calls':w['totals']['calls'],'graphs':w['graphs'],'graphed_calls':len(times),'median_graphed_draft_ms':statistics.median(x['draft_ms'] for x in times),'proposal_count_histogram':{str(k):sum(x['k']==k for x in w['timings']) for k in [1,2,3]}}
 out['runs'][name]=entry
# Every target-layout-equivalence claim uses the same 8300 GPU expert layout.
ref=json.loads((r/'quality-final/results.json').read_text())
checks=[]
for name in ['graph-main','adaptive-main']:
 rows=json.loads((r/name/'results.json').read_text())
 for control in ref:
  if control['case'] in ['story-0','chinese-0','code-0']:
   candidates=[x for x in rows if x['case'].startswith(control['case'][:-1])]
   assert len(candidates)==3
  else:
   candidates=[x for x in rows if x['case']==control['case']]
   assert len(candidates)==1,(name,control['case'])
  for got in candidates:
   assert got['generated_ids']==control['generated_ids'],(name,got['case'],'target IDs changed')
   checks.append({'run':name,'case':got['case'],'tokens':len(control['generated_ids']),'identical':True})
 out['runs'][name]['protocol']=json.loads((r/name/'protocol.json').read_text())
out['target_equivalence']=checks
for kind in ['story','chinese','code']:
 a=out['runs']['adaptive-main']['workloads'][kind]['median_tps'];b=out['runs']['mtp-final']['workloads'][kind]['median_tps']
 out['runs']['adaptive-main']['workloads'][kind]['change_vs_mtp_pct']=100*(a/b-1)
(r/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for name,entry in out['runs'].items():print(name,{k:round(v['median_tps'],2) for k,v in entry['workloads'].items()})
print('PASS target-equivalent requests',len(checks))
