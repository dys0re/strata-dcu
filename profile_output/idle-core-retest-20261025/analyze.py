"""Summarize completed retests and actual per-request thread queueing; no cross-period normalization."""
from pathlib import Path
import json,re,statistics,sys
R=Path(sys.argv[1]);summary={}
for d in sorted(R.iterdir()):
 if not d.is_dir() or not (d/'results.json').exists():continue
 results=json.loads((d/'results.json').read_text());tr=json.loads((d/'thread-usage.json').read_text()) if (d/'thread-usage.json').exists() else []
 select=json.loads((d/'idle-core-selection.json').read_text()) if (d/'idle-core-selection.json').exists() else None
 orig=select['original_threads'] if select else {};groups={};requests=[]
 for result,row in zip(results,tr):
  duration=(row['end']['time_ns']-row['start']['time_ns'])/1e9;workers=[];main=None;masks={}
  for tid,b in row['start']['threads'].items():
   e=row['end']['threads'].get(tid)
   if not e:continue
   q=(e['runqueue_ns']-b['runqueue_ns'])/1e9/duration
   runtime=(e['runtime_ns']-b['runtime_ns'])/1e9/duration
   if tid==str(row['pid']):main=dict(queue_fraction=q,runtime_fraction=runtime,affinity=b['affinity'])
   mask=orig.get(tid)
   if (mask and len(mask)==1 and mask[0]!=0) or (not select and b['affinity'].isdigit() and 1<=int(b['affinity'])<=12):
    workers.append(dict(tid=tid,queue_fraction=q,runtime_fraction=runtime,affinity=b['affinity']))
  requests.append(dict(case=result['case'],main=main,workers=workers,wall_seconds=duration))
 for label in ['story','chinese','code']:
  rows=[v for v in results if re.fullmatch(label+r'-\d+',v['case'])]
  if not rows:continue
  queue=[v['queue_fraction'] for x in requests if re.fullmatch(label+r'-\d+',x['case']) for v in x['workers']]
  mainq=[x['main']['queue_fraction'] for x in requests if re.fullmatch(label+r'-\d+',x['case'])]
  groups[label]=dict(tps=[v['decode_tps'] for v in rows],median_tps=statistics.median(v['decode_tps'] for v in rows),prefill_ms=[float(v['done'].split()[3]) for v in rows],median_prefill_ms=statistics.median(float(v['done'].split()[3]) for v in rows),worker_queue_median=statistics.median(queue),worker_queue_max=max(queue),main_queue_max=max(mainq))
 qa=json.loads((d/'qa-status.json').read_text()) if (d/'qa-status.json').exists() else None
 weight=json.loads((d/'weight-usage.json').read_text()) if (d/'weight-usage.json').exists() else None
 summary[d.name]=dict(workloads=groups,qa=qa,weights=weight,selection=select['chosen'] if select else None,all_quiet=select['all_quiet'] if select else None,requests=requests)
 print(d.name,{k:round(v['median_tps'],2) for k,v in groups.items()},'QA',qa)
(R/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
