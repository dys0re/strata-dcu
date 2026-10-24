"""Summarize actual same-host inference runs and validate fixed weight budgets."""
from pathlib import Path
import json,statistics,csv,re,hashlib
root=Path(__file__).resolve().parent
summary={}
for name in ['mtp-off','mtp-on','native-off','native-both']:
 p=root/name
 if not (p/'results.json').exists():continue
 rs=json.loads((p/'results.json').read_text());u=json.loads((p/'weight-usage.json').read_text())
 assert u['weights']<=u['limit']==u['total']//2
 assert u['experts']==2764800*(8500 if name.startswith('mtp') else 8300)
 info={'weights':u,'workloads':{},'qa_status':json.loads((p/'qa-status.json').read_text()),'engine_sha256':json.loads((p/'provenance.json').read_text())['engine_sha256']}
 for work in ['story','chinese','code']:
  values=[r['decode_tps'] for r in rs if re.fullmatch(work+r'-\d+',r['case'])]
  assert len(values)==3,(name,work,len(values))
  info['workloads'][work]={'median_tps':statistics.median(values),'min_tps':min(values),'max_tps':max(values),'values':values}
 assert all(r['done'].split()[5] in ['stop','length'] for r in rs)
 summary[name]=info
for base,new in [('mtp-off','mtp-on'),('native-off','native-both')]:
 if base in summary and new in summary:
  summary[new]['gain_pct']={k:100*(summary[new]['workloads'][k]['median_tps']/summary[base]['workloads'][k]['median_tps']-1) for k in ['story','chinese','code']}
if (root/'bf16-small-final.csv').exists():
 lines=(root/'bf16-small-final.csv').read_text().splitlines();assert lines[-1]=='PASS BF16_SMALL_MM'
 rows=list(csv.DictReader(lines[:-1]));assert len(rows)==104
 summary['gpu_micro']={'cases':len(rows),'max_scaled_error':max(float(r['sampled_scaled_error']) for r in rows),'note':'Production keeps BLAS for the full vocabulary head above two rows.'}
summary['scope']='BW/gfx936 BS=1; 3 repeats per workload per single engine start; no claims of statistical significance.'
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for k,v in summary.items():
 if isinstance(v,dict) and 'workloads' in v:print(k,{w:round(s['median_tps'],3) for w,s in v['workloads'].items()},v.get('gain_pct'))
