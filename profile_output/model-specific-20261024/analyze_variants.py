"""Summarize completed cases, without conflating exploratory single runs and repeated results."""
import json,re,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);summary={}
for p in sorted(root.iterdir()):
 if not p.is_dir() or not (p/'results.json').exists():continue
 results=json.loads((p/'results.json').read_text());groups={}
 for label in ['story','chinese','code']:
  rows=[x for x in results if re.fullmatch(label+r'-\d+',x['case'])]
  if not rows:continue
  accepted=sum(int(x['done'].split()[6]) for x in rows);offered=sum(int(x['done'].split()[7]) for x in rows)
  groups[label]=dict(tps=[x['decode_tps'] for x in rows],prefill_ms=[float(x['done'].split()[3]) for x in rows],median_prefill_ms=statistics.median(float(x['done'].split()[3]) for x in rows),median_tps=statistics.median(x['decode_tps'] for x in rows),draft_accepted=accepted,draft_offered=offered,draft_acceptance=accepted/offered if offered else None)
 qa=json.loads((p/'qa-status.json').read_text()) if (p/'qa-status.json').exists() else None
 weight=json.loads((p/'weight-usage.json').read_text()) if (p/'weight-usage.json').exists() else None
 summary[p.name]=dict(workloads=groups,qa=qa,weights=weight)
 print(p.name,{k:round(v['median_tps'],2) for k,v in groups.items()},'QA',qa)
(root/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
