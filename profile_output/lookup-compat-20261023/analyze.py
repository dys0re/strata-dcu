"""Summarize only numbered throughput cases; QA cases have a separate count."""
import json,re,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
rows={}
for base in [root.parent/'upstream-options-20261023',root]:
 for p in sorted(base.iterdir()):
  if not p.is_dir() or not (p/'results.json').exists():continue
  records=json.loads((p/'results.json').read_text())
  med={};samples={}
  for name in ['story','chinese','code']:
   samples[name]=[r['decode_tps'] for r in records if re.fullmatch(name+r'-\d+',r['case'])]
   if samples[name]:med[name]=statistics.median(samples[name])
  required={'ok','arithmetic','multiply','sum','capital','uppercase','sort-json','json','code-function','multi-turn','sample-0','sample-1','one_token','context_tail'}
  cases={r['case'] for r in records}
  report={'median_tps':med,'samples_tps':samples,'requests':len(records),
          'run_pass':(p/'qa-status.json').exists(),
          'full_qa_pass':required<=cases and (p/'qa-status.json').exists() and (p/'protocol.json').exists(),
          'qa_cases_present':sorted(required&cases)}
  if (p/'weight-usage.json').exists():report['weights']=json.loads((p/'weight-usage.json').read_text())
  if (p/'stderr.log').exists():
   log=(p/'stderr.log').read_text()
   report['hc']=[s for s in log.splitlines() if s.startswith('strata hc:')]
   report['pcie_policy']=[s for s in log.splitlines() if 'PCIe probe:' in s or 'PCIe share:' in s]
   report['suffix_windows']=[list(map(int,c)) for c in re.findall(r'suffix drafts: (\d+) windows, (\d+) of (\d+) drafts accepted',log)]
   report['discarded_feedback']=[int(v) for v in re.findall(r'discarded_feedback=(\d+)',log)]
  rows[base.name+'/'+p.name]=report
(root/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
for name,r in rows.items():
 if r['median_tps']:print(name,{k:round(v,2) for k,v in r['median_tps'].items()},'full QA',r['full_qa_pass'])
