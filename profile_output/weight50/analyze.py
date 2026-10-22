"""Summarize only this fixed-cache, <=50%-GPU-weight speed measurement."""
import json,re,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
result={'standard':'DCU GPU weight allocations <= 50% physical VRAM','gpu_experts':8500,'ram_experts':16076,'enhanced_mtp':'adaptive','runs':['run1','run2'],'workloads':{}}
all_records=[];usages=[]
for run in result['runs']:
 records=json.loads((root/run/'results.json').read_text());all_records+=records
 usage=json.loads((root/run/'weight-usage.json').read_text());usages.append(usage)
 assert usage['weights']<=usage['limit']==usage['total']//2
 assert 'PASS' in (root/(run+'.log')).read_text()
 events=json.loads((root/run/'memory-after.json').read_text())['memory.events']
 assert all(int(s.split()[1])==0 for s in events.splitlines() if s.split()[0] in ['oom','oom_kill'])
assert usages[0]['weights']==usages[1]['weights']
result['weight_usage']=usages
for kind in ['story','chinese','code']:
 records=[r for r in all_records if r['case'].startswith(kind+'-')]
 assert len(records)==6 and all(len(r['generated_ids'])==256 for r in records)
 rates=[r['decode_tps'] for r in records]
 # Fixed inputs/engine/settings should retain output IDs across repetitions and restarts.
 assert all(r['generated_ids']==records[0]['generated_ids'] for r in records)
 result['workloads'][kind]={'requests':len(records),'tokens_per_request':256,'decode_tps':rates,'median_tps':statistics.median(rates),'min_tps':min(rates),'max_tps':max(rates),'identical_output_ids':True}
result['checks']=json.loads((root/'run1/protocol.json').read_text())
result['over_budget']=json.loads((root/'over-budget-result.json').read_text())
(root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result['workloads'],ensure_ascii=False,indent=2))
