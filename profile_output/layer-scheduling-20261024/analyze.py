import json,statistics,re,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
rows={}
for folder in sorted(root.iterdir()):
 if not folder.is_dir() or not (folder/'results.json').exists():continue
 records=json.loads((folder/'results.json').read_text())
 thread=json.loads((folder/'thread-usage.json').read_text()) if (folder/'thread-usage.json').exists() else []
 usage=[]
 for x in thread:
  start,end=x['start'],x['end'];dt=end['time_ns']-start['time_ns']
  threads=[]
  for tid,b in end['threads'].items():
   if tid not in start['threads']:continue
   a=start['threads'][tid]
   threads.append(dict(tid=int(tid),affinity=b['affinity'],cpu_pct=100*(b['runtime_ns']-a['runtime_ns'])/dt))
  host=next((t['cpu_pct'] for t in threads if t['tid']==x['pid']),None)
  workers=[t['cpu_pct'] for t in threads if t['affinity'] in list(map(str,range(1,13)))]
  usage.append(dict(done=x['done'],host=host,worker_mean=statistics.mean(workers) if workers else None))
 qa=json.loads((folder/'qa-status.json').read_text()) if (folder/'qa-status.json').exists() else None
 stderr=(folder/'stderr.log').read_text() if (folder/'stderr.log').exists() else ''
 workloads={}
 for case in ['story','chinese','code']:
  data=[x for x in records if re.fullmatch(case+r'-\d+',x['case'])]
  if not data:continue
  us=[]
  for x in data:
   us.extend([u for u in usage if u['done']==x['done']])
  workloads[case]=dict(tps=[x['decode_tps'] for x in data],median_tps=statistics.median(x['decode_tps'] for x in data),
   host_cpu_pct=statistics.median(u['host'] for u in us) if us else None,worker_cpu_pct=statistics.median(u['worker_mean'] for u in us) if us else None)
 rows[folder.name]=dict(workloads=workloads,qa=qa,pcie_per_layer=[float(s) for s in re.findall(r'VRAM hits [0-9.]+, PCIe ([0-9.]+);',stderr)],
  gpu_plan_logged='resident groups start from a separate GPU plan' in stderr,cpu_pipeline_logged='per-expert gate/up -> quantize -> down pipeline enabled' in stderr,
  staged_logged='bounded DMA offload' in stderr)
(root/'analysis.json').write_text(json.dumps(rows,indent=2)+'\n')
for label,r in rows.items():
 print(label,[(k,round(v['median_tps'],2),round(v['worker_cpu_pct'],1) if v['worker_cpu_pct'] is not None else None) for k,v in r['workloads'].items()],r['qa'],max(r['pcie_per_layer'],default=0))
