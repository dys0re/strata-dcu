import json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
rs=json.loads((p/'results.json').read_text());out=[]
for r in rs:
 if r['case']=='warmup':continue
 a=r['decode_start'];b=r['after'];dt=b['time_ns']-a['time_ns'];threads=[]
 for tid,y in b['threads'].items():
  if tid not in a['threads']:continue
  x=a['threads'][tid]
  threads.append({'tid':int(tid),'affinity':y['affinity'],'runtime_ms':(y['runtime_ns']-x['runtime_ns'])/1e6,'cpu_pct':100*(y['runtime_ns']-x['runtime_ns'])/dt,'runqueue_ms':(y['runqueue_ns']-x['runqueue_ns'])/1e6})
 host=next(t for t in threads if t['tid']==r['pid'])
 workers=[t for t in threads if t['affinity'] in list(map(str,range(1,13)))]
 assert len(workers)==12,(r['case'],workers)
 fields=r['done'].split()
 def cg(v):return {s.split()[0]:int(s.split()[1]) for s in v.splitlines()}
 ca=cg(a['cgroup_cpu']);cb=cg(b['cgroup_cpu'])
 out.append({'case':r['case'],'sample_wall_ms':dt/1e6,'reported_decode_ms':float(fields[4]),'decode_tps':1000*int(fields[1])/float(fields[4]),'host':host,'worker_mean_pct':statistics.mean(t['cpu_pct'] for t in workers),'worker_min_pct':min(t['cpu_pct'] for t in workers),'worker_max_pct':max(t['cpu_pct'] for t in workers),'threads':threads,'cgroup_delta':{k:cb[k]-ca[k] for k in ca}})
sums={}
for key in ['story','chinese','code']:
 a=[r for r in out if r['case'].startswith(key+'-')]
 sums[key]={'host_cpu_pct_median':statistics.median(r['host']['cpu_pct'] for r in a),'worker_mean_pct_median':statistics.median(r['worker_mean_pct'] for r in a),'host_runqueue_ms':sum(r['host']['runqueue_ms'] for r in a),'throttled_usec':sum(r['cgroup_delta'].get('throttled_usec',0) for r in a),'nr_throttled':sum(r['cgroup_delta'].get('nr_throttled',0) for r in a)}
(p/'analysis.json').write_text(json.dumps({'summary':sums,'requests':out},indent=2)+'\n')
print(json.dumps(sums,indent=2))
