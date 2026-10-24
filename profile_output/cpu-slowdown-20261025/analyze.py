"""Join per-request engine timing, thread scheduler counters, NUMA snapshots and external CPU samples."""
from pathlib import Path
import json,os,sys
r=Path(sys.argv[1]);label=sys.argv[2];d=r/label
threads=json.loads((d/'thread-usage.json').read_text());cases=json.loads((d/'results.json').read_text());switches=json.loads((d/'affinity-switches.json').read_text());samples=json.loads((r/(label+'-samples.json')).read_text())
lines=[s for s in (d/'stderr.log').read_text().splitlines() if s.startswith('strata decode timing:')]
assert len(threads)==len(cases)==len(switches)==len(lines)
def kv(v):return {k:int(v) for k,v in [l.split() for l in v.splitlines()]}
result=[]
for row,c,m,line in zip(threads,cases,switches,lines):
 lo=row['start']['time_ns']/1e9;hi=row['end']['time_ns']/1e9;dt=hi-lo;ts=[]
 for tid,b in row['start']['threads'].items():
  e=row['end']['threads'].get(tid)
  if e and (e['runtime_ns']-b['runtime_ns'])/1e9/dt>.02:
   ts.append(dict(tid=tid,affinity=b['affinity'],cpu_fraction=(e['runtime_ns']-b['runtime_ns'])/1e9/dt,runqueue_fraction=(e['runqueue_ns']-b['runqueue_ns'])/1e9/dt))
 ss=[v for v in samples if lo<=v['time']<=hi];delta={};busy=[];used=None
 if len(ss)>1:
  b,e=kv(ss[0]['cgroup']),kv(ss[-1]['cgroup']);delta={k:e[k]-v for k,v in b.items()};span=ss[-1]['time']-ss[0]['time']
  for i in range(32):
   ds=[v-u for u,v in zip(ss[0]['cpu'][f'cpu{i}'],ss[-1]['cpu'][f'cpu{i}'])]
   busy.append((sum(ds)-ds[3]-ds[4])/os.sysconf('SC_CLK_TCK')/span)
  used=delta['usage_usec']/1e6/span
 large=[]
 for n in m['numa_maps'].splitlines():
  f=dict(s.split('=',1) for s in n.split() if '=' in s)
  if int(f.get('anon',0))>100000:large.append({k:int(v) for k,v in f.items() if k.startswith('N') or k=='anon'})
 item=dict(case=c['case'],tps=c['decode_tps'],wall_seconds=dt,threads=ts,cgroup_delta=delta,large_anon=large,timing=line,node0_busy_cpu_equivalents=sum(busy[:16]) if busy else None,node1_busy_cpu_equivalents=sum(busy[16:]) if busy else None,all_container_cpu_equivalents=used)
 if len(ss)>1:
  ticks=os.sysconf('SC_CLK_TCK')*span
  percore=[[v-u for u,v in zip(ss[0]['cpu'][f'cpu{i}'],ss[-1]['cpu'][f'cpu{i}'])] for i in range(16)]
  item.update(node0_user_cpu_equivalents=sum(v[0]+v[1] for v in percore)/ticks,
              node0_irq_cpu_equivalents=sum(v[5]+v[6] for v in percore)/ticks,
              node0_system_cpu_equivalents=sum(v[2] for v in percore)/ticks,
              container_user_cpu_equivalents=delta['user_usec']/1e6/span)
 result.append(item);print(c['case'],round(c['decode_tps'],2),'throttles',delta.get('nr_throttled'))
(r/(label+'-analysis.json')).write_text(json.dumps(result,indent=2)+'\n')
