"""Choose quiet physical cores after model load, preferring proximity to the existing expert pages."""
from pathlib import Path
import os,time,json

def cpu_snapshot():
 return {int(s.split()[0][3:]):list(map(int,s.split()[1:])) for s in Path('/proc/stat').read_text().splitlines() if s.startswith('cpu') and s.split()[0]!='cpu'}

def choose_and_pin(pid,allowed,participants,output):
 proc=Path(f'/proc/{pid}');original={}
 for task in (proc/'task').iterdir():
  try:original[int(task.name)]=sorted(os.sched_getaffinity(int(task.name)))
  except ProcessLookupError:pass
 old_cores=sorted({v[0] for v in original.values() if len(v)==1})
 assert len(old_cores)==participants,(old_cores,participants)
 before_numa=(proc/'numa_maps').read_text();largest={};largest_size=0
 for row in before_numa.splitlines():
  fields=dict(s.split('=',1) for s in row.split() if '=' in s)
  size=int(fields.get('anon',0))
  if size>largest_size:largest_size=size;largest={int(k[1:]):int(v) for k,v in fields.items() if k.startswith('N') and k[1:].isdigit()}
 topology={};seen=set()
 for c in sorted(allowed):
  p=Path(f'/sys/devices/system/cpu/cpu{c}');key=tuple((p/'topology'/n).read_text().strip() for n in ['physical_package_id','core_id'])
  if key in seen:continue
  seen.add(key);nodes=[int(n.name[4:]) for n in p.glob('node*') if n.name[4:].isdigit()]
  if nodes:topology.setdefault(nodes[0],[]).append(c)
 raw=[];a=cpu_snapshot()
 for _ in range(6):
  time.sleep(.5);b=cpu_snapshot();raw.append(dict(start=a,end=b));a=b
 busy={}
 for c in allowed:
  ds=[[v-u for u,v in zip(x['start'][c],x['end'][c])] for x in raw]
  busy[c]=sum(sum(d)-d[3]-d[4] for d in ds)/max(1,sum(sum(d) for d in ds))
 candidates=[]
 for node,cores in topology.items():
  if len(cores)<participants:continue
  chosen=sorted(sorted(cores,key=lambda c:(busy[c],c))[:participants])
  distance=list(map(int,Path(f'/sys/devices/system/node/node{node}/distance').read_text().split()))
  cost=sum(distance[n]*pages for n,pages in largest.items())/max(1,sum(largest.values()))
  candidates.append(dict(node=node,cores=chosen,max_busy=max(busy[c] for c in chosen),mean_busy=sum(busy[c] for c in chosen)/participants,weighted_expert_distance=cost))
 assert candidates,'No NUMA node has enough allowed physical cores'
 quiet=[c for c in candidates if c['max_busy']<=.15]
 if quiet:chosen=min(quiet,key=lambda c:(c['weighted_expert_distance'],c['mean_busy'],c['max_busy'],c['node']))
 else:chosen=min(candidates,key=lambda c:(c['max_busy'],c['mean_busy'],c['weighted_expert_distance']))
 mapping=dict(zip(old_cores,chosen['cores']));changes={}
 for tid,mask in original.items():
  if len(mask)==1 and mask[0] in mapping:
   try:
    os.sched_setaffinity(tid,{mapping[mask[0]]});actual=sorted(os.sched_getaffinity(tid));assert actual==[mapping[mask[0]]]
    changes[tid]=dict(before=mask,after=actual)
   except ProcessLookupError:pass
 assert pid in changes
 result=dict(chosen=chosen,all_quiet=bool(quiet),threshold=.15,participants=participants,old_cores=old_cores,mapping=mapping,candidates=candidates,per_cpu_busy=busy,changes=changes,expert_pages=largest,expert_anon_pages=largest_size,samples=raw,original_threads=original)
 Path(output).write_text(json.dumps(result,indent=2)+'\n')
 Path(str(output)+'.numa-before.txt').write_text(before_numa)
 print('IDLE_CORE_SELECTION',chosen,flush=True)
 return result
