import csv,json,re,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'v9'
summary={}
for folder in sorted(root.glob('trace-*')):
 if not (folder/'complete.json').exists():continue
 records=json.loads((folder/'results.json').read_text());config=json.loads((folder/'command.json').read_text())
 gpu_first=config['flags'].get('STRATA_VERIFY_HIT_PLAN')=='1'
 windows=[];current=None;request=-1
 for line in (folder/'stderr.log').read_text().splitlines():
  if line.startswith('strata trace: request '):request+=1
  m=re.match(r'verify graph: pos=(\d+) T=(\d+) before_ns=(\d+) after_ns=(\d+)',line)
  if m:
   if current is not None:windows.append(current)
   pos,T,before,after=map(int,m.groups());current=dict(request=request,pos=pos,T=T,before=before,after=after,cpu={},gpu={});continue
  if current is None:continue
  m=re.match(r'pool trace: layer (\d+) (\S+) (-?\d+) time_ns=(\d+)',line)
  if m:
   layer,what,n,ts=m.groups();c=current['cpu'].setdefault(int(layer),{});c[what]=int(ts)
   if what=='run':c['cpu_experts']=int(n)
   if what=='publish':c['pcie_experts']=int(n)
   continue
  m=re.match(r'verify ready: ring=(\d+)(?: layer=\d+ group=(\d+))? time_ns=(\d+) early=(\d+)',line)
  if m:
   ring,group,ts,early=m.groups();assert group in [None,'0'],'timeline expects the documented unsplit bs=1 graph'
   assert 1<=int(ring)<=48
   current['cpu'].setdefault(int(ring)-1,{})['early_ready' if early=='1' else 'driver_ready']=int(ts);continue
  m=re.match(r'  layer (\d+) group (\d+):(.*)',line)
  if m:
   layer,group,body=m.groups()
   if int(group)==0:current['gpu'][int(layer)]={int(i):float(v) for i,v in re.findall(r'\((\d+)\)=([0-9.]+)',body)}
 if current is not None:windows.append(current)
 rows=[]
 for idx,w in enumerate(windows):
  assert 0<=w['request']<len(records),(folder,idx,w['request'])
  record=records[w['request']];prompt_n=int(record['done'].split()[2]);phase='decode' if w['pos']>=prompt_n-1 else 'prompt'
  assert set(w['cpu'])==set(range(48)),(folder,idx,'CPU layers',set(w['cpu']))
  assert all(l in w['gpu'] for l in range(48)),(folder,idx,'GPU layers')
  for layer in range(48):
   c=w['cpu'][layer];g=w['gpu'][layer]
   for mark in [0,17,18,19,22,24,33,34]:assert mark in g,(folder,idx,layer,mark)
   for mark in ['begin','publish','run','ran','driver_ready']:assert mark in c,(folder,idx,layer,mark)
   assert c['begin']<=c['publish']<=c['run']<=c['ran']<=c['driver_ready'],(folder,idx,layer,c)
   assert (g[33]<=g[19] if gpu_first else g[19]<=g[33]),(folder,idx,layer,'resident/plan order',g)
   resident_us=g[33]-(g[18] if gpu_first else g[19])
   plan_us=g[19]-(g[33] if gpu_first else g[18])
   assert min(resident_us,plan_us,g[34]-g[22],g[24]-g[34])>=-.11
   row=dict(window=idx,request=w['request'],case=record['case'],phase=phase,pos=w['pos'],T=w['T'],layer=layer,
    host_launch_before_ns=w['before'],host_launch_after_ns=w['after'],cpu_experts=c.get('cpu_experts'),pcie_experts=c.get('pcie_experts'),
    cpu_begin_us=(c['begin']-w['before'])/1000,cpu_plan_published_us=(c['publish']-w['before'])/1000,
    cpu_pool_start_us=(c['run']-w['before'])/1000,cpu_pool_end_us=(c['ran']-w['before'])/1000,
    cpu_early_ready_us=(c['early_ready']-w['before'])/1000 if 'early_ready' in c else None,
    cpu_driver_ready_us=(c['driver_ready']-w['before'])/1000,
    gpu_begin_us=g[0],gpu_router_end_us=g[17],gpu_shared_quant_end_us=g[18],gpu_resident_end_us=g[33],
    gpu_plan_ready_us=g[19],gpu_pcie_end_us=g[22],gpu_cpu_rows_available_us=g[34],gpu_combine_end_us=g[24],
    cpu_plan_us=(c['publish']-c['begin'])/1000,cpu_pool_us=(c['ran']-c['run'])/1000,
    gpu_resident_us=resident_us,gpu_plan_wait_read_us=plan_us,gpu_cpu_wait_copy_us=g[34]-g[22],
    gpu_merge_us=g[24]-g[34],early_lead_us=(c['ran']-c['early_ready'])/1000 if 'early_ready' in c else None)
   rows.append(row)
 with (folder/'layers.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 story=[r for r in rows if r['case']=='story' and r['phase']=='decode']
 assert story
 by_layer=[]
 for layer in range(48):
  data=[r for r in story if r['layer']==layer]
  by_layer.append(dict(layer=layer,windows=len(data),**{k:statistics.mean(r[k] for r in data) for k in ['cpu_plan_us','cpu_pool_us','gpu_resident_us','gpu_plan_wait_read_us','gpu_cpu_wait_copy_us','gpu_merge_us']}))
 summary[folder.name]=dict(total_windows=len(windows),story_decode_windows=len(story)//48,rows=len(rows),gpu_first=gpu_first,
  early_events=sum(r['early_lead_us'] is not None for r in story),
  early_events_before_return=sum(r['early_lead_us'] is not None and r['early_lead_us']>=0 for r in story),
  mean_ms_per_story_window={k:sum(r[k] for r in story)/(len(story)//48)/1000 for k in ['cpu_plan_us','cpu_pool_us','gpu_resident_us','gpu_plan_wait_read_us','gpu_cpu_wait_copy_us','gpu_merge_us']},layers=by_layer,
  clock_note='CPU offsets are from the CPU graph-launch clock; GPU offsets are from the GPU first marker. Axes are independent. Trace nodes and logging add overhead; not production TPS evidence.')
 (folder/'analysis.json').write_text(json.dumps(summary[folder.name],indent=2)+'\n')
(root/'timeline-analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
for label,x in summary.items():print(label,x['story_decode_windows'],x['mean_ms_per_story_window'],'early',x['early_events_before_return'])
