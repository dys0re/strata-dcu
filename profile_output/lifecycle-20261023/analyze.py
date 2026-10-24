"""Analyze measured lifecycle clocks; CPU and GPU totals are overlapping timelines."""
import collections,csv,json,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
names=['-','hc0','qkv_or_query','conv','ab','z','rec','kv_index','kv_norm_rope','kv_append','query','scores_topk','kv_resolve','attention','gate','unused','out_proj','hc1_router_bell','shared_enqueue_quant','wait_plan','gpu_experts','wait_bounce_fetch','pcie_experts','wait_cpu','copy_combine_join']

def gap(a,b):return max(0,a-b) if a and b else 0

def windows(path):
 out=collections.OrderedDict()
 with path.open() as f:
  for row in csv.reader(f):
   v=list(map(int,row)); end,T,pos,seq,l=v[:5]; assert len(v)==43,len(v)
   key=(end,T,pos,seq);out.setdefault(key,{})[l]=v[5:]
 return out

def read_request_trace(d,reqs):
 ws=windows(d/'lifecycle.csv');result=[];details=[]
 for r in reqs:
  selected=[(k,ls) for k,ls in ws.items() if r['started_ns']<=k[0]<=r['ended_ns'] and k[2]>=len(r['prompt_ids'])-1]
  sums=collections.Counter();byT=collections.defaultdict(list);bylayer=collections.defaultdict(collections.Counter)
  for key,layers in selected:
   end,T,pos,seq=key;one=collections.Counter(); assert len(layers)==49,(key,len(layers))
   for l,v in layers.items():
    if l==48:continue
    st=v[:33];sub=collections.Counter();prev=st[0]
    for i in range(1,25):
     if not st[i] or st[i]<prev:continue
     sub[names[i]]+=st[i]-prev;prev=st[i]
    # hc1 is part of router_bell in the coarse stamps; split only when present.
    if st[30] and st[31] and st[32] and st[16]<=st[30]<=st[31]<=st[32]<=st[17]:
     z=st[32]-st[16];sub['hc1_router_bell']-=z;sub['hc1']=z
    # Shared stream is an overlapping measurement, never added to main-stream totals.
    sub['shared_parallel']=gap(v[34],v[33])
    sub['host_wait']=gap(v[36],v[35]);sub['host_layer']=gap(v[37],v[36])
    sub['transition']=gap(layers[l+1][0],st[24])
    bylayer[l].update(sub);one.update(sub)
   one['head']=gap(layers[48][1],layers[48][0]);one['gpu_span']=gap(layers[48][1],layers[0][0])
   # transition includes the final layer-to-head transition and completes the main-stream partition.
   parts=sum(v for k,v in one.items() if k not in ['shared_parallel','host_wait','host_layer','gpu_span'])
   assert abs(parts-one['gpu_span'])<100000,(key,parts,one['gpu_span'])
   sums.update(one);byT[T].append(one)
   details.append({'case':r['case'],'end_ns':end,'T':T,'pos':pos,'seq':seq,**{k:v/1e6 for k,v in one.items()}})
  n=len(selected);tokens=len(r['generated_ids'])
  result.append({'case':r['case'],'tokens':tokens,'windows':n,'decode_tps':r['decode_tps'],'decode_ms':r['decode_ms'],
    'ms_per_output':{k:v/1e6/tokens for k,v in sums.items()},
    'ms_per_window':{k:v/1e6/n for k,v in sums.items()} if n else {},
    'by_T':{str(T):{'windows':len(vs),'ms_per_window':{k:sum(v.get(k,0) for v in vs)/1e6/len(vs) for k in sums}} for T,vs in byT.items()},
    'layers':{str(l):{k:v/1e6/n for k,v in sub.items()} for l,sub in bylayer.items()} if n else {}})
 (d/'window-summary.json').write_text(json.dumps(details,indent=2))
 return result

all_results={}
for d in sorted(ROOT.iterdir()):
 if not d.is_dir() or not (d/'results.json').exists():continue
 rs=json.loads((d/'results.json').read_text())
 result={'requests':[{k:r[k] for k in ['case','decode_ms','decode_tps']} for r in rs]}
 if (d/'lifecycle.csv').exists():result['lifecycle']=read_request_trace(d,rs)
 if (d/'draft.csv').exists():
  ds=list(csv.reader((d/'draft.csv').open()));draft=[]
  for r in rs:
   sums=collections.Counter();counts=collections.Counter();byK=collections.defaultdict(list)
   for row in ds:
    kind=row[0];pos,k,begin,end=map(int,row[1:5]);st=list(map(int,row[5:]));
    if not(r['started_ns']<=begin<=r['ended_ns']) or pos<len(r['prompt_ids'])-1:continue
    counts[kind]+=1;sums[kind+'_host']+=end-begin;sums[kind+'_gpu']+=gap(st[-1],st[0])
    if kind=='draft':
     sums['network_gpu']+=gap(st[11],st[0]);sums['head_gpu']+=gap(st[12],st[11]);sums['selector_projection']+=gap(st[13],st[12]);sums['topk']+=gap(st[14],st[13]);sums['selector_path']+=gap(st[15],st[14]);byK[k].append((st[-1]-st[0],st[12]-st[11]))
   draft.append({'case':r['case'],'counts':dict(counts),'ms_per_output':{k:v/1e6/len(r['generated_ids']) for k,v in sums.items()},'by_candidates':{str(k):{'calls':len(v),'gpu_ms':statistics.mean(x[0] for x in v)/1e6,'head_ms':statistics.mean(x[1] for x in v)/1e6} for k,v in byK.items()}})
  result['draft']=draft
 all_results[d.name]=result
(ROOT/'analysis.json').write_text(json.dumps(all_results,indent=2))
for name,r in all_results.items():
 print(name)
 for a in r.get('lifecycle',[]):
  if a['case'] not in ['story-0','chinese-0','code-0']:continue
  print(a['case'],a['windows'],round(a['decode_tps'],2),{k:round(v,3) for k,v in sorted(a['ms_per_output'].items(),key=lambda kv:-kv[1])[:14]})
 for a in r.get('draft',[]):
  if a['case'] in ['story-0','chinese-0','code-0']:print('draft',a['case'],a['counts'],{k:round(v,3) for k,v in a['ms_per_output'].items()})
