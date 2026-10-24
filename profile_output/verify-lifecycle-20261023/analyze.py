"""Current verifier traces. Main GPU stages partition time; CPU/shared overlap."""
from pathlib import Path
from collections import OrderedDict,Counter,defaultdict
import csv,json,re,bisect,statistics,gzip
ROOT=Path(__file__).resolve().parent
NAMES=['-','hc0','qkv_or_query','conv','ab','z','rec','kv_index','kv_norm_rope','kv_append','query','scores_topk','kv_resolve','attention','gate','unused','out_proj','hc1_router_bell','shared_enqueue_quant','wait_plan','gpu_experts','wait_bounce_fetch','pcie_experts','wait_cpu','copy_combine_join']
OVERLAP={'shared_parallel','host_wait','host_layer','gpu_span'}
CPU_FIELDS=['plan_ms','actq_ms','jobs_ms','compute_ms','cpu_experts','cpu_entries','gpu_hits','pcie_experts']
def gap(a,b):return max(0,a-b) if a and b else 0
def trace_open(path):
 return path.open() if path.exists() else gzip.open(str(path)+".gz","rt")

def analyze_run(d):
 reqs=json.loads((d/'results.json').read_text());result={'requests':[]}
 timings=[line for line in (d/'stderr.log').read_text().splitlines() if line.startswith('strata decode timing:')]
 assert len(timings)>=len(reqs)
 weights=json.loads((d/'weight-usage.json').read_text())
 assert weights['weights']<=weights['limit']==weights['total']//2
 result['weights']=weights
 trace=OrderedDict();cpu=[]
 if (d/'lifecycle.csv').exists() or (d/'lifecycle.csv.gz').exists():
  for row in csv.reader(trace_open(d/'lifecycle.csv')):
   v=list(map(int,row));assert len(v)==43
   end,T,pos,seq,l=v[:5];trace.setdefault((end,T,pos,seq),{})[l]=v[5:]
  cpu=[{k:(int(v) if k in ['begin_ns','end_ns','layer','T'] else float(v)) for k,v in r.items()} for r in csv.DictReader(trace_open(d/'cpu.csv'))]
  assert all(cpu[i]['begin_ns']<=cpu[i+1]['begin_ns'] for i in range(len(cpu)-1))
 begins=[r['begin_ns'] for r in cpu];details=[]
 for req,tl in zip(reqs,timings):
  m=re.search(r'(\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = verify ([\d.]+).*?\+ commit/emit ([\d.]+) \+ draft ([\d.]+);',tl)
  assert m,tl
  nw,Tmean,emitted,wall,verify,commit,draft=map(float,m.groups())
  tokens=len(req['generated_ids'])
  item=dict(case=req['case'],tokens=tokens,decode_tps=req['decode_tps'],windows=int(nw),mean_T=Tmean,
   tokens_per_window=emitted,host_ms_per_window=dict(total=wall,verify=verify,commit=commit,draft=draft))
  if trace:
   chosen=[(key,layers) for key,layers in trace.items() if req['started_ns']<=key[0]<=req['ended_ns'] and key[2]>=len(req['prompt_ids'])-1]
   assert len(chosen)==int(nw),(d,req['case'],len(chosen),nw)
   sums=Counter();csums=Counter();byT=defaultdict(list);layersums=defaultdict(Counter);layercpu=defaultdict(Counter);classsum=defaultdict(Counter)
   emitted_total=0
   for wi,(key,layers) in enumerate(chosen):
    end,T,pos,seq=key;assert set(layers)==set(range(49)),key
    one=Counter();cone=Counter()
    for l in range(48):
     v=layers[l];st=v[:33];sub=Counter();prev=st[0];assert prev>0
     for i in range(1,25):
      if not st[i]:continue
      assert st[i]>=prev,(key,l,i,st[i],prev)
      sub[NAMES[i]]+=st[i]-prev;prev=st[i]
     if st[30] and st[31] and st[32] and st[16]<=st[30]<=st[31]<=st[32]<=st[17]:
      hc=st[32]-st[16];sub['hc1_router_bell']-=hc;sub['hc1']=hc
     sub['shared_parallel']=gap(v[34],v[33]);sub['host_wait']=gap(v[36],v[35]);sub['host_layer']=gap(v[37],v[36]);sub['transition']=gap(layers[l+1][0],st[24])
     one.update(sub);layersums[l].update(sub);classsum['QSA' if st[7] else 'GDN'].update(sub)
     start=bisect.bisect_left(begins,v[36]);finish=bisect.bisect_right(begins,v[37])
     matches=[c for c in cpu[start:finish] if c['layer']==l and c['T']==T and c['end_ns']<=v[37]]
     assert len(matches)==1,(key,l,len(matches))
     c=matches[0];cnt={k:c[k] for k in CPU_FIELDS};cnt['callback_ms']=(c['end_ns']-c['begin_ns'])/1e6
     cone.update(cnt);layercpu[l].update(cnt)
    one['head']=gap(layers[48][1],layers[48][0]);one['gpu_span']=gap(layers[48][1],layers[0][0])
    parts=sum(v for k,v in one.items() if k not in OVERLAP)
    assert parts==one['gpu_span'],(key,parts,one['gpu_span'])
    emitted_now=chosen[wi+1][0][2]-pos if wi+1<len(chosen) else tokens-emitted_total
    assert 1<=emitted_now<=T,(key,emitted_now,T)
    emitted_total+=emitted_now
    sums.update(one);csums.update(cone);byT[T].append((one,cone,emitted_now))
    details.append(dict(case=req['case'],T=T,pos=pos,seq=seq,end_ns=end,emitted=emitted_now,
       gpu_ms={k:v/1e6 for k,v in one.items()},cpu=dict(cone)))
   assert emitted_total==tokens
   n=len(chosen)
   item['gpu_ms_per_output']={k:v/1e6/tokens for k,v in sums.items()}
   item['gpu_ms_per_window']={k:v/1e6/n for k,v in sums.items()}
   item['cpu_per_output']={k:v/tokens for k,v in csums.items()}
   item['cpu_per_window']={k:v/n for k,v in csums.items()}
   item['by_T']={str(t):dict(windows=len(rows),outputs=sum(x[2] for x in rows),tokens_per_window=sum(x[2] for x in rows)/len(rows),gpu_ms_per_window={k:sum(x[0][k] for x in rows)/len(rows)/1e6 for k in sums},cpu_per_window={k:sum(x[1][k] for x in rows)/len(rows) for k in csums}) for t,rows in byT.items()}
   item['layers']={str(l):dict(gpu_ms_per_window={k:v/1e6/n for k,v in sub.items()},cpu_per_window={k:v/n for k,v in layercpu[l].items()}) for l,sub in layersums.items()}
   item['layer_classes_ms_per_output']={cl:{k:v/1e6/tokens for k,v in sub.items()} for cl,sub in classsum.items()}
  result['requests'].append(item)
 if details:(d/'windows.json').write_text(json.dumps(details,indent=2)+'\n')
 return result

results={}
for d in sorted(ROOT.iterdir()):
 if d.is_dir() and (d/'qa-status.json').exists() and (d/'results.json').exists():
  q=json.loads((d/'qa-status.json').read_text());assert q['engine_rc']==0 and q['functional_assertions_passed']
  results[d.name]=analyze_run(d)
(ROOT/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
for name,data in results.items():
 for r in data['requests']:
  if r['case'] not in ['story-0','chinese-0','code-0']:continue
  if 'gpu_ms_per_output' not in r:
   print(name,r['case'],round(r['decode_tps'],2),'tok/win',r['tokens_per_window'],'T',r['mean_T'],'no GPU instrumentation');continue
  g=r['gpu_ms_per_output'];c=r['cpu_per_output']
  print(name,r['case'],round(r['decode_tps'],2),'tok/win',r['tokens_per_window'],'T',r['mean_T'],
    'gpu_ms/token', {k:round(g.get(k,0),3) for k in ['gpu_span','wait_cpu','wait_plan','hc0','hc1','qkv_or_query','out_proj','head','gpu_experts']},
    'cpu_ms/token',round(c.get('compute_ms',0),3))
print('PASS VERIFIER_LIFECYCLE_PARTITION',len(results))
