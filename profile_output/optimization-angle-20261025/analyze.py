"""Summarize observed CPU wait and the isolated Q4 optimization candidate."""
from pathlib import Path
import csv,hashlib,json,re
R=Path(__file__).resolve().parent
rows=list(csv.DictReader((R/'trace-current/layers.csv').open()))
records=json.loads((R/'trace-current/results.json').read_text())
timing=re.findall(r'per layer-window: CPU experts ([0-9.]+) \(([0-9.]+) entries\)',(R/'trace-current/stderr.log').read_text())
assert len(timing)==len(records)
stages=['cpu_plan_us','cpu_pool_us','gpu_resident_us','gpu_plan_wait_read_us','gpu_cpu_wait_copy_us','gpu_merge_us']
out={'lifecycle':{},'primitives':{}}
for record,(jobs,entries) in zip(records,timing):
    case=record['case']
    if case not in ['story','chinese','code']:continue
    data=[x for x in rows if x['case']==case and x['phase']=='decode']
    windows=len(data)//48
    assert windows and len(data)==windows*48
    for window in {x['window'] for x in data}:
        assert {int(x['layer']) for x in data if x['window']==window}==set(range(48))
    queues={}
    for tid,b in record['threads_before'].items():
        if tid in record['threads_after']:
            a=record['threads_after'][tid]
            queues[tid]=(a['schedstat'][1]-b['schedstat'][1])/record['wall_ns']
    out['lifecycle'][case]=dict(windows=windows,mean_ms_per_window={k:sum(float(x[k]) for x in data)/windows/1000 for k in stages},
        max_thread_queue_fraction=max(queues.values()),cpu_experts_per_layer=float(jobs),cpu_entries_per_layer=float(entries),
        singleton_fraction_lower_bound_approx=max(0,2-float(entries)/float(jobs)),
        bound_note='Derived lower bound: singleton_jobs >= 2*jobs-entries. Rounded request averages, not an observed group-size histogram.')
for tag in ['q4','q4-stream']:
    data=[json.loads(x) for x in (R/(tag+'-results.jsonl')).read_text().splitlines()]
    assert data[-1]=={'kind':'complete','failures':0}
    assert json.loads((R/(tag+'-status.json')).read_text())['returncode']==0
    numeric=[x for x in data if x['kind']=='numeric']
    assert len(numeric)==(16 if tag=='q4-stream' else 32)
    assert any(x['edge'] for x in numeric)
    results=[]
    for nt in [1,2,3,4]:
        tiles=[0] if tag=='q4-stream' else [0,1]
        for tile in tiles:
            stages_={}
            for stage in [0,1]:
                d={x['method']:x['us'] for x in data if x['kind']=='median' and x['nt']==nt and x['tile']==tile and x['stage']==stage}
                assert set(d)=={0,1,2}
                for method in d:
                    assert len([x for x in data if x['kind']=='timing' and x['nt']==nt and x['tile']==tile and x['stage']==stage and x['method']==method])==9
                stages_['gate_up' if stage==0 else 'down']=d
            baseline=0 if nt==1 else 1
            results.append(dict(nt=nt,tile=tile,median_us=stages_,production_baseline_method=baseline))
    out['primitives'][tag]=dict(cases=results,numerical_cases=len(numeric),max_relative_to_existing=max(x['relative'] for x in numeric),
         max_relative_to_ggml_normal=max(x['ggml_relative'] for x in numeric),method_names={0:'ggml per-token (production nt=1)',1:'existing fused Q4 (production nt>=2)',2:'unsigned nibble with shared input correction'},
         scope='Synthetic Q4_0 weights and Q8_0 inputs; single core; separate gate/up and down probes. No intermediate quantization, expert pool, GPU or decode TPS measurement.')
selection=json.loads((R/'trace-current/idle-core-selection.json').read_text())
out['placement']=dict(chosen=selection['chosen'],expert_pages=selection['expert_pages'])
weight_line=re.search(r'strata weight usage: (.*)',(R/'trace-current/stderr.log').read_text()).group(1)
out['weight_usage']={k:float(v) if '.' in v else int(v) for k,v in re.findall(r'(\w+)=([0-9.]+)',weight_line)}
assert out['weight_usage']['weights']==29373360352
assert out['weight_usage']['weights']<=out['weight_usage']['limit']
assert records[-1]['case']=='recovery' and records[-1]['generated_ids']==[3793,248046]
out['scope']='Unchanged production binary, 8500 GPU experts, enhanced MTP, full-layer trace (not normal TPS); isolated arithmetic candidate only. No production patch or new full functional QA.'
(R/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
sources=[p for p in R.iterdir() if p.suffix in ['.py','.cpp']]
(R/'source-manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},indent=2)+'\n')
print(json.dumps(out,indent=2))
