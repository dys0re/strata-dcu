"""Audit fixed-resource ABBA throughput and executable QA; no significance claim."""
from pathlib import Path
import json,math,re,statistics
R=Path(__file__).resolve().parent
QA={'ok','arithmetic','multiply','sum','capital','uppercase','sort-json','json','code-function','multi-turn','sample-0','sample-1','one_token','context_tail','code-search','code-intervals','code-encoding','data-transform'}
status=json.loads((R/'bench-status.json').read_text())
assert [s['label'] for s in status]==['baseline-a','candidate-b','candidate-b2','baseline-a2']
assert all(s['returncode']==0 for s in status)
candidate=json.loads((R/'candidate.json').read_text())['sha256']
result={'runs':{},'pooled':{},'scope':'Four sequential cold starts in ABBA order, 3x256 tokens/task/start, quiet-core selection after load. Descriptive medians, not statistical significance.'}
samples={kind:{k:[] for k in ['story','chinese','code']} for kind in ['baseline','candidate']}
for s in status:
    label=s['label'];p=R/label;new=label.startswith('candidate')
    rows=json.loads((p/'results.json').read_text());by={r['case']:r for r in rows}
    assert QA<=set(by)
    qa=json.loads((p/'qa-status.json').read_text())
    assert qa['functional_assertions_passed'] and qa['qa_scope']=='full' and qa['extended_qa'] and qa['engine_rc']==0
    protocol=json.loads((p/'protocol.json').read_text())
    assert protocol['cancel'].split()[5]=='cancel' and protocol['one_token_limit'] and protocol['context_tail']
    assert int(protocol['recovery'].split()[1])>0
    stderr=(p/'stderr.log').read_text()
    assert 'strata cpu q4: shared input bias and single-position rows active' in stderr
    assert ('strata verify: direct CPU result combine selected' in stderr)==new
    command=json.loads((p/'command.json').read_text())
    assert not command['profile'] and command['timing_only'] and not command['dump_logits']
    assert not any(k in command['overrides'] for k in ['STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_POOL_TRACE'])
    assert command['overrides']['STRATA_CPU_Q4_INPUT_BIAS']=='1'
    assert command['overrides']['STRATA_CPU_RESULT_DIRECT']==('1' if new else '0')
    assert command['overrides']['STRATA_CPU_Q4_MT']=='1'
    assert '--mtp-adaptive' in command['command']
    assert command['command'][command['command'].index('--expert-cache')+1]=='8500'
    prov=json.loads((p/'provenance.json').read_text())
    assert prov['engine_sha256']==(candidate if new else '4fc8304ec17c7e6b0df4a843f7f60ebe21934e6a7e88004a358e6b2356edb8cd')
    weights=json.loads((p/'weight-usage.json').read_text())
    assert weights['weights']==29373360352 and weights['experts']==23500800000
    assert weights['weights']<=weights['limit']==34351349760 and weights['mtp_proxy']==0
    selection=json.loads((p/'idle-core-selection.json').read_text())
    assert selection['participants']==13
    threads=json.loads((p/'thread-usage.json').read_text())
    assert len(threads)>=len(rows)
    metrics={}
    trace=stderr.split('strata trace: request ')[1:]
    assert len(trace)>=len(rows)
    for workload in ['story','chinese','code']:
        indexes=[i for i,r in enumerate(rows) if re.fullmatch(workload+r'-\d+',r['case'])]
        assert len(indexes)==3
        values=[];queues=[];cpu=[];prefill=[];windows=[]
        for i in indexes:
            row=rows[i];assert len(row['generated_ids'])==256 and row['done'].split()[5]=='length'
            tps=256000/row['decode_ms'];assert abs(tps-row['decode_tps'])<1e-8
            values.append(tps);prefill.append(float(row['done'].split()[3]))
            th=threads[i];wall=th['end']['time_ns']-th['start']['time_ns']
            assert th['done']==row['done']
            for tid,b in th['start']['threads'].items():
                if tid not in th['end']['threads']:continue
                a=th['end']['threads'][tid];assert a['affinity']==b['affinity']
                queues.append((a['runqueue_ns']-b['runqueue_ns'])/wall)
            timing=re.search(r'strata decode timing: (\d+) windows, avg T ([0-9.]+), ([0-9.]+) tokens/window,.*? CPU ([0-9.]+)\]',trace[i])
            assert timing,(label,row['case'])
            windows.append(dict(windows=int(timing[1]),avg_T=float(timing[2]),tokens_per_window=float(timing[3])))
            cpu.append(float(timing[4]))
        samples['candidate' if new else 'baseline'][workload]+=values
        metrics[workload]=dict(tps=values,median_tps=statistics.median(values),median_prefill_ms=statistics.median(prefill),cpu_ms_per_window=cpu,windows=windows,max_thread_queue_fraction=max(queues))
    result['runs'][label]=dict(metrics=metrics,qa_cases=len(QA),protocol_pass=True,weights=weights,selection=selection['chosen'],all_quiet=selection['all_quiet'],engine_sha256=prov['engine_sha256'])
ratios=[]
for workload in ['story','chinese','code']:
    old=statistics.median(samples['baseline'][workload]);new=statistics.median(samples['candidate'][workload]);ratio=new/old;ratios.append(ratio)
    result['pooled'][workload]=dict(baseline_samples=samples['baseline'][workload],candidate_samples=samples['candidate'][workload],baseline_median=old,candidate_median=new,ratio=ratio)
result['geometric_mean_ratio']=math.prod(ratios)**(1/3)
result['enable_machine_default']=result['geometric_mean_ratio']>=1.01 and min(ratios)>=.97
result['default_rule']=json.loads((R/'plan.json').read_text())['default_rule']
(R/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['pooled','geometric_mean_ratio','enable_machine_default']},indent=2))
