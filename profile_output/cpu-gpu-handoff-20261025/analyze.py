"""Validate retained raw evidence and summarize the handoff hypothesis."""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics as st

R = Path(__file__).resolve().parent
postflight = json.loads((R / 'postflight.json').read_text())
assert postflight['gpu_used_mib'] == 2
assert not postflight['live_job_handles'] and not postflight['engine_or_probe_processes']
assert postflight['memory_events']['oom'] == postflight['memory_events']['oom_kill'] == 0
assert postflight['engine_sha256'] == '4fc8304ec17c7e6b0df4a843f7f60ebe21934e6a7e88004a358e6b2356edb8cd'
origins = json.loads((R / 'source-origins.json').read_text())
for path, digest in origins.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert postflight['source_hashes']['/root/strata-dcu/Strata/' + path] == digest
clock = json.loads((R / 'clock-rate.json').read_text())
assert clock['returncode'] == 0 and clock['rate_khz'] == 25000 and clock['ns_per_tick'] == 40
rows = list(csv.DictReader((R / 'trace-current/layers.csv').open()))
records = json.loads((R / 'trace-current/results.json').read_text())
assert json.loads((R / 'trace-current/complete.json').read_text())['returncode'] == 0
assert records[-1]['generated_ids'] == [3793, 248046]
cfg = json.loads((R / 'trace-current/command.json').read_text())
assert cfg['engine_sha256'] == '4fc8304ec17c7e6b0df4a843f7f60ebe21934e6a7e88004a358e6b2356edb8cd'
assert cfg['trace_flags']['STRATA_CPU_Q4_INPUT_BIAS'] == '1'
assert cfg['flags']['STRATA_VERIFY_HIT_PLAN'] == '1'
assert cfg['flags']['STRATA_CPU_EARLY_RESULT'] == '1'
assert cfg['flags']['STRATA_CPU_EXPERT_PIPELINE'] == '1'
assert cfg['command'][cfg['command'].index('--expert-cache') + 1] == '8500'
assert '--mtp-adaptive' in cfg['command']
stderr = (R / 'trace-current/stderr.log').read_text()
assert 'strata cpu q4: shared input bias and single-position rows active' in stderr
weight_line = re.search(r'strata weight usage: (.*)', stderr)[1]
weights = {k: float(v) if '.' in v else int(v) for k, v in re.findall(r'(\w+)=([0-9.]+)', weight_line)}
assert weights['weights'] == 29373360352 <= weights['limit'] == 34351349760

# Existing CSV deliberately groups the wait and copy. Read existing marker 23
# to split them, without comparing CPU and GPU clocks or adding instrumentation.
window = -1
gpu = {}
for line in stderr.splitlines():
    if line.startswith('verify graph: '):
        window += 1
    m = re.match(r'  layer (\d+) group 0:(.*)', line)
    if m:
        marks = {int(k): float(v) for k, v in re.findall(r'\((\d+)\)=([0-9.]+)', m[2])}
        if all(k in marks for k in [22, 23, 34, 24]):
            gpu[window, int(m[1])] = marks

out = dict(lifecycle={}, weight_usage=weights)
details = []
for row in rows:
    marks = gpu[int(row['window']), int(row['layer'])]
    wait = marks[23] - marks[22]
    copy = marks[34] - marks[23]
    merge = marks[24] - marks[34]
    assert min(wait, copy, merge) >= -.11
    assert abs(wait + copy - float(row['gpu_cpu_wait_copy_us'])) < .11
    details.append(dict(row, gpu_cpu_wait_us=wait, gpu_cpu_copy_us=copy))
with (R / 'handoff-layers.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(details[0]))
    w.writeheader()
    w.writerows(details)
for case in ['story', 'chinese', 'code']:
    data = [r for r in details if r['case'] == case and r['phase'] == 'decode']
    windows = {r['window'] for r in data}
    assert windows
    for win in windows:
        assert {int(r['layer']) for r in data if r['window'] == win} == set(range(48))
    keys = ['cpu_pool_us', 'gpu_cpu_wait_us', 'gpu_cpu_copy_us', 'gpu_merge_us', 'gpu_resident_us']
    queues = []
    record = next(x for x in records if x['case'] == case)
    for tid, before in record['threads_before'].items():
        if tid in record['threads_after']:
            after = record['threads_after'][tid]
            queues.append((after['schedstat'][1] - before['schedstat'][1]) / record['wall_ns'])
    out['lifecycle'][case] = dict(
        windows=len(windows),
        mean_ms_per_window={k: sum(float(r[k]) for r in data) / len(windows) / 1000 for k in keys},
        median_us_per_layer={k: st.median(float(r[k]) for r in data) for k in keys},
        max_thread_queue_fraction=max(queues))

out['methods'] = {
    0: 'current gfx936 compat dispatch: standalone wait, mapped copy, hit add, weighted combine',
    1: 'existing kernels with optional wait+copy and hit+combine fusion (not current dispatch)',
    2: 'new standalone wait+direct mapped read+weighted combine; no VRAM result intermediate'}
out['probes'] = {}
for tag, kind in [('probe', 'timing'), ('probe-fresh', 'fresh')]:
    data = [json.loads(x) for x in (R / (tag + '-results.jsonl')).read_text().splitlines()]
    assert json.loads((R / (tag + '-status.json')).read_text())['returncode'] == 0
    assert data[-1]['kind'] == 'complete' and data[-1]['failures'] == 0
    assert data[-1]['numeric_cases'] == (975 if tag == 'probe' else 525)
    numeric = data[-1]['numeric_cases']
    data = [x for x in data if x['kind'] == kind]
    assert len(data) == 25 * 7 * 3 * (1 if tag == 'probe' else 30)
    summary = []
    for t in range(1, 6):
        for pct in [0, 30, 60, 90, 100]:
            methods = {}
            rounds = {}
            for method in range(3):
                group = [x for x in data if x['t'] == t and x['hits'] == t * 10 * pct // 100 and x['method'] == method]
                assert {x['round'] for x in group} == set(range(7))
                round_medians = []
                for round_ in range(7):
                    values = [x['us'] for x in group if x['round'] == round_]
                    assert len(values) == (1 if tag == 'probe' else 30)
                    round_medians.append(st.median(values))
                methods[method] = st.median(round_medians)
                rounds[method] = round_medians
            summary.append(dict(t=t, synthetic_hit_percent=pct, median_us=methods, round_medians_us=rounds,
                                direct_vs_current_ratio=methods[2] / methods[0],
                                direct_vs_existing_fused_ratio=methods[2] / methods[1]))
    out['probes'][tag] = dict(cases=summary, counted_numerical_checks=numeric,
                              measured='GPU event time per hot replay' if tag == 'probe' else 'fresh payload wall latency, graph launch through stream completion')
out['limits'] = [
    'Full-layer tracing adds overhead. CPU/GPU clocks are independent. No traced TPS claim.',
    'Synthetic hit fractions and seeded float outputs; not real-model output acceptance or expert timing.',
    'All-hit synthetic case is correctness coverage; production may bypass the CPU path entirely.',
    'Standalone fusion not integrated into the engine. No full answer/code/cancel QA for the candidate.',
    'Existing optional fused kernels are measured separately to distinguish already-available arithmetic from new direct consumption.',
    'No promise to remove the much larger CPU expert compute wait. No decode TPS gain measured.',
    'No GPU expert-count, weight format, environment, production dispatch or defaults changed.']
(R / 'audit.json').write_text(json.dumps(dict(
    objective='Identify an evidence-supported CPU/GPU handoff optimization direction; no production implementation requested',
    current_engine_and_flags_verified=True, full_layer_trace_verified=True,
    raw_wait_and_copy_markers_verified=True, three_way_probe_verified=True,
    fresh_payload_probe_verified=True, independent_numerical_reference_passed=True,
    fixed_experts_and_half_vram_verified=True, production_sources_unchanged=True,
    remote_idle_and_binary_unchanged_verified=True,
    implementation_and_decode_tps_gain='not performed or claimed'), indent=2) + '\n')
(R / 'analysis.json').write_text(json.dumps(out, indent=2) + '\n')
sources = [p for p in R.iterdir() if p.suffix in ['.py', '.cpp', '.inc']]
(R / 'source-manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}, indent=2) + '\n')
print(json.dumps(out['lifecycle'], indent=2))
for tag, probe in out['probes'].items():
    for row in probe['cases']:
        if row['synthetic_hit_percent'] in [30, 60] and row['t'] in [1, 3, 5]:
            print(tag, row['t'], row['synthetic_hit_percent'], row['median_us'])
