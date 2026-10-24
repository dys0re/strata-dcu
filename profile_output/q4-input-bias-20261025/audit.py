"""Final evidence audit, including independent entry validation and postflight."""
from pathlib import Path
import hashlib,json,re,subprocess,sys
R=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(R/'analyze_bench.py')],check=True,stdout=subprocess.DEVNULL)
analysis=json.loads((R/'analysis.json').read_text());assert analysis['enable_machine_default']
assert len(json.loads((R/'build-status.json').read_text()))==3
assert all(x['returncode']==0 for x in json.loads((R/'build-status.json').read_text()))
for name,weight in [('native-qa',32827238400),('joint-reject-qa',34195377152)]:
    p=R/name;q=json.loads((p/'qa-status.json').read_text());w=json.loads((p/'weight-usage.json').read_text())
    cmd=json.loads((p/'command.json').read_text());records=json.loads((p/'results.json').read_text())
    assert q['functional_assertions_passed'] and q['qa_scope']=='full' and q['extended_qa'] and q['engine_rc']==0
    assert w['weights']==weight and w['weights']<=w['limit']==34351349760 and w['experts']==22947840000
    assert cmd['overrides']['STRATA_CPU_Q4_INPUT_BIAS']=='1' and cmd['overrides']['STRATA_CPU_Q4_MT']=='1'
    assert cmd['command'][cmd['command'].index('--expert-cache')+1]=='8300'
    if name.startswith('joint'):assert cmd['command'][cmd['command'].index('--spec-corrupt')+1]=='1'
    assert len(records)==20 and (p/'lookup-counters.json').exists()
    protocol=json.loads((p/'protocol.json').read_text());assert protocol['cancel'].split()[5]=='cancel'
    assert 'strata cpu q4: shared input bias and single-position rows active' in (p/'stderr.log').read_text()
assert len(json.loads((R/'modes-status.json').read_text()))==2
assert all(x['returncode']==0 for x in json.loads((R/'modes-status.json').read_text()))
meta=json.loads((R/'deployment.json').read_text())
assert meta['sha256']=='4fc8304ec17c7e6b0df4a843f7f60ebe21934e6a7e88004a358e6b2356edb8cd'
assert meta['mtp_input_bias_default'] and json.loads((R/'selected-model.json').read_text())['cpu_q4_input_bias']
entry=json.loads((R/'entry/checks.json').read_text())
assert [x['name'] for x in entry]==['mtp-default','mtp-override','native-opt-in','joint-opt-in']
assert [x['input_bias'] for x in entry]==[True,False,True,True]
assert all(x['answer_correct'] and x['lookup_exercised'] for x in entry)
transcript=(R/'entry-validation.txt').read_text()
assert 'PASS Q4 INPUT BIAS PUBLIC ENTRIES; GPU idle' in transcript
post=json.loads((R/'postflight.json').read_text())
assert post['engine_sha256']==meta['sha256'] and post['engine']==meta['engine']
assert post['model_config']['engine_sha256']==meta['sha256']
assert post['gpu_used_mib']==2 and not post['engine_processes'] and not post['job_handles_alive']
assert post['memory_events']['oom']==0 and post['memory_events']['oom_kill']==0
repo=R.parents[1]
for name,digest in json.loads((R/'deployed-source-manifest.json').read_text()).items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
for p in R.rglob('*'):
    if p.is_file():
        b=p.read_bytes();assert not b.startswith(b'\x7fELF'),p
        assert (b'-----BEGIN '+b'OPENSSH PRIVATE KEY-----') not in b,p
checks=dict(abba_four_runs=True,mtp_18qa_each=True,native_and_joint_20qa_each=True,weights_and_experts_fixed=True,
            numerical_and_pool_gates=True,independent_entry_four_cases=True,default_and_override_verified=True,
            final_sources_match=True,gpu_idle_and_jobs_exited=True,scope='HIP runtime and x86 CPU fallback only; no full CUDA/SYCL/Metal/multi-GPU runtime claim')
(R/'audit.json').write_text(json.dumps(checks,indent=2)+'\n')
print('FINAL AUDIT PASS')
