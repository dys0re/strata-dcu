"""Check actual outputs, deployed options, source identity and the independent entry pass."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
R = Path(__file__).resolve().parent
repo = R.parents[1]
subprocess.run([sys.executable, str(R / 'analyze_bench.py')], check=True, stdout=subprocess.DEVNULL)
a = json.loads((R / 'analysis.json').read_text())
for filename, count in [('build-status.json', 2), ('extra-gates.json', 2), ('after-bench-status.json', 4)]:
    states = json.loads((R / filename).read_text())
    assert len(states) == count and all(x['returncode'] == 0 for x in states), filename
assert 'MAPPED_MOE_PARITY cases=600 failures=0' in (R / 'gpu-tests-full.log').read_text()
assert 'MAPPED_MOE_PARITY cases=960 failures=0' in (R / 'expanded-test-full.log').read_text()
cuda = json.loads((R / 'cuda-compile.json').read_text())
assert cuda['returncode'] == 0 and cuda['source_sha256'] == hashlib.sha256((repo / 'src/kernels/cuda/native_moe.cu').read_bytes()).hexdigest()
mode_states = json.loads((R / 'modes-status.json').read_text())
assert [x['label'] for x in mode_states] == ['native-qa', 'joint-reject-qa', 'disabled-qa', 'device-plan-fallback-qa']
assert all(x['returncode'] == 0 for x in mode_states)
for state in mode_states:
    name = state['label']; p = R / name
    qa = json.loads((p / 'qa-status.json').read_text())
    weights = json.loads((p / 'weight-usage.json').read_text())
    command = json.loads((p / 'command.json').read_text())
    rows = json.loads((p / 'results.json').read_text())
    protocol = json.loads((p / 'protocol.json').read_text())
    stderr = (p / 'stderr.log').read_text()
    assert qa['functional_assertions_passed'] and qa['qa_scope'] == 'full' and qa['extended_qa'] and qa['engine_rc'] == 0
    assert len(rows) == 20 and (p / 'lookup-counters.json').exists()
    by = {x['case']: x for x in rows}
    for case, answer in [('ok', 'OK'), ('arithmetic', '4'), ('multiply', '391'), ('sum', '5050'), ('capital', '北京'), ('uppercase', 'ABC'), ('multi-turn', 'ORBIT')]:
        assert by[case]['text'].strip() == answer, (name, case)
    assert json.loads(by['json']['text']) == dict(ok=True, count=3)
    assert json.loads(by['sort-json']['text']) == [-1, 2, 2, 9]
    assert protocol['cancel'].split()[5] == 'cancel' and int(protocol['recovery'].split()[1]) > 0
    assert 'PASS' in (R / (name + '.log')).read_text()
    enabled = name in ['native-qa', 'joint-reject-qa']
    assert ('strata verify: direct CPU result combine selected' in stderr) == enabled
    requested = name != 'disabled-qa'
    assert command['overrides']['STRATA_CPU_RESULT_DIRECT'] == str(int(requested))
    if name == 'device-plan-fallback-qa':
        assert state['environment']['STRATA_VERIFY_DEVICE_PLAN'] == '1'
        assert state['environment']['STRATA_VERIFY_HIT_PLAN'] == '0'
    expected_weights = 32827238400 if name == 'native-qa' else 34195377152 if name == 'joint-reject-qa' else 33542205952
    slots = 8300 if enabled else 8500
    assert weights['weights'] == expected_weights <= weights['limit'] == 34351349760
    assert weights['experts'] == (22947840000 if slots == 8300 else 23500800000)
    assert command['command'][command['command'].index('--expert-cache') + 1] == str(slots)
    if name == 'joint-reject-qa':
        assert command['command'][command['command'].index('--spec-corrupt') + 1] == '1'
trace = R / 'trace-current'
assert json.loads((trace / 'complete.json').read_text())['returncode'] == 0
assert json.loads((trace / 'command.json').read_text())['trace_flags']['STRATA_CPU_RESULT_DIRECT'] == '1'
assert 'strata verify: direct CPU result combine selected' in (trace / 'stderr.log').read_text()
assert json.loads((trace / 'results.json').read_text())[-1]['generated_ids'] == [3793, 248046]
meta = json.loads((R / 'deployment.json').read_text())
assert meta['sha256'] == '81f79177f7e9b15a1d4506bed0211db95b369594868f51bcc19e64fa4017ce9b'
assert meta['mtp_direct_result_default'] == a['enable_machine_default']
model = json.loads((R / 'selected-model.json').read_text())
assert model['cpu_result_direct'] == a['enable_machine_default'] and model['cpu_q4_input_bias'] is True
entry = json.loads((R / 'entry/checks.json').read_text())
assert [x['name'] for x in entry] == ['mtp-default', 'mtp-override', 'native-opt-in', 'joint-opt-in']
assert [x['direct_result'] for x in entry] == [a['enable_machine_default'], not a['enable_machine_default'], True, True]
assert all(x['answer_correct'] and x['lookup_exercised'] for x in entry)
assert 'PASS CPU RESULT DIRECT PUBLIC ENTRIES; GPU idle' in (R / 'entry-validation.txt').read_text()
post = json.loads((R / 'postflight.json').read_text())
assert post['engine_sha256'] == meta['sha256'] == model['engine_sha256']
assert post['engine'] == meta['engine'] and post['gpu_used_mib'] == 2
assert not post['engine_processes'] and not post['job_handles_alive']
assert post['memory_events']['oom'] == post['memory_events']['oom_kill'] == 0
assert post['model_config'] == model and post['ple'].startswith('/root/private_data/')
for name, digest in json.loads((R / 'deployed-source-manifest.json').read_text()).items():
    assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
for p in R.rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
        data = p.read_bytes()
        assert not data.startswith(b'\x7fELF'), p
        assert (b'-----BEGIN ' + b'OPENSSH PRIVATE KEY-----') not in data, p
checks = dict(implemented_and_built=True, actual_gpu_entry_and_960_cases=True,
              mtp_four_runs_18qa_and_recovery_each=True, native_joint_and_fallback_20qa_each=True,
              fixed_experts_and_weight_budget=True, full_layer_execution_record=True,
              independent_public_entry_four_cases=True, source_identity_and_defaults_verified=True,
              idle_and_process_exit_verified=True,
              speed_claim='no aggregate TPS benefit established; default follows predeclared gate',
              scope='gfx936 compatibility runtime; CUDA native_moe unit compiled; no SYCL/Metal/multi-GPU runtime claim')
(R / 'audit.json').write_text(json.dumps(checks, indent=2) + '\n')
print('FINAL CPU RESULT DIRECT AUDIT PASS')
