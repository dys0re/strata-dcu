"""Read-only final resource/deployment snapshot after independent validation."""
from pathlib import Path
import hashlib,json,os,re,subprocess
R=Path('/root/private_data/strata-dcu/cpu-result-direct-20261025')
assert len(json.loads((R/'entry/checks.json').read_text()))==4
engine=Path('/root/strata-dcu/build-gfx936/strata')
gpu=subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
ps=subprocess.check_output(['ps','-eo','pid=,comm='],text=True)
jobs={134463,134466,134467,134468,134469}
for f in ['active.json','active-modes.json','active-profile.json','after-bench-active.json']:
    d=json.loads((R/f).read_text());jobs.update(d[k] for k in ['pid','driver_pid'] if k in d)
alive=sorted(pid for pid in jobs if Path(f'/proc/{pid}').exists())
events=dict((k,int(v)) for k,v in (s.split() for s in Path('/sys/fs/cgroup/memory.events').read_text().splitlines()))
data=dict(engine=str(engine.resolve()),engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest(),
    gpu_used_mib=int(re.search(r'Used Memory \(MiB\): (\d+)',gpu)[1]),gpu_output=gpu,
    engine_processes=[line.strip() for line in ps.splitlines() if line.split()[1].startswith('strata') or line.split()[1]=='q4-probe'],
    job_handles_alive=alive,checked_job_handles=sorted(jobs),memory_events=events,
    cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
    system_sizes=subprocess.check_output(['du','-sh','/root/strata-dcu','/root/.cache'],text=True),
    ple=str(Path('/root/strata-dcu/autoround/ple.gguf').resolve()),
    model_config=json.loads(Path('/root/private_data/strata-dcu/model-specific-20261024/selected-model.json').read_text()))
assert data['gpu_used_mib']==2 and not data['engine_processes'] and not data['job_handles_alive']
assert data['memory_events']['oom']==0 and data['memory_events']['oom_kill']==0
assert data['model_config']['cpu_q4_input_bias'] is True
assert data['model_config']['cpu_result_direct']==json.loads((R/'deployment.json').read_text())['mtp_direct_result_default'] and data['ple'].startswith('/root/private_data/')
assert data['model_config']['engine_sha256']==data['engine_sha256']
(R/'postflight.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
