"""Persistent build and CPU correctness gates; no public engine replacement."""
from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
R=Path('/root/private_data/strata-dcu/q4-input-bias-20261025')
S=Path('/root/strata-dcu/Strata');B=Path('/root/private_data/strata-dcu/p1/build')
status=[]
def run(label,cmd,env=None):
    start=time.monotonic()
    with (R/(label+'.log')).open('w') as log:
        rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    status.append(dict(label=label,command=cmd,returncode=rc,wall_seconds=time.monotonic()-start))
    (R/'build-status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(label,rc,flush=True);assert rc==0,label
assert 'Used Memory (MiB): 2' in subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
files=['CMakeLists.txt','include/strata/kernels/cpu/q4_avx512.hpp','src/kernels/cpu/q4_avx512.cpp','src/kernels/cpu/native_expert.cpp','src/kernels/cpu/q4_avx512_parity.cpp','tools/autoround_pack.py','tools/bench_dflash2_native.py']
(R/'source-manifest.json').write_text(json.dumps({f:hashlib.sha256((S/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
run('build',['cmake','--build',str(B),'--target','strata','q4_avx512_parity','pool_tasks_test','-j','4'])
shutil.copy2(B/'strata',R/'strata-candidate')
(R/'candidate.json').write_text(json.dumps(dict(path=str(R/'strata-candidate'),sha256=hashlib.sha256((R/'strata-candidate').read_bytes()).hexdigest()),indent=2)+'\n')
run('cpu-tests',['ctest','--test-dir',str(B),'-R','^(q4_cpu_|pool_expert_pipeline_q4_bias)','--output-on-failure','-j','1'])
pack=Path('/root/private_data/strata-dcu/autoround/pack')
with (pack/'experts.bin').open('rb') as f:blob=f.read(2764800)
assert len(blob)==2764800
(R/'real-expert-l0e0.bin').write_bytes(blob)
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_CPU_Q4_INPUT_BIAS='1',STRATA_CPU_EXPERT_PIPELINE='1',STRATA_POOL_IDLE_US='0',STRATA_POOL_TEST_DELAY_US='2000')
run('real-expert-pool',[str(B/'pool_tasks_test'),'2','2',str(R/'real-expert-l0e0.bin')],env)
print('BUILD_AND_CPU_GATES_PASS',flush=True)
