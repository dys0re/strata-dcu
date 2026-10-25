"""Persistent build and CPU correctness gates; no public engine replacement."""
from pathlib import Path
import hashlib,json,os,resource,shutil,subprocess,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
R=Path('/root/private_data/strata-dcu/cpu-result-direct-20261025')
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
files=['CMakeLists.txt', 'include/strata/kernels/native_moe.hpp', 'src/kernels/cuda/native_moe.cu', 'src/core/verify.cpp', 'src/kernels/mapped_moe_parity.cpp', 'tools/autoround_pack.py', 'tools/bench_dflash2_native.py']
(R/'source-manifest.json').write_text(json.dumps({f:hashlib.sha256((S/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
run('build',['cmake','--build',str(B),'--target','strata','mapped_moe_parity','native_multi_parity','resident_plan_parity','-j','4'])
shutil.copy2(B/'strata',R/'strata-candidate')
(R/'candidate.json').write_text(json.dumps(dict(path=str(R/'strata-candidate'),sha256=hashlib.sha256((R/'strata-candidate').read_bytes()).hexdigest()),indent=2)+'\n')
run('gpu-tests',['ctest','--test-dir',str(B),'-R','^(mapped_moe_parity|native_multi_parity|resident_plan_parity)$','--output-on-failure','-j','1'])
print('BUILD_AND_GPU_GATES_PASS',flush=True)
