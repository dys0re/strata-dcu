"""External-only reproduction with CPU scheduling/cgroup sampling; no engine edits."""
from pathlib import Path
import subprocess,os,time,json
R=Path('/root/private_data/strata-dcu/cpu-slowdown-20261025')
P=Path('/root/private_data/strata-dcu/model-specific-20261024')
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_VERIFY_HIT_PLAN='1',STRATA_CPU_EXPERT_PIPELINE='1',STRATA_CPU_EARLY_RESULT='1',STRATA_HC_Q8='1',STRATA_DRAFT_HEAD_PROXY='0')
for k in ['STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US','STRATA_VERIFY_PROFILE','STRATA_POOL_TRACE','STRATA_VERIFY_TRACE','STRATA_PCIE_STAGED_MAX']:env.pop(k,None)
label=os.environ.get('CASE','reproduce'); out=R/label
cmd=['python3','-u','/root/private_data/strata-dcu/cpu-slowdown-20261025/selected_affinity_bench.py','--engine',str(P/'final-v1/strata-final'),'--mode','adaptive','--width','5','--graph','--draft-adaptive','--no-resident-pin','--smoke','--timing-only','--process-placement','--thread-usage','--out',str(out)]
cmd += ['--dense-gguf',str(P/'dense-q8.gguf'),'--mtp-runtime',str(P/'mtp-tuned')]
if os.environ.get('AFFINITY'):cmd=['taskset','-c',os.environ['AFFINITY']]+cmd
samples=[]
with (R/(label+'.log')).open('w') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 (R/(label+'-pid.json')).write_text(json.dumps(dict(pid=p.pid,command=cmd)))
 while p.poll() is None:
  s=dict(time=time.monotonic(),cpu={x.split()[0]:list(map(int,x.split()[1:])) for x in Path('/proc/stat').read_text().splitlines() if x.startswith('cpu')},cgroup=Path('/sys/fs/cgroup/cpu.stat').read_text(),pressure=Path('/sys/fs/cgroup/cpu.pressure').read_text())
  samples.append(s);time.sleep(.2)
 (R/(label+'-samples.json')).write_text(json.dumps(samples))
 (R/(label+'-status.json')).write_text(json.dumps(dict(returncode=p.returncode)))
 print(label,'RC',p.returncode,flush=True)
 raise SystemExit(p.returncode)
