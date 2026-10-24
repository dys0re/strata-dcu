"""Serial retest of paths affected by shared-host CPU contention. Fixed experts/threads/weight cap."""
from pathlib import Path
import os,subprocess,json,time,hashlib
R=Path('/root/private_data/strata-dcu/idle-core-retest-20261025');P=Path('/root/private_data/strata-dcu/model-specific-20261024');E=P/'final-v1/strata-final'
assert hashlib.sha256(E.read_bytes()).hexdigest()=='3ae138ead7fdbab0324bb17b423022d32c497eb18e3fb5aa82901011b98c9450'
assert 'Used Memory (MiB): 2' in subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
subprocess.run(['python3',str(R/'prepare.py')],check=True)
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_VERIFY_HIT_PLAN='1',STRATA_CPU_EXPERT_PIPELINE='1',STRATA_CPU_EARLY_RESULT='1',STRATA_HC_Q8='0',STRATA_DRAFT_HEAD_PROXY='0')
for k in ['STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US','STRATA_POOL_TRACE','STRATA_VERIFY_TRACE','STRATA_VERIFY_PROFILE']:env.pop(k,None)
quant=['--dense-gguf',str(P/'dense-q8.gguf'),'--mtp-runtime',str(P/'mtp-tuned')]
cases=[('fixed-bf16-smoke',{},[],False),('idle-bf16',{},[],True),('idle-bf16-proxy',{'STRATA_DRAFT_HEAD_PROXY':'1'},[],True),('idle-selected',{'STRATA_HC_Q8':'1'},quant,True),('idle-selected-proxy',{'STRATA_HC_Q8':'1','STRATA_DRAFT_HEAD_PROXY':'1'},quant,True)]
status=[]
for label,flags,args,idle in cases:
 script=R/'idle_bench.py' if idle else Path('/root/strata-dcu/Strata/tools/bench_dflash2_native.py')
 cmd=['python3','-u',str(script),'--engine',str(E),'--mode','adaptive','--width','5','--graph','--draft-adaptive','--no-resident-pin','--timing-only','--process-placement','--thread-usage','--out',str(R/label)]+args
 cmd+=['--repeat','3','--qa-after-bench','--extended-qa'] if idle else ['--smoke']
 print('START',label,flush=True);start=time.monotonic()
 with (R/(label+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,env=dict(env,**flags),stdout=log,stderr=subprocess.STDOUT)
  (R/'active.json').write_text(json.dumps(dict(label=label,pid=p.pid,command=cmd)))
  rc=p.wait()
 status.append(dict(label=label,returncode=rc,wall_seconds=time.monotonic()-start));(R/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 print('END',label,rc,flush=True)
 assert rc==0,(label,rc)
 assert 'Used Memory (MiB): 2' in subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
print('ALL_COMPLETE',flush=True)
