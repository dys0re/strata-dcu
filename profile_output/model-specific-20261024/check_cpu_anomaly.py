"""Short paired diagnostics after an unexpected CPU-expert slowdown; not a replacement benchmark."""
from pathlib import Path
import os,subprocess,json
P=Path('/root/private_data/strata-dcu/model-specific-20261024');R=P/'final-v1'
assert (P/'bf16-proxy-status.log').read_text().strip()=='PROXY_RC=0'
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_VERIFY_HIT_PLAN='1',STRATA_CPU_EXPERT_PIPELINE='1',STRATA_CPU_EARLY_RESULT='1',STRATA_HC_Q8='0')
status=[]
for name,on in [('cpu-check-plain','0'),('cpu-check-proxy','1')]:
 cmd=['python3','-u','/root/strata-dcu/Strata/tools/bench_dflash2_native.py','--engine',str(R/'strata-candidate'),'--mode','adaptive','--width','5','--graph','--draft-adaptive','--no-resident-pin','--smoke','--timing-only','--process-placement','--thread-usage','--out',str(R/name)]
 print('START',name,flush=True)
 with (R/(name+'.log')).open('w') as f:rc=subprocess.run(cmd,env=dict(env,STRATA_DRAFT_HEAD_PROXY=on),stdout=f,stderr=subprocess.STDOUT).returncode
 status.append(dict(label=name,returncode=rc));(R/'cpu-check-status.json').write_text(json.dumps(status,indent=2)+'\n')
 print('END',name,rc,flush=True)
 assert rc==0
print('CPU_CHECK_COMPLETE',flush=True)
