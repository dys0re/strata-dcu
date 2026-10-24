"""Standalone BF16-head proxy end-to-end measurement, after all other GPU jobs finish."""
from pathlib import Path
import subprocess,os,json
P=Path('/root/private_data/strata-dcu/model-specific-20261024');R=P/'final-v1'
assert (P/'recover-final-status.log').read_text().strip()=='RECOVERY_RC=0'
probe=subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
assert 'Used Memory (MiB): 2' in probe,probe
cmd=['python3','-u','/root/strata-dcu/Strata/tools/bench_dflash2_native.py','--engine',str(R/'strata-candidate'),'--mode','adaptive','--width','5','--graph','--draft-adaptive','--no-resident-pin','--timing-only','--repeat','3','--qa-after-bench','--extended-qa','--out',str(R/'proxy-bf16')]
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_VERIFY_HIT_PLAN='1',STRATA_CPU_EXPERT_PIPELINE='1',STRATA_CPU_EARLY_RESULT='1',STRATA_HC_Q8='0',STRATA_DRAFT_HEAD_PROXY='1')
for name in ['STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_VERIFY_TRACE','STRATA_POOL_TRACE']:env.pop(name,None)
with (P/'proxy-kernel-production-flags.log').open('w') as f:
 test=subprocess.run(['/root/private_data/strata-dcu/p1/build/draft_head_proxy_test'],env=dict(env,STRATA_BF16_Q8_VEC4='1',STRATA_BF16_Q8_COLS='4'),stdout=f,stderr=subprocess.STDOUT)
assert test.returncode==0
with (R/'proxy-bf16.log').open('w') as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
(R/'proxy-bf16-status.json').write_text(json.dumps(dict(returncode=rc))+'\n')
raise SystemExit(rc)
