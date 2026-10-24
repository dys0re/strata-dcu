"""Sequential QA and exploratory throughput; fixed expert slots, no profiling/build in flight."""
from pathlib import Path
import subprocess,os,json,hashlib
P=Path('/root/private_data/strata-dcu/model-specific-20261024')
R=P/'pilot-v1';R.mkdir(exist_ok=False)
B=Path('/root/private_data/strata-dcu/p1/build')
E=R/'strata-candidate';E.write_bytes((B/'strata').read_bytes());E.chmod(0o755)
base='/root/private_data/strata-dcu/layer-scheduling-20261024/v9/strata-final'
(R/'binary-sha256.json').write_text(json.dumps({str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [base,str(E)]},indent=2)+'\n')
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_VERIFY_HIT_PLAN='1',STRATA_CPU_EXPERT_PIPELINE='1',STRATA_CPU_EARLY_RESULT='1',STRATA_HC_Q8='0',STRATA_DRAFT_HEAD_PROXY='0')
for k in ['STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_TRACE','STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_VERIFY_PROFILE']:
 env.pop(k,None)
status=[]
def run(label,engine,flags={},extra=(),qa=False):
 cmd=['python3','-u','/root/strata-dcu/Strata/tools/bench_dflash2_native.py','--engine',str(engine),'--mode','adaptive','--width','5','--graph','--draft-adaptive','--no-resident-pin','--timing-only','--out',str(R/label)]
 cmd+=['--qa-only'] if qa else ['--repeat','1','--qa-after-bench']
 cmd+=list(extra)
 print('START',label,flush=True)
 with (R/(label+'.log')).open('w') as f:rc=subprocess.run(cmd,env=dict(env,**flags),stdout=f,stderr=subprocess.STDOUT).returncode
 status.append(dict(label=label,returncode=rc));(R/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 print('END',label,rc,flush=True)
run('plain-qa',E,qa=True)
run('baseline',base)
run('hc-q8',E,{'STRATA_HC_Q8':'1'})
quant=['--dense-gguf',str(P/'dense-q8.gguf')];tuned=['--mtp-runtime',str(P/'mtp-tuned')]
run('dense-q8',E,{'STRATA_HC_Q8':'1'},quant)
run('tuned',E,{},tuned)
run('proxy',E,{'STRATA_DRAFT_HEAD_PROXY':'1'})
run('quant-tuned',E,{'STRATA_HC_Q8':'1'},quant+tuned)
run('all',E,{'STRATA_HC_Q8':'1','STRATA_DRAFT_HEAD_PROXY':'1'},quant+tuned)
print('PILOT_COMPLETE',flush=True)
