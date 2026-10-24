from pathlib import Path
import os,sys,subprocess,json,time,hashlib,statistics
ROOT=Path('/root/private_data/strata-dcu/layer-scheduling-20261024')
BUILD=Path('/root/private_data/strata-dcu/p1/build')
ENGINE=ROOT/'strata-candidate'
DRIVER='/root/strata-dcu/Strata/tools/bench_dflash2_native.py'
ENV=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2')
FLAGS=['STRATA_VERIFY_HIT_PLAN','STRATA_CPU_EXPERT_PIPELINE','STRATA_CPU_EARLY_RESULT','STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US']
for k in FLAGS:ENV.pop(k,None)
status=[]
def run(label,cmd,env):
 print('START',label,flush=True)
 with (ROOT/(label+'.log')).open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 status.append(dict(label=label,returncode=r.returncode));(ROOT/'status.json').write_text(json.dumps(status,indent=2))
 print('END',label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
run('resident-plan-final',[str(BUILD/'resident_plan_parity')],ENV)
for label,extra in [('pool-plain',{}),('pool-pipeline',{'STRATA_CPU_EXPERT_PIPELINE':'1'}),('pool-sleep',{'STRATA_CPU_EXPERT_PIPELINE':'1','STRATA_POOL_IDLE_US':'0'})]:
 run(label,[str(BUILD/'pool_tasks_test')],dict(ENV,**extra))
ENGINE.write_bytes((BUILD/'strata').read_bytes());ENGINE.chmod(0o755)
(ROOT/'engine-sha256.txt').write_text(hashlib.sha256(ENGINE.read_bytes()).hexdigest()+'\n')
settings=[('baseline',{}),('gpu-plan',{'STRATA_VERIFY_HIT_PLAN':'1'}),
 ('cpu-pipeline',{'STRATA_CPU_EXPERT_PIPELINE':'1'}),
 ('both',{'STRATA_VERIFY_HIT_PLAN':'1','STRATA_CPU_EXPERT_PIPELINE':'1'}),
 ('early',{'STRATA_VERIFY_HIT_PLAN':'1','STRATA_CPU_EXPERT_PIPELINE':'1','STRATA_CPU_EARLY_RESULT':'1'}),
 ('staged-2',{'STRATA_VERIFY_HIT_PLAN':'1','STRATA_CPU_EXPERT_PIPELINE':'1','STRATA_CPU_EARLY_RESULT':'1','STRATA_PCIE_STAGED_MAX':'2'}),
 ('idle-200',{'STRATA_VERIFY_HIT_PLAN':'1','STRATA_CPU_EXPERT_PIPELINE':'1','STRATA_CPU_EARLY_RESULT':'1','STRATA_POOL_IDLE_US':'200'})]
for label,extra in settings:
 cmd=['python3','-u',DRIVER,'--engine',str(ENGINE),'--mode','adaptive','--width','5',
      '--repeat','1','--qa-after-bench','--timing-only','--thread-usage','--no-resident-pin',
      '--out',str(ROOT/label)]
 run(label,cmd,dict(ENV,**extra))
 print('RESULT',label,[(x['case'],round(x['decode_tps'],2)) for x in json.loads((ROOT/label/'results.json').read_text()) if x['case'] in ['story-0','chinese-0','code-0']],flush=True)
print('MATRIX_COMPLETE',flush=True)
