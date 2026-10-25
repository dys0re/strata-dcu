"""Check the shared CPU change in native DFlash2 and forced-rejection joint mode."""
from pathlib import Path
import json,os,resource,subprocess,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
R=Path('/root/private_data/strata-dcu/cpu-result-direct-20261025')
assert len(json.loads((R/'bench-status.json').read_text()))==4
assert all(x['returncode']==0 for x in json.loads((R/'bench-status.json').read_text()))
env=dict(os.environ,STRATA_CPU_RESULT_DIRECT='1',STRATA_CPU_Q4_MT='1',STRATA_CPU_Q4_INPUT_BIAS='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_HC_Q8='0',STRATA_DRAFT_HEAD_PROXY='0',STRATA_VERIFY_HIT_PLAN='0',STRATA_CPU_EXPERT_PIPELINE='0',STRATA_CPU_EARLY_RESULT='0')
for k in ['STRATA_VERIFY_PROFILE','STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_POOL_TRACE','STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US','STRATA_POOL_TEST_DELAY_US']:
    env.pop(k,None)
status=[]
for label,mode,extra in [('native-qa','native',[]),('joint-reject-qa','joint',['--corrupt','1']),('disabled-qa','adaptive',[]),('device-plan-fallback-qa','adaptive',[])]:
    assert 'Used Memory (MiB): 2' in subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
    cmd=['python3','-u',str(R/'idle_bench.py'),'--engine',str(R/'strata-candidate'),'--mode',mode,'--width','8' if mode in ['native','joint'] else '5','--graph','--draft-adaptive','--no-resident-pin','--timing-only','--process-placement','--thread-usage','--qa-only','--extended-qa','--suffix-draft','3','--lookup-checks','--out',str(R/label),*extra]
    runenv=dict(env)
    if label=='disabled-qa':runenv['STRATA_CPU_RESULT_DIRECT']='0'
    if label=='device-plan-fallback-qa':runenv['STRATA_VERIFY_DEVICE_PLAN']='1'
    print('START',label,flush=True)
    with (R/(label+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,env=runenv,stdout=log,stderr=subprocess.STDOUT)
        (R/'active-modes.json').write_text(json.dumps(dict(label=label,pid=p.pid,driver_pid=os.getpid())))
        rc=p.wait()
    status.append(dict(label=label,returncode=rc,environment={k:runenv.get(k) for k in ['STRATA_CPU_RESULT_DIRECT','STRATA_VERIFY_DEVICE_PLAN','STRATA_VERIFY_HIT_PLAN','STRATA_CPU_Q4_INPUT_BIAS']}));(R/'modes-status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('END',label,rc,flush=True);assert rc==0,label
print('MODES_QA_COMPLETE',flush=True)
