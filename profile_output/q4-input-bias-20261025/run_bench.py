"""ABBA comparison, fixed experts/weights; quiet-core selection before warmup."""
from pathlib import Path
import ast,hashlib,json,os,resource,subprocess,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
R=Path('/root/private_data/strata-dcu/q4-input-bias-20261025')
S=Path('/root/strata-dcu/Strata');P=Path('/root/private_data/strata-dcu/model-specific-20261024')
assert all(x['returncode']==0 for x in json.loads((R/'build-status.json').read_text()))
assert len(json.loads((R/'build-status.json').read_text()))==3
source=S/'tools/bench_dflash2_native.py';text=source.read_text()
text='import sys\nsys.path.insert(0,"/root/strata-dcu/Strata/tools")\n'+text
needle='  for name,prompt,n,expected,temperature in cases:'
assert text.count(needle)==1
text=text.replace(needle,"""  # [PROFILE] Select quiet cores after load; never switch during decode.
  from idle_cores import choose_and_pin
  choose_and_pin(p.pid,os.sched_getaffinity(0),13,out/'idle-core-selection.json')
  for name,prompt,n,expected,temperature in cases:""")
ast.parse(text);(R/'idle_bench.py').write_text(text)
(R/'driver-source.json').write_text(json.dumps(dict(base_driver_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generated_sha256=hashlib.sha256(text.encode()).hexdigest()),indent=2)+'\n')
old=P/'final-v1/strata-final';new=R/'strata-candidate'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='3ae138ead7fdbab0324bb17b423022d32c497eb18e3fb5aa82901011b98c9450'
env=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_VERIFY_HIT_PLAN='1',STRATA_CPU_EXPERT_PIPELINE='1',STRATA_CPU_EARLY_RESULT='1',STRATA_HC_Q8='1',STRATA_DRAFT_HEAD_PROXY='0')
for key in ['STRATA_VERIFY_PROFILE','STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_POOL_TRACE','STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US','STRATA_POOL_TEST_DELAY_US']:
    env.pop(key,None)
status=[]
cases=[('baseline-a',old,'0'),('candidate-b',new,'1'),('candidate-b2',new,'1'),('baseline-a2',old,'0')]
(R/'plan.json').write_text(json.dumps(dict(order=[x[0] for x in cases],normal_throughput='3 repetitions of 256tokens per workload per cold start; no full-layer tracing',default_rule='geometric-mean TPS ratio >=1.01 and no task below 0.97, using pooled 6 samples/task/config; descriptive result only, not statistical significance',experts_gpu=8500,experts_ram=16076,weight_limit=34351349760),indent=2)+'\n')
for label,engine,bias in cases:
    assert 'Used Memory (MiB): 2' in subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
    cmd=['python3','-u',str(R/'idle_bench.py'),'--engine',str(engine),'--mode','adaptive','--width','5','--graph','--draft-adaptive','--no-resident-pin','--timing-only','--process-placement','--thread-usage','--repeat','3','--qa-after-bench','--extended-qa','--dense-gguf',str(P/'dense-q8.gguf'),'--mtp-runtime',str(P/'mtp-tuned'),'--out',str(R/label)]
    print('START',label,flush=True);start=time.monotonic()
    with (R/(label+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,env=dict(env,STRATA_CPU_Q4_INPUT_BIAS=bias),stdout=log,stderr=subprocess.STDOUT)
        (R/'active.json').write_text(json.dumps(dict(label=label,pid=p.pid,driver_pid=os.getpid(),command=cmd)))
        rc=p.wait()
    status.append(dict(label=label,returncode=rc,seconds=time.monotonic()-start,engine=str(engine),input_bias=bias))
    (R/'bench-status.json').write_text(json.dumps(status,indent=2)+'\n')
    print('END',label,rc,flush=True);assert rc==0,label
print('ABBA_COMPLETE',flush=True)
