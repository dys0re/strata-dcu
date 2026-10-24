from pathlib import Path
import os,subprocess,json,hashlib,statistics,math
PARENT=Path('/root/private_data/strata-dcu/layer-scheduling-20261024')
ROOT=PARENT/'v9';ROOT.mkdir(exist_ok=False)
BUILD=Path('/root/private_data/strata-dcu/p1/build');ENGINE=ROOT/'strata-candidate'
DRIVER='/root/strata-dcu/Strata/tools/bench_dflash2_native.py'
FLAGS=['STRATA_VERIFY_HIT_PLAN','STRATA_CPU_EXPERT_PIPELINE','STRATA_CPU_EARLY_RESULT','STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US','STRATA_POOL_TEST_DELAY_US','STRATA_POOL_TRACE','STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_VERIFY_PROFILE']
ENV=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2')
for flag in FLAGS:ENV.pop(flag,None)
status=[]
def launch(label,cmd,env):
 print('START',label,flush=True)
 with (ROOT/(label+'.log')).open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 status.append(dict(label=label,returncode=r.returncode));(ROOT/'status.json').write_text(json.dumps(status,indent=2))
 print('END',label,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
def bench(label,flags,mode='adaptive',repeat=3,extra=()):
 cmd=['python3','-u',DRIVER,'--engine',str(ENGINE),'--mode',mode,'--width','5' if mode=='adaptive' else '8',
      '--graph','--draft-adaptive','--repeat',str(repeat),'--timing-only','--thread-usage','--no-resident-pin','--out',str(ROOT/label)]
 cmd+=['--qa-after-bench'] if repeat else ['--qa-only']
 cmd+=list(extra)
 launch(label,cmd,dict(ENV,**flags))
launch('resident-plan',[str(BUILD/'resident_plan_parity')],ENV)
for name,flags in [('pool-plain',{}),('pool-pipeline',{'STRATA_CPU_EXPERT_PIPELINE':'1'}),('pool-sleep',{'STRATA_CPU_EXPERT_PIPELINE':'1','STRATA_POOL_IDLE_US':'0','STRATA_POOL_TEST_DELAY_US':'2000'})]:
 launch(name,[str(BUILD/'pool_tasks_test')],dict(ENV,**flags))
ENGINE.write_bytes((BUILD/'strata').read_bytes());ENGINE.chmod(0o755)
(ROOT/'engine-sha256.txt').write_text(hashlib.sha256(ENGINE.read_bytes()).hexdigest()+'\n')
CORE={'STRATA_VERIFY_HIT_PLAN':'1','STRATA_CPU_EXPERT_PIPELINE':'1','STRATA_CPU_EARLY_RESULT':'1'}
CPU={'STRATA_CPU_EXPERT_PIPELINE':'1','STRATA_CPU_EARLY_RESULT':'1'}
def med(folder):
 r=json.loads((folder/'results.json').read_text())
 return {k:statistics.median(x['decode_tps'] for x in r if x['case'] in [f'{k}-{i}' for i in range(3)]) for k in ['story','chinese','code']}
base=med(PARENT/'v7/baseline')
bench('core',CORE);bench('cpu-early',CPU)
rows={name:med(ROOT/name) for name in ['core','cpu-early']}
score=lambda row:math.prod(row[k]/base[k] for k in base)**(1/3)
eligible=[name for name,row in rows.items() if min(row[k]/base[k] for k in base)>=.97]
if eligible:chosen=max(eligible,key=lambda name:score(rows[name]));flags=dict(CORE if chosen=='core' else CPU)
else:
 bench('plain',{});rows['plain']=med(ROOT/'plain');chosen='plain';flags={}
bench('idle-500',dict(flags,STRATA_POOL_IDLE_US='500'))
rows['idle-500']=med(ROOT/'idle-500')
# Accept the lower-CPU setting only without a measured task regression against production, and no more than
# 2% median task loss relative to the chosen full-spin implementation. These are selection rules, not significance.
if all(rows['idle-500'][k]>=base[k] and rows['idle-500'][k]>=.98*rows[chosen][k] for k in base):
 chosen='idle-500';flags['STRATA_POOL_IDLE_US']='500'
def busy(folder):
 r=json.loads((folder/'results.json').read_text());u=json.loads((folder/'thread-usage.json').read_text());values=[]
 for x in r:
  if x['case'] not in [f'{k}-{i}' for k in ['story','chinese','code'] for i in range(3)]:continue
  z=next(z for z in u if z['done']==x['done']);a,b=z['start'],z['end'];dt=b['time_ns']-a['time_ns']
  v=[100*(y['runtime_ns']-a['threads'][tid]['runtime_ns'])/dt for tid,y in b['threads'].items() if tid in a['threads'] and y['affinity'] in list(map(str,range(1,13)))]
  assert len(v)==12
  values.append(statistics.mean(v))
 return statistics.median(values)
# Also require an observed reduction, rather than selecting a sleep knob from its name alone.
full_spin=max([name for name in rows if name!='idle-500' and min(rows[name][k]/base[k] for k in base)>=.97],key=lambda name:score(rows[name]),default='plain')
if chosen=='idle-500' and busy(ROOT/full_spin)-busy(ROOT/'idle-500')<10:
 chosen=full_spin;flags=dict(CORE if chosen=='core' else CPU if chosen=='cpu-early' else {})
selection=dict(chosen=chosen,flags=flags,baseline=base,medians=rows,scores={k:score(v) for k,v in rows.items()},worker_busy={name:busy(ROOT/name) for name in rows},engine_sha256=hashlib.sha256(ENGINE.read_bytes()).hexdigest())
(ROOT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print('SELECTION',json.dumps(selection),flush=True)
bench('forced-staged-2',dict(CORE,STRATA_PCIE_STAGED_MAX='2'),repeat=0,extra=['--request-pcie-frac','1','--suffix-draft','3','--lookup-checks'])
# Exercise the same target execution at up to eight rows, including accepted drafts and complete rejection.
bench('native-qa',dict(flags,STRATA_VERIFY_HIT_PLAN='1'),mode='native',repeat=0,extra=['--suffix-draft','3','--lookup-checks'])
bench('joint-reject-qa',dict(flags,STRATA_VERIFY_HIT_PLAN='1'),mode='joint',repeat=0,extra=['--suffix-draft','3','--lookup-checks','--corrupt','1'])
print('FINAL_MATRIX_COMPLETE',flush=True)
