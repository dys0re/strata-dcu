"""Repeated normal-throughput comparison and executable QA. All expert slots stay fixed."""
from pathlib import Path
import subprocess,os,json,hashlib,statistics,math,re
P=Path('/root/private_data/strata-dcu/model-specific-20261024');R=P/'final-v1';R.mkdir(exist_ok=False)
assert (P/'status-v8.log').read_text().strip()=='TEST_RC=0'
E=R/'strata-candidate';E.write_bytes(Path('/root/private_data/strata-dcu/p1/build/strata').read_bytes());E.chmod(0o755)
OLD='/root/private_data/strata-dcu/layer-scheduling-20261024/v9/strata-final'
ENV=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_VERIFY_HIT_PLAN='1',STRATA_CPU_EXPERT_PIPELINE='1',STRATA_CPU_EARLY_RESULT='1',STRATA_HC_Q8='0',STRATA_DRAFT_HEAD_PROXY='0')
for k in ['STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_TRACE','STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_VERIFY_PROFILE']:
 ENV.pop(k,None)
QUANT=['--dense-gguf',str(P/'dense-q8.gguf')];TUNED=['--mtp-runtime',str(P/'mtp-tuned')]
status=[]
def run(label,engine,flags={},extra=(),mode='adaptive',repeat=3):
 cmd=['python3','-u','/root/strata-dcu/Strata/tools/bench_dflash2_native.py','--engine',str(engine),'--mode',mode,'--width','5' if mode=='adaptive' else '8','--graph','--draft-adaptive','--no-resident-pin','--timing-only','--extended-qa','--out',str(R/label)]
 cmd+=['--repeat',str(repeat),'--qa-after-bench'] if repeat else ['--qa-only']
 cmd+=list(extra)
 print('START',label,flush=True)
 with (R/(label+'.log')).open('w') as f:rc=subprocess.run(cmd,env=dict(ENV,**flags),stdout=f,stderr=subprocess.STDOUT).returncode
 status.append(dict(label=label,returncode=rc));(R/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 print('END',label,rc,flush=True)
 return rc
run('baseline',OLD)
configs=[('q8',{},QUANT),('q8-tuned',{},QUANT+TUNED),('hc-q8-tuned',{'STRATA_HC_Q8':'1'},QUANT+TUNED),('q8-tuned-proxy',{'STRATA_DRAFT_HEAD_PROXY':'1'},QUANT+TUNED)]
for name,flags,args in configs:run(name,E,flags,args)
def scores(label):
 results=json.loads((R/label/'results.json').read_text())
 return {k:statistics.median(x['decode_tps'] for x in results if re.fullmatch(k+r'-\d+',x['case'])) for k in ['story','chinese','code']}
assert (R/'baseline/qa-status.json').exists(),'Baseline QA must pass before choosing defaults'
base=scores('baseline');eligible=[];medians={};flags_for={n:f for n,f,args in configs}
for name,flags,args in configs:
 if (R/name/'qa-status.json').exists() and next(s['returncode'] for s in status if s['label']==name)==0:
  medians[name]=scores(name)
  if all(medians[name][k]>=base[k] for k in base):eligible.append(name)
score=lambda x:math.prod(x[k]/base[k] for k in base)**(1/3)
assert eligible,'No variant improved every measured median; inspect before deployment'
chosen=max(eligible,key=lambda n:score(medians[n]))
if 'proxy' in chosen:
 plain=[n for n in eligible if 'proxy' not in n]
 if plain:
  best=max(plain,key=lambda n:score(medians[n]))
  if score(medians[chosen])<1.01*score(medians[best]):chosen=best
selection=dict(chosen=chosen,flags=flags_for[chosen],dense_gguf=str(P/'dense-q8.gguf'),mtp_runtime=str(P/'mtp-tuned') if 'tuned' in chosen else None,baseline=base,medians=medians,engine_sha256=hashlib.sha256(E.read_bytes()).hexdigest(),selection_rule='All task medians >= baseline; highest geometric ratio; proxy needs >=1% over best non-proxy. Not a significance claim.')
(R/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('SELECTION',json.dumps(selection),flush=True)
# Exercise the feature combination at up to eight positions without changing native DFlash's shared BF16 head.
proj=['--dense-gguf',str(P/'dense-projections-q8.gguf'),'--suffix-draft','3','--lookup-checks']
run('native-projections-qa',E,{'STRATA_HC_Q8':'1'},proj,mode='native',repeat=0)
run('joint-proxy-reject-qa',E,{'STRATA_HC_Q8':'1','STRATA_DRAFT_HEAD_PROXY':'1'},proj+TUNED+['--corrupt','1'],mode='joint',repeat=0)
assert all(x['returncode']==0 for x in status),'At least one final QA/config failed; inspect raw evidence'
print('FINAL_MATRIX_COMPLETE',flush=True)
