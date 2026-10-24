"""Preserve throughput runs; repeat full QA after correcting the added semantic wrapper check."""
from pathlib import Path
import subprocess,os,json,hashlib,statistics,math,re,time
P=Path('/root/private_data/strata-dcu/model-specific-20261024');R=P/'final-v1';E=R/'strata-candidate'
handle=Path('/proc/68406/stat');identity=handle.read_text().split()[21] if handle.exists() else None
while handle.exists() and identity is not None:
 try:
  if handle.read_text().split()[21]!=identity:break
 except FileNotFoundError:break
 time.sleep(5)
status=json.loads((R/'status.json').read_text());assert len(status)==5,status
assert not (R/'selection.json').exists(),'Selection exists; inspect rather than overwrite'
ENV=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_VERIFY_HIT_PLAN='1',STRATA_CPU_EXPERT_PIPELINE='1',STRATA_CPU_EARLY_RESULT='1',STRATA_HC_Q8='0',STRATA_DRAFT_HEAD_PROXY='0')
for k in ['STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_TRACE','STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_VERIFY_PROFILE']:ENV.pop(k,None)
OLD='/root/private_data/strata-dcu/layer-scheduling-20261024/v9/strata-final'
Q=['--dense-gguf',str(P/'dense-q8.gguf')];T=['--mtp-runtime',str(P/'mtp-tuned')]
configs={'baseline':(OLD,{},[]),'q8':(E,{},Q),'q8-tuned':(E,{},Q+T),'hc-q8-tuned':(E,{'STRATA_HC_Q8':'1'},Q+T),'q8-tuned-proxy':(E,{'STRATA_DRAFT_HEAD_PROXY':'1'},Q+T)}
rechecks=[]
def qa(label,engine,flags,args,mode='adaptive'):
 probe=subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
 assert re.search(r'Used Memory \(MiB\): 2\b',probe),probe
 cmd=['python3','-u','/root/strata-dcu/Strata/tools/bench_dflash2_native.py','--engine',str(engine),'--mode',mode,'--width','5' if mode=='adaptive' else '8','--graph','--draft-adaptive','--no-resident-pin','--qa-only','--extended-qa','--timing-only','--out',str(R/label),*args]
 print('START',label,flush=True)
 with (R/(label+'.log')).open('w') as f:rc=subprocess.run(cmd,env=dict(ENV,**flags),stdout=f,stderr=subprocess.STDOUT).returncode
 rechecks.append(dict(label=label,returncode=rc));(R/'qa-recheck-status.json').write_text(json.dumps(rechecks,indent=2)+'\n')
 print('END',label,rc,flush=True);return rc
valid={}
for item in status:
 name=item['label']
 if item['returncode']==0:valid[name]=True;continue
 engine,flags,args=configs[name]
 valid[name]=qa(name+'-qa-recheck',engine,flags,args)==0
assert valid['baseline'],'Baseline full QA did not pass after checker correction'
def scores(label):
 rows=json.loads((R/label/'results.json').read_text())
 return {k:statistics.median(x['decode_tps'] for x in rows if re.fullmatch(k+r'-\d+',x['case'])) for k in ['story','chinese','code']}
base=scores('baseline');medians={n:scores(n) for n in configs if n!='baseline'}
score=lambda x:math.prod(x[k]/base[k] for k in base)**(1/3)
eligible=[n for n in medians if valid[n] and all(medians[n][k]>=base[k] for k in base)]
assert eligible,'No variant passes QA and all median speed comparisons'
chosen=max(eligible,key=lambda n:score(medians[n]))
if 'proxy' in chosen:
 plain=[n for n in eligible if 'proxy' not in n]
 if plain:
  best=max(plain,key=lambda n:score(medians[n]))
  if score(medians[chosen])<1.01*score(medians[best]):chosen=best
selection=dict(chosen=chosen,flags=configs[chosen][1],dense_gguf=str(P/'dense-q8.gguf'),mtp_runtime=str(P/'mtp-tuned') if 'tuned' in chosen else None,baseline=base,medians=medians,effective_qa_pass=valid,engine_sha256=hashlib.sha256(E.read_bytes()).hexdigest(),selection_rule='All task medians >= baseline; highest geometric ratio; proxy needs >=1% over best non-proxy; not significance. Full QA rechecked after semantic Markdown-wrapper correction; original return codes preserved.')
(R/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');print('SELECTION',json.dumps(selection),flush=True)
proj=['--dense-gguf',str(P/'dense-projections-q8.gguf'),'--suffix-draft','3','--lookup-checks']
assert qa('native-projections-qa',E,{'STRATA_HC_Q8':'1'},proj,mode='native')==0
assert qa('joint-proxy-reject-qa',E,{'STRATA_HC_Q8':'1','STRATA_DRAFT_HEAD_PROXY':'1'},proj+T+['--corrupt','1'],mode='joint')==0
assert all(valid.values()),'One tested configuration still has a real QA failure'
print('RECOVERY_COMPLETE; ALL QA PASS',flush=True)
