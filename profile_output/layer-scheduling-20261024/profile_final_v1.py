from pathlib import Path
import os,subprocess,json,time,hashlib
from transformers import AutoTokenizer
ROOT=Path('/root/private_data/strata-dcu/layer-scheduling-20261024/v9')
selection=json.loads((ROOT/'selection.json').read_text())
engine=ROOT/'strata-candidate'
cmd=json.loads((ROOT/'core/command.json').read_text())['command'];cmd[0]=str(engine)
base=dict(os.environ,STRATA_CPU_Q4_MT='1',STRATA_DFLASH_SMALL_GEMV='1',STRATA_HC_SPLIT='2',STRATA_TRACE='1',STRATA_DECODE_TIMING='1')
for key in ['STRATA_VERIFY_HIT_PLAN','STRATA_CPU_EXPERT_PIPELINE','STRATA_CPU_EARLY_RESULT','STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US','STRATA_POOL_TEST_DELAY_US','STRATA_VERIFY_PROFILE','STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_POOL_TRACE']:
 base.pop(key,None)
tok=AutoTokenizer.from_pretrained('/root/private_data/Qwen3.8-F',local_files_only=True)
CORE={'STRATA_VERIFY_HIT_PLAN':'1','STRATA_CPU_EXPERT_PIPELINE':'1','STRATA_CPU_EARLY_RESULT':'1'}
variants=[('trace-plain',{}),('trace-selected',selection['flags'])]
if selection['flags']!=CORE:variants.append(('trace-gpu-core',CORE))
for name,flags in variants:
 out=ROOT/name;out.mkdir(exist_ok=False)
 env=dict(base,**flags,STRATA_POOL_TRACE='1',STRATA_VERIFY_TRACE='1',STRATA_VERIFY_TRACE_FULL='1')
 (out/'command.json').write_text(json.dumps(dict(command=cmd,flags=flags,trace_flags={'STRATA_POOL_TRACE':'1','STRATA_VERIFY_TRACE':'1','STRATA_VERIFY_TRACE_FULL':'1'},engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest()),indent=2)+'\n')
 records=[]
 with (out/'stderr.log').open('w') as err,(out/'stdout.log').open('w') as log:
  p=subprocess.Popen(cmd,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
  def line():
   s=p.stdout.readline()
   if not s:raise RuntimeError(f'EOF {p.poll()}')
   log.write(s);log.flush()
   if s.startswith('ERR '):raise RuntimeError(s)
   return s.strip()
  try:
   while not line().startswith('READY '):pass
   for case,file,n in [('warmup','warmup',128),('story','story-0',64),('recovery','ok',16)]:
    req=(ROOT/'core'/(file+'.request.txt')).read_text()
    req=req.replace(req.split(' ',2)[0]+' '+req.split(' ',2)[1]+' ',f'GEN {n} ',1)
    p.stdin.write(req);p.stdin.flush();ids=[];start=time.monotonic_ns()
    while True:
     s=line()
     if s.startswith('T '):ids.append(int(s.split()[1]))
     if s.startswith('DONE '):done=s;break
    fields=done.split();assert int(fields[1])==len(ids) and int(fields[8])==0 and fields[5] in ['length','stop']
    text=tok.decode(ids,skip_special_tokens=True)
    if case=='recovery':assert text.strip()=='OK',text
    records.append(dict(case=case,request=req,done=done,generated_ids=ids,text=text,wall_ns=time.monotonic_ns()-start))
    (out/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
   p.stdin.write('QUIT\n');p.stdin.flush();p.wait(timeout=30);assert p.returncode==0
  finally:
   if p.poll() is None:p.terminate();p.wait(timeout=30)
 (out/'complete.json').write_text(json.dumps(dict(returncode=p.returncode,qa='64-token profile generation and OK recovery; not throughput evidence'))+'\n')
 print('PROFILE_COMPLETE',name,flush=True)
