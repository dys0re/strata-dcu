"""Fresh documented-entry validation. Run once on the idle remote GPU, exactly as ENTRY.md states."""
import os,re,json,hashlib,subprocess,time
from pathlib import Path
PARENT=Path('/root/private_data/strata-dcu/model-specific-20261024')
ROOT=PARENT/'final-v1'
SELECT=json.loads((ROOT/'selection.json').read_text())
EXPECTED=json.loads((ROOT/SELECT['chosen']/'weight-usage.json').read_text())
OUT=ROOT/'entry';OUT.mkdir(exist_ok=False)
CACHE=Path('/root/private_data/strata-dcu/autoround')
ENGINE=Path('/root/strata-dcu/build-gfx936/strata')
SHA=SELECT['engine_sha256']
assert ENGINE.resolve()==ROOT/'strata-final'
with ENGINE.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==SHA
probe=subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
assert re.search(r'Total Memory \(MiB\): 65520',probe)
assert re.search(r'Used Memory \(MiB\): 2\b',probe),probe
(OUT/'gpu-before.txt').write_text(probe)
env=os.environ.copy()
for key in ['STRATA_DCU_MODEL_DEFAULTS','STRATA_DCU_MODEL_CONFIG','STRATA_DENSE_Q8_VARIANT','STRATA_HC_Q8','STRATA_DRAFT_HEAD_PROXY','STRATA_DCU_SCHEDULING_DEFAULTS','STRATA_VERIFY_HIT_PLAN','STRATA_CPU_EXPERT_PIPELINE','STRATA_CPU_EARLY_RESULT','STRATA_PCIE_STAGED_MAX','STRATA_POOL_IDLE_US','STRATA_POOL_SPIN_US','STRATA_POOL_TEST_DELAY_US','STRATA_POOL_TRACE','STRATA_VERIFY_TRACE','STRATA_VERIFY_TRACE_FULL','STRATA_VERIFY_PROFILE','STRATA_PCIE_BALANCE','STRATA_SH_STREAM','STRATA_HC_SPLIT','STRATA_CPU_Q4_MT','STRATA_DFLASH_SMALL_GEMV']:
 env.pop(key,None)
phrase='The copper owl watches the quiet river.'
expected=' '.join([phrase]*8)
prompt='Copy the text between <text> tags exactly, without tags or commentary.\n<text>'+expected+'</text>'
checks=[]
for mode,args,weight,expert_bytes,slots in [
 ('mtp',[],EXPECTED['weights'],23500800000,8500),
 ('native',['--dflash2','/root/private_data/strata-dcu/dflash2/checkpoint'],32827238400,22947840000,8300),
 ('joint',['--dflash2','/root/private_data/strata-dcu/dflash2/checkpoint','--dflash-mtp-prefix','2'],34195377152,22947840000,8300)]:
 out=OUT/mode;out.mkdir()
 cmd=['bash','/root/strata-dcu/run-autoround.sh',*args,'--context','512','--suffix-draft','3','--prompt',prompt,'--max-new','128']
 (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/'invocation.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert r.returncode==0,(mode,r.returncode)
 for name in ['generation.json','run-command.json','engine.stdout.log','engine.stderr.log']:
  (out/name).write_bytes((CACHE/name).read_bytes())
 x=json.loads((out/'generation.json').read_text());stderr=(out/'engine.stderr.log').read_text();stdout=(out/'engine.stdout.log').read_text()
 assert x['text'].strip()==expected,(mode,x['text'])
 w=x['weight_vram'];assert w['total']==68702699520 and w['weights']==weight and w['experts']==expert_bytes
 assert w['weights']<=w['limit']==34351349760
 assert sum(w.get(k,0) for k in ['pack','dense','head','experts','mtp','external_draft','native_draft','staging'])==w['weights']
 assert x['kernel_options']==dict(hc_split='2',cpu_q4_mt=True,dflash_small_gemv=True)
 v=x['model_variants']
 if mode=='mtp':
  assert v['dense_gguf']==SELECT['dense_gguf'] and v['mtp_runtime']==SELECT['mtp_runtime'],v
  assert v['hc_q8']==(SELECT['flags'].get('STRATA_HC_Q8')=='1'),v
  assert v['draft_head_proxy']==(SELECT['flags'].get('STRATA_DRAFT_HEAD_PROXY')=='1'),v
  assert w.get('mtp_proxy',0)==EXPECTED.get('mtp_proxy',0)
 else:
  assert v['dense_gguf']==str(CACHE/'dense.gguf') and v['mtp_runtime'] is None,v
  assert not v['hc_q8'] and not v['draft_head_proxy'],v
 assert ('STRATA_HC_Q8=1:' in stderr)==v['hc_q8']
 assert ('draft-only 2-bit head screening' in stderr)==v['draft_head_proxy']
 s=x['scheduling_options'];enabled=(mode=='mtp')
 assert all(s[k]==enabled for k in ['gpu_hit_plan','cpu_expert_pipeline','early_result']),s
 assert s['staged_experts_max']==0 and s['pool_idle_us'] is None and s['shared_stream']=='auto'
 assert x['expert_execution']==dict(resident_pin=False,adapt_every=0,adapt_swaps=96,pcie_mode='auto',pcie_frac=0)
 assert 'participating threads: 13' in stderr and 'runs as staged' in stderr
 assert f'expert_slots={slots}' in stderr
 assert bool('resident groups start from a separate GPU plan' in stderr)==enabled
 assert bool('per-expert gate/up -> quantize -> down pipeline enabled' in stderr)==enabled
 assert sum(map(int,re.findall(r'suffix drafts\s*:?\s+(\d+) windows',stderr+stdout)))>0
 if mode=='mtp':assert x['mtp_mode']=='adaptive' and x['mtp_max_t']==5
 if mode=='native':assert x['draft_method']=='dflash2' and x['draft_backend']=='native'
 if mode=='joint':assert x['draft_method']=='mtp+dflash2' and x['joint_mtp_prefix']==2 and x['draft_backend']=='native'
 checks.append(dict(mode=mode,answer_correct=True,lookup_exercised=True,weights=weight,expert_slots=slots,scheduled=enabled,engine_sha256=SHA))
 (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
 print('PASS ENTRY',mode,flush=True)
for i in range(20):
 probe=subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmeminfo','vram'],text=True)
 if re.search(r'Used Memory \(MiB\): 2\b',probe):break
 time.sleep(.25)
assert re.search(r'Used Memory \(MiB\): 2\b',probe),probe
(OUT/'gpu-after.txt').write_text(probe)
ps=subprocess.check_output(['ps','-eo','pid,comm'],text=True)
assert not any(re.search(r'\bstrata(?:-|$)',line) for line in ps.splitlines()),ps
with ENGINE.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==SHA
print('PASS ALL PUBLIC ENTRIES; GPU idle; no remaining engine',flush=True)
