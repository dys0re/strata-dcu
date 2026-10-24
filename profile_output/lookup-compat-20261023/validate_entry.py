"""Verbatim public-entry check; run on the idle remote GPU after selecting the candidate."""
import json,re,subprocess,hashlib
from pathlib import Path
OUT=Path('/root/private_data/strata-dcu/lookup-compat-20261023/entry');OUT.mkdir(exist_ok=False)
CACHE=Path('/root/private_data/strata-dcu/autoround')
ENGINE=Path('/root/strata-dcu/build-gfx936/strata')
assert hashlib.sha256(ENGINE.read_bytes()).hexdigest()=='1c5b8a268585ab62bfe53a098c114b7e645c8befa91384d58c645fec9a8ff396'
phrase='The copper owl watches the quiet river.'
expected=' '.join([phrase]*8)
prompt='Copy the text between <text> tags exactly, without tags or commentary.\n<text>'+expected+'</text>'
results=[]
for mode,args,weight,experts in [
 ('mtp',[],33542205952,23500800000),
 ('native',['--dflash2','/root/private_data/strata-dcu/dflash2/checkpoint'],32827238400,22947840000),
 ('joint',['--dflash2','/root/private_data/strata-dcu/dflash2/checkpoint','--dflash-mtp-prefix','2'],34195377152,22947840000)]:
 out=OUT/mode;out.mkdir()
 cmd=['bash','/root/strata-dcu/run-autoround.sh',*args,'--suffix-draft','3','--prompt',prompt,'--max-new','128']
 (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 with (out/'invocation.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,(mode,r.returncode)
 for name in ['generation.json','run-command.json','engine.stdout.log','engine.stderr.log']:
  (out/name).write_bytes((CACHE/name).read_bytes())
 x=json.loads((out/'generation.json').read_text());log=(out/'engine.stderr.log').read_text()
 assert x['text'].strip()==expected,(mode,x['text'])
 assert x['weight_vram']['weights']==weight and x['weight_vram']['experts']==experts
 assert x['weight_vram']['weights']<=x['weight_vram']['limit']==34351349760
 assert x['kernel_options']['hc_split']=='2' and x['kernel_options']['cpu_q4_mt'] and x['kernel_options']['dflash_small_gemv']
 assert x['suffix_draft']==3 and x['expert_execution']==dict(resident_pin=False,adapt_every=0,adapt_swaps=96,pcie_mode='auto',pcie_frac=0)
 assert 'runs as staged' in log
 assert sum(map(int,re.findall(r'suffix drafts\s*:?\s+(\d+) windows',log+(out/'engine.stdout.log').read_text())))>0
 if mode=='mtp':assert x['mtp_mode']=='adaptive' and x['mtp_max_t']==5
 if mode=='native':assert x['draft_method']=='dflash2' and x['draft_backend']=='native'
 if mode=='joint':assert x['draft_method']=='mtp+dflash2' and x['joint_mtp_prefix']==2
 results.append(dict(mode=mode,answer_correct=True,lookup_exercised=True,weights=weight,staged=True))
 (OUT/'checks.json').write_text(json.dumps(results,indent=2)+'\n')
 print('PASS ENTRY',mode,flush=True)
print('PASS ALL PUBLIC ENTRIES',flush=True)
