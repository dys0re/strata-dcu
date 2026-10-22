"""Verify the completed DFlash2 runs and select the tested engine, retaining MTP as default."""
from pathlib import Path
import hashlib,json,os,re,subprocess,datetime
root=Path('/root/private_data/strata-dcu/dflash2');repo=Path('/root/strata-dcu/Strata')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
provenance=json.loads((root/'adaptive-main/provenance.json').read_text())
source={n:sha(repo/n) for n in provenance['source_sha256']}
assert source==provenance['source_sha256'],'tested core/worker source changed'
for n in ['tools/autoround_pack.py','tools/download_dflash2.py']:source[n]=sha(repo/n)
engine=root/'strata-final';assert sha(engine)==provenance['engine_sha256']
for name in ['graph-main','adaptive-main','quality-final','mtp-final']:
 assert 'PASS' in (root/(name+'.log')).read_text(),name
 u=json.loads((root/name/'weight-usage.json').read_text());assert u['weights']<=u['limit']==u['total']//2
 events=json.loads((root/name/'memory-after.json').read_text())['memory.events']
 assert all(int(s.split()[1])==0 for s in events.splitlines() if s.split()[0] in ['oom','oom_kill'])
cli=json.loads((root/'cli-cache/generation.json').read_text())
assert cli['text'].strip()=='OK' and cli['engine_exit']==0 and cli['draft_method']=='dflash2'
assert cli['draft_worker']['graphs']==12 and cli['draft_worker']['adaptive'] and cli['draft_worker']['worker_error'] is None
assert cli['weight_vram']['external_draft']==2477529600 and cli['weight_vram']['weights']<=cli['weight_vram']['limit']
log=(root/'cli-cache/engine.stderr.log').read_text()
assert all(int(n)==0 for n in re.findall(r'route tail skip: (\d+) missed experts',log))
running=[]
engines={str(root/x) for x in ['strata','strata-final','strata-bridge-v1']}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
 except OSError:continue
 if a and (a[0] in engines or (a[0].endswith('python3') and any(x in a for x in [str(repo/'tools/strata_dflash2.py'),str(repo/'tools/dflash2_worker.py'),str(root/'benchmark.py')]))):running.append(int(p.name))
assert not running,running
old=Path('/root/private_data/strata-dcu/weight50/strata');assert sha(old)=='07f55bdc54abae0f3a1078068095ae69cfeb61a270870e4c3db7d107b7e3a214'
p1=Path('/root/private_data/strata-dcu/weight50/strata-p1-before-budget');assert sha(p1)=='48f710e8e467aa6544d9f22faad187d7c189be2e0822e2edead2b749d1035ed6'
link=Path('/root/strata-dcu/build-gfx936/strata');before=str(link.resolve());assert before in [str(old),str(engine)]
tmp=link.with_name('strata.dflash2-'+str(os.getpid())+'.link');os.symlink(engine,tmp);os.replace(tmp,link)
manifest=json.loads((root/'checkpoint/download-manifest.json').read_text());assert manifest['files']['model.safetensors']['sha256']=='d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f'
assert (root/'checkpoint/model.safetensors').stat().st_size==1206140008
record={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'engine_sha256':sha(engine),'source_sha256':source,'default_engine_before':before,'default_engine_after':str(link.resolve()),'default_drafter':'adaptive MTP; DFlash2 requires --dflash2','weight50_engine_preserved':str(old),'p1_engine_preserved':str(p1),'download_manifest':manifest,'checkpoint_load_sha256_verified_by_workers':True,'checkpoint_size':1206140008,'cli_passed':True,'remaining_inference_processes':running,'memory_events':Path('/sys/fs/cgroup/memory.events').read_text(),'system_root_usage':subprocess.check_output(['du','-xsh','/root'],text=True).strip(),'limitations':['HIP/gfx936, BS=1, context <=512 only','Only the recorded request set, not a full quality benchmark','GPU device-used snapshots are not peak memory measurements']}
(root/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
print('PASS sources, binary, CLI, budgets, no OOM; selected tested engine with MTP default')
print('engine',record['engine_sha256']);print(record['system_root_usage'])
