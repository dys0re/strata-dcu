"""Verify the tested source/cache and atomically select the persistent P1 engine."""
from pathlib import Path
import hashlib,json,os,sys,subprocess,datetime
p1=Path('/root/private_data/strata-dcu/p1');repo=Path('/root/strata-dcu/Strata')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
expected=json.loads((p1/'source-sha256.json').read_text())
actual={}
for name,value in expected.items():
 path=Path('/root/strata-dcu/run-autoround.sh') if name=='strata-test/run-autoround.sh' else repo/name
 actual[name]=sha(path);assert actual[name]==value,name
running=[]
for d in Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:args=[x.decode(errors='replace') for x in (d/'cmdline').read_bytes().split(b'\0') if x]
 except OSError:continue
 if not args:continue
 if args[0] in [str(p1/'build/strata'),str(p1/'strata-baseline-089c5b4'),'/root/strata-dcu/build-gfx936/strata']:
  running.append(int(d.name))
assert not running,running
sys.path.insert(0,str(repo/'tools'))
import autoround_pack as ap
cache=Path('/root/private_data/strata-dcu/autoround');ck=ap.AutoRoundCheckpoint(Path('/root/private_data/Qwen3.8-F'))
assert ck.inventory()==json.loads((cache/'source.json').read_text())
assert ap.check_mtp(ck,cache)==cache/'mtp'
ready=json.loads((cache/'mtp/ready.json').read_text());mtp_hashes={}
for name,info in ready['files'].items():
 mtp_hashes[name]=sha(cache/'mtp'/name)
 assert mtp_hashes[name]==info['sha256'],name
assert mtp_hashes==json.loads((p1/'expected-mtp-sha256.json').read_text())
assert (cache/'pack/experts.bin').stat().st_size==67947724800
assert (cache/'ple.gguf').stat().st_size==318
old=Path('/root/strata-dcu/build-gfx936/strata');new=p1/'build/strata';baseline=p1/'strata-baseline-089c5b4'
baseline_hash=sha(baseline)
assert baseline_hash=='fdd0cbc67dbffd9bce59a180174d7dad5ccb3dd40452ffb8131aed797a8abc24'
new_hash=sha(new);previous=sha(old)
assert previous in [baseline_hash,new_hash]
link=old.with_name('strata.p1-'+str(os.getpid())+'.link')
os.symlink(new,link);os.replace(link,old)
assert old.is_symlink() and old.resolve()==new.resolve() and sha(old)==new_hash
events=Path('/sys/fs/cgroup/memory.events').read_text()
assert all(int(line.split()[1])==0 for line in events.splitlines() if line.split()[0] in ['oom','oom_kill'])
record={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hashes_match':actual,
 'checkpoint_inventory_verified':True,'mtp_cache_validation_passed':True,'mtp_runtime_sha256':mtp_hashes,
 'main_expert_bytes':(cache/'pack/experts.bin').stat().st_size,'ple_descriptor_bytes':318,
 'remaining_inference_processes':running,'engine_sha256':new_hash,'baseline_089c5b4_sha256':baseline_hash,
 'selected_engine_path':str(old),'persistent_engine':str(new),'previous_engine_sha256':previous,
 'system_root_usage':subprocess.check_output(['du','-xsh','/root'],text=True).strip(),
 'memory_events':events,'cache_symlink':str(Path('/root/strata-dcu/autoround').resolve()),
 'limits':['checkpoint inventory checked; no new full SHA-256 scan of all original model or main expert bytes',
 'CUDA/SYCL compilation and other hardware not tested; current execution is gfx936/HIP only']}
(p1/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
print('PASS: source hashes, checkpoint inventory, MTP SHA-256 and cache validation; no running inference; P1 selected atomically')
print('engine',new_hash)
print(record['system_root_usage'])
