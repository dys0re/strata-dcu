"""Record source/binary provenance and select the validated <=50% runtime."""
from pathlib import Path
import hashlib,json,os,datetime,subprocess
root=Path('/root/private_data/strata-dcu/weight50');repo=Path('/root/strata-dcu/Strata')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
paths=['CMakeLists.txt','src/program/generate.cpp','src/core/mtp.cpp','include/strata/core/mtp.hpp','src/kernels/cuda/native_mmvq.cu','include/strata/kernels/native_mmvq.hpp','tools/autoround_pack.py','strata-test/run-autoround.sh']
source={name:sha(repo/name) for name in paths}
assert json.loads((root/'over-budget-result.json').read_text())['rejected_before_generation']
for run in ['run1','run2']:
 usage=json.loads((root/run/'weight-usage.json').read_text());assert usage['weights']<=usage['limit']==usage['total']//2
 assert 'PASS' in (root/(run+'.log')).read_text()
running=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:argv=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\0') if s]
 except OSError:continue
 if argv and argv[0] in [str(root/'strata'),'/root/strata-dcu/build-gfx936/strata']:
  running.append(int(p.name))
assert not running,running
# Keep the historical P1 binary unchanged and move only the deployment symlink.
old=Path('/root/private_data/strata-dcu/p1/build/strata')
assert sha(old)==sha(root/'strata-p1-before-budget')=='48f710e8e467aa6544d9f22faad187d7c189be2e0822e2edead2b749d1035ed6'
link=Path('/root/strata-dcu/build-gfx936/strata');tmp=link.with_name('strata.weight50-'+str(os.getpid())+'.link')
os.symlink(root/'strata',tmp);os.replace(tmp,link)
record={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'engine_sha256':sha(root/'strata'),'historical_p1_sha256':sha(old),'source_sha256':source,'inference_processes':running,'selected_engine':str(link.resolve()),'memory_events':Path('/sys/fs/cgroup/memory.events').read_text(),'system_root_usage':subprocess.check_output(['du','-xsh','/root'],text=True).strip(),'limitations':['Only HIP/gfx936 built and executed; CUDA/SYCL not built','Device-used snapshot is at initialization, not a peak GPU memory measurement','This is BS=1 short-context decode speed, not a full model quality evaluation']}
(root/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
