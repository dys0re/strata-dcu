"""Freeze native DFlash deployment provenance; omit large local-only fixtures from Git."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root=Path('/root/private_data/strata-dcu/dflash-native')
source=Path('/root/strata-dcu/Strata')
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

names=['CMakeLists.txt','include/strata/core/dflash_bridge.hpp','include/strata/core/dflash_native.hpp',
       'include/strata/core/verify.hpp','include/strata/prefill/prefill.hpp','src/program/generate.cpp',
       'src/prefill/dflash.cu','src/prefill/dflash_witness.cpp','tools/dflash2_pack.py',
       'tools/autoround_pack.py','tools/bench_dflash2_native.py','tools/test_dflash2_native.py',
       'tools/report_dflash2_native.py','tools/audit_dflash2_native.py']
jobs=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
    except (OSError,ProcessLookupError):
        continue
    if not args:
        continue
    executable=Path(args[0]).name
    is_engine=executable in ['strata','strata-final','dflash_native_witness']
    is_driver=executable.startswith('python') and any(Path(a).name in
        ['bench_dflash2_native.py','strata_dflash2.py','dflash2_worker.py'] for a in args[1:3])
    if is_engine or is_driver:
        jobs.append(dict(pid=int(proc.name),args=args))
assert not jobs,jobs
fixtures=sorted((root/'witness').glob('features-*.f32'))
audit=dict(binary_sha256=sha(root/'strata-final'),
           default_engine=os.path.realpath('/root/strata-dcu/build-gfx936/strata'),
           source_sha256={n:sha(source/n) for n in names},
           pack_weight_sha256=sha(root/'pack/weights.bin'),pack_bytes=(root/'pack/weights.bin').stat().st_size,
           active_jobs=jobs,memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),
           root_layer=subprocess.check_output(['du','-xsh','/root'],text=True).strip(),
           remote_only_fixtures=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in fixtures],
           old_binary_retained=Path('/root/private_data/strata-dcu/dflash2/strata-final').is_file())
(root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ['source_sha256','remote_only_fixtures']},indent=2))
