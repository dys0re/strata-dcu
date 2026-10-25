"""Run the standalone probe after the real engine has exited; persist every result."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

R = Path('/root/private_data/strata-dcu/cpu-gpu-handoff-20261025')
assert 'Used Memory (MiB): 2' in subprocess.check_output(
    ['/opt/hyhal/bin/hy-smi', '--showmeminfo', 'vram'], text=True)

def cpus():
    data = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if re.fullmatch('cpu[0-9]+', fields[0]):
            values = list(map(int, fields[1:9]))
            data[int(fields[0][3:])] = (sum(values), values[3] + values[4])
    return data

a = cpus()
time.sleep(.5)
b = cpus()
allowed = os.sched_getaffinity(0)
busy = {c: 1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]) for c in allowed}
core = min(allowed, key=lambda c: (busy[c], c))
os.sched_setaffinity(0, {core})
meta = dict(core=core, busy=busy[core], command=[str(R / 'handoff-probe')],
            sha256=hashlib.sha256((R / 'handoff-probe').read_bytes()).hexdigest(),
            flags={k: os.getenv(k) for k in ['STRATA_FUSE_WAIT', 'STRATA_FUSE_HITS']})
(R / 'probe-command.json').write_text(json.dumps(meta, indent=2) + '\n')
with (R / 'probe-results.jsonl').open('w') as out, (R / 'probe-stderr.log').open('w') as err:
    p = subprocess.Popen(meta['command'], stdout=out, stderr=err)
    (R / 'active-probe.json').write_text(json.dumps(dict(pid=p.pid, driver_pid=os.getpid())))
    p.wait()
    (R / 'probe-status.json').write_text(json.dumps(dict(returncode=p.returncode)))
    print('PROBE_RC', p.returncode, flush=True)
    raise SystemExit(p.returncode)
