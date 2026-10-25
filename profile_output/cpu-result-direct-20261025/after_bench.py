"""Wait for the existing benchmark driver, then run the remaining gates once."""
from pathlib import Path
import json
import os
import resource
import subprocess
import time
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
R = Path('/root/private_data/strata-dcu/cpu-result-direct-20261025')
pid = json.loads((R / 'active.json').read_text())['driver_pid']
while Path(f'/proc/{pid}').exists():
    time.sleep(3)
bench = json.loads((R / 'bench-status.json').read_text())
assert len(bench) == 4 and all(s['returncode'] == 0 for s in bench)
status = []
for name in ['finish_gpu_test', 'run_modes_qa', 'profile_current', 'analyze_bench']:
    print('START', name, flush=True)
    with (R / (name + '-driver.log')).open('w') as log:
        p = subprocess.Popen(['python3', '-u', str(R / (name + '.py'))], stdout=log, stderr=subprocess.STDOUT)
        (R / 'after-bench-active.json').write_text(json.dumps(dict(name=name, pid=p.pid, driver_pid=os.getpid())))
        rc = p.wait()
    status.append(dict(name=name, returncode=rc))
    (R / 'after-bench-status.json').write_text(json.dumps(status, indent=2) + '\n')
    print('END', name, rc, flush=True)
    assert rc == 0, name
print('AFTER_BENCH_COMPLETE', flush=True)
