"""Replay the old nine-case native witness and compare its actual saved outputs."""
import json
import subprocess
from pathlib import Path

root = Path('/root/private_data/strata-dcu/joint-draft-20261023/legacy-cache')
previous = Path('/root/private_data/strata-dcu/q4-multitoken-20261023/witness')
root.mkdir(exist_ok=False)
names = [line.split()[-1] for line in (previous/'script.txt').read_text().splitlines()
         if line.startswith('draft ')]
for mode in ['eager', 'graph']:
    out = root/mode
    out.mkdir()
    command = ['/root/private_data/strata-dcu/p1/build/dflash_native_witness',
               '/root/private_data/strata-dcu/dflash-native/pack',
               '/root/private_data/strata-dcu/autoround/dense.gguf',
               str(previous/'script.txt'), '4096', mode, str(out)]
    r = subprocess.run(command, capture_output=True, text=True)
    (out/'stdout.log').write_text(r.stdout)
    (out/'stderr.log').write_text(r.stderr)
    assert r.returncode == 0, r.stderr
    for name in names:
        for suffix in ['.ids', '.f32']:
            assert (out/(name+suffix)).read_bytes() == (previous/mode/(name+suffix)).read_bytes(), (mode,name,suffix)
report = dict(cases=len(names), modes=['eager','graph'], byte_equal_previous=True,
              reference=str(previous))
(root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print('PASS LEGACY_NATIVE_BYTE_EQUAL', len(names))
