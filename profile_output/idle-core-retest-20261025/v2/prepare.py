"""Generate a bench copy with selection before the first request; engine remains unchanged."""
from pathlib import Path
import ast,hashlib,json
R=Path('/root/private_data/strata-dcu/idle-core-retest-20261025/v2');source=Path('/root/strata-dcu/Strata/tools/bench_dflash2_native.py');s=source.read_text()
s='import sys\nsys.path.insert(0,"/root/strata-dcu/Strata/tools")\n'+s
needle="   for name,prompt,n,expected,temperature in cases:" # actual loop indentation is two spaces
needle='  for name,prompt,n,expected,temperature in cases:'
assert s.count(needle)==1
s=s.replace(needle,"""  # [PROFILE] Select once after load, before warmup. No active selection during decode.
  from idle_cores import choose_and_pin
  selected=choose_and_pin(p.pid,os.sched_getaffinity(0),13,out/'idle-core-selection.json')
  for name,prompt,n,expected,temperature in cases:""")
ast.parse(s);(R/'idle_bench.py').write_text(s)
(R/'driver-source.json').write_text(json.dumps(dict(base_driver_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),generated_driver_sha256=hashlib.sha256(s.encode()).hexdigest()),indent=2)+'\n')
