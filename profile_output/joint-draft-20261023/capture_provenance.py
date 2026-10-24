"""Capture the real remote build inputs and executable hashes."""
from pathlib import Path
import hashlib
import json
import subprocess

source = Path('/root/strata-dcu/Strata')
build = Path('/root/private_data/strata-dcu/p1/build')
out = Path('/root/private_data/strata-dcu/joint-draft-20261023')
files = ['CMakeLists.txt', 'src/program/generate.cpp', 'src/prefill/dflash.cu',
         'src/prefill/dflash_witness.cpp', 'src/core/mtp.cpp', 'src/core/verify.cpp',
         'include/strata/core/dflash_native.hpp', 'include/strata/core/dflash_bridge.hpp',
         'include/strata/spec/joint_budget.hpp', 'src/spec/joint_budget_test.cpp',
         'src/kernels/cpu/native_expert.cpp', 'src/kernels/cpu/q4_avx512.cpp',
         'include/strata/kernels/cpu/q4_avx512.hpp', 'src/kernels/cuda/bf16_small.cu',
         'include/strata/kernels/bf16_small.hpp', 'tools/autoround_pack.py',
         'tools/bench_dflash2_native.py', 'tools/test_joint_dflash.py',
         'strata-test/run-autoround.sh']
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
binary = out/'strata-final'
assert sha(binary) == sha(out/'strata-prob') == sha(build/'strata')
commands = subprocess.check_output(['ninja','-C',str(build),'-t','commands'], text=True)
selected = [line for line in commands.splitlines()
            if 'generate.cpp.o -MF' in line or 'dflash.cu.o -MF' in line]
assert selected and any('gfx936' in line for line in selected)
(out/'compile-commands.txt').write_text('\n'.join(selected)+'\n')
manifest = dict(engine_sha256=sha(binary), witness_sha256=sha(build/'dflash_native_witness'),
                source_sha256={name:sha(source/name) for name in files},
                entry=str(binary), runtime_spec='dflash2@52f8bc84',
                compiler='/opt/dtk/llvm/bin/clang++', backend='HIP', arch='gfx936')
(out/'build-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest['engine_sha256'])
