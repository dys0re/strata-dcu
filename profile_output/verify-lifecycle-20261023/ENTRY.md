# Current verifier lifecycle diagnostic entry

Runtime specification is unchanged: dflash2@52f8bc84, existing DTK/HYHAL/system Python, no package installs. The production engine and its symlink remain unchanged. This diagnostic binary includes the current CPU Q4 multi-token and GPU small-projection optimizations plus the patch saved in this directory. Actual build input hashes are in build-provenance.json; driver provenance records the restored production source tree.

Run this exact invocation once, while the GPU is idle, from the local repository:

```bash
ssh -S /tmp/strata-dcu-control -o BatchMode=yes -p 11754 root@ssh.zzai.scnet.cn 'ulimit -c 0; source /opt/dtk/env.sh >/dev/null 2>&1; source /opt/hyhal/env.sh >/dev/null 2>&1; STRATA_CPU_Q4_MT=1 STRATA_DFLASH_SMALL_GEMV=1 python3 -u /root/private_data/strata-dcu/verify-lifecycle-20261023/run.py --engine /root/private_data/strata-dcu/verify-lifecycle-20261023/strata-profile --out /root/private_data/strata-dcu/verify-lifecycle-20261023/mtp-profile --mode adaptive --width 5 --repeat 1 --timing-only --lifecycle --checks'
```

Expected: exit 0, final PASS, 11 request records including warmup, three 256-token workloads, OK/4/北京 checks, two sampled requests and output/context limits; cancellation/recovery pass in protocol.json. All GPU weights <=50%, 8500 GPU experts, CPU Q4 and DFlash small projection flags are 1. lifecycle.csv has 43 integer columns per row: five window/layer identifiers, 33 main GPU clocks, two overlapping shared-stream clocks, three host timestamps. Every decode window has 49 rows (48 layers plus output head). cpu.csv has named columns for per-layer planning, activation quantization, job construction, expert compute, and routing counts. Timestamps are GPU nanoseconds from the existing gfx936 wall-clock conversion and host steady-clock nanoseconds; do not subtract GPU clocks from host clocks. The two timelines overlap and must not be added together.

This command profiles the real verifier, not a synthetic kernel. Preserve the default fusion/shared-stream choices. Do not set STRATA_VERIFY_PROFILE, which has a different execution path. Record any divergence without fixing it during the verbatim check. Evidence goes under this directory's entry/ folder. The command runs one instance and exits it; postflight GPU should return idle.
