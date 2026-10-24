# SCNet DFlash2 runtime

### env: dflash2@52f8bc84
how: existing system Python; source /opt/dtk/env.sh and /opt/hyhal/env.sh; no pip installs
tier: 15 CPUs, 59 GiB RAM, 1 BW gfx936 with 63.984375 GiB VRAM
weights: persistent target and draft paths from profile_output/dflash2/env-spec.json
validated: 2026-10-22 seeded witness + fresh agent verbatim invocation passed (exit 0, 12 graphs, IDs [553,393,85], incremental cache equal)
gotcha: noninteractive SSH lacks platform proxy; download via authenticated login environment, suppress its startup diagnostics; never log proxy secrets. ModelScope download verified against HF SHA-256.

From /home/dry/data/strara-dcu/Strata run this exact command after ensuring no inference process is running:

```bash
ssh -S /tmp/strata-dcu-control -p 11754 root@ssh.zzai.scnet.cn 'source /opt/dtk/env.sh; source /opt/hyhal/env.sh; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false STRATA_DFLASH_GRAPH=1; python3 -u /root/strata-dcu/Strata/tools/dflash2_worker.py --checkpoint /root/private_data/strata-dcu/dflash2/checkpoint --target /root/private_data/Qwen3.8-F --out /root/private_data/strata-dcu/dflash2/agent-witness'
```

Expected: exit 0, WITNESS DFLASH2, finite_forward=true, repeat_equal=true, incremental_cache_equal=true, two identical [553,393,85] proposals, weight_bytes=2477529600, graphs=12. Shader/attention warnings can appear on this HIP build. This checks actual draft kernels and checkpoint loading, not target-model acceptance or inference speed.

## Native DFlash2 application validation

Runtime spec is unchanged (`dflash2@52f8bc84`); no packages installed or upgraded.
Native HIP application built in persistent `/root/private_data/strata-dcu/p1/build`.
Default engine symlink points to `/root/private_data/strata-dcu/dflash-native/strata-final`.
Binary SHA-256: `2aba1cff1291d1af0479188e50050a7b78ff4b619fd90ab5f0e155e4e8f5c42d`.
Native pack: `/root/private_data/strata-dcu/dflash-native/pack`; default AutoRound cache
has a symlink `dflash2-native` to this persistent pack. Existing target/cache paths
are the same ones used by the earlier validated IPC application.

Verbatim public-entry invocation (run only when the GPU is idle):

```bash
ssh -S /tmp/strata-dcu-control -p 11754 root@ssh.zzai.scnet.cn 'bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --prompt "Reply with exactly OK." --max-new 16'
```

Expected: exit 0, generated text `OK`. Inspect
`/root/private_data/strata-dcu/autoround/generation.json`: `draft_method=dflash2`,
`draft_backend=native`, `draft_graph=true`, `draft_adaptive=true`, `draft_max_t=8`,
`weight_vram.native_draft=1206131200`, `external_draft=0`,
`weights=32783001600`, `weights <= limit=34351349760`. No independent
PyTorch drafter process. Source of invocation: `docs/DFLASH2_NATIVE.md`.

Validated native public entry: 2026-10-22, fresh agent `/root/validate_native_entry`, exit 0, OK, native/graph/adaptive/max_t=8, weight bill matched, no doc divergence.

## Current native entry after DFlash-only cleanup (2026-10-23)

Runtime specification unchanged: dflash2@52f8bc84. No dependency installs.
Tested candidate: /root/private_data/strata-dcu/dflash-native/cleanup-20261023/strata
SHA-256: 745c912b923cec9696517e9a898609c1e4483fb4baca3b411f1f52c0dc8a154a
The build-gfx936/strata entry will point to this candidate before the following public-entry check.
The earlier native section above is a historical validation record.

Verbatim current public-entry check (GPU must be idle):

```bash
ssh -S /tmp/strata-dcu-control -o BatchMode=yes -p 11754 root@ssh.zzai.scnet.cn 'ulimit -c 0; bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --prompt "Reply with exactly OK." --max-new 16'
```

Expected: exit 0, answer OK. In /root/private_data/strata-dcu/autoround/generation.json, draft_method=dflash2, draft_backend=native, draft_graph=true, draft_adaptive=true, draft_max_t=8; weight_vram.native_draft=1206131200, external_draft=0, staging=44236800, weights=32827238400 <= limit=34351349760. No independent Python drafter.

Validated current public entry: 2026-10-23, fresh validator validate_dflash_clean_entry, exit 0, answer OK; native/graph/adaptive/max_t and complete weight bill matched. No doc divergence or remaining inference processes.

## Q4 CPU / native DFlash2 small projections (2026-10-23)

Runtime spec unchanged: dflash2@52f8bc84. No package/dependency changes.
Current selected engine: `/root/private_data/strata-dcu/q4-multitoken-20261023/strata-final`.
SHA-256: `d8e293371f4bc399baf7b1562ef516af475b7870791ecd60f2d8f9bb4a1b9ac5`.
The `/root/strata-dcu/run-autoround.sh` wrapper now defaults CPU Q4 multi-token and DFlash2 small projection options to 1; environment overrides of 0 keep the previous kernels. All model/cache paths and default MTP choice remain as before.

Run exactly (GPU must be idle):

```bash
ssh -S /tmp/strata-dcu-control -o BatchMode=yes -p 11754 root@ssh.zzai.scnet.cn 'ulimit -c 0; bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --prompt "Reply with exactly OK." --max-new 16'
```

Expected: exit 0, answer OK. `/root/private_data/strata-dcu/autoround/generation.json` has `kernel_options.cpu_q4_mt=true`, `kernel_options.dflash_small_gemv=true`, `draft_backend=native`, graph/adaptive true, max_t=8, `native_draft=1206131200`, `staging=44236800`, `weights=32827238400 <= limit=34351349760`. No independent Python drafter. Engine symlink and SHA must match the above.
Source invocation: docs/DCU_KERNEL_OPTIMIZATIONS.md. CPU independent checks + 104 GPU cases + full semantic/code QA, cancellation/recovery, nine eager/graph cache cases, and 2100-token recovery have passed. Fresh documented-entry pass: PASS, 2026-10-23, independent validator validate_kernel_entry. Verbatim invocation exit 0 / OK; all 16 JSON checks, symlink/SHA, process exit and GPU idle checks passed. Evidence: profile_output/q4-multitoken-20261023/entry/.

## MTP + native DFlash2 joint entry (2026-10-23)

Runtime spec unchanged: dflash2@52f8bc84. No package/dependency changes.
Current selected engine: `/root/private_data/strata-dcu/joint-draft-20261023/strata-final`.
SHA-256: `1551c2b1cc1624a29ab9b17f2276599c2a35648c58babc417e2c6ad98947b1b1`.
Default remains enhanced MTP. The new joint path is explicit, documented in `docs/JOINT_DRAFT.md`.
Full semantic/protocol QA, forced-rejection QA, 15 seeded-prefix/cache cases,
9 unchanged standalone cache cases and 2100-token recovery have passed.
Performance does not improve in this test; it remains opt-in.

Verbatim current joint public-entry check (GPU must be idle):

```bash
ssh -S /tmp/strata-dcu-control -o BatchMode=yes -p 11754 root@ssh.zzai.scnet.cn 'ulimit -c 0; bash /root/strata-dcu/run-autoround.sh --dflash2 /root/private_data/strata-dcu/dflash2/checkpoint --dflash-mtp-prefix 2 --prompt "Reply with exactly OK." --max-new 16'
```

Expected: exit 0 and answer OK. In `/root/private_data/strata-dcu/autoround/generation.json`:
`draft_method=mtp+dflash2`, `joint_mtp_prefix=2`, `mtp_mode=adaptive`, `mtp_max_t=5`,
`draft_backend=native`, graph/adaptive true, max_t=8, both kernel options true.
Weight usage: native_draft=1206131200, mtp=1368138752, external_draft=0,
staging=44236800, experts=22947840000, weights=34195377152 <= limit=34351349760.
Inspect `engine.stderr.log` for actual MTP loading and `strata joint draft:` counters.
No independent Python drafter. All engine processes must exit and VRAM return idle.
The previous entries above are historical validations; the last block is current.

Fresh documented-entry validation: PASS, 2026-10-23, validate_joint_entry. Verbatim command exit 0 / OK; all 32 checks passed, no documentation divergence. Both processes exited, GPU returned to 2 MiB / 0% use. Evidence: profile_output/joint-draft-20261023/entry/.

## HC / lookup public entry (2026-10-24)

Runtime spec unchanged: dflash2@52f8bc84. No dependency installs. Container restarted;
GPU total 65520 MiB, CPU quota 15, RAM limit 59 GiB rechecked. Source and candidate survived unchanged.
Current selected engine: `/root/private_data/strata-dcu/lookup-compat-20261023/strata-final`.
SHA256: `1c5b8a268585ab62bfe53a098c114b7e645c8befa91384d58c645fec9a8ff396`.
The public wrapper selects HC staged (override STRATA_HC_SPLIT), keeps enhanced MTP,
and defaults cache exchange/PCIe/lookup off based on measured results.
Explicit suffix lookup now supports enhanced MTP, native DFlash2, joint and IPC.
Strict HC parity, full functional QA for all native combinations, mixed-probability
verification, forced rejection, 2100-token lookup and IPC QA passed. Prior sections are historical.

Fresh validator invocation and expected outputs are in
`profile_output/lookup-compat-20261023/ENTRY.md`; run only when the root agent confirms GPU idle.
Fresh documented-entry validation: PASS, 2026-10-24, validate_options_entry. Verbatim invocation ran once and exited 0; all three entries returned correct copied text, exercised lookup, selected staged HC and met the weight limits. SHA/symlink matched; GPU returned to 2 MiB and no engine/draft processes remained. ENTRY.md now states the full public symlink path after the validator caught an ambiguity. Evidence: profile_output/lookup-compat-20261023/entry/.

## Layer scheduling public entry (2026-10-24)

Runtime spec unchanged: dflash2@52f8bc84; no dependency changes.
Selected engine: `/root/private_data/strata-dcu/layer-scheduling-20261024/v9/strata-final`.
Public symlink: `/root/strata-dcu/build-gfx936/strata`.
SHA256: `a2ac48e5fe51fb18d93db50de4bfdb9fdf1d3682877284b1a7b2d4ff217336c2`.
The machine-specific wrapper opts into GPU resident planning, per-expert CPU pipeline and early result publication
only for enhanced MTP. Each individual environment override still wins. Other draft modes keep these opt-in.
Small DMA staging and reduced layer-end spin remain off by default, based on actual measurements.
GPU experts remain8500 for MTP,8300 for native/joint; all weight bills stay under34351349760B. PLE stays on disk.
Seeded GPU plan cases1601 and CPU numerical/concurrency cases552 per configuration passed. Forced late-wake
old code RC139 versus corrected RC0. Current candidate full functional QA passed for enhancedMTP,
forced two-expert DMA, nativeDFlash2 and forced-rejection joint mode. Raw profile records contain48 CPU/GPU layers per window.

Fresh independent public-entry invocation and expected outputs:
`profile_output/layer-scheduling-20261024/ENTRY.md`.
Fresh documented-entry validation: PASS, 2026-10-24, validate_layer_entry. Verbatim invocation ran exactly once,
exit 0; enhanced MTP, native and joint entry checks all passed. No edits/retries or observed divergences.
Evidence: profile_output/layer-scheduling-20261024/entry-validation.txt and v9/entry/.
Postflight: GPU2MiB, no remaining engine, memory.events oom/oom_kill both0.
