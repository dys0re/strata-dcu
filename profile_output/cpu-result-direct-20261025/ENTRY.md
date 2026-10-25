# Independent public-entry validation — corrected logging assertion

Read `/home/dry/.codex/skills/run-experiment/SKILL.md` and the provider ledger
`/home/dry/data/strara-dcu/Strata/.aris/compute/scnet-dflash2.md`.
Runtime is unchanged `dflash2@52f8bc84`; no dependency installs.

Run only after the root confirms deployment and GPU idle. Execute this invocation
exactly once; do not edit configuration, retry, or launch additional inference:

```bash
ssh -S /tmp/strata-direct-20261025-control -o BatchMode=yes -p 11754 root@ssh.zzai.scnet.cn 'ulimit -c 0; python3 -u /root/private_data/strata-dcu/cpu-result-direct-20261025/validate_entry.py'
```

The helper calls the actual public wrapper for four cold starts: enhanced-MTP
machine default, its explicit inverse environment override, native DFlash2 opt-in,
and joint MTP+DFlash2 opt-in. The chosen machine default is recorded in
`deployment.json.mtp_direct_result_default`, not inferred from the request flag.
Each copies eight sentences exactly, exercises suffix lookup, checks the requested
option and actual capture-selection marker, existing model/dispatch defaults,
13 participants and the fixed expert count. It requires 8500 GPU experts for MTP,
8300 for native/joint, all weights <=34351349760B, unchanged selected binary digest
`81f79177f7e9b15a1d4506bed0211db95b369594868f51bcc19e64fa4017ce9b`,
and the deployed persistent symlink target.

Expected exit 0 and:

```text
PASS ENTRY mtp-default
PASS ENTRY mtp-override
PASS ENTRY native-opt-in
PASS ENTRY joint-opt-in
PASS CPU RESULT DIRECT PUBLIC ENTRIES; GPU idle
```

Raw evidence goes to persistent `cpu-result-direct-20261025/entry/`; it must not
already exist before this one-shot invocation. Report exact command, exit code,
observed output, and any documentation/runtime divergence to the root. If it
fails, preserve the first failure and stop rather than correcting or retrying.

The first independent invocation failed in the validation helper: its expected Q4
log string had accidentally been renamed from `shared input bias` to `shared input direct`.
The first public call had already passed its answer and option checks. That failed
helper and transcript are retained as `validate_entry_failed_v1.py` and
`entry-validation-failed-v1.txt`; raw output is retained in `entry-failed-v1/`.
This revision changes only the expected log string, with engine/config unchanged.
The invocation above is the corrected revision's one-shot validation.
