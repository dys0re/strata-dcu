# Fresh public-entry validation

Read the run-experiment skill and `.aris/compute/scnet-dflash2.md`.
Run only the invocation below and the read-only checks described here. No other experiments or notifications.
The root agent will select the candidate and tell the validator when the GPU is idle.

From `/home/dry/data/strara-dcu/Strata`, run verbatim:

```bash
ssh -S /tmp/strata-dcu-control -o BatchMode=yes -p 11754 root@ssh.zzai.scnet.cn 'ulimit -c 0; python3 -u /root/private_data/strata-dcu/lookup-compat-20261023/validate_entry.py'
```

The script runs the public `run-autoround.sh` entry three times: enhanced MTP,
native DFlash2, and joint MTP/DFlash2. Each explicitly enables suffix lookup and
copies a repeated sentence. This also exercises the separate non-serve MTP path.
It saves exact invocation arguments and outputs, and verifies the answer,
lookup use, HC staged selection, kernel options, expert options and full weight bill.

Expected: exit 0, `PASS ENTRY mtp`, `PASS ENTRY native`, `PASS ENTRY joint`,
then `PASS ALL PUBLIC ENTRIES`. Inspect
`/root/private_data/strata-dcu/lookup-compat-20261023/entry/checks.json`:
all three modes have answer_correct=true, lookup_exercised=true, staged=true.
Weights are respectively 33542205952, 32827238400, 34195377152 bytes, each at most
34351349760. GPU expert bytes are 23500800000 for MTP and 22947840000 for the others.

The public symlink `/root/strata-dcu/build-gfx936/strata` must resolve to
`/root/private_data/strata-dcu/lookup-compat-20261023/strata-final`, SHA256
`1c5b8a268585ab62bfe53a098c114b7e645c8befa91384d58c645fec9a8ff396`.
Afterward, check no engine/draft process remains and GPU memory returns to idle
(around 2 MiB; use `/opt/hyhal/bin/hy-smi --showmeminfo vram` after sourcing DTK/HyHAL).

Copy the entry results to the local sibling `entry/` directory and write a short
`validation.txt` listing checked facts and any doc-vs-reality divergence.
Do not silently fix a failed command, modify sources, or restart tests under a new command.
Report the failure and exact observed output to the root agent.
