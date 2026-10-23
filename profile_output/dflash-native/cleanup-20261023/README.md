# DFlash2-only cleanup validation — 2026-10-23

Relative to `80ea84d`, the retained source changes implement native DFlash2, its CLI and QA, and complete weight-budget accounting. Uncommitted expert predictors, training/TTT, CPU/GPU miss splitting, lifecycle hooks and their experimental artifacts were removed. Existing expert scheduling and MTP sources are restored to the baseline.

Passed on BW/gfx936: full HIP engine/witness build; 14 semantic/protocol QA requests (198 output tokens); a 2100-token prompt followed by request recovery; four enhanced-MTP smoke requests; nine native draft cases in both eager and graph modes covering repeat, reset, incremental recomputation and ring wrap. The public CLI was independently executed verbatim and returned OK. Raw commands, source/binary hashes, answers and checks are in the accompanying JSON/log files.

Native GPU weights, including 44,236,800 staging bytes, total 32,827,238,400 bytes (47.781584% of physical VRAM). The tested binary SHA-256 is `745c912b923cec9696517e9a898609c1e4483fb4baca3b411f1f52c0dc8a154a`. It is the selected remote entry.

The new CUDA translation unit also compiled with CUDA 13.3, g++-15 and sm_80. Full CUDA runtime, SYCL and Metal were not tested. PyTorch/native draft differences remain diagnostic; no cross-backend numerical parity or new performance gain is claimed.

All tests use existing persistent model data; core dumps are disabled.
