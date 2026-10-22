# P1 evidence: gfx936 BF16 multi-column reuse

The human report is `docs/P1_BF16_MULTICOL.md`. Base revision: `089c5b4`.

- `a0`, `b0`, `b1`, `a1`: primary ABBA sequence, adaptive MTP in both arms, six requests per workload per arm, two engine starts per arm. `primary-results.json` reports decode TPS; `analyze.py` verifies matched settings and token outputs.
- `baseline-checks`, `final-checks`, `fixed8-checks`: independent numerical/protocol checks. Profiling and first-logit dumps are excluded from the primary comparison. `verify_equivalence.py` checks all saved outputs and first-token arrays.
- `inline-control`, `fusion-qdedup`, `fusion-qfuse`: exploratory one-startup fusion tests. QDEDUP needs the inline control; QFUSE produced non-equivalent output and was not adopted. `fusion-results.json` preserves that failure.
- `micro-cols*` and `micro-threshold8*`: exploratory layouts, including regressions. `micro-tuned-cols*` is the final 96-shape dispatch measurement. Microseconds are kernel time, not model TPS.
- `audit.json`, `source-sha256.json`, `wrapper-smoke.json`: source/cache verification, deployed binary hash, process cleanup and the original launcher's 64-token smoke test.
- `p1-core.patch`: exact implementation patch; applies to an isolated index at `089c5b4`.
- `build.log.gz`: lossless archive of the initial complete build log; uncompressed size and both SHA-256 hashes are recorded in `build-log-archive.json`.
- `artifacts.tsv`: complete evidence file sizes and SHA-256, excluding the manifest itself and Python bytecode caches. No model weights or build binaries are included.

All model runs are single-request, 512-context, INT8 KV on the documented BW/gfx936 machine. Other hardware/backends and full model-quality evaluations were not run. Original model/main expert files were inventory/size checked, not rehashed in full; the four MTP runtime files were fully hashed.
