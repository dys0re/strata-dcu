"""Write the human-readable report from verified raw measurements."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text());u=s['weight_usage'][0]
labels={'pack':'主模型 pack','dense':'主模型 dense 投影','head':'主输出头','experts':'8,500 个主模型路由专家','mtp':'MTP（含 512 个专家、dense 和独立输出头）'}
rows='\n'.join(f'| {labels[k]} | {u[k]:,} | {u[k]/2**30:.6f} |' for k in labels)
names={'story':'英文故事','chinese':'中文解释','code':'Python LRU 代码与说明'}
rates='\n'.join(f'| {names[k]} | {v["median_tps"]:.2f} | {v["min_tps"]:.2f}–{v["max_tps"]:.2f} |' for k,v in s['workloads'].items())
text=f'''# GPU 权重不超过 50%：速度重测

2026-10-22，远程 BW/gfx936、Qwen3.8-F AutoRound。执行[当前测试标准](../../docs/DCU_TEST_STANDARD.md)。固定 8,500 个 GPU 主模型专家，其余 16,076 个放 RAM（41.394424 GiB），没有进行专家数扫描。

## 权重账单

| GPU 权重 | 字节 | GiB |
| --- | ---: | ---: |
{rows}
| 合计 | {u['weights']:,} | {u['weights']/2**30:.6f} |

物理总显存 {u['total']:,} 字节（{u['total']/2**30:.6f} GiB），上限 {u['limit']:,} 字节（{u['limit']/2**30:.6f} GiB）。实际模型权重占 **{100*u['weights']/u['total']:.4f}%**。两次启动账单相同。

主机映射 embedding 使用 RAM，PLE 保留在磁盘。KV、状态、激活、索引、临时工作区和驱动开销不混入模型权重。初始化时设备总占用 {u['device_used']/2**30:.6f} GiB；扣除权重的余量为 {(u['device_used']-u['weights'])/2**30:.6f} GiB，包含分配器及驱动开销，不是精确 KV 用量，也不是峰值测量。

## Decode 速度

| 工作负载 | 中位数 token/s | 六次范围 token/s |
| --- | ---: | ---: |
{rates}

两次独立引擎启动，每次先预热 128 tokens，再按英文/中文/代码顺序重复三轮，每个长请求实际生成 256 tokens。BS=1、512 上下文上限、INT8 KV、12 个 CPU workers、PLE direct IO、关闭 prompt/conversation cache。**增强 MTP adaptive 开启**，验证上限 T=5，允许在线选择较短窗口；VEC4=1、COLS=4，QFUSE/QDEDUP 关闭。完整开关见各次 command.json。

Decode TPS = 实际生成 token 数 / 引擎 DONE 的 decode_ms，不包含加载及 prefill。每类六次输出 token IDs 一致。STRATA_TRACE=1 启动/窗口诊断也记录在命令中。历史 19,000/自动满显存专家的 P1 数据不符合新预算，不与本轮混用，也不据此声称算子收益变化。

## 校验

- 三个短问答 OK / 4 / 北京正确；固定 seed 重复、单 token、501+3 合法上下文边界通过。
- STOP 取消及下一请求恢复通过。两轮没有 OOM 或被拒绝的内核启动。
- 旧的 19,000 专家配置在 50% 预算下退出 2，在开始生成及分配 RAM 专家补集前拒绝。
- 仅在此 HIP/gfx936 硬件编译、运行；未构建 CUDA/SYCL，未做长上下文或全面质量评测。

## 文件与实现

- `src/program/generate.cpp`：可选 STRATA_MAX_WEIGHT_VRAM_FRACTION；先约束缓存预算，绑定 MTP 后检查实际 GPU 权重账单，超限拒绝。
- `include/strata/core/mtp.hpp`、`src/core/mtp.cpp`：单独记录 MTP 所有拥有的权重分配，不把 KV/激活混入权重；额外 Q4 副本分别计入。
- `tools/autoround_pack.py`、`strata-test/run-autoround.sh`：本机入口固定 50% 预算，默认 8,500 专家及增强 MTP；输出 generation.json 带权重账单，旧二进制缺少账单时拒绝发布有效结果。
- 本目录 `benchmark.py`：请求前核验预算，记录提示、token IDs、DONE、内存事件。`analyze.py` / `summary.json`：核验并汇总；`check_budget.py` / `over-budget-result.json`：超限拒绝；`audit_remote.py` / `audit.json`：来源、二进制 SHA-256 及退出进程检查。

`weight-budget.patch` 是在 P1 源码之上的增量；复现先使用 ../p1/p1-core.patch，再应用它。远程构建产物及原 P1 二进制副本位于持久盘 `/root/private_data/strata-dcu/weight50/`。正式入口选用该目录的 strata，原 P1 二进制保留；大文件不写系统盘。
'''
(root/'README.md').write_text(text)
