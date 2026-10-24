"""Generate tables from the measured verifier lifecycle, not estimated timings."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
s=json.loads((P/'analysis.json').read_text())
MODES={'mtp-profile':'增强 MTP','native-profile':'DFlash2','joint-profile':'MTP + DFlash2'}
TASKS={'story-0':'英文','chinese-0':'中文','code-0':'代码'}
def request(mode,task):return next(r for r in s[mode]['requests'] if r['case']==task)
def projection(g):return sum(g.get(k,0) for k in ['hc0','hc1','qkv_or_query','ab','z','kv_index','kv_norm_rope','query','out_proj','head'])
lines=['# 当前验证流程生命周期（2026-10-23）','',
'以当前生产二进制 `1551c2b1…` 为基线，保留 Q4_0 CPU 多 token 和 gfx936 GPU 投影优化。诊断二进制 `7060a916…` 仅增加计时，完整哈希及构建输入见 `build-provenance.json`。没有拿优化前的旧 trace 充当当前结果。','',
'BW/gfx936、BS=1、12 CPU workers、512 上下文、INT8 KV、PLE direct IO。增强 MTP 8500 GPU 专家，独立 DFlash2/联合模式 8300，所有 GPU 模型权重 ≤50%。各启动先预热 128 tokens，再测英文/中文/代码各 256 tokens。','',
'## 结论','',
'验证主链的两个大头仍是 CPU 专家结果依赖，以及 GPU 投影/HC 读/输出头。CPU 生成计划的计算时间很小；“GPU 等计划及复制计划”不能等同于 CPU 规划算法的耗时。联合模式还有候选窗口扩大、实际提交增长不足的损失。','',
'每层的实际流程是：GPU 读残差并做 mixer/attention → 路由及发布激活/专家 ID → CPU 生成计划并计算 RAM 专家，同时 GPU 计算驻留专家 → GPU 等 CPU 结果、复制与合并 → 下一层。48 层后执行输出头；接受判断/提交和下一轮草稿在这条主模型验证链外。','',
'## 增强 MTP 主流分段','',
'单位均为 ms/实际输出 token。这是诊断主流跨度，不是单 token 独立推理时间。投影组包含相邻量化/归一化处理，不是孤立 GEMM 内核时间。','',
'| GPU 主流阶段 | 英文 | 中文 | 代码 |','| --- | ---: | ---: | ---: |']
rows=[('等 CPU 专家结果',lambda g:g['wait_cpu']),('等计划及计划复制',lambda g:g['wait_plan']),('投影、HC 读及输出头',projection),('GPU 驻留专家',lambda g:g['gpu_experts']),('结果复制/合并/侧流汇合',lambda g:g['copy_combine_join']),('其余主流阶段',lambda g:g['gpu_span']-g['wait_cpu']-g['wait_plan']-projection(g)-g['gpu_experts']-g['copy_combine_join']),('主流总跨度',lambda g:g['gpu_span'])]
for label,fn in rows:lines.append('| '+label+' | '+' | '.join(f'{fn(request("mtp-profile",t)["gpu_ms_per_output"]):.2f}' for t in TASKS)+' |')
lines+=['','计划与 CPU 结果等待合计占三类诊断主流约 42% / 31% / 37%。CPU 计算与 GPU 阶段有重叠，不能把下面的 CPU 时间再次加到上表。共享专家侧流也单独计数，不加到主流总跨度。','',
'## CPU 分支','',
'| CPU 阶段（ms/输出 token） | 英文 | 中文 | 代码 |','| --- | ---: | ---: | ---: |']
for key,label in [('plan_ms','计划计算'),('actq_ms','激活量化'),('jobs_ms','任务分配'),('compute_ms','专家池执行（含内部同步/隐藏量化）'),('callback_ms','完整层回调')]:
 lines.append('| '+label+' | '+' | '.join(f'{request("mtp-profile",t)["cpu_per_output"][key]:.3f}' for t in TASKS)+' |')
lines+=['','计时口径补充（源码复核）：`compute_ms` 来自 `expert_source.cpp` 的 `ms_run`，包围整个 `run_split_multi_native`，包含 gate/up、down、线程池阶段栅栏及主线程隐藏激活量化；它不是纯 GEMV 时间。表内 `actq_ms` 是进入专家池前的输入量化，不包含上述隐藏激活量化。当前 trace 尚未拆分这些内部成本，因此也不能单靠该列判定内存带宽已经饱和。','',
'英文 CPU 计划计算约 0.16 ms/token，而 GPU “等计划及复制”约 3.75 ms/token。后者还包含同步、计划读取和内核执行间隔；本轮未将等待与复制拆成两个独立 GPU 区间，不能把差值直接判定为纯握手延迟。',
'','以 distinct CPU experts × 每专家 2,764,800 字节 / 专家计算时间折算，三类有效权重处理率约 56.0/50.6/46.5 GB/s。这不是 PMU 测得的 DRAM 总线带宽。',
'','热态专家文件读取及 PCIe 专家分流计数均为 0。CPU 与 GPU 仍交换激活、ID、计划和结果。trace 中 PCIe 分支存在少量跨度，是零工作量分支及相邻节点开销，不能解释为专家权重传输。PLE 的主机等待另计；当前正常 MTP 的该项约 0.18–0.37 ms/window。','',
'## 三种草稿路径下的验证成本','',
'以下都是本轮诊断数据；没有将带插桩的 TPS 当作新的生产测速结果。每条路径只启动一次。','',
'| 模式 | 请求 | 平均验证宽度 T | 输出/窗口 | 验证位置/输出 | GPU 主流 ms/输出 | CPU 专家池 ms/输出 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for mode,label in MODES.items():
 for task,tl in TASKS.items():
  r=request(mode,task);positions=sum(int(t)*v['windows'] for t,v in r['by_T'].items())
  lines.append(f'| {label} | {tl} | {r["mean_T"]:.2f} | {r["tokens_per_window"]:.2f} | {positions/r["tokens"]:.3f} | {r["gpu_ms_per_output"]["gpu_span"]:.2f} | {r["cpu_per_output"]["compute_ms"]:.2f} |')
lines+=['','代码任务里，DFlash2 单独平均 T=4.15、产出 3.56；联合平均 T=5.13、产出 3.41。联合主模型 GPU 验证成本由 23.19 增到 30.80 ms/输出 token，约 +33%；这是草稿计算之外的额外成本。本轮现象支持“验更多位置而提交不足”的判断，不是只把两种草稿时间相加就能解释。','',
'## 窗口宽度与层热点','',
'增强 MTP 的代码请求，按实际运行窗口分组。不同宽度可能对应不同上下文/路由，不是固定输入的因果消融。','',
'| T | 窗口数 | 输出/窗口 | GPU 主流 ms/窗口 | CPU 专家池 ms/窗口 | CPU entries/distinct expert |','| ---: | ---: | ---: | ---: | ---: | ---: |']
for t,x in sorted(request('mtp-profile','code-0')['by_T'].items(),key=lambda p:int(p[0])):
 c=x['cpu_per_window'];lines.append(f'| {t} | {x["windows"]} | {x["tokens_per_window"]:.2f} | {x["gpu_ms_per_window"]["gpu_span"]:.2f} | {c["compute_ms"]:.2f} | {c["cpu_entries"]/max(1,c["cpu_experts"]):.2f} |')
lines+=['','英文中 CPU 结果等待最高的三层（从 0 编号）为：','',
'| 层 | GPU 等 CPU ms/窗口 | CPU 专家数/窗口 | CPU 专家池 ms/窗口 |','| ---: | ---: | ---: | ---: |']
hot=sorted(request('mtp-profile','story-0')['layers'].items(),key=lambda p:-p[1]['gpu_ms_per_window'].get('wait_cpu',0))[:3]
for l,x in hot:lines.append(f'| {l} | {x["gpu_ms_per_window"]["wait_cpu"]:.3f} | {x["cpu_per_window"]["cpu_experts"]:.2f} | {x["cpu_per_window"]["compute_ms"]:.3f} |')
lines+=['','完整逐层、GDN/QSA、按 T 以及按窗口结果见 `analysis.json` 和各运行目录的 `windows.json`。','',
'## 观察开销及 QA','',
'| 请求 | 正常 TPS | 诊断 TPS | 正常平均 T / 输出每窗口 | 诊断平均 T / 输出每窗口 |','| --- | ---: | ---: | --- | --- |']
for task,label in TASKS.items():
 a=request('mtp-normal',task);b=request('mtp-profile',task)
 lines.append(f'| {label} | {a["decode_tps"]:.2f} | {b["decode_tps"]:.2f} | {a["mean_T"]:.2f} / {a["tokens_per_window"]:.2f} | {b["mean_T"]:.2f} / {b["tokens_per_window"]:.2f} |')
lines+=['','时间戳节点有开销，自适应控制器也会因测得的成本不同而改变窗口；代码诊断甚至比这次正常运行快，因此上表差异不能全部称为纯插桩开销，也不能直接把诊断比例当成生产比例。这里保留生产融合/跨流选择，未使用会改变这些路径的 STRATA_VERIFY_PROFILE。','',
'三种诊断均通过 OK/4/北京、采样、输出/上下文边界及 STOP/恢复检查。每个有请求时间边界的 decode 窗口，均核对了 48 层+head、CPU 层回调一一对应、GPU 主流各分段精确加回总跨度，以及实际输出计数。没有要求长文本逐 token 相同。独立会话按 ENTRY.md 原样跑 MTP 诊断，31/31 检查通过。','',
'## 后续工程优先级','',
'1. 缩短 CPU 未命中专家计算与结果交接的关键路径。这部分确实让 GPU 停住；CPU 计划算法本身不是主要耗时。',
'2. 继续降低 GPU BF16 投影/HC 读的每输出 token 成本。它在中文中比 CPU 结果等待更大，不能只优化 CPU。',
'3. 把计划等待与计划复制分开测量，再决定是否优化映射内存交接或合并节点。当前 3.75 ms/token 是二者合计，不能预先承诺全部可消除。',
'4. 联合草稿的扩大窗口只有在实际提交增长足够时才值得；本轮代码已经观察到多验少出的成本。','',
'## 插桩与产物清单','',
'| 文件 | 操作 | 内容 |','| --- | --- | --- |',
'| src/core/verify.cpp | 仅远程诊断构建临时修改，已恢复 | 保留融合/跨流的逐层 GPU/CPU 生命周期，补丁见 instrumentation.patch |',
'| src/program/generate.cpp | 仅远程诊断构建临时修改，已恢复 | 每层 CPU 计划/量化/分配/计算增量及专家计数 |',
'| instrumentation.patch | 新增 | 相对当前生产源码的全部诊断差异 |',
'| run.py、ENTRY.md | 新增 | 当前 benchmark 驱动的时间边界采集、显式诊断环境变量、原样复跑命令 |',
'| analyze.py、report.py | 新增 | 时间分区与 CPU 行匹配、按层/T/实际输出汇总、生成此报告 |',
'| */lifecycle.csv.gz、*/cpu.csv.gz | 新增 | 无损压缩的原始 trace；远程持久盘保留 CSV 原文件 |',
'| */results.json、*/windows.json、analysis.json、entry/、audit.json | 新增 | 请求、逐窗、汇总、独立复验和资源/哈希审计 |','',
'诊断文件与二进制位于 `/root/private_data/strata-dcu/verify-lifecycle-20261023/`。正式入口和共享 build/strata 均恢复为生产二进制 `1551c2b1…`，两个临时修改的源码 SHA 与本地原文件一致。所有推理进程已退出，GPU 回到 2 MiB，OOM 计数为 0。','',
'复算：`python3 profile_output/verify-lifecycle-20261023/analyze.py`，再运行同目录 `report.py`。无需重新启动模型。']
(P/'README.md').write_text('\n'.join(lines)+'\n')
print('WROTE verifier lifecycle report')
