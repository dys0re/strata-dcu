# 启动时独占CPU的权限检查

2026-10-25，当前实例容器内不能为推理保留宿主独占核心。
原始检查结果见 permissions.json。

- UID0，但CAP_SYS_NICE与CAP_SYS_ADMIN均未授予。
- cgroup2挂载在容器内为ro；cpuset.cpus及cpuset.cpus.partition不可写，后者为member。
- cpu.max仍1500000/100000，可运行核心0–127；这是总时间配额与可用掩码，没有独占分区。
- RLIMIT_RTPRIO、RLIMIT_NICE的软硬限制都为0。
- 仅在一个短命自有子进程尝试nice(-1)，实际返回EPERM；未改推理优先级或平台配置。

真正的CPU独占须由宿主/平台分配排他的CPU集合。单纯sched_setaffinity只约束自己的线程在哪里运行，
不能阻止其他租户线程使用同一核心。提前启动占位线程也不能建立独占所有权。
容器内可另行做启动时负载选核、运行中检查调度等待并换核；它没有宿主独占保证，本轮未将其冒充独占实现。

核对内核官方说明：https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#cpuset
所有动作均为权限/配置读取及自有子进程权限探测。未启动推理任务，未修改引擎、权重或生产入口。
