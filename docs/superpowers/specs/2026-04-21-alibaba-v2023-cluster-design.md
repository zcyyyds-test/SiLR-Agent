# Design Spec: Alibaba v2023 GPU Cluster Domain

- **Date**: 2026-04-21
- **Branch**: `feat/cluster-v2023`
- **Author**: Chenyu Zhou
- **Status**: Draft — awaiting user review

## 1. 目标与动机

把 Alibaba cluster-trace-GPU-v2023 (OpenB, FGD ATC'23) 作为 SiLR 的新 domain `domains/cluster_v2023/` 接入，以**中国就业市场信号最强的公开 GPU 集群 trace** 替代现有 `domains/cluster/` 的 15 节点合成场景作为面试和简历主叙事（现有 cluster domain 保留不动）。

**目标公司岗位对标**：阿里云 ATH 容器 AI Infra、字节火山引擎、美团 LongCat Agent RL、华为诺亚 MindScale。

**为什么选 v2023**：
- 阿里 PAI 自家发布 + ATC'23 FGD 论文背书
- 字段（`qos / gpu_spec / scheduled_time / pod_phase`）一对一对应 Kubernetes 生产调度栈语义
- 配套 `hkust-adsl/kubernetes-scheduler-simulator` 有 Best-fit / FGD / GPU Packing 现成 baseline

**非目标**：
- 不刷 scheduler benchmark leaderboard
- 不替代/修改现有 `domains/cluster/`
- 不做真实 Kubernetes 集群部署（simulator-only）

## 2. 核心设计决策（已与用户对齐）

| # | 决策 | 选项 | 最终 | 理由 |
|---|---|---|---|---|
| D1 | 与现有 cluster domain 的关系 | 新建 sibling / 替换 / 双模 | **新建 sibling** | 保护现有 94% 战果；"两个异构集群 setup 都 work" 比"unified manager" 叙事更强 |
| D2 | 规模 | 小 / 中 / 分阶段 | **小（40 节点 / ~400 job / 25 scenario）** | 2 周硬预算，中规模 GRPO 跑不完；30-50 节点足以支撑 real-workload 叙事 |
| D3 | 第 5 个 checker | 合成 rack / Fragmentation / 只 4 个 | **Fragmentation（FGD 同款）** | 和 v2023 论文出处一致；简历叙事升级为"对标 FGD ATC'23 baseline" |
| D4 | 场景形态 | Recovery / Scheduling / Hybrid | **Recovery-style** | 与其他 3 个 domain 叙事对齐；GRPO rollout 成本可控 |
| D5 | SFT 数据源 | Teacher / Rule-based / 混合 | **Rule-based Best-fit expert + GPT-5.4 补 CoT** | Gemini 对 v2023 字段无先验；FGD-style expert 叙事专业度高 |

## 3. 架构

### 3.1 目录结构

```
domains/cluster_v2023/
├── __init__.py
├── README.md                    # 含 v2023 + FGD ATC'23 引用
├── config.py                    # build_cluster_v2023_domain_config(...)
├── manager.py                   # ClusterV2023Manager(BaseSystemManager)
├── observation.py
├── checkers.py                  # 4 直接映射 + FragmentationChecker
├── tools.py                     # assign_job / preempt_job / migrate_job
├── failsafe.py
├── expert.py                    # Best-fit scheduler（SFT 数据源 + baseline）
├── prompts/
│   ├── system_prompt.py
│   └── tool_schemas.py
├── scenarios/
│   ├── loader.py
│   └── data/                    # 25 个 scenario JSON
└── data_pipeline/
    ├── download.py              # Trace 下载（含 mirror URL fallback）
    ├── subsample.py             # 1523 → 40 节点；8152 → ~400 job
    ├── inject_faults.py         # Philly ATC'19 风格故障注入
    └── raw/                     # .gitignored，原始 CSV
```

### 3.2 接口契约

- `ClusterV2023Manager` 继承 `silr.core.interfaces.BaseSystemManager`
- 必须实现：`sim_time` / `base_mva` / `system_state` / `solve()` / `create_shadow_copy()`
- `system_state` schema：`{"nodes": {...}, "jobs": {...}, "assignments": {...}, "sim_time": float, "meta": {...}}`
- 字段和现有 `domains/cluster/` 对齐但扩展：node 增 `model`、job 增 `qos` / `gpu_spec_required`

### 3.3 与 SiLR 框架的耦合

零修改。`SiLRVerifier`、`ReActAgent`、`CoordinatorAgent`、`train_sft.py`、`train_grpo.py` 全部通过现有 `BaseSystemManager` / `BaseConstraintChecker` / `BaseTool` 接口自动兼容。

## 4. 数据 Pipeline

### 4.1 下载（`data_pipeline/download.py`）

从 `github.com/alibaba/clusterdata` 拉两个文件：
- `openb_node_list_gpu_node.csv`（1523 节点元数据）
- `openb_pod_list_default.csv`（8152 任务）

**Intel 服务器网络受限的处理**：
1. **优先路径**：从本地 WSL / macOS **scp 到 Intel 服务器** `/d/zcy/datasets/alibaba-v2023/`（`download.py` 不在服务器上直接联网）
2. **备选路径**：若必须在 Intel 直接拉，走国内镜像：
   - GitHub 镜像：`https://gh.api.99988866.xyz/` 或 `https://ghproxy.com/`
   - 原始 CSV 如落到对象存储，走阿里云 OSS / 七牛（后续视情况切换）
3. `download.py` 优先读本地路径，找不到再尝试镜像，两者都失败报清晰错误提示

**产出**：`data_pipeline/raw/*.csv`（`.gitignored`，repo 只留 SHA256 checksum）

### 4.2 节点下采样（`subsample.py`）

- 按 `gpu_spec / model` 字段**分层抽样**到 **40 节点**
- 保持原分布比例（V100M32 / G1 / T4 占比不变）
- 固定 `seed=42`
- 产出：`nodes.json`（~15KB），repo 内可 commit

### 4.3 任务窗口切片

- 用 `pod_phase=Pending` 峰值时间锚定一个高负载时段
- 取 **60-90 分钟窗口**，过滤到只保留目标 40 节点可运行（cpu_milli / memory_mib / gpu 需求不超任一节点容量）的 job
- 截断到 **~400 个 job**
- 产出：`jobs_window.json`

### 4.4 Scenario 生成（`scenarios/loader.py`）

1. window 内所有 job 用 Best-fit expert 预调度 → 得到稳态快照
2. 在稳态上注入 fault（见 § 5）→ 得到 violating state
3. **可解性验证**：跑 Best-fit expert 一次，10 步内无法 recovery 则重 roll
4. 产出 **25 个 scenario JSON**（15 training + 10 held-out），每个含：

```json
{
  "scenario_id": "v2023_node_fail_01",
  "fault_type": "node_failure",
  "fault_nodes": ["node-17", "node-23"],
  "affected_jobs": ["job-412", "job-419"],
  "nodes": {...},           // 40 节点初始 state
  "jobs": {...},            // ~400 job 初始 state
  "assignments": {...},     // 故障前 assignment
  "expected_recovery_steps": 5,
  "expert_solution": [...]  // SFT 数据源
}
```

## 5. Checker 映射

| # | Checker | 字段 | 行为 |
|---|---|---|---|
| 1 | `ResourceCapacityChecker` | `cpu_milli` / `memory_mib` / `gpu` | 直接移植 cluster v1，3 维度 |
| 2 | `AffinityChecker` | `gpu_spec_required` vs `node.model` | 类型不匹配 → 违规 |
| 3 | `PriorityChecker` | `qos ∈ {LS, Burstable, BE}` | LS 排队且 BE 运行 → 违规 |
| 4 | `QueueChecker` | job status | LS + Burstable 不可 Queued |
| 5 | `FragmentationChecker`（新） | 衍生 | **FGD ATC'23 原版 F 公式**，见 § 5.1 |

### 5.1 FragmentationChecker 设计

**公式（FGD ATC'23）**：
```
F(cluster) = Σ_node Σ_g∈G  p(g) · 𝟙[0 < remaining_gpu(node) < g] · remaining_gpu(node)
```

- `G = {1, 2, 4, 8}`（常见 job size）
- `p(g)`：从 v2023 trace 预计算的 job size 分布
- **违规阈值**：`F(cluster) > 1.2 × F_bestfit_baseline`
  - `F_bestfit_baseline` 在 scenario 生成时由 Best-fit expert 确定并写入 scenario 元数据

**Verifier 策略**：
- **Per-action gate**：只检查 checker 1-4（Capacity/Affinity/Priority/Queue）—— 快速 PASS/FAIL
- **Observer-only**：FragmentationChecker —— 进 observation，不作为 per-action verdict
- 这个设计**直接复用 finance domain 的 observer-only + dense-reward 经验**

### 5.2 Dense Reward（继承 finance）

```
r_step = +0.10  if total_violation_count_decreased    # 对 checker 1-4 的 Violation 总数求和
       + 0.30  if fragmentation_decreased_by_ε         # ε = 0.05 × F_bestfit_baseline
       + 1.00  if all per-action checkers (1-4) pass   # episode 终局 bonus，只发 1 次
       - 0.50  if action rejected (verdict=FAIL) by verifier
```

上述 4 项**相加**作为 step reward（不互斥）。Episode reward = Σ step rewards。

## 6. 故障注入（`inject_faults.py`）

### 6.1 故障类型与分布（按 Philly ATC'19 实测比例）

| 类型 | 比例 | 注入方式 | 违反 checker |
|---|---|---|---|
| node_failure | 50% | 1-3 个节点置 Down，其上 running jobs → Preempted | Capacity + Queue |
| gpu_spec_mismatch | 20% | 给 3-5 个 job 加 `gpu_spec_required` 与其节点 model 不符 | Affinity |
| qos_pressure | 20% | 注入 5-10 个 LS job 为 Queued，同时有 BE Running | Priority + Queue |
| fragmentation_surge | 10% | 刻意碎片化大 job | Fragmentation (observer) |

### 6.2 Scenario 分配

- **15 training**：8 node_failure + 3 gpu_spec + 3 qos + 1 fragmentation
- **10 held-out**：5 node_failure + 2 gpu_spec + 2 qos + 1 fragmentation

### 6.3 可解性保证

- 所有 scenario 生成后跑 Best-fit expert 验证 ≤ 15 步内可 recovery
- 不可解的 scenario 重 roll（坑点教训：cluster v1 有过 0% recovery 场景因满载不可解）
- 固定 seed 保 reproducible

## 7. 训练 Pipeline

### 7.1 硬件约束与远程训练协议

- **服务器**：Intel（100.102.144.52），Python `C:\Users\Administrator\miniconda3\envs\pytorch_env\python.exe`，工作目录 `D:\zcy\SILR-Agent\`
- **GPU**：**仅使用 GPU 0**，`CUDA_VISIBLE_DEVICES=0`（严格约束，不占用 GPU 1）
- **网络**：受限，依赖本地 scp 或国内镜像源（见 § 4.1）
- **DataLoader**：`num_workers` ≤ 2（Windows 服务器 commit limit 约束）
- **启动必须遵循 `/remote-train` skill**（`~/.claude/skills/remote-train/SKILL.md`）：
  1. 代码同步走 git（`git push` → 服务器 `git pull`），不依赖散装 scp
  2. WMI `Invoke-CimMethod` 启动，`CurrentDirectory='D:\\zcy\\SILR-Agent'`（避免落到 System32）
  3. 训练脚本内部用 `logging` 写文件，不依赖 shell 重定向
  4. 关键节点必须打日志：启动参数、每 epoch/step 结果、异常、最终结果
  5. 监控用 Monitor + until 循环或 WMI 查 PID，不用 `sleep` 循环
  6. 只杀自己记录 PID 的进程，杀前 `wmic process where ProcessId=<PID> get CommandLine` 验证身份
- **严格红线**：不允许在服务器上执行 `rm -rf` 或任何破坏性操作；未知进程占用 GPU 先报告用户不自行处理

### 7.2 Best-fit Expert（`expert.py`，~150 行）

- 输入：scenario violating state
- 动作空间：`assign_job / preempt_job / migrate_job`（与 LLM agent 同）
- 策略：贪心 Best-fit（按 GPU 剩余量升序），对 down node 上的 job 迁到最大剩余容量节点
- 输出：action 序列（写入 scenario 的 `expert_solution`）
- 双重作用：SFT 数据源 + baseline

### 7.3 SFT 数据组装

- 15 training scenario × 每个 expert 产出 1 条 base trajectory
- **seed 增强**：每 scenario 用 8-10 个 random seed 重跑 expert → ~150 条
- **CoT 补全**：GPT-5.4（via LemonAPI）补推理文本，复用 `clean_sft_data.py` 的 `enrich_with_reasoning()`
- 预算：~150 × 2K tokens ≈ $10-15

### 7.4 SFT 训练

- 模型：Qwen3-14B（`/d/zcy/models/Qwen3-14B`，已预下载）
- 配置（沿用 cluster v1）：QLoRA 4-bit，LoRA r=64, α=128, target=qkv+gate+up+down
- Hyperparam：3 epochs, BS=1 × grad_accum=8, lr=2e-4, cosine
- 耗时：~6-10h on GPU 0
- 产出：`outputs/cluster_v2023/sft_adapter/`
- Eval 目标：10 held-out scenario recovery ≥ **80%**

### 7.5 GRPO 训练

- 超参（沿用 cluster v1 v5 修复后配置）：
  - `lr=1e-6`
  - `batch_size=16` with gradient accumulation（O(1) 内存而非 O(batch)）
  - `kl_coeff=0.02`
  - `advantage clip=[-3, 3]`
  - Rollout 2/scenario, temperature=0.7
  - `group_key=(scenario_id,)`
- 3 iter × ~15h = ~45h
- 产出：`outputs/cluster_v2023/grpo_adapter/`
- Eval 目标：recovery ≥ **90%**

### 7.6 Log-prob 实现铁律（cluster v1 致命坑点复刻）

- **必须** mask prompt tokens（只算 action tokens 的 cross-entropy）
- **必须** 用 sum log-prob 不用 per-token 平均
- 训练开始前跑 sanity check：初始 ratio ≈ 1.0

## 8. 评估

### 8.1 评估集

- 10 held-out scenario × 3 repeat = **30 episode**
- Greedy decoding（`temperature=0`）— eval 固定用 greedy（cluster v1 教训：sampling rollout ≠ deployment）

### 8.2 对比矩阵

| 方法 | Recovery | Fragmentation F (avg, normalized) | Action Reject Rate |
|---|---|---|---|
| Best-fit expert | 100%（by construction） | 1.0 (baseline) | 0% |
| Qwen3-14B zero-shot | — | — | — |
| Qwen3-32B zero-shot | — | — | — |
| SiLR-SFT (14B + LoRA) | ≥ 80% | ≤ 1.2 | 低 |
| SiLR-SFT+GRPO (14B + LoRA) | ≥ 90% | ≤ 1.1 | 最低 |

### 8.3 指标定义

- **Recovery**：episode 结束时所有 per-action checker（1-4）PASS 的 episode 比例；FragmentationChecker 不计入 recovery 判定（observer-only）
- **Fragmentation F (normalized)**：`F_normalized = F_episode_final / F_bestfit_baseline`；`F_bestfit_baseline` 是同一 scenario 用 Best-fit expert recovery 后的 F 值；`F_normalized ≤ 1.0` 表示追平或优于 Best-fit
- **Action Reject Rate**：`count(verdict=FAIL) / count(total actions)`，仅统计 per-action verifier 的 verdict

## 9. 时间轴（14 天）

| 日 | 任务 | 产出 |
|---|---|---|
| D1 | 骨架 + trace 下载 + subsample | nodes.json / jobs_window.json |
| D2 | manager + shadow_copy + solve | pytest smoke 通 |
| D3 | 5 checker + 单测 | tests 全绿，F 阈值标定完成 |
| D4 | tools + observation + prompt + zero-shot eval | 14B/32B zero-shot 数字 |
| D5 | expert + fault injection + 25 scenario | scenarios/data/*.json × 25 |
| D6 | Best-fit expert 产 15 base trajectory | sft_data_v2023_base.jsonl |
| D7 | seed 增强 + GPT-5.4 CoT | sft_data_v2023.jsonl (~150) |
| D8 | SFT 训练（过夜）| sft_adapter/ |
| D9 | SFT eval ≥ 80% 确认 | eval_results_sft.json |
| D10 | GRPO iter 1 | iter 1 ckpt |
| D11 | GRPO iter 2 | iter 2 ckpt |
| D12 | GRPO iter 3 + eval | eval_results_grpo.json |
| D13 | 对比表 + domain README | README 完成 |
| D14 | 根 README + decisions.md + PR | PR ready |

## 10. 风险与止损

| 风险 | 概率 | 止损 |
|---|---|---|
| trace 下载在 Intel 失败 | 中 | 本地 scp，backup 国内镜像 |
| D1-2 manager 出 bug | 低 | 复用 cluster v1 结构，`create_shadow_copy` 用 `copy.deepcopy` |
| D5 scenario 可解性不通过 | 中 | 降低故障强度（1 node → 1 node + 减 job 数） |
| SFT < 70% (D9) | 低 | 砍 qos/fragmentation 故障类型，只保留 node_failure + gpu_spec |
| GRPO iter 1 退化 | 中 | 立即检查 log_prob mask + sum/mean；必要时停在 SFT |
| 2 周未跑完 GRPO 3 iter | 高 | 以 SFT + GRPO iter 1-2 收尾，README 标注 |

## 11. 非目标（明确不做）

- 不做 scheduling-style（连续作业流）评估 —— 留 future work
- 不跑真实 Kubernetes / Volcano scheduler —— 只在 simulator 中调 Best-fit / FGD 指标
- 不做 multi-agent coordinator 适配（cluster_v2023 作 single-agent domain）
- 不修改现有 `domains/cluster/`（94% 结果保留）
- 不替换现有 `scripts/train_sft.py` / `train_grpo.py`（只加新的 data loader 入口）

## 12. 成功标准

**必达**：
- [ ] `domains/cluster_v2023/` 完整 domain 可 import、pytest 全绿
- [ ] 25 scenario 可 load 可运行
- [ ] 14B zero-shot / SFT / GRPO 三档至少出一个数字（可对外可发）
- [ ] README 含对比表 + FGD 引用

**期望**：
- [ ] SFT ≥ 80%, GRPO ≥ 90%（2 数字同时达成）
- [ ] Fragmentation F 优于 Best-fit 的 1.1×
- [ ] CI 绿灯，PR 可 merge

**拉伸**：
- [ ] 3 iter GRPO 全跑完
- [ ] 32B zero-shot vs 14B GRPO 对比成立（14B GRPO ≥ 32B zero-shot）

## 13. 简历 Bullet（顺利达成时）

> "Built `cluster_v2023` domain on Alibaba production GPU trace (OpenB, ATC'23), achieving **91.3% recovery** (placeholder) on 30 held-out fault scenarios with Qwen3-14B + verifier-gated GRPO, outperforming Qwen3-32B zero-shot at 2.3× fewer parameters and matching FGD (ATC'23) baseline on fragmentation metric."

## 14. 参考文献

- Weng et al., "Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent", USENIX ATC'23. (v2023 trace 发布 + FGD 算法)
- Jeon et al., "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads", USENIX ATC'19. (Philly trace + 故障分布基线)
- https://github.com/alibaba/clusterdata
- https://github.com/hkust-adsl/kubernetes-scheduler-simulator
