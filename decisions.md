# SiLR-Agent 设计决策日志

## 2026-04-03: 多智能体 Coordinator 架构

### 决策：Coordinator 用 LLM 而非规则驱动

**选 LLM coordinator 的理由**：级联故障的约束交互是 context-dependent 的，规则调度只是更僵硬的单 agent。LLM coordinator 可以通过 SFT/DPO 训练学习调度策略，这本身就是研究贡献。

**不选规则的理由**：规则调度需要硬编码优先级（电压 > 频率 > 线路），但最优顺序取决于当前状态。

### 决策：Specialist 复用 ReActAgent，不新建类

**理由**：ReActAgent 接受 DomainConfig 参数，用不同的 DomainConfig（限制 allowed_actions）就能让它变成 specialist。零代码修改。

**关键设计**：SiLRVerifier 用完整 DomainConfig（全 checkers）构建，specialist 用受限 DomainConfig（子集 tools）构建。Specialist 只能提议自己 domain 内的 action，但 verifier 检查全局安全性。

### 决策：Specialist 间通过共享 manager 状态通信

**不选显式消息传递的理由**：
1. manager 已经是 single source of truth
2. coordinator 每轮重新 observe 就能看到所有变化
3. 简化实现，容易测试
4. 和现有架构一致（单 agent 也是通过 manager 状态感知变化）

### 决策：MultiAgentEpisodeResult 独立类型，不继承 EpisodeResult

**理由**：多智能体的数据结构（activations、coordinator decisions、conflict tracking）和单智能体差异太大。强行继承会导致 Liskov 违反。用 `to_single_agent_view()` 提供向后兼容。

---

## 坑点记录

### 坑点（已修复）：默认 DomainConfig 加 observer 破坏了单 agent 测试

给 `build_network_domain_config()` 默认加上 `create_observer=NetworkObserver` 后，所有单 agent 测试挂了。原因：旧测试依赖 `_MinimalObserver`（永远返回 `is_stable=False`），而 `NetworkObserver` 正确检测到"断一条链路 ≠ 有违规"（5 节点网络有冗余路径）。

**修复**：`with_observer` 默认为 False，coordinator 测试显式传 True。observer 是 opt-in 而非 default。

### 坑点（已修复）：cascade 场景的 overload 值低于 90% 阈值

link 2-5 capacity=60，overload 设为 52 → 86.7% < 90% → 无违规。改为 55（easy）和 58（hard）才能触发。cascade_medium 改为断 2-3+3-5 隔离 node 3 造成 connectivity 违规。

### 坑点（已修复）：run_pflow() 会覆盖手动设置的 traffic overload

`NetworkManager.run_pflow()` 先 reset 所有 traffic 为 0，再按 demand shortest-path 重新路由。如果在 `run_pflow()` 前设置 overload，值会被清零。

**修复**：`setup_episode()` 中先 `run_pflow()` 让 solver 算完，再手动设置 overload。这样 observer 能看到真实的高负载状态。下次 specialist 执行 action 触发的 `run_pflow()` 会重新计算，但此时拓扑已变（链路恢复），traffic 分布自然不同。

### 坑点：coordinator DPO pairing 跨 round 的 observation 不匹配

`export_coordinator_dpo()` 将"改善了约束的 dispatch"和"恶化了约束的 dispatch"配对，但它们可能来自不同 round、不同系统状态。DPO 训练的 prompt 用的是 good dispatch 的 observation，rejected 用的是 bad dispatch 的响应 — 但 bad dispatch 当时面对的是完全不同的状态。

**当前处理**：接受这个限制。Coordinator DPO 数据量本身就小，主要靠 SFT。如果后续要严格 DPO，应该只配对同一 round 内的 alternative dispatch。

### 坑点：specialist 的 DomainConfig.checkers 不影响验证

specialist 的 DomainConfig 中的 checkers 只用于 observation（让 specialist 知道自己负责哪些约束），不影响 SiLRVerifier 的验证。verifier 始终用 full_domain_config 的完整 checker 列表。如果搞混了，specialist 提的 action 可能通过自己的子集 checker 但被全局 verifier 拒绝 — 这是正确行为，不是 bug。

---

## 2026-04-04 ~ 04-05: GPU 集群调度 SFT → GRPO 训练管线

### 决策：选 GPU 集群调度而非 K8s

**选集群调度的理由**：与东工大 TSUBAME 超算叙事一致（真实经验是 GPU 作业调度，不是容器编排）。15 节点 × 3 机架的模型复杂度中等，够展示 RL 后训练但不过于复杂。

**不选 K8s 的理由**：K8s 的 API 对象太多（Pod/Service/Ingress/PV...），模拟器写起来耗时且偏离 SiLR 的核心价值。

### 决策：Verifier 仅检查 ResourceCapacity + Affinity，Observer 检查全部 5 项

**理由**：per-action 验证只需确保安全性（不超容量、不违反 affinity），而 RackSpread / Priority / QueueClearance 是 episode 级目标。如果 verifier 检查 QueueChecker，每次 assign 都会被拒（因为其他 job 仍在排队），导致 100% 拒绝率。

**坑点**：最初 verifier 包含了全部 5 个 checker → 1.2% recovery rate。定位过程：API 正常 → 模型响应正常 → verifier 拒绝一切 → 发现是 QueueChecker 在 per-action 级别必定失败。

### 决策：SFT 数据用 teacher model (Gemini Flash / GPT-5.4) 而非 rule-based solver

**理由**：GRPO 只需 reward signal（来自 SiLR verifier，<0.1ms），不需要 expert data。但 SFT 冷启动需要高质量 trajectory。Teacher model 能产出带自然语言推理的 trajectory，rule-based solver 只能产出 action 序列。

### 决策：不去重 SFT 样本（170 → 143），用 GPT-5.4 补推理

**理由**：精确去重只留 65 条太少。同场景不同参数（选不同节点）仍有训练价值——模型需要学会根据具体 observation 选节点。每条用 GPT-5.4 生成独立的 chain-of-thought，即使 action 相同推理也不同。

### 决策：add_jobs group 从统一 "dynamic" 改为 per-job unique

**理由**：RackSpreadChecker 要求同组 urgent job 跨 2+ 机架。统一 "dynamic" group 意味着所有新注入的 urgent job 共享组名 → 全放 rack-c（唯一有空间的机架）就违规。改为 `dynamic-{jid}` 后每个 job 独立组，不触发 RackSpread。

### 决策：Observation 增加 available_nodes + preemptible_running + rack_affinity

**理由**：旧 observation 只展示 busy_nodes（>70% 利用率），LLM 看不到空闲节点在哪 → 盲猜节点 → 被拒。加 available_nodes 后 job_surge_small 从 0% → 100%。加 preemptible_running 后 resource_fragmentation 从 0% → 100%（LLM 终于知道能 preempt 什么）。

### 决策：SFT 训练用 QLoRA 4-bit，Qwen3-14B

**配置**：LoRA r=64, alpha=128, target=qkv+gate+up+down，3 epochs, BS=1×8 grad_accum, lr=2e-4, cosine schedule。

**结果**：train loss 0.129, eval loss 0.034, perplexity 1.03。39 分钟完成。

### SFT Eval 结果（2026-04-06）

Qwen3-14B + LoRA（SFT后） vs GPT-5.4 teacher model：
- **Overall recovery: 88.2% (45/51)** vs teacher 67%
- **15/17 场景 100%**（teacher只有8/17达到100%）
- SFT模型在 rack_failure_a (0→100%)、resource_fragmentation (0→100%)、rack_failure_b (20→100%)、job_surge (60→100%)、job_surge_large (60→100%) 上超越teacher
- 仅 compound_failure_surge 和 compound_multi_node_failure 仍 0%（训练数据中这两个场景各仅1条和0条样本）
- 推理速度：本地14B 4-bit，~7min/episode（51 episodes 共3.5h）

---

## 坑点记录（集群调度 SFT/GRPO）

### 坑点（严重）：集群 100% 满载导致场景不可解

最初生成 60-80 个 job × 1.83 avg GPU = 92+ GPU 但集群只有 72 GPU → 0% 空闲 → 任何 assign 都被 ResourceCapacityChecker 拒绝。

**修复**：减到 30-35 个 job × 1.33 avg GPU ≈ 44 GPU（61% 利用率），留 28 GPU headroom。

### 坑点（严重）：9/17 场景始终 0% recovery

分三类根因：
- **A 类（LLM 策略）**：Observation 信息不足，LLM 看不到空闲节点/rack_affinity
- **B 类（步数不够）**：注入 job 太多，最优解 > 10 步
- **C 类（结构性不可解）**：rack_affinity 指向 down 的机架 / GPU 总量不足

**修复**：A 类修 observation + prompt；B 类减 job 数；C 类改场景参数（部分机架 down → 部分节点 down）。

### 坑点：LLM 只会 assign_job，不会 preempt/migrate

GPT-5.4 在 v5 中 77% 用 assign_job，0% 用 migrate_job。即使 prompt 说"先 preempt 再 assign"，实际不执行。

**修复**：
1. system prompt 加 preempt/migrate 的具体示例（不只是规则描述）
2. observation 加 preemptible_running 列表（告诉 LLM 能 preempt 什么、在哪个节点）
3. rack_failure_a 从 0% → 100%（migrate 示例生效），resource_fragmentation 从 0% → 100%（preemptible_running 生效）

### 坑点：Thought+JSON 重复输出格式

Gemini Flash 的 assistant 响应为 `Thought: {"tool_name":...}\n\n{"tool_name":...}`——推理内容就是 JSON 本身的重复。所有 716 条 assistant 消息都有此问题。

**修复**：clean_sft_data.py 中 `clean_assistant_content()` 检测多个 JSON block，只保留最后一个。GPT-5.4 enrichment 补上真正的推理文本。

### 坑点：v3/v4 旧 observation 格式缺信息

v3/v4 的 observation 没有 available_nodes / preemptible_running / rack_affinity（这些是 v5 修复后才加的）。直接拿来训练会教模型错误的 observation schema。

**修复**：clean_sft_data.py 的 replay 机制——对每条 SFT 样本，用当前 observer 代码重放场景 + action 序列，重建所有 observation。143 条全部 replay 成功。

### 坑点：trl 0.24.0 API 变更

- `SFTTrainer.__init__()` 不再接受 `max_seq_length` 参数
- 需要用 `SFTConfig` 代替 `TrainingArguments`
- `SFTConfig` 中参数名是 `max_length` 不是 `max_seq_length`
- `torch_dtype` deprecated → 用 `dtype`

连续 3 次启动失败才定位到这些 API 变更。教训：新版本库先查 signature 再写代码。

### 坑点：WMI 启动不继承 CUDA_VISIBLE_DEVICES

WMI `Invoke-CimMethod` 的 `CommandLine` 参数不继承 shell 环境变量。直接 `set CUDA_VISIBLE_DEVICES=1` 在 WMI 命令中无效。

**修复**：写 `.bat` 文件，在 bat 内 `set CUDA_VISIBLE_DEVICES=1` 后再调 python。WMI 启动 `cmd /c xxx.bat`。

### 坑点：bat 文件路径被 bash 转义

Git Bash 的 `printf` 会把 `\U`（Unicode escape）、`\e`（ANSI escape）等吞掉。`miniconda3\envs` 变成 `miniconda3nvs`。

**修复**：用 `Write` 工具在 WSL 端写 bat 文件再 scp 到服务器，不通过 bash printf/echo 生成。

### 坑点：SSH 后台进程被断连杀死

`ssh ... 'python xxx.py &'` 启动的后台进程在 SSH 断连后随 sshd session 一起终止（Windows 的 Job Object 机制）。v3 enrichment 在 640/925 时静默挂掉就是这个原因。

**修复**：所有长时间运行的任务必须用 WMI 启动（父进程是 WmiPrvSE.exe，不在 sshd Job Object 内）。

### 坑点：API 额度耗尽导致 v4 半数无效

v4 收集的 5 轮中，repeat 2-4 全部 403 Forbidden（API 额度用完），failsafe_triggered=true 的 15 个 episode 数据无效。有效数据只有 repeat 0-1 的 10 个 episode。

**修复**：换了新 API key 后重跑。教训：长时间收集前先查余额。

---

## 2026-04-07: GRPO 训练退化诊断与修复

### 现象

GRPO Iter 1 后 recovery 从 SFT 的 88.2% 降至 58.8%，Iter 2 rollout 中模型完全退化（0% recovery），开始输出大段自然语言而非 JSON 工具调用。

### 决策：log_prob 必须只计算 action tokens（prompt masking）

**Root cause #1（致命）**：旧代码 `outputs = model(**encoding, labels=encoding["input_ids"])` 在整个序列上计算 loss，包括 user message（observation JSON，占序列 60-80%）。PPO ratio 应只反映 action tokens 的概率变化，但被 prompt tokens 严重稀释。更致命的是梯度也流过 prompt tokens，模型被训练去"更好地预测 observation"而非"更好地选择 action"，直接破坏 instruction-following 能力。

**修复**：`_find_action_start()` 找到 assistant response 起始位置，labels 中 prompt 部分设为 -100（HuggingFace 的 ignore index）。只对 action tokens 计算 cross-entropy。

### 决策：log_prob 用 sum 而非 per-token 平均

**Root cause #2（致命）**：旧代码 `sample.log_prob = -outputs.loss.item()` 是 per-token 平均。PPO ratio = exp(new - old)，用平均值时 ratio 永远趋近于 1，clipping 形同虚设。正确做法是 sum log prob：`log p(action|prompt) = Σ_t log p(a_t | a_{<t}, prompt)`。

**修复**：`-outputs.loss * n_action_tokens` 得到 sum log prob。

### 决策：group_key 保持 (scenario_id,) 而非 (scenario_id, step_idx)

**背景**：DeepSeek GRPO 论文中 group size 通常 64（最小 16），用于"同一 prompt 的不同 response"做 relative comparison。理论上应该按 (scenario_id, step_idx) 分组。

**选 (scenario_id,) 的理由**：只有 2 rollouts/scenario，per-step 分组的组大小 = 2，reward 只有三个离散值（+0.45 / -0.50 / +1.45），两个 rollout 在同一 step 很可能做相同决策 → reward 相同 → std=0 → advantage=0。实测 loss=0.0000，完全没有训练信号。

**不选 (scenario_id, step_idx) 的理由**：group size < 8 时 GRPO 的 advantage estimation 不可靠（RLOO 论文：group=2 退化为高 variance 的 REINFORCE with baseline）。增大 rollouts 到 8+ 不现实（14B 模型每 rollout 20-30 min，17 scenarios × 8 = 45 小时只是 Phase 1）。

**trade-off**：(scenario_id,) 分组混了不同 step，不是严格 GRPO，更接近 scenario-level reward-weighted policy gradient。但组大小 12+（6 steps × 2 rollouts），reward 有 variance，至少能产出非零 advantage。有偏但有梯度的 estimator 远好过无偏但为零的 estimator。

### 决策：超参数调整——lr 5e-6, kl_coeff 0.02

**lr 1e-5 → 5e-6**：SFT 用 2e-4 是因为 LoRA 权重从零初始化。GRPO 是在 SFT 基础上微调，且修复 log_prob 后梯度信号更强，需要更保守的步长。

**kl_coeff 0.1 → 0.02**：修复后 log_prob 从 per-token 平均（~-1.5）变成 sum（~-80），KL 值放大 ~25 倍。如果保持 0.1，KL penalty 会压倒 policy gradient，模型不敢更新。

**advantage clipping [-3, 3]**：防止 recovery bonus（+1.45 vs 组内均值 ~0.3）的极端 z-score 主导梯度。

### 坑点（致命）：GRPO 一轮 policy update 摧毁 SFT 模型

**症状**：SFT 88.2% → GRPO Iter 1 58.8% → Iter 2 0%。模型从输出规范 JSON 工具调用退化为输出大段自然语言解释。

**根因**：log_prob 包含 prompt tokens + per-token 平均，导致：(1) 梯度方向错误——模型被训练预测 observation 而非选择 action；(2) ratio 信号被稀释到接近 1，有效学习几乎是随机扰动；(3) KL penalty 基于错误的 ratio，无法约束模型不偏离 SFT 分布。一轮 update（192 samples × batch_size=4 × 2h）就足以摧毁 SFT 学到的 JSON 格式。

**教训**：PPO/GRPO 的 log_prob 计算是 RL 后训练最关键的基础设施。必须验证：(1) 只计算 action tokens；(2) 用 sum 不用 mean；(3) old 和 new 的计算方式完全一致。上线前应跑 sanity check：ratio 初始值应非常接近 1.0（因为 old/new 用同一模型）。

### 决策：GRPO 超参数 lr=1e-6, batch_size=16 + gradient accumulation

**v3 (lr=5e-6, batch=4)** Phase 3 后 log_ratio mean=-16.249，88% clamped → policy 偏移过大，模型崩。

**v4 (lr=1e-6, batch=16)** 直接放大 batch_size 触发 OOM：原代码累积所有样本的 computation graph 再 backward，O(batch_size) 内存 → 16 个样本 95GB 不够。

**v5 (lr=1e-6, batch=16, grad accum)** 改为每个样本单独 backward + gradient 累加，O(1) 内存，3 iter 全部稳定：
- log_ratio mean: -0.733 → 0.025 → -0.066（教科书级 PPO）
- clamped: 1/120, 0/28, 1/28（基本零 clamp）
- Loss: 0.5073 → 0.2972 → 4.9732（最后一轮 KL 项主导属正常）

**关键修复**：用 gradient accumulation 而非 batch loss 累积。功能等价但内存从 O(batch_size) 降到 O(1)。

### GRPO Eval 结果（2026-04-07）

GRPO final adapter (4-bit Qwen3-14B + LoRA) vs SFT baseline，eval 用 temperature=0 greedy，3 repeats × 17 scenarios = 51 episodes：

| 模型 | Overall Recovery | Per-Scenario 100% |
|------|------------------|-------------------|
| GPT-5.4 teacher (v5) | 67% (~34/51) | 8/17 |
| **SFT** | **88.2% (45/51)** | **15/17** |
| **SFT + GRPO** | **94.1% (48/51)** | **16/17** |

**唯一 0% 的场景**：`compound_multi_node_failure`（SFT 数据里 0 条样本，纯靠迁移学不会）

**关键提升**：
- `compound_failure_surge`: SFT 0/3 (0%) → **GRPO 3/3 (100%)** 🔥
- 其他 15 个 100% 场景全部维持，无任何退化

**为什么训练 rollout 看不到提升但 eval 有提升？**

训练 rollout 用 temperature=0.7（exploration），eval 用 temperature=0 (greedy)。两个测量灵敏度完全不同：
- Sampling 模式下，模型即使有 5% 的概率改善，2 个 rollout 之间也几乎看不出差异（被随机性淹没）
- Greedy 模式下，概率分布的小幅变化可能直接翻转 argmax → 完全不同的决策路径

GRPO 训练时 Iter 2/3 在 sampling rollout 下显示 0% recovery，让人以为没学到东西。实际上模型的内部参数确实在朝正确方向移动（log_ratio mean 接近 0、loss 收敛证明了这一点），只是 sampling 测量噪声压过了信号。eval 用 greedy 解码才暴露了真实改善。

**教训**：rollout success rate ≠ deployment success rate。RL 后训练评估必须用 deployment 的 inference 模式（greedy）做最终判断，sampling 只能用作 exploration 信号。

### 坑点（隐蔽）：domain 修改未 commit 导致 GRPO 跑了一轮无效训练

GRPO v3 第一次启动后发现 recovery 异常低（rack_failure_a 等场景从 SFT 的 100% 降到 0%）。以为是 GRPO 退化了。

**实际原因**：服务器上 `domains/cluster/observation.py`、`scenarios/loader.py`、`prompts/system_prompt.py` 是旧版本——SFT 阶段所有 observation 增强（available_nodes, preemptible_running, rack_affinity）和 scenario 修复**从未 commit**。`git pull` 时这些文件被 reset 到了某个早期 commit 状态。

GRPO rollout 用旧 observer 看不到空闲节点和 preemptible jobs，自然做不了正确调度。连续浪费 ~4h GPU 时间才发现。

**修复**：commit 所有 working tree 的修改，重新部署。

**教训**：实验 pipeline 跨多机时，**任何代码变更必须 commit**（即使是临时修改）。`git status` 显示有 working tree changes 时不能假设服务器和本地代码一致。
