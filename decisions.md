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

### 坑点：specialist 的 DomainConfig.checkers 不影响验证

specialist 的 DomainConfig 中的 checkers 只用于 observation（让 specialist 知道自己负责哪些约束），不影响 SiLRVerifier 的验证。verifier 始终用 full_domain_config 的完整 checker 列表。如果搞混了，specialist 提的 action 可能通过自己的子集 checker 但被全局 verifier 拒绝 — 这是正确行为，不是 bug。
