# Cross-family 8B comparison (step-8, seeds 1000-1002, 3 mined scenarios)

Qwen3-8B and DeepSeek-R1-Distill-Llama-8B (Llama-3 base + R1 distillation) are independent families at matched 8B parameter scale. The comparison tests whether the terminal-deadlock / progress_mag-advances pattern is Qwen-specific or generalizes across families.

## Cross-family headline (overall recovery / 9 ep per policy)

| Family | Family-of-origin | Terminal | Progress_mag |
|---|---|---|---|
| Qwen3-8B (canonical) | Alibaba Qwen3 | 0/9 | 3/9 |
| DSR1-Distill-Llama-8B (cross-family 8B, altfix) | Meta Llama-3 + DeepSeek R1 distill | 0/9 | 0/9 |
| Qwen3-14B bare-text (elicitation ablation) | Alibaba Qwen3 | 0/9 | 8/9 |
| Gemma-3-12B-IT (cross-family 12B, altfix) | Google Gemma-3 | 0/9 | 9/9 |
| Gemma-4-31B-IT (cross-family 31B, altfix) | Google Gemma-4 | 0/9 | 8/9 |
| Qwen3-32B step=16 (TSUBAME rescue) | Alibaba Qwen3 | 0/9 | 7/9 |


### Qwen3-8B (canonical)

| Scenario | Policy | Recovery | Final pen. mean | Reject/prop |
|---|---|---|---|---|
| m1 | `terminal` | 0/3 | 14.701 | 1.000 |
| m1 | `progress_mag` | 0/3 | 1.602 | 0.436 |
| m2 | `terminal` | 0/3 | 5.061 | 1.000 |
| m2 | `progress_mag` | 1/3 | 1.674 | 0.500 |
| m3 | `terminal` | 0/3 | 17.592 | 1.000 |
| m3 | `progress_mag` | 2/3 | 0.172 | 0.318 |

**Overall (9 episodes per policy):**

| Policy | Recovery | Wilson 95% CI |
|---|---|---|
| `terminal` | 0/9 | [0.00, 0.30] |
| `progress_mag` | 3/9 | [0.12, 0.65] |


### DSR1-Distill-Llama-8B (cross-family 8B, altfix)

| Scenario | Policy | Recovery | Final pen. mean | Reject/prop |
|---|---|---|---|---|
| m1 | `terminal` | 0/3 | 14.701 | 1.000 |
| m1 | `progress_mag` | 0/3 | 7.753 | 0.077 |
| m2 | `terminal` | 0/3 | 5.061 | 1.000 |
| m2 | `progress_mag` | 0/3 | 4.531 | 0.512 |
| m3 | `terminal` | 0/3 | 17.592 | 1.000 |
| m3 | `progress_mag` | 0/3 | 10.201 | 0.000 |

**Overall (9 episodes per policy):**

| Policy | Recovery | Wilson 95% CI |
|---|---|---|
| `terminal` | 0/9 | [0.00, 0.30] |
| `progress_mag` | 0/9 | [0.00, 0.30] |


### Qwen3-14B bare-text (elicitation ablation)

| Scenario | Policy | Recovery | Final pen. mean | Reject/prop |
|---|---|---|---|---|
| m1 | `terminal` | 0/3 | 14.701 | 1.000 |
| m1 | `progress_mag` | 2/3 | 0.117 | 0.045 |
| m2 | `terminal` | 0/3 | 5.061 | 1.000 |
| m2 | `progress_mag` | 3/3 | 0.000 | 0.000 |
| m3 | `terminal` | 0/3 | 17.592 | 1.000 |
| m3 | `progress_mag` | 3/3 | 0.000 | 0.300 |

**Overall (9 episodes per policy):**

| Policy | Recovery | Wilson 95% CI |
|---|---|---|
| `terminal` | 0/9 | [0.00, 0.30] |
| `progress_mag` | 8/9 | [0.56, 0.98] |


### Gemma-3-12B-IT (cross-family 12B, altfix)

| Scenario | Policy | Recovery | Final pen. mean | Reject/prop |
|---|---|---|---|---|
| m1 | `terminal` | 0/3 | 14.701 | 1.000 |
| m1 | `progress_mag` | 3/3 | 0.000 | 0.000 |
| m2 | `terminal` | 0/3 | 5.061 | 1.000 |
| m2 | `progress_mag` | 3/3 | 0.000 | 0.000 |
| m3 | `terminal` | 0/3 | 17.592 | 1.000 |
| m3 | `progress_mag` | 3/3 | 0.000 | 0.000 |

**Overall (9 episodes per policy):**

| Policy | Recovery | Wilson 95% CI |
|---|---|---|
| `terminal` | 0/9 | [0.00, 0.30] |
| `progress_mag` | 9/9 | [0.70, 1.00] |


### Gemma-4-31B-IT (cross-family 31B, altfix)

| Scenario | Policy | Recovery | Final pen. mean | Reject/prop |
|---|---|---|---|---|
| m1 | `terminal` | 0/3 | 14.701 | 1.000 |
| m1 | `progress_mag` | 3/3 | 0.000 | 0.000 |
| m2 | `terminal` | 0/3 | 5.061 | 1.000 |
| m2 | `progress_mag` | 2/3 | 0.802 | 0.462 |
| m3 | `terminal` | 0/3 | 17.592 | 1.000 |
| m3 | `progress_mag` | 3/3 | 0.000 | 0.000 |

**Overall (9 episodes per policy):**

| Policy | Recovery | Wilson 95% CI |
|---|---|---|
| `terminal` | 0/9 | [0.00, 0.30] |
| `progress_mag` | 8/9 | [0.56, 0.98] |


### Qwen3-32B step=16 (TSUBAME rescue)

| Scenario | Policy | Recovery | Final pen. mean | Reject/prop |
|---|---|---|---|---|
| m1 | `terminal` | 0/3 | 14.701 | 1.000 |
| m1 | `progress_mag` | 2/3 | 0.117 | 0.386 |
| m2 | `terminal` | 0/3 | 5.061 | 1.000 |
| m2 | `progress_mag` | 2/3 | 0.802 | 0.604 |
| m3 | `terminal` | 0/3 | 17.592 | 1.000 |
| m3 | `progress_mag` | 3/3 | 0.000 | 0.300 |

**Overall (9 episodes per policy):**

| Policy | Recovery | Wilson 95% CI |
|---|---|---|
| `terminal` | 0/9 | [0.00, 0.30] |
| `progress_mag` | 7/9 | [0.45, 0.94] |


