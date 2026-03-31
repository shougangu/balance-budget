# Follow-Up Questions for SFT-DPO-GRPO Pipeline
## Inspired by SimpleRL-Zoo (arXiv:2503.18892)

**Paper:** [SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild](https://arxiv.org/abs/2503.18892)
**Authors:** Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, Junxian He

**Context:** Our project (Balance Budget, arXiv:2502.11284) studies trade-offs between SFT and preference-based finetuning (DPO/KTO). SimpleRL-Zoo investigates zero RL training across 10 diverse base models and finds that SFT can *hurt* subsequent RL performance. Their findings raise several follow-up questions for our SFT-DPO-GRPO pipeline.

---

## 1. Does the SFT-Hurts-RL Effect Extend to DPO/KTO?

**SimpleRL-Zoo finding:** More SFT steps leads to diminished RL performance. Even 10 SFT steps showed some negative effects, and >20 steps substantially reduced RL potential.

**Follow-up questions:**
- Does the inverse SFT→RL relationship also hold for SFT→DPO? Our pipeline currently assumes SFT is a beneficial warm-up for DPO — is that actually true across all model families?
- At what SFT checkpoint does DPO performance peak vs. degrade? Is there an optimal "SFT budget" that differs from what we currently use?
- Does the effect differ between DPO and KTO, given KTO doesn't require preference pairs and may be less sensitive to the policy's starting point?
- **Experiment:** Run DPO from multiple SFT checkpoints (0, 10, 20, 50, 100 steps) and measure pass@k on GSM8K and IFEval. Compare against our current full-SFT→DPO baseline.

## 2. Should We Support Zero-RL / Zero-DPO (Base Model → GRPO/DPO Directly)?

**SimpleRL-Zoo finding:** Zero RL (RL directly from base model, no SFT) produces the best reasoning gains.

**Follow-up questions:**
- Can we run DPO directly from base models (Qwen2, Llama3) without any SFT stage? How does this compare to our SFT→DPO pipeline?
- Is GRPO (which we have infrastructure for via `UnslothGRPOTrainer` but don't actively use) a better fit for the zero-training paradigm than DPO, since GRPO uses on-policy reward signals similar to the rule-based rewards in SimpleRL-Zoo?
- **Experiment:** Add a `--skip-sft` flag to `unified_early_pipeline.py` that runs DPO/GRPO directly from the base model. Compare pass@k curves against SFT→DPO.

## 3. How Does Data Difficulty Interact with Our SFT-PFT Budget Allocation?

**SimpleRL-Zoo finding:** Data difficulty must align with model capability. Weak models collapse on hard data; strong models plateau on easy data.

**Follow-up questions:**
- Our GSM8K data is a single difficulty level. Should we stratify GSM8K (and future datasets) by difficulty and allocate different difficulty tiers to the SFT vs. DPO stages?
- For the SFT stage: should weaker models (Llama3-1B, Qwen2-2B) train on easy subsets while larger models (Llama3-8B, Qwen2-7B) train on harder subsets?
- For the DPO stage: should preference pairs be drawn from problems at the model's capability frontier (where it sometimes gets right and sometimes wrong), rather than uniformly?
- Does the optimal SFT-to-DPO data ratio depend on data difficulty? E.g., easy data may need less SFT budget before DPO kicks in.
- **Experiment:** Tag GSM8K problems by difficulty (e.g., number of reasoning steps). Run separate SFT-DPO sweeps per difficulty tier per model size. Measure whether difficulty-aware allocation changes the optimal budget split.

## 4. What Role Do Exploration Hyperparameters Play in DPO/GRPO?

**SimpleRL-Zoo finding:** Sampling size and temperature during RL training significantly affect reasoning capability emergence.

**Follow-up questions:**
- Our current `PassAtKConfig` uses `temperature=0.5` for evaluation. But what temperature should we use during DPO/GRPO *training* data generation? SimpleRL-Zoo uses `temperature=1.0` for RL rollouts.
- For GRPO specifically: how many rollouts per prompt (SimpleRL-Zoo uses 8) are needed to get reliable group reward estimates? Does this differ by model size?
- Should we generate DPO preference pairs on-the-fly during training (with temperature-controlled exploration) rather than using pre-generated static datasets from GPT-4o?
- **Experiment:** Sweep DPO training with preference data generated at different temperatures (0.5, 0.7, 1.0, 1.2). For GRPO, sweep rollout counts (4, 8, 16) per prompt.

## 5. Should Format Rewards Be Relaxed or Tightened?

**SimpleRL-Zoo finding:** Models trained without strict formatting constraints show better exploration and higher final performance, especially for weaker models.

**Follow-up questions:**
- Our GSM8K scoring uses strict regex extraction (`#### (-?[0-9.,]+)`) with a flexible fallback. Is the strict format hurting weaker models' training signal?
- For IFEval, strict instruction following is the whole point — but does enforcing it from step 0 hurt exploration early in training?
- Should we use a curriculum: relaxed format rewards early → strict format rewards later?
- **Experiment:** Compare two DPO/GRPO runs: one with strict-only scoring, one with a two-phase approach (loose early, strict late). Measure final pass@k.

## 6. Do Qwen2.5 Models Already "Know" the Answers? (Representation vs. Generation)

**SimpleRL-Zoo finding:** Qwen2.5 base models already exhibit strong instruction-following and self-reflection, making them unrepresentative for studying zero RL.

**Follow-up questions:**
- Our codebase trains both Qwen2 and Llama3 families. Are our Qwen2 results inflated because the base model already has strong capabilities, making the SFT-DPO trade-off appear different than it really is?
- Should we report results stratified by model family to avoid Qwen-specific effects dominating conclusions?
- Do the optimal SFT-DPO budget ratios differ between Qwen2 (strong base) and Llama3 (weaker base)?
- **Experiment:** Run identical SFT-DPO budget sweeps on Qwen2-3B vs. Llama3-3B (similar sizes). Check if optimal ratios diverge.

## 7. Can We Integrate GRPO as a Third Stage (SFT → DPO → GRPO)?

**SimpleRL-Zoo finding:** Rule-based reward RL (GRPO-style) elicits emergent reasoning behaviors that supervised methods cannot.

**Follow-up questions:**
- Our pipeline supports SFT→DPO. Should we add a GRPO stage after DPO to push reasoning further?
- Or should we replace DPO with GRPO entirely, given SimpleRL-Zoo's findings that RL from base models outperforms SFT→RL?
- What reward function should GRPO use? For GSM8K, exact-match correctness is natural. For IFEval, instruction adherence score works. But how do we handle partial credit?
- What is the three-way budget allocation: given N total training examples, how should we split across SFT, DPO, and GRPO?
- **Experiment:** Extend `unified_early_pipeline.py` to support a third GRPO stage. Compare: (a) SFT→DPO, (b) SFT→GRPO, (c) SFT→DPO→GRPO, (d) Base→GRPO.

## 8. Does the "Aha Moment" (Self-Reflection Emergence) Occur in Our Pipeline?

**SimpleRL-Zoo finding:** Non-Qwen models can exhibit the "aha moment" (spontaneous self-reflection) during RL training. Increased response length doesn't always correlate with this emergence.

**Follow-up questions:**
- Do we see self-reflection or verification behaviors emerge during our DPO training? Are we even measuring this?
- Should we add a metric to our `PassAtKStoppingCallback` that detects self-reflection patterns (e.g., "wait, let me check", "actually", "I made an error")?
- Does the DPO stage encourage or suppress these behaviors compared to GRPO?
- **Experiment:** Add regex-based detection of self-reflection patterns in raw generations logged to W&B tables. Track emergence across training steps.

## 9. How Does KL Penalty Interact with the SFT Budget?

**SimpleRL-Zoo finding:** KL loss coefficient matters (1e-4 for small models, 1e-3 for >14B).

**Follow-up questions:**
- Our DPO uses `beta=0.1` as the KL penalty. Should this vary with (a) model size and (b) how many SFT steps preceded DPO?
- Hypothesis: if SFT has already moved the model far from the base, a lower beta allows DPO to move further. If starting from base (zero DPO), a higher beta prevents collapse.
- **Experiment:** Sweep `beta` in {0.01, 0.05, 0.1, 0.5} crossed with SFT steps {0, 10, 50, full}. Measure pass@k and generation diversity.

## 10. On-Policy vs. Off-Policy Preference Data

**SimpleRL-Zoo finding:** On-policy RL (generating fresh rollouts during training) is central to their approach.

**Follow-up questions:**
- Our DPO uses off-policy preference data (pre-generated by GPT-4o from HuggingFace). Does switching to on-policy preference generation (model generates its own chosen/rejected pairs scored by rule-based rewards) change the SFT-DPO trade-off?
- On-policy data should be more aligned with the current policy's capability, naturally implementing the "difficulty alignment" that SimpleRL-Zoo advocates.
- **Experiment:** Implement an online DPO variant where the model generates candidate responses at each training step, scores them with GSM8K exact-match, and uses the best/worst as chosen/rejected pairs. Compare against static GPT-4o preferences.

---

## Priority Ranking

| Priority | Question | Effort | Expected Impact |
|----------|----------|--------|-----------------|
| P0 | Q1: SFT-hurts-DPO effect | Low | High — directly validates/invalidates our core assumption |
| P0 | Q2: Zero-DPO from base model | Medium | High — could fundamentally change pipeline design |
| P1 | Q7: GRPO as third stage | Medium | High — activates unused infrastructure |
| P1 | Q3: Difficulty-aware data allocation | Medium | Medium-High — refines budget allocation |
| P1 | Q9: KL penalty × SFT budget interaction | Low | Medium — simple sweep, actionable results |
| P2 | Q6: Qwen vs. Llama family differences | Low | Medium — important for paper validity |
| P2 | Q4: Exploration hyperparameters | Medium | Medium — temperature/rollout sweeps |
| P2 | Q5: Format reward relaxation | Low | Medium — easy to test |
| P3 | Q8: Self-reflection emergence detection | Low | Low-Medium — observability improvement |
| P3 | Q10: On-policy DPO | High | High but risky — significant infrastructure change |

---

## Implementation Notes

**Existing infrastructure to leverage:**
- `unified_early_pipeline.py` — extend with `--skip-sft`, `--run-grpo` flags
- `UnslothGRPOTrainer` — already cached, needs integration into pipeline
- `PassAtKStoppingCallback` — extend to track self-reflection patterns
- `EvalStrategy` ABC — add difficulty-stratified evaluation variants
- W&B table logging (from recent plan) — use for exploration analysis

**New infrastructure needed:**
- GRPO stage in `unified_early_pipeline.py` with rule-based reward functions
- Difficulty tagging for GSM8K dataset
- On-policy preference data generation loop (for Q10)
- Multi-checkpoint SFT→DPO sweep harness (for Q1)
