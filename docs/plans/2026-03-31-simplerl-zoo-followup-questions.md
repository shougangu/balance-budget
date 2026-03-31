# Follow-Up Questions for SFT-DPO-GRPO Pipeline

## Grounding: What We Already Know

### Our findings (Balance Budget, [arXiv:2502.11284](https://arxiv.org/abs/2502.11284), ACL 2025)
- **Cold start problem:** DPO directly from base models fails on reasoning tasks like GSM8K due to distribution shift — the base model can't produce step-by-step reasoning, so preference pairs are out-of-distribution.
- **Small SFT fixes it:** Allocating even <10% of the data budget to SFT first yields 15-20% absolute improvement on GSM8K over zero-SFT DPO.
- **SFT dominates low-data:** Below ~1000 examples, pure SFT outperforms any SFT+DPO mix.
- **Shift toward PFT at scale:** With larger budgets, optimal performance requires increasing the preference data allocation.
- **Task-dependent:** The optimal SFT-to-PFT ratio varies across tasks (math, instruction following, helpfulness, summarization) and model sizes.

### SimpleRL-Zoo ([arXiv:2503.18892](https://arxiv.org/abs/2503.18892), COLM 2025)
- **Zero RL works:** GRPO with rule-based rewards directly from base models yields 10-20pt accuracy gains across 10 diverse models — *without any SFT*.
- **SFT hurts RL:** More SFT steps → diminished GRPO performance. Even 10 SFT steps showed negative effects.
- **Data difficulty alignment:** Weak models collapse on hard data; strong models plateau on easy data.
- **Format reward relaxation:** Strict formatting constraints hurt exploration for weaker models.
- **"Aha moment" emergence:** Self-reflection behaviors emerged in non-Qwen models during zero RL for the first time.

### Key tension to resolve
Our paper says "some SFT is necessary" (cold start for offline DPO). SimpleRL-Zoo says "SFT hurts" (for on-policy GRPO). These are **not contradictory** — the cold start problem is likely specific to *offline* preference optimization (DPO), where the algorithm cannot explore and depends on the reference policy being close to the data distribution. On-policy RL (GRPO) overcomes this through active exploration. This distinction — and how to exploit it — is the central thread of the follow-up questions below.

### Current post-training landscape (2025-2026)
- **"SFT memorizes, RL generalizes"** ([arXiv:2501.17161](https://arxiv.org/abs/2501.17161), ICML 2025): SFT is essential as a format-stabilizing cold start, but RL develops the actual generalizable reasoning.
- **On-policy methods dominate:** Iterative DPO, GRPO, DAPO all outperform offline DPO, with distribution shift identified as offline DPO's core weakness.
- **DAPO** ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)): Improved GRPO with dynamic sampling and decoupled clipping, achieving 50% on AIME 2024 with Qwen2.5-32B.
- **Curriculum RL is emerging:** VCRL (variance-based difficulty), SPEED-RL (online difficulty estimation), SAI-DPO (self-aware iterative DPO) all show that adaptive difficulty scheduling during RL significantly improves reasoning.
- **NFT bridges SL and RL:** Negative feedback training achieves RL-like results from supervised learning, with theoretical equivalence to GRPO under strict on-policy conditions.

---

## Novel Follow-Up Questions

### Q1. Is the Cold Start Problem an Artifact of Offline DPO, Not a Property of the Task?

**Why this matters:** Our core finding is that cold start SFT is necessary for math. But SimpleRL-Zoo shows GRPO doesn't need it. If the cold start problem is an artifact of *offline* preference optimization rather than a fundamental property of reasoning tasks, then the "optimal budget allocation" question changes entirely depending on which PFT method you use.

**Research question:** Does iterative (on-policy) DPO eliminate the cold start problem that we observed with offline DPO? If so, the budget allocation framework from our paper needs a third axis: not just SFT-vs-PFT ratio, but also offline-vs-online PFT.

**Experiment:**
- Run iterative DPO from the base model on GSM8K: at each iteration, the current policy generates responses, scores them with exact-match, and constructs fresh preference pairs. Compare against our offline DPO baseline (with and without SFT cold start).
- If iterative DPO from base matches SFT→offline-DPO, the cold start was indeed a distribution shift artifact.
- Measure across model families (Llama3 vs Qwen2) to test whether the "strong base model" confound matters.

**Connection to current work:** This directly tests the hypothesis from the on-policy literature (iterative DPO, DAPO) that distribution shift is the root cause, while extending our budget allocation framework to online settings.

---

### Q2. Three-Way Budget Allocation: SFT vs. DPO vs. GRPO

**Why this matters:** Our paper studied a two-way budget split (SFT vs. DPO). But the post-training landscape has shifted — GRPO/DAPO are now standard for reasoning tasks, and "SFT memorizes, RL generalizes" suggests each stage plays a distinct role. The natural extension is a three-way budget allocation: SFT (format stabilization) → DPO (preference alignment) → GRPO (generalizable reasoning via exploration).

**Research question:** Given a fixed annotation budget of N examples, what is the optimal three-way split across SFT, DPO, and GRPO? Does each stage have diminishing returns, and does the optimal split differ by task type and model capability?

**Hypotheses:**
- For math/reasoning: minimal SFT (format only, ~5%) → skip DPO → maximal GRPO. The "SFT memorizes" finding predicts DPO won't add value over GRPO for verifiable tasks.
- For instruction following/helpfulness: moderate SFT (~25%) → substantial DPO → no GRPO. These tasks lack verifiable rewards, making GRPO less applicable.
- The three-way frontier will show that DPO and GRPO are *substitutes* for verifiable tasks but *complements* for subjective tasks.

**Experiment:**
- Extend the budget sweep from `collections.yaml` to include a GRPO fraction: `sft_ratio × dpo_ratio × grpo_ratio` grid.
- Use GSM8K exact-match as the GRPO reward. For IFEval, use the strict instruction-following score.
- Build Pareto frontiers per task to identify the efficient allocation.

---

### Q3. Variance-Based Difficulty Curriculum Meets Budget Allocation

**Why this matters:** SimpleRL-Zoo shows data difficulty must match model capability. VCRL ([arXiv:2509.19803](https://arxiv.org/abs/2509.19803)) shows that *reward variance* during GRPO training is a real-time proxy for difficulty — problems where the model sometimes succeeds and sometimes fails (high variance) are the most informative. Our paper showed that the SFT-DPO ratio depends on the data budget. But neither paper asked: **does the optimal budget ratio change dynamically as the model improves during training?**

**Research question:** Should the SFT→DPO→GRPO pipeline use a *dynamic* budget allocation that shifts based on the model's evolving capability, rather than a static pre-determined split?

**Concrete idea — Variance-gated stage transitions:**
- Start in SFT until output format stabilizes (measured by format compliance rate reaching ~90%).
- Transition to DPO/GRPO, but use rollout reward variance to select training problems: high-variance problems (at the model's capability frontier) get GRPO, while low-variance problems (consistently correct or incorrect) get skipped or downweighted.
- As the model improves, the difficulty frontier shifts automatically — this implements SimpleRL-Zoo's "difficulty alignment" without manual difficulty categorization.

**Experiment:**
- Implement a variance-based prompt selector in the GRPO training loop. At each batch, generate rollouts, compute per-prompt reward variance, and filter to prompts with variance above a threshold.
- Compare against static difficulty tiers (Easy/Medium/Hard from SimpleRL-Zoo) and uniform sampling.
- Measure both final accuracy and training efficiency (accuracy per gradient step).

---

### Q4. Does the Cold Start SFT Budget Scale with Model Size, or Is It Constant?

**Why this matters:** Our paper found <10% SFT is sufficient to fix the cold start. SimpleRL-Zoo found that the KL coefficient needs to scale with model size (1e-4 for ≤14B, 1e-3 for >14B). Neither paper systematically studied whether the *amount* of SFT needed for cold start depends on model scale.

**Research question:** Is the cold start SFT budget (in absolute examples or relative %) a function of model size? Specifically: do larger models need fewer SFT examples to learn the output format, because their pretraining already encodes more structure?

**Hypotheses:**
- Larger models (Qwen2-7B, Llama3-8B) need fewer SFT examples for format stabilization, because they have stronger instruction-following from pretraining.
- The cold start budget is roughly *constant in absolute terms* (say, ~100-500 examples) rather than constant as a percentage — meaning for large training budgets, the SFT fraction should shrink toward zero.
- Qwen2 models need less cold start SFT than Llama3 at equivalent sizes (because Qwen2.5 already has latent instruction-following, per SimpleRL-Zoo).

**Experiment:**
- For each model in our sweep (1B, 2B, 3B, 7B, 8B, 14B), run DPO with SFT warm-up of {0, 50, 100, 250, 500, 1000} examples (absolute, not percentage).
- Plot the "cold start recovery curve" — DPO accuracy as a function of SFT examples — per model size.
- Check if there's a universal "minimum viable SFT" count that works across sizes.

---

### Q5. SFT as Format Teacher vs. Knowledge Teacher: Disentangling the Cold Start

**Why this matters:** "SFT memorizes, RL generalizes" argues SFT's role is format stabilization, not knowledge acquisition. Our cold start finding could mean either: (a) the base model needs SFT to learn the *format* of step-by-step math solutions, or (b) the base model needs SFT to acquire *math knowledge* that DPO then refines. These have very different implications for budget allocation.

**Research question:** Can we replace cold start SFT with a cheaper format-only intervention (e.g., SFT on a few dozen format templates with dummy content), freeing the entire data budget for DPO/GRPO?

**Experiment:**
- Create a "format-only SFT" dataset: 50-100 examples with correct step-by-step *format* but random/trivial math content (e.g., "What is 2+2? Let's think step by step...").
- Compare three conditions on GSM8K: (a) no SFT → DPO, (b) format-only SFT → DPO, (c) full math SFT → DPO.
- If format-only SFT eliminates the cold start, then the budget allocation problem simplifies: spend a negligible fixed cost on format SFT, then allocate the entire budget to PFT.
- This also tests whether GRPO's ability to skip SFT entirely (SimpleRL-Zoo) is because on-policy exploration naturally discovers the format, while offline DPO cannot.

---

### Q6. Offline DPO as a Compression of On-Policy RL: What's the Pareto Frontier?

**Why this matters:** Current trends show on-policy methods (GRPO, DAPO, iterative DPO) outperform offline DPO. But on-policy methods are *much more expensive* — they require generating rollouts during training. Our paper's budget framework assumed a fixed annotation cost. The real question practitioners face is: **given a fixed compute budget (not just data budget), when is cheap offline DPO + SFT cold start better than expensive on-policy GRPO?**

**Research question:** What is the compute-normalized Pareto frontier of SFT→offline-DPO vs. SFT→GRPO vs. direct GRPO? At what compute budget does GRPO start dominating?

**Hypotheses:**
- For small compute budgets (e.g., single GPU, hours not days), SFT→offline-DPO with our optimal budget ratios wins — it uses pre-generated data and is cheap to train.
- For large compute budgets, GRPO from base model wins — the exploration more than compensates for the overhead.
- There's a crossover point that depends on model size (smaller models → lower crossover, GRPO becomes practical sooner).

**Experiment:**
- Fix total GPU-hours at several levels (e.g., 1h, 4h, 16h, 64h per model size).
- For each budget: compare best-achievable accuracy from (a) SFT→offline DPO (varying the ratio), (b) SFT→GRPO, (c) zero GRPO.
- Plot the compute-accuracy Pareto frontier per approach and model size.

---

### Q7. Cross-Task Transfer in the SFT Cold Start: Can Math Format SFT Help Instruction Following (and Vice Versa)?

**Why this matters:** Our paper studied each task independently. But if the cold start SFT primarily teaches *format* rather than *task knowledge* (Q5), then format SFT from one task might transfer to another. This would mean the cold start cost is amortized across tasks in a multi-task setting.

**Research question:** Does SFT cold start on one task (e.g., math step-by-step) transfer to reduce the cold start on a different task (e.g., instruction following)?

**Experiment:**
- Train SFT on GSM8K (math format), then run DPO on TuluIF (instruction following). Does math-format SFT reduce the IFEval cold start?
- Vice versa: SFT on TuluIF, then DPO on GSM8K.
- Compare against: (a) no SFT → DPO, (b) in-domain SFT → DPO, (c) cross-domain SFT → DPO.
- If cross-task transfer works, it suggests the cold start is a general "output structure" problem, not task-specific.

---

### Q8. Does DPO Beta Need to Co-vary with SFT Duration? (Interaction Effect)

**Why this matters:** Our beta ablation (`beta_ablation.py`) sweeps beta from existing SFT checkpoints, but treats beta and SFT duration as independent. SimpleRL-Zoo found the KL coefficient must scale with model size. There may be an analogous interaction: the *distance* the SFT has moved the policy from the base model should affect how much KL constraint DPO needs.

**Research question:** Is there a principled relationship between SFT duration and optimal DPO beta? Specifically: should beta decrease as SFT duration increases (because the reference policy is already far from the base, so less KL constraint is needed to allow further movement)?

**Concrete prediction:**
- Short SFT (near base model) + high beta → prevents DPO from collapsing into degenerate modes.
- Long SFT (far from base) + low beta → allows DPO to make meaningful updates despite the already-shifted reference.
- The optimal beta × SFT-duration product may be approximately constant.

**Experiment:**
- Cross beta ∈ {0.01, 0.05, 0.1, 0.5} with SFT fractions ∈ {0.01, 0.1, 0.25, 0.5, 1.0} for a full grid.
- Measure both final accuracy and training stability (loss variance, generation diversity).
- Fit a regression to check if `optimal_beta ≈ c / sqrt(sft_steps)` or similar scaling law.

---

### Q9. NFT as a Drop-In DPO Replacement: Does Negative Feedback Eliminate the Cold Start?

**Why this matters:** NFT (Negative Feedback Training) achieves RL-like results from supervised learning and is theoretically equivalent to GRPO under strict on-policy conditions. If NFT can replace DPO in our pipeline, it might combine the *computational efficiency* of offline methods with the *generalization* of RL, potentially eliminating the cold start altogether.

**Research question:** Can NFT replace DPO in our SFT→PFT pipeline, and if so, does it reduce or eliminate the cold start problem?

**Experiment:**
- Implement NFT as an alternative PFT method alongside DPO and KTO in the pipeline.
- Run the full budget allocation sweep with NFT instead of DPO.
- Test zero-SFT → NFT on GSM8K to check if the cold start persists.
- If NFT eliminates the cold start while being computationally cheaper than GRPO, it becomes the recommended default for practitioners.

---

### Q10. Self-Aware Difficulty Scheduling for Budget-Constrained Post-Training

**Why this matters:** SAI-DPO uses real-time model feedback (attempts-to-correct, solution steps, output length) to dynamically construct training batches. Our paper showed the budget ratio matters. Combining these: rather than a static budget split, use the model's *own performance signals* to decide when to stop SFT and start DPO/GRPO, and which examples to prioritize at each stage.

**Research question:** Can we build an adaptive pipeline that uses online performance signals to automatically determine stage transitions (SFT→DPO→GRPO) and per-example difficulty selection, rather than requiring a pre-specified budget ratio?

**Design:**
1. Start in SFT. Monitor format compliance and pass@1 on a held-out set every N steps.
2. When pass@1 improvement plateaus (our existing `early_tuples` callback already detects this), automatically transition to DPO or GRPO.
3. During DPO/GRPO, use rollout reward variance to select training examples (VCRL-style).
4. The "budget ratio" emerges from the data rather than being prescribed.

**Experiment:**
- Implement the adaptive pipeline and compare against the best static ratios from our paper.
- Measure both final accuracy and robustness to different total budgets (does the adaptive approach find a good ratio regardless of budget size?).
- This could unify our budget allocation findings with the curriculum RL literature.

---

## Priority Ranking

| Priority | Question | Effort | Expected Impact | Novelty |
|----------|----------|--------|-----------------|---------|
| P0 | Q1: Cold start as offline-DPO artifact | Medium | Very High — reframes our core finding | High |
| P0 | Q5: Format vs. knowledge disentanglement | Low | Very High — could simplify the entire pipeline | Very High |
| P0 | Q2: Three-way SFT-DPO-GRPO budget | Medium | High — natural extension of our paper | Medium-High |
| P1 | Q4: Cold start scaling with model size | Low | High — practical insight, easy to test | Medium |
| P1 | Q3: Variance-based difficulty curriculum | Medium | High — connects to VCRL/SPEED-RL | High |
| P1 | Q8: Beta × SFT duration interaction | Low | Medium — extends our beta ablation | Medium |
| P2 | Q6: Compute-normalized Pareto frontier | Medium | High — practitioner-relevant | High |
| P2 | Q7: Cross-task cold start transfer | Low | Medium — amortization insight | High |
| P2 | Q9: NFT as DPO replacement | Medium | Medium-High — depends on NFT maturity | Medium |
| P3 | Q10: Self-aware adaptive pipeline | High | Very High — unifies budget + curriculum | Very High |

---

## Implementation Notes

**Existing infrastructure to leverage:**
- `unified_early_pipeline.py` — extend with `--run-grpo`, `--skip-sft` flags
- `beta_ablation.py` — extend for beta × SFT-duration grid (Q8)
- `warmup_ablation.py` — repurpose for cold start scaling experiments (Q4)
- `PassAtKStoppingCallback` with `early_tuples` — already detects plateaus for adaptive transitions (Q10)
- `EvalStrategy` ABC — add variance-based difficulty metrics (Q3)
- `UnslothGRPOTrainer` — integrate into pipeline for Q2, Q3, Q6
- W&B table logging — track format compliance, reward variance, self-reflection emergence

**New infrastructure needed:**
- Iterative DPO loop with on-policy preference generation (Q1)
- Format-only SFT dataset generator (Q5)
- GRPO stage in `unified_early_pipeline.py` with rule-based reward functions (Q2)
- Compute budget tracking and normalization (Q6)
- NFT trainer integration (Q9)
- Variance-based prompt selector for GRPO batches (Q3)
