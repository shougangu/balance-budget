# R128 4/16/64 H100-hour grid status

Last full refresh: **2026-07-17 19:10 EDT**.
Live W&B run states, endpoint evals, and Killarney Slurm values were sampled
between 18:50 and 19:10 EDT.

This is the operational reference for the Gemma-3 4B, Gemma-3 12B,
Llama-3 3B, and Llama-3 8B `r128` compute grid.

## Reading the grid

- The three total-compute targets are 240, 960, and 3840 H100-minutes
  (4, 16, and 64 H100-hours).
- The SFT starting points are:

  | Total target | 0% SFT | 25% SFT | 50% SFT | 75% SFT | 100% SFT |
  |---:|---:|---:|---:|---:|---:|
  | 240m | 0m | 60m | 120m | 180m | SFT at 240m |
  | 960m | 0m | 240m | 480m | 720m | SFT at 960m |
  | 3840m | 0m | 960m | 1920m | 2880m | SFT at 3840m |

- For two-H100 GRPO, minimum wall time remaining is
  `(target H100-minutes - checkpoint H100-minutes) / 120`.
- For one-H100 SFT, it is `(target - checkpoint) / 60`.
- The estimates below exclude startup, checkpoint rollback, and final
  MATH500/GSM8K/AMC evaluation overhead.
- `✅` means the target evaluation/snapshot is complete. `▶` means running,
  `⏳` means queued, `⏸` means deliberately held, and `⚠` means action is
  still required.
- A cell counts as complete only when the run logged a MATH500 pass@1 eval
  within the budget's +/-30m tolerance window (`budget_table.py` accepts an
  eval at `>= target - 1m`). A run that crashed past the target but never
  logged that eval is **not** complete and needs a resume-to-endpoint or an
  eval-resume.

## Executive status

- **57/60 cells** now have a durable endpoint eval. Since the 2026-07-16
  refresh, four formerly-open 64h cells closed: `k0oi912e` (g4b 0%),
  `pm6fdbzm` (l3b 0%), `klmbedtb` (l3b 50%), and `9ou3bxvy` (l8b 0%). The
  g12b 4h/16h 0% provenance caveat is resolved: the fresh `vmoh3nah` lineage
  logged clean 240m (16.5%) and 960m (58.5%) endpoints.
- **Three cells remain open**, all in the 64h row:
  - **g4b 64h/25% (`wpsjue9e`)** — was crashed at 3021m (ck1704 @3016m);
    **resumed** on Killarney as `4304033` (qos=high, 12h) at 19:35 EDT, now
    RUNNING. ~824m / ~6.9h min to the endpoint.
  - **g12b 64h/0% (`vmoh3nah`)** — crashed at 2286m (ck504 @2250m; ~1590m /
    ~13.25h min left). Now a two-job relay: fresh `4304051` (qos=high, 12h)
    resumes from ck504 and covers most of the remainder; `4278908` (24h) was
    re-chained `afterany:4304051` so it finishes the last stretch without ever
    running concurrently (no two-writer hazard).
  - **g12b 64h/25% (`mv7y3xap`)** — crashed at 3843m with its last eval at
    3553m (no endpoint). Eval-resume job `4303263` is RUNNING: it restored
    ck1144 @3827m, logged a baseline (MATH500 56.0%), and is training the last
    ~13m to cross 3840m so the forced endpoint eval logs.
- **Tag-loss fix (data integrity):** `k0oi912e` (g4b 0%) and `wpsjue9e`
  (g4b 25%) had lost the `try3` tag and carried a malformed `240.960` tag.
  `budget_table.py --tag try3` filtered them out, which blanked the **entire
  g4b 0% column** (not just the new cell) on regenerate. Re-added `try3` to
  both run IDs. `k0oi912e`'s history is also contaminated by a divergent second
  writer (~1879-2496m); its 240/960/3840 endpoints read clean, so the cell
  values are valid but its mid-run curve is not.

## Complete 60-cell grid

### Gemma-3 4B

| Total target | 0% SFT | 25% SFT | 50% SFT | 75% SFT | 100% SFT |
|---:|---|---|---|---|---|
| 4h / 240m | ✅ `k0oi912e` | ✅ `kqlan2ae` (251.78m) | ✅ `zhn17te2` (321.52m) | ✅ `2f22vn9u` (388.41m) | ✅ `dcoa9vsp` SFT snapshot |
| 16h / 960m | ✅ `k0oi912e` | ✅ `fgdhqk89` (960.42m) | ✅ `bohcwj92` (1034.42m) | ✅ `wkzlqrzk` (1035.81m) | ✅ `dcoa9vsp` SFT snapshot |
| 64h / 3840m | ✅ `k0oi912e` (crashed@3921m; 3840m eval landed, avg 33.5; `try3` re-added) | ▶ `wpsjue9e` resume `4304033` RUN (from ck1704 @3016m; qos=high 12h; ~6.9h to endpoint; `try3` re-added) | ✅ `8tdbsj8s` (4341m) | ✅ `8um6gxvd` (4112m) | ✅ `dcoa9vsp` (3862.82m) |

### Llama-3 3B

| Total target | 0% SFT | 25% SFT | 50% SFT | 75% SFT | 100% SFT |
|---:|---|---|---|---|---|
| 4h / 240m | ✅ `pm6fdbzm` | ✅ `xo8ppeei` (281.66m) | ✅ `87dprku7` (344.06m) | ✅ `refrrdkn` (403.60m) | ✅ `1swanogm` SFT snapshot |
| 16h / 960m | ✅ `pm6fdbzm` | ✅ `zrpfbnwa` (1165.03m) | ✅ `hbl3j927` (1066.89m) | ✅ `5p36cm1d` (960.95m) | ✅ `1swanogm` SFT snapshot |
| 64h / 3840m | ✅ `pm6fdbzm` (3864m; endpoint@3841m) | ✅ `gx6ux0m4` (3901m) | ✅ `klmbedtb` (3871m; endpoint@3841m) | ✅ `pwp7xcd8` (4102m) | ✅ `1swanogm` (3845m) |

### Gemma-3 12B

| Total target | 0% SFT | 25% SFT | 50% SFT | 75% SFT | 100% SFT |
|---:|---|---|---|---|---|
| 4h / 240m | ✅ `vmoh3nah` (240m@16.5%) | ✅ `clr8aodj` (271.69m) | ✅ `44vrscyh` (362.81m) | ✅ `xjxm20hk` (420.43m) | ✅ `hc8s6t6m` SFT snapshot |
| 16h / 960m | ✅ `vmoh3nah` (962m@58.5%) | ✅ `rr58zq3g` (1232.86m) | ✅ `s88emh6a` (1200.81m) | ✅ `fy6x3gou` (962.08m) | ✅ `hc8s6t6m` SFT snapshot |
| 64h / 3840m | ▶ `vmoh3nah` relay: `4304051` (12h, qos=high) from ck504 @2250m, then `4278908` (24h, `afterany:4304051`) finishes (~1590m total left) | ▶ `mv7y3xap` re-eval `4303263` running (ck1144 @3827m; baseline m500 56.0%; crossing 3840m) | ✅ `6oshr446` (3903m) | ✅ `inj34k54` (4119m) | ✅ `hc8s6t6m` (3847m) |

The old `0ntohyza` GRPO lineage is corrupted and not counted; `mv7y3xap` is its
clean replacement from the same 960m SFT checkpoint. The old 10k `0xy679uw`
lineage is superseded by the fresh full-25k `vmoh3nah` for the 0% column.

### Llama-3 8B

| Total target | 0% SFT | 25% SFT | 50% SFT | 75% SFT | 100% SFT |
|---:|---|---|---|---|---|
| 4h / 240m | ✅ `9ou3bxvy` | ✅ `8m6m2qkv` (240m) | ✅ `cskocfi6` (240m) | ✅ `kqf9wovo` (240m) | ✅ `0i93c6my` SFT snapshot |
| 16h / 960m | ✅ `9ou3bxvy` | ✅ `gwfoyk7s` (1428.85m) | ✅ `fqx0e3ym` (960m) | ✅ `ar368tkg` (961.65m) | ✅ `0i93c6my` SFT snapshot |
| 64h / 3840m | ✅ `9ou3bxvy` (3921m; endpoint@3842m) | ✅ `c80lwmn1` (3909m) | ✅ `ar9r252k` (3979m; W&B state failed but endpoint@3840m saved) | ✅ `haoeksck` (4450m) | ✅ `0i93c6my` (3938m) |

## Active/open-cell progress

Snapshot: **2026-07-17 19:10 EDT**. Live values came from each run's latest
W&B summary; checkpoint values came from local `trainer_state.json`
(`OffsetAwareWandbCallback.total_seconds`). GRPO jobs use two H100s, so minimum
wall time is checkpoint H100-minutes remaining divided by 120.

| Model/cell | Active lineage | Coverage | Live W&B total | Latest checkpoint | Target remaining from checkpoint | Status |
|---|---|---|---:|---:|---:|---|
| g4b 64h/25% | `wpsjue9e` | `4304033` RUN, 12h qos=high | 3021m (was crashed) | ck1704, 3016m | 824m / 6.87h | ▶ resuming to endpoint; `try3` re-added |
| g12b 64h/0% | `vmoh3nah` fresh 25k | `4304051` PD 12h + `4278908` PD 24h (`afterany:4304051`) | 2286m (crashed) | ck504, 2250m | 1590m / 13.25h | ▶ two-job relay queued |
| g12b 64h/25% | `mv7y3xap` | `4303263` RUN, eval-resume | 3827m | ck1144, 3827m | 13m to cross 3840m | ▶ endpoint eval imminent |

`mv7y3xap`'s eval-resume (`4303263`, qos=high, isolated `VLLM_CACHE_ROOT`) uses
a dedicated metadata copy
`tuning/models_metadata/eval_resume_mv7y3xap.jsonl` so the shared claim row is
not mutated. It restores ck1144 (3827m), runs the on-train-begin baseline eval
(does not satisfy the >=3839m endpoint reader), then trains the last few steps
to cross 3840m where the forced eval logs the countable endpoint.

## Current queue snapshot (Killarney, 19:10 EDT)

| Job | State | Limit | Lineage/cell | Notes |
|---:|---|---:|---|---|
| `4303263` | RUN | 1.5h | `mv7y3xap`, g12b 64h/25% eval-resume | ck1144 @3827m; crossing 3840m |
| `4304033` | RUN | 12h | `wpsjue9e` resume, g4b 64h/25% | from ck1704 @3016m; qos=high; started 19:35 |
| `4301482` | RUN | 12h | `[1]math-l8b` | unrelated l8b job on kn175 |
| `4304051` | PD | 12h | `vmoh3nah` resume, g12b 64h/0% | qos=high; resumes ck504 @2250m |
| `4278908` | PD | 24h | `vmoh3nah` finisher, g12b 64h/0% | re-chained `afterany:4304051`; kept, not cancelled |

Fir and Nibi were not re-audited (their SSH still requires interactive MFA);
their queues are historical context only — see the 2026-07-16 revision.

## Resume-bug / tag-loss lineage audit

The GRPO resume-policy fix is commit `3ba3a0c` (present in current HEAD). Newly
relevant this refresh:

| Lineage | Classification | Current disposition |
|---|---|---|
| `k0oi912e` | affected/recovered + tag-loss | completed; 3840m endpoint eval landed (avg 33.5); lost `try3` (malformed `240.960`), re-added; mid-run divergent-writer contamination ~1879-2496m, endpoints clean |
| `wpsjue9e` | unfinished + tag-loss | crashed@3021m; lost `try3`, re-added; needs ~824m resume |
| `pm6fdbzm` | unfinished continuation | completed; endpoint@3841m |
| `klmbedtb` | recovery/continued | completed; endpoint@3841m |
| `9ou3bxvy` | unfinished continuation | completed; endpoint@3842m |
| `vmoh3nah` | fresh full-25k lineage | crashed@2286m; successor `4278908` queued |
| `mv7y3xap` | clean replacement of `0ntohyza` | eval-resume `4303263` in flight to log 3840m endpoint |

Do not count or resume abandoned alternatives `ei6smspb`, `0ntohyza`, or
`h2yyuilo`.

## Recovery and data locations

Cluster roots:

- Killarney: `/project/6105902/shougan/balance-budget`
- Fir: `/scratch/shougan/balance-budget`
- Nibi: `/project/6105696/shougan/balance-budget`

Canonical grid metadata rows:
`tuning/models_metadata/[1]{gemma3-4B,gemma3-12B,llama3-3B,llama3-8B}-r128.json`.
Recovery row sets: `recovery_killarney_20260713/`, `recovery_grid_20260714/`,
`fir_clones_20260715/`, and (Fir) `recovery_fir_2026071{3,4}/`.

Restartable local checkpoints for the open cells:

- `tuning/models/gemma3-4B_math500-p@1-960m_sft-1041696_dcoa9vsp_rlvr-simplerl-25000_grpo_wpsjue9e/checkpoint-1704` (3016m)
- `tuning/models/gemma3-12B_math500-p@1-0m_sft-0_rbd754r8_rlvr-simplerl-25000_grpo_vmoh3nah/checkpoint-504` (2250m)
- `tuning/models/gemma3-12B_math500-p@1-960m_sft-404112_hc8s6t6m_rlvr-simplerl-25000_grpo_mv7y3xap/checkpoint-1144` (3827m)

SimpleRL caches: `tuning/data/datasets/rlvr-simplerl-25000` is the full 18,972
unique prompts (10k old + 8,972 unseen); `rlvr-simplerl-10000` is the old 10k.

The read-only W&B status helper is
`wandb_repair/audit_grid_status_20260715.py`.

## Next actions

1. Let eval-resume `4303263` finish; confirm the `mv7y3xap` 3840m endpoint eval
   logged, then regenerate the g12b raw tables and downstream to fill 64h/25%.
2. Monitor the `vmoh3nah` relay: `4304051` (12h, qos=high) covers ~1440 of the
   ~1590 H100-min remaining; `4278908` (`afterany:4304051`) picks up the last
   ~150 H100-min (~1.25h wall) plus endpoint-eval headspace. The `afterany`
   chain guarantees they never write concurrently.
3. Monitor `wpsjue9e` resume `4304033` (g4b 64h/25%); ~6.9h min to the endpoint
   under a 12h qos=high allocation. Regenerate g4b 64h/25% once it lands.
4. Re-establish Fir/Nibi SSH (interactive MFA) if their queues need auditing;
   cancel any duplicate writer whose Killarney copy is already running to avoid
   the `k0oi912e`-style divergent-writer contamination.
