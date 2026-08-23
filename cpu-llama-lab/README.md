# CPU llama.cpp Lab v5.4.9
## Resume brief (read this first)
- Goal: find a CPU-only Claude-Code/agent config that is simultaneously correct, responsive, and cheap-memory friendly.
- Quality veto: speed never wins unless deterministic repo-repair/tool-use tests pass; failed quants promote to better fidelity.
- Core rule: main sweeps use one worker per physical core in the selected locality; only one cheap model probes 1/2x,1x,2x (+4x KNL SMT).
- Context rule: vanilla calibration stops at 2K and promotes only usable candidates; 8K/16K/32K/64K are deeper follow-up work, never automatic calibration tax.
- Prefix rule: depth sweeps recycle one server prefix; repeated 0->N prefill is invalid and reuse is proved before expensive work.
- Bootstrap/decode rule: prove PP + F16 decode first; Q8/Q4 KV are separate ISA-sensitive ablations and cannot poison platform screening.
- Clock rule: every performance-bearing test samples loaded CPU MHz; authoritative sustained clocks below 90% of base are marked `clock-throttled` and cannot win.
- KNL fact: Granite4.1-3B measured ~2.4-2.9x TG from MCDRAM vs one-channel DDR while PP changed only ~2-4%.
- KNL target: cross the 16 GiB MCDRAM residency boundary, then attack remaining compute/dequant/kernel overhead.
- KNL quant rule: prioritize Q3_K_S full-HBM, one distinct IQ3, Q4_K_M quality/spill control, and at most one truly different low-bit surprise.
- KNL memory rule: strict DDR, strict MCDRAM, and MCDRAM-preferred spill/tiering are separate controls; repack/no-repack stays orthogonal.
- Calibration rule: default uses Granite350M causal platform probes + one Granite3B 512->2K promise check in ~5 benchmark minutes; `calibrate --full` restores legacy exhaustive sweeps.
- Primary models: Qwen3.6-35B-A3B first, Nemotron3.5-30B-A3B, Qwen3.8-27B, Qwen3.6-27B; live inventory selects representative quants.
- Qwen3.6 hypothesis: >10 TG/s at useful short context is the stretch target; Q3 full-HBM vs Q4 spill is the decisive causal comparison.
- Cross-platform/NUMA: KNL alone gets source patches/HBM; Xeon/EPYC/TR use legal native/ISA/repack/LTO/BLAS/NUMA controls, with physical cores counted per locality.
- Repro/stale rule: Python >=3.8; pin identical `LLAMA_REF=<SHA>` across hosts; release/session/run tags prevent stale rows from contaminating results.
- Run: `calibrate` for the ~5-minute platform decision pass; `ballpark` jumps straight to Qwen3.6-35B + Nemotron3.5 + an HBM-fitting dense Qwen3.8-27B check; `batch-report` salvages old batch evidence; `calibrate --full` restores exhaustive work.



## v5.4.9 HBM-fitting Qwen3.8 reality check + surplus-platform coverage

`./cpu-llama-lab.sh ballpark all` now includes a deliberately tiny dense Qwen3.8-27B `Q3_K_M` probe after Qwen3.6-35B and Nemotron. On KNL the Qwen3.8 probe is fail-closed unless the discovered GGUF is a `comfortable` or `nominal` full-MCDRAM fit; it will not silently turn the dense comparison into a DDR-spill result. With the current Unsloth inventory Q3_K_M is about 13.8 GB (~12.85 GiB), leaving roughly 3 GiB of the 16-GiB MCDRAM node before runtime allocations. The default dense test is only PP32/TG8 at depths 0/128 with F16 KV and three causal KNL passes: DDR baseline, strict MCDRAM, then combo/no-repack strict MCDRAM. `ballpark qwen38` runs only this check; `LAB_BALLPARK_Q38_REPACK=1` opts into the fourth repack load.

The same commands remain topology-driven rather than platform-specific scripts. Regression coverage now includes Threadripper 1920X/1950X/2990WX, Threadripper Pro, EPYC 7551P (Naples), Rome, Milan NPS, Genoa NPS, Broadwell-EP, and single/dual-socket Xeon. Main runs use physical cores in the selected locality; multi-NUMA/socket policies are derived from detected NUMA nodes/packages and ISA builds from detected feature flags.

## v5.4.8 big-model-first ballpark + batch salvage

`./cpu-llama-lab.sh ballpark` deliberately jumps to the actual target MoEs before any deep sweep. By default it tests exactly one quant per family: Qwen3.6-35B-A3B `UD-Q3_K_S`, Nemotron-3.5-Lightning-30B-A3B `MXFP4_MOE`, and dense Qwen3.8-27B `Q3_K_M`. Required downloads are announced separately from the benchmark budget. The default total benchmark budget is 300 seconds (`LAB_BALLPARK_BUDGET_S`) and is divided across the requested models; per-model subprocess timeouts are clipped to the remaining share so the budget is a real benchmark-work ceiling rather than only an ETA. `ballpark qwen`, `ballpark nemotron`, and `ballpark qwen38` spend the whole budget on one family.

Each Qwen3.6/Nemotron build/policy configuration loads the model once and asks `llama-bench` for PP128/TG32 at depth 0/512 with F16 KV; the dense Qwen3.8 reality check is deliberately smaller at PP32/TG8 and depth 0/128. On KNL the causal toggles are: base/no-repack DDR, base/no-repack MCDRAM (or MCDRAM-preferred spill when the quant cannot fit), combo/no-repack on the same memory tier, then combo+repack if budget remains. Thus the first three passes directly estimate the value of HBM/tiering and the KNL combo kernel on the real model; repack is the first optional pass pruned when time is tight. No batch factorial, agent test, extra quant, or 2K/8K/16K sweep is hidden behind this command. A model that clears PP>=20 and TG>=5 at depth 512 is marked worth deeper follow-up; TG>=10 is the responsive target. Results are written to `calibration/primary-ballpark-<model>.json` plus the normal JSONL evidence.

`./cpu-llama-lab.sh batch-report` reads *all* historical `batch-tune` JSONL rows, recovers real PP throughput from `raw[].avg_ts`, ranks `(batch,ubatch)` settings within each model/build/policy/thread group, and writes `calibration/batch-salvage.json` plus TSV. This specifically fixes misleading ad-hoc summaries that printed top-level `tps=0` even though `llama-bench` stored valid `avg_ts`. Legacy rows without per-run clock evidence are explicitly labeled as heuristic-only rather than publication-grade.

## v5.4.7 five-minute decision calibration

Vanilla `./cpu-llama-lab.sh calibrate` is now a bounded decision pass rather than a factorial research sweep. After required models/builds are present, its benchmark budget defaults to 300 seconds (`LAB_CAL_BUDGET_S`). It prints the complete upcoming model/quant/build/policy/depth/KV/batch plan and an ETA before each stage, then prints every individual pass before launch and a DONE row with wall time, throughput and loaded clock.

The default path uses Granite 4.0 350M Q4_K_M to establish load/decode safety, the sparse SMT probe, F16/Q8/Q4 KV capability, prefix reuse, and a small causal platform map. It then runs exactly one best platform configuration on Granite 4.1 3B Q4_K_M through 512->2048 using F16-preferred KV and fixed batch/ubatch; there is no 7-way batch search. A 2K result is promoted only when PP >=20 tok/s and TG >=5 tok/s; TG >=10 tok/s is labeled responsive. `calibration/promise.json` records the decision and projects 8K/16K cost. Vanilla calibration stops there.

On KNL the lean platform map is: KNL base/no-repack DDR, KNL base/no-repack strict MCDRAM, generic AVX2/no-repack strict MCDRAM, KNL combo/no-repack strict MCDRAM, and one combo/no-repack MCDRAM-preferred spill control. On generic CPUs it tests native no-repack, native repack, explicit AVX2, and one whole-machine NUMA/interleave control. Optional controls are the first to be pruned when the five-minute budget is tight.

`./cpu-llama-lab.sh calibrate --full` retains the v5.4.6 calibration behavior for research runs that deliberately want the full platform map, all four calibration models, batch tuning, shallow 8K and 16K continuation. Cold builds/downloads are explicitly announced and estimated separately from the five-minute benchmark budget; fast setup compiles only the minimal build variants and only `llama-bench`/`llama-server`.

## v5.4.6 per-test loaded-clock validation

CPU frequency is now a first-class benchmark invariant. Every direct `llama-bench` result and every persistent-server PP/TG endpoint is sampled while the workload is running; NUMA worker, agent-task, Claude-Code smoke, perplexity/review/EvalPlus quality paths also record loaded-clock evidence. Result rows retain min/mean/p90/max MHz, reference source, ratio and throttle status. A row with an authoritative reference that falls below `LAB_CLOCK_MIN_RATIO` (default 0.90 of base) becomes `clock-throttled` by default and is excluded from platform shortlists and winners while retaining its raw throughput for diagnosis.

`preflight` now runs a short physical-core loaded-frequency probe before an expensive sweep and writes it into `preflight.json`; `clock-check` exposes the same probe directly. The preferred reference is sysfs `base_frequency`; known stable base clocks are supplied for Xeon Phi 7250 (1.4 GHz), Threadripper 1950X (3.4 GHz), and 1920X (3.5 GHz); max-frequency-only references are recorded but deliberately non-gating. Override controls are `LAB_CLOCK_REFERENCE_MHZ`, `LAB_CLOCK_MIN_RATIO`, `LAB_CLOCK_MIN_MHZ`, `LAB_CLOCK_SAMPLE_MS`, `LAB_CLOCK_MAX_CPUS`, and `LAB_CLOCK_ENFORCE=0` for diagnostic-only collection.

Visible thread/KV/platform calibration summaries now print sampled PP/TG MHz alongside tok/s, and final reports carry clock columns. Standalone quality defaults to F16 KV rather than the known-broken KNL Q4 KV path. Agent temporary repositories are kept under the suite-local root rather than system `/tmp`.

## v5.4.5 Python 3.8 + cross-host reproducibility hotfix

Pop!_OS 20.04 exposed a Python-3.8 parse failure in `scripts/build.py`: a nested f-string was valid on newer Python but failed before CMake on the 1920X/1950X comparison host. The package is now Python-3.8-grammar clean. `selftest` now fails immediately if `py_compile`, shell syntax, unit tests, patcher tests, or the codegen verifier fail instead of printing a later PASS after an earlier syntax error.

`LLAMA_REF` now applies to an existing clean llama.cpp checkout as well as a fresh clone. This prevents a Threadripper host cloned from current `master` from silently being compared with a KNL host at a different SHA. First-generation 1920X/1950X are explicitly detected as Zen1 Threadripper, retaining the four-channel AVX2/native/no-repack/LTO platform matrix and the 1/2x,1x,2x cheap thread probe. The old bare `pro` substring check is fixed so the word `Processor` can no longer mislabel X399 as eight-channel Threadripper PRO. Cpuset hygiene now compares effective affinity count rather than textual CPU-ID ranges, avoiding false restriction warnings on kernels that expose offline/possible IDs.

## v5.4.4 decode-safe bootstrap + neutral KV baseline

Real KNL calibration exposed `rc=132` (SIGILL) only on TG while PP succeeded on the same `knl-base-norepack` binary. v5.4.4 therefore requires bootstrap to prove **both PP32 and a tiny F16-KV decode**, not merely model load/prefill. F16 KV is now the neutral quick-platform and thread-SMT baseline; Q8_0/Q4_0 are isolated in a cheap `kv-probe` and remain later finalist ablations rather than being allowed to invalidate otherwise-good CPU/build configurations.

The calibration order is bootstrap -> thread probe -> KV capability probe -> prefix-cache proof -> full 350M platform map. Exit status 128+signal is normalized (`132 => SIGILL/4`) in benchmark records. `node-workers` also no longer passes unsupported `-tb` to `llama-bench`. If the KNL decode backend SIGILLs even with F16 KV, bootstrap automatically advances to the conservative generic AVX2/no-repack control, turning that failure into ablation evidence instead of a global abort.

## v5.4.3 load-safe bootstrap + clean-run diagnostics

The calibration now establishes a runnable baseline before thread scaling, prefix-cache testing, or the platform factorial map. On KNL it prefers `knl-base-norepack`; if that fails it tries `knl-generic-avx2-norepack`, which disables both the KNL AVX-512 backend selection and CPU repacking while retaining a conservative AVX2 CPU path. Repacked KNL builds are then tested as ablations rather than assumed to be load-safe.

Server startup failures now record exit status/signal and distinguish an actual process crash from a startup timeout. Prefix smoke starts from the load-safe bootstrap combination, then tries bounded fallbacks across other small calibration models/builds/policies. Failure summaries only count the current experiment tag, platform shortlisting uses only the latest quick run, and the final calibration report uses rows created after the current calibration session started.

## v5.4.2 benchmark-command and prefix-smoke hotfix

- fixes a release-blocking bug where `llama-bench` was passed unsupported `-tb`, causing every quick PP trial to fail before inference;
- prefix-cache smoke now uses ordinary mmap/no-mmap first, tries safe placement fallbacks, uses F16 KV by default to isolate cache semantics from low-bit-KV support, and caps each smoke startup at 90 s;
- failed benchmark rows retain stderr tails and failure summaries show representative failing commands;
- all v5.4.1 KNL HBM-aware quant selection, physical-core defaults, isolated 34/68/136/272 KNL SMT probe, repack controls, and cross-platform Xeon/EPYC/TR ablations remain unchanged.

## v5.4.1 physical-core default + isolated SMT probe + representative quants

Main sweeps no longer cross every build/model/quant with a thread-count list. Each memory/NUMA policy receives exactly the physical-core count available inside that locality. Full-machine policies use all machine cores; `socketN-local` uses one socket's cores; NPS/SNC node-local policies use that node's estimated physical-core share. This prevents dual-socket controls from being accidentally oversubscribed.

`calibrate` runs one separate Granite 4.0 350M thread sanity probe: 1/2x, 1x, and 2x physical cores on ordinary SMT2 systems; KNL additionally tests all four hardware-thread occupancies, 34/68/136/272 on a 68-core 7250. The result is diagnostic only and is not propagated across every expensive candidate; set `LAB_THREAD_SURPRISE_THRESHOLD` (default 0.10) to control the reported-surprise threshold.

Primary-model quant planning is representative rather than exhaustive. KNL/MCDRAM defaults to one conventional Q3 full-HBM candidate, one different IQ3 family, a higher-quality Q4 spill control, and at most one genuinely different low-bit surprise. `LAB_KNL_EDGE_DUPLICATE_Q3=1` explicitly restores one edge-fit same-family Q3 experiment. Generic CPUs similarly keep one representative per useful low-bit family; `LAB_INCLUDE_HIGH_BIT_CONTROLS=1` or explicit `LAB_WEIGHT_QUANTS=...` restores Q6/Q8 controls when wanted.

## v5.4.0 KNL MCDRAM-capacity search + platform-pruned ablations

On KNL Flat mode, model planning is now **HBM-aware rather than bit-count-first**. The planner records raw GGUF size versus visible MCDRAM, reserves 1 GiB by default, labels each quant `comfortable`, `nominal`, `near`, or `oversize`, and keeps a small hypothesis-rich shortlist instead of every same-bit variant. Qwen3.6-35B-A3B explicitly prioritizes `UD-Q3_K_S`, `UD-IQ3_S`, `UD-Q3_K_M`, and one higher-quality `UD-Q4_K_M` partial-HBM control when present; other dense/MoE families are shortlisted dynamically from the live inventory so a surprising fit is not excluded.

KNL Flat mode also gets explicit `knl-ddr-strict`, `knl-mcdram-strict`, and `knl-mcdram-preferred` placement controls. Strict MCDRAM is only attempted for raw-weight candidates classified as comfortable/nominal fits; preferred-HBM is the controlled spill/tiering experiment for edge/oversize candidates. KNL builds now include `knl-base-norepack` and `knl-combo-norepack`, while every CPU retains explicit repack/no-repack and mmap/no-mmap controls.

To keep the factorial search tractable, `calibrate` first runs the complete build × memory/NUMA quick map on Granite 4.0 350M at the physical-core default; thread/SMT scaling is a separate four-point-or-smaller probe. It preserves mandatory baseline/repack/HBM controls and writes `calibration/platform-shortlist.json`; the larger calibration rungs and primary 27–35B models reuse that shortlist. Set `LAB_AUTO_CALIBRATE=0` to deliberately bypass this pruning.

Shortlists are release-versioned and the launcher validates that every non-optional build in the current plan actually has both `llama-bench` and `llama-server`. Upgrading over an older `.cpu-llama-lab` therefore triggers only the missing incremental builds instead of silently omitting new ablations.

`prefix-smoke` now proves persistent prompt-cache reuse before an expensive screen. A failure stops early and prints the cache/timing fields plus the server-log tail rather than ending later with an unexplained “no 8K winner.”

Useful KNL overrides: `LAB_KNL_MCDRAM_RESERVE_MIB`, `LAB_KNL_MCDRAM_NEAR_MIB`, `LAB_KNL_MCDRAM_QUANT_BUDGET`, `LAB_KNL_MCDRAM_CONTROL_MAX_RATIO`, and explicit `LAB_WEIGHT_QUANTS` (which disables automatic HBM shortlist selection).

## v5.3.5 agent/coding calibration ladder + cross-platform contract

The calibration ladder is now chosen for the actual target workload rather than generic small-model quality:

- **Granite 4.0 350M Q4_K_M** — sub-0.5B coding/tool-use floor;
- **Qwen3 0.6B Q4_0** — <=1B reasoning + agent/tool-use rung;
- **Granite 4.0 1B Q4_K_M** — stronger small coding/tool-use agent rung (IBM's H-1B label; ~1.5B active parameters);
- **Granite 4.1 3B Q4_K_M** — realistic small coding-agent anchor and direct comparison with the prior KNL/Ollama measurements.

`calibrate` still uses the exact same fixed GGUF quant for every CPU, the same <=8K screen, the same recycled prefixes, and the same one-winner 8K->16K continuation. It additionally runs the repository-repair/tool-use agent gate in **diagnostic-only** mode: pass/fail and latency are recorded, but a tiny calibration model failing quality does not abort timing calibration. The production `overnight`, `full-sweep`, and `full` workflows retain the hard quality/responsiveness gate.

Qwen-family agent-gate requests use non-greedy sampling (temperature 0.6, top-p 0.95, top-k 20) because Qwen3 explicitly warns that greedy decoding can degrade quality or repeat indefinitely. Other calibration families remain deterministic by default.

Cross-platform fairness is an explicit invariant. KNL alone receives the KNL source patch. Xeon, EPYC, Threadripper and other x86 CPUs use pristine upstream behavior and receive architecture-legal native/repack/LTO/OpenBLAS/ISA ablations. Multi-NUMA systems test node-local, whole-physical-socket (`socketN-local`), distribute and interleave policies where available. The context points and model/quants are unchanged across CPU platforms. `test_platforms` now regression-tests KNL, dual Xeon 8168, Milan NPS4, Genoa NPS4 and WRX80/Threadripper-Pro plan invariants.


## v5.3.4 agentic-coding winner gate

The primary objective is now **responsive agentic coding**, not maximum tokens/s in isolation. `full-sweep` and `overnight` use a funnel:

```text
all fitting model/quant/runtime candidates
  -> <=2K broad performance ablations
  -> recycled-prefix 512 -> 2K -> 8K screen
  -> reject candidates below the human-responsiveness floor
  -> deterministic Anthropic Messages API repository-repair/tool-use gate
  -> if a fast quant fails, promote higher-fidelity quants
  -> if installed, real `claude -p` repository-repair smoke
  -> exactly one quality-accepted winner per family
  -> recycled-prefix 8K -> 16K -> 32K -> 64K deep characterization
```

The default hard gate is intentionally demanding because the built-in fixtures are small: all fixtures must pass (`LAB_AGENT_MIN_PASS_RATE=1.0`), 8K PP must be at least 20 tok/s, 8K TG at least 3 tok/s, and median repair wall time at most 90 seconds. These are human-usability floors, not relative benchmark scores, and are configurable with `LAB_AGENT_MIN_PP_TPS`, `LAB_AGENT_MIN_TG_TPS`, and `LAB_AGENT_MAX_MEDIAN_TASK_S`. A slow candidate is rejected **before** spending wall time on the agent suite.

The built-in agent uses safe repository tools and does not expose arbitrary shell execution. When Claude Code is installed, the suite also launches an actual non-interactive `claude -p` repair against the candidate's local llama-server. An installed Claude Code integration that fails is a candidate failure; set `LAB_REQUIRE_CLAUDE_CODE=1` if the CLI's absence should fail the run as well. The older `quality` command remains a complementary perplexity/EvalPlus/human-review stage; HumanEval-style scores alone do not decide the production winner.

On multi-socket or multi-NUMA-per-socket hosts, v5.3.4 additionally maps NUMA nodes to physical packages and generates `socketN-local` policies. This prevents an EPYC NPS2/NPS4 or Intel SNC benchmark from confusing "one NUMA node" with "one complete socket" and allows the same script to test complete-socket locality versus cross-socket distribute/interleave.

## v5.3.3 calibration ladder (superseded by v5.3.5)

Before committing to the expensive primary sweep, run:

```bash
./cpu-llama-lab.sh calibrate
```

The original v5.3.3 ladder used Gemma 3 270M IT, Qwen3.5 0.8B, Gemma 3 1B IT, and Granite 4.1 3B. v5.3.5 replaces the first three with agent/coding-oriented calibration models. The calibration uses the same build/NUMA/thread/batch/KV search, persistent cached-prefix 512 -> 2K -> 8K screen, and exactly one winner continuation to 16K. The four families are calibration-only and are not repeated by ordinary `overnight` unless `LAB_INCLUDE_CALIBRATION=1` or they are explicitly named in `LAB_MODELS`.

After calibration, use:

```bash
./cpu-llama-lab.sh models-plan
./cpu-llama-lab.sh estimate
```

`estimate` combines the live primary model/quant plan with measured calibration wall times to provide a best-case planning bound before large GGUF downloads/benchmarking.

## V5.3.2 short-context-first unattended path

`./cpu-llama-lab.sh overnight` is the no-prompt benchmark/ablation path for a machine whose dependencies are already installed. It deliberately does **not** run package installation. It runs preflight, builds only if no benchmark binary exists, ABI/backend/codegen validation, automatic GGUF discovery/download, broad <=2K screening, batch tuning, recycled-prefix finalist screening through 8K, then exactly one deeper winner per model family. Interrupted runs are resumable by experiment-tagged per-measurement keys; `.cpu-llama-lab/overnight-state.json` records the current coarse stage while the JSONL result log provides fine-grained restart state.

Every actual depth sweep uses a persistent `llama-server` and recycled token-array prefix. Candidate finalists are screened only at 512 -> 2K -> 8K. At 8K the suite chooses exactly one combined quant/build/NUMA/thread/KV/batch winner for each model family. Only that winner is allowed to continue, by default 8K -> 16K -> 32K -> 64K. `cache_n` must prove prefix reuse; PP records only the newly evaluated suffix (`prompt_n`/`prompt_ms`) and TG128 is sampled at each endpoint. There is no repeated 0→N fill for measured depth points.

Remote model acquisition is also fail-soft per family/quant: one unavailable repository or repeatedly failed payload does not abort the rest of the night, while an experiment whose complete planned download set cannot fit available disk fails before payload transfer begins. Stalled `curl` transfers are bounded by a low-speed timeout and retain resumable `.part` files.

Default unattended safety bounds are configurable: `LAB_QUICK_TIMEOUT_S=180`, `LAB_BATCH_TIMEOUT_S=300`, `LAB_SERVER_START_TIMEOUT_S=300`, `LAB_PP_INTERVAL_TIMEOUT_S=1200`, `LAB_TG_ENDPOINT_TIMEOUT_S=180`, `LAB_MIN_QUICK_PP_TPS=10`, and `LAB_MIN_QUICK_TG_TPS=1`. Timeouts/failures/skipped higher-depth points are written to the result JSONL and report rather than silently omitted.

For a metadata/disk-size preview without downloading model payloads, run `./cpu-llama-lab.sh models-plan`.

# Detailed documentation

A pushbutton, architecture-aware CPU inference laboratory for `llama.cpp`, with a narrowly scoped, reversible Knights Landing (KNL / Xeon Phi x200) backend patch and a universal Xeon / EPYC / Threadripper tuning harness.

The project is intentionally split into two layers:

1. **`knl/` — upstream-oriented llama.cpp changes.** Applied only when the detected CPU is Knights Landing. These changes add a correct KNL AVX-512 feature bundle plus individually disableable q3/q4/q6 prefetch and q4_0 ILP experiments. Non-KNL source behavior is untouched.
2. **The lab harness — external tuning and validation.** Pristine upstream `llama.cpp` is used on Xeon, EPYC and Threadripper. The lab detects hardware, chooses legal compilers/ISA variants, searches NUMA/thread/batch policies, benchmarks PP/TG/KV, checks quality, audits ABI linkage, and produces reproducible reports.

That separation is deliberate: the KNL diff can be submitted upstream without coupling it to a large experimental framework.

## What v5.3 adds

V5.3 restores automatic GGUF discovery/download as a first-class part of the CPU sweep. V5.3.2 sets the default admission floor to **8,192 tokens**: the suite queries live Hugging Face tree metadata for the established four-model catalog, groups sharded GGUFs, rejects only candidates that cannot fit the 8K screen with the lowest-memory configured KV, downloads/reuses every fitting weight quant on ordinary CPUs; on KNL with visible MCDRAM it keeps a small HBM-capacity-aware shortlist plus a higher-quality spill control. All selected candidates live beneath `LAB_ROOT/models`, and one registry is shared across sweep, manifest, preflight, and quality commands. Manual `MODEL_*` paths remain optional controls rather than prerequisites.

## What v5.2.2 adds

V5.2.2 fixes the KNL q4_0 unroll generator against current upstream `//` comments inside the AVX2 hot loop. The unroller emits the loop body as a continued C macro; retaining a `//` line followed by `\` is invalid because backslash-newline splicing occurs before line-comment removal and can comment out following macro statements. The generator now strips `//` comments outside string/character literals before emitting continuation lines, and rejects source preprocessor directives or pre-existing continuation backslashes inside the copied loop body.

The patcher regression fixture now contains line comments inside the q4_0 loop and compiles with `GGML_KNL_Q4_0_UNROLL` enabled, covering the exact class of failure that made `knl-unroll`, `knl-combo`, and `knl-instrument` fail while prefetch-only variants succeeded.

## What v5.2.1 adds

V5.2.1 fixes a function-boundary bug exposed by llama.cpp commit `d775b8967a46d8beb110d444aa3b8938179e0dd8`. After `ggml_vec_dot_q6_K_q8_K()` upstream now places a global AVX/AVX2 preprocessor block before the next `void` function. The old chunker used the next `void` as the function boundary, accidentally absorbed that unrelated block, and reported two `__AVX2__` branches.

Function chunks are now bounded by their matching braces. The AVX2 selector remains fail-closed inside the true function body, and a regression fixture reproduces the exact post-q6 layout that caused the failure.

## What v5.2 adds

V5.2 fixes an additional current-upstream ambiguity in the x86 K-quant functions: q3_K/q4_K/q6_K can contain the same outer block loop in both the AVX2 implementation and a scalar/fallback branch. The KNL patcher now parses preprocessor branch nesting and injects prefetch/instrumentation only into the unique `__AVX2__` branch. It never resolves this ambiguity by arbitrary first/last-match selection.

This is specifically fail-closed: if the AVX2 branch itself has zero or multiple candidate model-block loops, or the chosen loop no longer references both `x[i]` and `y[i]`, `patch-check` stops without modifying the source tree.

## What v5.1 adds

V5.1 hardens the KNL source-patch integration against harmless upstream source reformatting while remaining fail-closed on real semantic changes:

- semantic/regex anchors instead of byte-for-byte CMake whitespace anchors;
- transactional patching: every source transformation validates before any upstream file is written;
- `patch-check` / `--dry-run` compatibility validation against the exact cloned llama.cpp revision;
- clean upstream-drift failures with the commit and log instead of a Python traceback;
- regression tests for current aligned CMake formatting and for all-or-nothing patch failure;
- KNL compilation uses `-march=knl -mtune=knl` directly.

## What v5 adds

V5 hardens the v4 workflow for current Linux distributions and modern Ubuntu kernels:

- kernel-matched `perf` installation/probing (`linux-tools-$(uname -r)` first on Ubuntu, with generic fallbacks);
- KNL compiler selection by **real C and C++ `-march=knl` probes**, not version strings alone;
- GCC 14/13/etc. coexists with the host's current GCC; the default compiler is never globally replaced;
- optional distro-native `memkind` installation only on KNL and only when the package actually exists;
- private pinned `uv` under `LAB_ROOT/tools/bin`, no shell-profile edits and no `$HOME` cache/tool pollution;
- suite-local ccache, uv and XDG caches;
- sanitized `CFLAGS`, `LDFLAGS`, `LD_LIBRARY_PATH`, `CMAKE_PREFIX_PATH`, include/library paths by default;
- sanitized external OpenMP/OpenBLAS/MKL thread/affinity variables for benchmark subprocesses by default;
- exact compiler/CMake/build metadata per variant;
- `apt-get check`, `dpkg --audit`, held-package recording on apt hosts;
- post-build `ldd` + `readelf` audit of every produced executable/shared GGML library;
- failure on unresolved dynamic libraries;
- required GLIBC/GLIBCXX version recording per binary;
- RPATH/RUNPATH recording;
- benchmark preflight covering load, swap, CPU governor, turbo/boost, automatic NUMA balancing, THP policy, cpuset restrictions, `perf`, disk/RAM pressure, model filesystem type, and KNL MCDRAM exposure;
- kernel command line, microcode, CPU vulnerability/mitigation state, and clocksource in the manifest;
- optional generic `perf stat` counters (`LAB_PERF=1`) after a real usability probe succeeds;
- an optional KNL `knl-combo-hbm` build when distro `memkind` metadata is available;
- strict publication mode (`LAB_STRICT=1`) that stops on high-severity preflight conditions;
- a richer debug bundle containing ABI, preflight, build-environment and quality evidence.

All suite-managed artifacts remain below `$PWD/.cpu-llama-lab` by default. System package installation is the only deliberate exception.

---

## Quick start

```bash
chmod +x cpu-llama-lab.sh

./cpu-llama-lab.sh plan
./cpu-llama-lab.sh install
./cpu-llama-lab.sh selftest
./cpu-llama-lab.sh preflight

# No MODEL_* variables are required. The suite queries Hugging Face metadata,
# rejects weight quants that cannot satisfy the usable-context RAM floor,
# downloads/reuses the fitting GGUFs beneath .cpu-llama-lab/models, and sweeps them.
./cpu-llama-lab.sh full
```

The default automatic model catalog is, in likely-fastest-first order:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`
- `unsloth/Qwen3.8-27B-GGUF`
- `unsloth/Qwen3.6-27B-GGUF`

Before downloading payloads, the suite queries the live Hugging Face tree API for GGUF filenames and byte sizes. `auto` considers every discovered model-weight GGUF quant, prioritizes conventional CPU-friendly K-quants for first results, and retains every candidate whose weights plus the lowest-memory configured KV cache can satisfy the default **8,192-token** screening floor. The estimated maximum planned context is recorded. Candidate search never exceeds 8K; only the one 8K winner per model family is asked to continue deeper, with a default 64K hard cap.

Inspect or prefetch the automatic set explicitly with:

```bash
./cpu-llama-lab.sh models
```

Manual model variables (`MODEL_Q36`, `MODEL_Q38`, `MODEL_NEMO`, `MODEL`, or `EXTRA_MODELS_FILE`) remain supported as additive controls. Set `LAB_AUTO_MODELS=0` only when you deliberately want a manual-only run. `EXTRA_MODELS_FILE` accepts `name|/absolute/path/model.gguf`, one entry per line.

For four independent machines:

```bash
LAB_WORLD=4 LAB_RANK=0 ./cpu-llama-lab.sh quick
LAB_WORLD=4 LAB_RANK=1 ./cpu-llama-lab.sh quick
LAB_WORLD=4 LAB_RANK=2 ./cpu-llama-lab.sh quick
LAB_WORLD=4 LAB_RANK=3 ./cpu-llama-lab.sh quick
```

A stable SHA-256 experiment key deterministically assigns non-overlapping benchmark work.

---

## KNL BIOS modes

### Primary explicit-placement configuration

Use:

- **MCDRAM: Memory (Intel Flat mode)**
- **Cluster mode: Quadrant**

On BIOSes that offer `Cache / Memory / Hybrid`, **Memory is Intel Flat mode**. Flat/Memory mode exposes MCDRAM as addressable memory and is required for rigorous HBM-versus-DDR placement experiments. The detector looks for one or more **memory-only NUMA nodes** and records an MCDRAM mode hint.

A one-NUMA-node KNL usually means MCDRAM is in **Cache mode** or otherwise not exposed as Flat memory. That is not an error: Cache + Quadrant is an important independent ablation, especially for sparse MoE models whose total model exceeds 16 GiB but whose frequently reused expert working set may be smaller.

Changing MCDRAM/cluster mode requires a reboot. Compiler, thread, NUMA, quant, KV and benchmark ablations do not.

SNC-2/SNC-4 should be treated as a later topology experiment after Quadrant is characterized.

---

## Compiler and distro policy

### KNL

GCC 15 removed Knights Landing target support. V5 therefore searches installed GCC 14/13/12/etc. and accepts a compiler only after **both C and C++ source files compile with `-march=knl`**.

On apt systems, if no installed compiler passes, the installer tries the newest available distro package from GCC 14 downward. It does **not** add PPAs or repositories from another Ubuntu release.

The intended configuration on a modern Ubuntu KNL is therefore perfectly normal:

```text
system compiler: current distro GCC
KNL compiler:    gcc-14 / g++-14 selected only for KNL builds
system glibc:    current distro glibc
system kernel:   current distro kernel
libnuma/BLAS:    current distro libraries
```

No `update-alternatives` changes are made.

### Xeon / EPYC / Threadripper

The current system compiler is preferred with `GGML_NATIVE=ON`. Explicit ISA-ceiling builds are generated independently so native dispatch is tested rather than trusted blindly.

### Why this avoids library mismatch problems

The suite compiles `llama.cpp` **on the target host against that host's headers and shared libraries**. It does not import an old glibc or copy compiler-runtime libraries from another distro release.

After each build, `abi-audit` runs `ldd` and `readelf` against the produced binaries and records:

- unresolved shared objects;
- maximum required `GLIBC_*` symbol level;
- maximum required `GLIBCXX_*` symbol level;
- RPATH/RUNPATH entries.

An unresolved dependency makes `abi-audit` fail.

---

## Package installation details

`install` detects apt/dnf/pacman and installs only normal build/measurement dependencies from the configured repositories.

On Ubuntu/apt it attempts `perf` tools in this order:

1. `linux-tools-$(uname -r)` when the exact package exists;
2. `linux-tools-generic`;
3. `linux-tools-common`.

It then runs a real:

```bash
perf stat -e cycles,instructions -- true
```

probe and records `ok`, `permission-or-policy`, `installed-but-unusable`, or `missing`.

`perf` failure is not confused with a llama.cpp failure.

### memkind

On KNL only, the installer checks whether the host package manager actually provides `libmemkind-dev` / `memkind-devel`. When available, the plan can build `knl-combo-hbm` with `GGML_CPU_HBM=ON`. If unavailable, NUMA-based Flat-memory placement remains usable and memkind-specific experiments are simply skipped.

### uv / EvalPlus

By default v5 installs a pinned standalone `uv` release beneath:

```text
.cpu-llama-lab/tools/bin/
```

using an unmanaged installation that does not modify shell profiles. Caches and EvalPlus environments live under:

```text
.cpu-llama-lab/cache/
.cpu-llama-lab/venvs/evalplus/
```

EvalPlus defaults to a pinned `evalplus==0.3.1`; override deliberately with `EVALPLUS_SPEC=...`.

Use `LAB_USE_SYSTEM_UV=1` only when deliberately choosing an existing system `uv`.

---

## Reproducible build environment

External variables that commonly cause accidental compiler/library mismatches are **not inherited by default**:

```text
CFLAGS
CXXFLAGS
CPPFLAGS
LDFLAGS
CPATH
C_INCLUDE_PATH
CPLUS_INCLUDE_PATH
LIBRARY_PATH
LD_LIBRARY_PATH
PKG_CONFIG_PATH
CMAKE_PREFIX_PATH
```

If any are present, preflight records them and `build.py` sanitizes them.

To deliberately test a custom toolchain/library environment:

```bash
LAB_INHERIT_BUILD_ENV=1 ./cpu-llama-lab.sh build
```

The resulting report prominently records that the environment was inherited.

Ccache is local to the lab root and uses compiler-content checking:

```text
CCACHE_DIR=.cpu-llama-lab/cache/ccache
CCACHE_COMPILERCHECK=content
```

This prevents unrelated user ccache state from being part of the experiment.

---

## Runtime environment hygiene

Benchmark subprocesses similarly remove externally inherited:

```text
OMP_NUM_THREADS
OMP_PROC_BIND
OMP_PLACES
OPENBLAS_NUM_THREADS
GOTO_NUM_THREADS
MKL_NUM_THREADS
KMP_AFFINITY
```

unless explicitly requested with:

```bash
LAB_INHERIT_RUNTIME_ENV=1 ./cpu-llama-lab.sh quick
```

The variables seen before sanitization are preserved in each benchmark record.

This matters because an unnoticed `OPENBLAS_NUM_THREADS=1` or stale OpenMP affinity setting can completely change PP results.

---

## Preflight / benchmark hygiene

Run:

```bash
./cpu-llama-lab.sh preflight
```

It records and evaluates:

- current load average;
- swap enablement and actual swap use;
- cpufreq driver/governor per policy;
- turbo/boost state when exposed by sysfs;
- kernel automatic NUMA balancing;
- Transparent Huge Page enable/defrag policies;
- process cpuset restrictions;
- `perf` usability and `perf_event_paranoid`;
- available RAM relative to configured model size;
- free disk space under `LAB_ROOT`;
- network/distributed model filesystems;
- KNL visible MCDRAM nodes/mode hint;
- potentially contaminating build environment variables.

Warnings are preserved in `preflight.json` and `preflight.md`.

For publication-style runs:

```bash
LAB_STRICT=1 ./cpu-llama-lab.sh full
```

High-severity conditions such as active swap, heavy system load or critically low memory/disk space then stop the run rather than quietly contaminating the result.

The suite records rather than automatically changing governor, turbo, THP, automatic NUMA balancing, or kernel mitigations. Those are system-wide policy choices and silently changing them would make results less auditable.

---

## CPU / ISA plan

The detector records actual CPUID flags and emits only legal experiments.

| Platform | Typical generated ISA ablations |
|---|---|
| KNL | dedicated `AVX512_KNL` base; prefetch distances; q4_0 unroll; optional HBM/memkind build |
| Broadwell-EP / Zen 2 / Zen 3 | native vs explicit AVX2 |
| Skylake-SP | native vs AVX2 vs AVX-512 |
| Cascade Lake | above + VNNI when exposed |
| Ice Lake | native/AVX2/AVX-512/VNNI as supported |
| Sapphire/Emerald/Granite Rapids | native plus legal AVX-512/VNNI/BF16/AMX paths |
| Zen 4 / Zen 5 | native vs AVX2 vs legal AVX-512 family paths |

Every ordinary platform also gets applicable repack ON/OFF, LTO and OpenBLAS PP candidates.

“Optimal” means the best empirically observed configuration in the tested search space for the supplied model/workload, not a claim that one thread count or backend is universally best.

---

## NUMA search

On one NUMA node the plan remains simple. On multiple nodes, it adds:

- upstream `--numa distribute` with mmap / first-touch;
- distribute + `--no-mmap`;
- `numactl --interleave=all` + `--numa numactl`;
- strict node-local runs for every CPU NUMA node;
- weights-only mirror only if the actual built CLI exposes it;
- independent node-local workers as an aggregate-throughput control.

Interleave, independent workers and true mirrored weights are never mislabeled as equivalent.

### Cold page-cache NUMA validation

For an otherwise idle lab machine:

```bash
LAB_DROP_CACHES=1 ./cpu-llama-lab.sh quick
```

This runs `sync` and drops page cache only before experiments explicitly marked as cold-first-touch tests. It is off by default because it affects the entire host.

---

## KNL source patch

The patch defines:

```text
GGML_AVX512_KNL = AVX512F + AVX512CD + AVX512ER + AVX512PF (+ PREFETCHWT1 capability)
```

rather than generic upstream `GGML_AVX512`, whose feature bundle assumes later AVX-512 subsets such as BW/DQ/VL that KNL does not implement.

Independent KNL configurations include:

- architecture-correct base;
- q3_K/q4_K/q6_K software prefetch;
- prefetch distance 1/2/4;
- q4_0 four-block independent-accumulator unroll;
- combined candidate;
- combined + memkind/HBM when available;
- first-call instrumentation (`GGML_KNL_TRACE=1`).

The patcher is idempotent and reversible.

Generate the isolated upstream diff with:

```bash
./cpu-llama-lab.sh export-knl-patch
```

The universal lab logic is intentionally not included in that diff.

---

## Correctness / ABI / code generation proof

A performance win is accepted only after multiple layers of evidence:

1. Python/shell/planner/hardening unit tests.
2. KNL patch apply/check/reapply/revert test.
3. Standalone q4 numerical comparison.
4. GCC KNL target compilation and macro audit.
5. Disassembly audit proving expected KNL code generation and rejecting forbidden KNL ISA assumptions.
6. Upstream `test-backend-ops` for every successful build.
7. Post-build ABI audit (`ldd`, `readelf`, GLIBC/GLIBCXX, RPATH/RUNPATH).
8. Compile commands and build metadata retained per build.
9. Independent optimization-off controls.
10. Per-model quality gates.

Useful commands:

```bash
./cpu-llama-lab.sh selftest
./cpu-llama-lab.sh backend-tests
./cpu-llama-lab.sh abi-audit
./cpu-llama-lab.sh codegen-audit
./cpu-llama-lab.sh debug-bundle
```

---

## Optional perf counters

After `install` has proven `perf stat` works, enable generic counters with:

```bash
LAB_PERF=1 ./cpu-llama-lab.sh full-sweep
```

Default events are:

```text
cycles,instructions,cache-misses,branches,branch-misses,
context-switches,cpu-migrations,page-faults
```

Override with `LAB_PERF_EVENTS=...`.

This is opt-in because `perf` adds measurement overhead and some systems restrict counters.

---

## PP/TG sweep semantics

Quick pruning records `pp512` once per exact configuration plus a short-context TG checkpoint.

Batch/ubatch tuning uses **PP1024**, deliberately outside the publication progression, so tuning does not duplicate a reported depth.

Full finalist PP then covers only the remaining configured depths:

```text
512, 2048, 8192 (screen), then 16384, 32768, 65536 for the single family winner
```

TG is measured at configured depths for `q4_0`, `q8_0` and `f16` KV by default. Deep configurations that fail/OOM remain explicit failures rather than disappearing from the dataset.

---

## Quality gates

`agent-quality` is the hard **pre-deep** gate for the unattended/full sweep. It evaluates real repository repair, tool use, test immutability, wall time, PP/TG responsiveness, quant fallback, and (when installed) Claude Code itself. A model family that cannot produce an acceptable quant gets no deep sweep.

`quality` is the complementary post-selection/reference suite and performs, where supported:

- fixed-corpus perplexity for every configured model/quant;
- deterministic coding/debug/repository-review prompts;
- exact model file hash in quality records;
- HumanEval+ generation through pinned EvalPlus;
- MBPP+ additionally under `QUALITY_LEVEL=weekend`;
- blind-by-default human response scoring.

```bash
./cpu-llama-lab.sh quality
./cpu-llama-lab.sh judge
```

The blind TUI uses persistent randomized candidate aliases until explicitly revealed.

---

## One-DIMM KNL design

A strong four-sled physical sweep remains:

| sled | populated DDR channels |
|---|---:|
| A | 1 |
| B | 2 |
| C | 4 |
| D | 6 |

Keep CPU, DIMM type/speed, BIOS, model/quant and software revision otherwise identical.

The key proof is not just STREAM bandwidth. It is the context-dependent inference transition:

1. weights + hot KV entirely in MCDRAM → TG should be comparatively insensitive to DDR population;
2. controlled KV/weight spill into DDR → TG becomes increasingly DDR-bandwidth-sensitive;
3. one-channel versus six-channel curves diverge exactly where spill begins.

That directly demonstrates why KNL can be useful with minimal DRAM population when the inference-critical working set fits MCDRAM.

---

## Commands

```text
plan             hardware/NUMA/ISA detection and deterministic experiment plan
install          distro dependencies, legal compiler selection, private uv, perf probe
preflight        benchmark-hygiene checks
manifest         exact OS/compiler/library/kernel/model metadata
fetch            clone/reuse upstream llama.cpp
patch-check      validate all KNL patch anchors without modifying source
build            architecture-specific build ablations
models           live HF inventory + RAM fit gate + automatic GGUF download/reuse
abi-audit        dynamic-link/GLIBC/GLIBCXX audit; fails unresolved libraries
backend-tests    upstream test-backend-ops
codegen-audit    compile_commands/build metadata/disassembly
export-knl-patch isolated upstream-oriented KNL diff
quick            broad PP512/TG<=2K pruning sweep
batch-tune       PP1024 batch/ubatch tuning
full-sweep       cached-prefix <=8K finalist screen + one <=64K deep winner per model family
node-workers     independent per-NUMA-node replica throughput control
quality          perplexity + coding review + EvalPlus generation
judge            blind human scoring TUI
report           final Markdown report
selftest         package + KNL + hardening tests; no model required
debug-bundle     archive hardware/preflight/ABI/build/results evidence
overnight        unattended benchmark/ablation pipeline (no package install)
full             complete pushbutton pipeline including install/quality
```

---

## Important environment controls

```text
LAB_ROOT=/path                 default: $PWD/.cpu-llama-lab
LAB_STRICT=1                   fail on high-severity benchmark preflight issues
LAB_INHERIT_BUILD_ENV=1        deliberately retain custom compiler/link environment
LAB_INHERIT_RUNTIME_ENV=1      deliberately retain OMP/BLAS runtime environment
LAB_DROP_CACHES=1              disruptive cold first-touch NUMA experiment
LAB_PERF=1                     attach generic perf stat counters
LAB_HASH_MODELS=1              hash models in manifest (quality hashes them regardless)
LAB_AUTO_MODELS=0              disable automatic catalog; use only manual MODEL_* inputs
LAB_MODELS=alias[,alias...]    restrict automatic catalog (default: all four families)
LAB_WEIGHT_QUANTS=auto         all discovered fitting GGUF weight quants (or comma-list exact quants)
LAB_MODEL_MIN_CTX=8192         hard screen-fit floor for pre-download admission
LAB_SCREEN_CAP=8192             maximum context before per-family winner selection (hard-capped at 8192)
LAB_DEEP_CAP=65536              default maximum context for the one deep winner per family
LAB_DEEP_DEPTHS=16384,32768,65536
LAB_WINNER_TG_WEIGHT=0.65       normalized winner score weight for TG; PP receives the remainder
LAB_MODEL_RESERVE_MIB=N        RAM reserve override; default is max(4 GiB, 5% of physical RAM)
LAB_MODEL_PREFLIGHT_OVERHEAD_MIB=N optional extra conservative RAM allowance before download
LAB_REFRESH_MODELS=1           refresh HF metadata/recompute registry instead of reusing it
LAB_WORLD=4 / LAB_RANK=0..3    deterministic multi-machine sharding
UV_VERSION=...                 override pinned private uv installer version
EVALPLUS_SPEC=...              override pinned EvalPlus package specification
```

---

## Publication / upstream checklist

Before publishing a result or opening a KNL performance PR, preserve at least:

- `hardware.json`
- `plan.json`
- `manifest.json`
- `preflight.json`
- `build-environment.json`
- `llama-commit.txt`
- `abi/ABI.md`
- `codegen/*compile_commands.json`
- relevant `objdump`
- `test-backend-ops` logs
- raw benchmark JSONL
- model SHA-256 from quality output
- final `REPORT.md`

For the upstream KNL PR itself, submit only the focused KNL backend/source changes and associated correctness/codegen evidence. Do not mix universal installer/tuning changes into the same patch.
