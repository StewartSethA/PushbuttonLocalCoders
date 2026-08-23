# v5.4.9

- add Qwen3.8-27B `Q3_K_M` to `ballpark all` plus `ballpark qwen38`; on KNL this dense reality-check is fail-closed unless the discovered GGUF fits wholly inside visible 16-GiB MCDRAM;
- keep the dense 27B probe intentionally tiny: PP32/TG8 at depth 0 and 128, F16 KV, with only DDR -> strict-MCDRAM -> combo/no-repack causal passes by default; repack is opt-in for Qwen3.8;
- prefer current Unsloth Qwen3.8 `Q3_K_M` (~13.8 GB / ~12.85 GiB) over unnecessarily lower Q3_K_S while retaining >2 GiB raw MCDRAM headroom in the 16-GiB fit test;
- broaden EPYC generation detection to Naples/Rome/Milan/Genoa/Turin SKU families including F-series forms; add explicit regression coverage for EPYC 7551P, Rome 7742, X399 1920X/1950X/2990WX, Broadwell-EP, single- and dual-socket Xeon, Milan/Genoa NPS, and Threadripper Pro;
- preserve one script and one physical-core/NUMA-aware planning path across KNL, Threadripper, EPYC, and 1+ socket Xeon; architecture-specific ISA/memory policies are selected from detected hardware rather than separate scripts.

# v5.4.8

- added `ballpark [all|qwen|nemotron]`: a ~5-minute big-model-first smoke that goes directly to Qwen3.6-35B-A3B `UD-Q3_K_S` and Nemotron-3.5-Lightning `MXFP4_MOE`;
- each ballpark config uses one llama-bench model load to measure PP128/TG32 at depth 0 and 512 with F16 KV; KNL tests HBM/tiering, combo-kernel, and repack toggles causally;
- ballpark is budget-aware: the first three causal KNL passes are mandatory, repack is optional/pruned first, and no deeper/quality/factorial work runs automatically;
- added `batch-report` to salvage historical batch-tune rows by reading llama-bench `raw[].avg_ts`, rank `(batch,ubatch)`, and preserve clock-unknown legacy evidence as heuristic-only;
- added benchmark-only build setup (`LAB_BENCH_ONLY_BUILD=1`) so ballpark does not compile llama-server/CLI/perplexity/test binaries it does not use;
- primary model ballpark prints complete model/quant/config/depth/KV/pass plans, cold-read ETA assumptions, per-pass progress, loaded clock, and promotion decisions before doing anything deeper.

# v5.4.7

- Make `calibrate` a deliberately lean ~5-minute decision pass after required builds/models are present; `calibrate --full` restores the v5.4.6 legacy factorial behavior.
- Default calibration uses only Granite 4.0 350M Q4_K_M for platform/thread/KV/prefix causality and Granite 4.1 3B Q4_K_M for one representative cached-prefix 512->2K usability check. Remove Qwen-0.6B, Granite-1B, 7-way batch tuning, 8K/16K depth work, and agent diagnostics from vanilla calibration.
- Add a five-config KNL causal map (same-backend DDR/HBM, generic-vs-KNL backend, base-vs-combo kernel, one preferred-HBM spill control) and a four-config generic CPU map (native no-repack/repack, explicit AVX2, one NUMA/interleave control).
- Add selective fast builds via `LAB_BUILD_NAMES`; fast calibration builds only `llama-bench` and `llama-server` for the minimal build set. Cold build/download time is announced separately from the benchmark budget.
- Print an explicit calibration overview and, before every stage/pass, model/quant, build, memory/NUMA policy, threads, depth, KV type, batch/ubatch, token count, timeout, and ETA. Print DONE rows with status, tok/s, wall time, and sampled loaded MHz.
- Add a hard fast-calibration wall budget (`LAB_CAL_BUDGET_S`, default 300 s). Optional spill/NUMA controls are pruned first to reserve time for the 3B promise check.
- Define the default promotion gate at 2K as PP >=20 tok/s and TG >=5 tok/s; TG >=10 tok/s is the responsive target. Write `calibration/promise.json` with the decision and projected standalone 8K/16K cost; default calibration does not launch those deeper depths.
- Remove interrupted-selftest orphan risk by giving fake test servers Linux parent-death cleanup; when invoked through the harness, test temporary directories remain under the suite-local root.

# v5.4.6

- Make loaded CPU frequency a first-class benchmark invariant. Every direct `llama-bench` row and every measured persistent-server PP/TG request records min/mean/p90/max MHz, reference source, loaded/reference ratio, and throttle status.
- Default fail-closed behavior: an otherwise successful performance row becomes `clock-throttled` when an authoritative base-frequency reference exists and loaded p90 is below `LAB_CLOCK_MIN_RATIO` (default 0.90). Throttled rows are retained as evidence but cannot enter quick finalists, platform shortlists, thread winners, or deep-model candidate selection.
- Add a short physical-core loaded-frequency probe to `preflight` and a standalone `clock-check` command. Preflight evidence is carried into manifests/calibration reports.
- Prefer sysfs `base_frequency`; add stable nominal-base fallbacks for Xeon Phi 7250 (1400 MHz), Threadripper 1950X (3400 MHz), and 1920X (3500 MHz). Max-frequency-only references are recorded but intentionally non-gating.
- Add `LAB_CLOCK_SAMPLE_MS`, `LAB_CLOCK_MAX_CPUS`, `LAB_CLOCK_MIN_RATIO`, `LAB_CLOCK_MIN_MHZ`, `LAB_CLOCK_REFERENCE_MHZ`, and `LAB_CLOCK_ENFORCE` controls.
- Thread, KV, platform-shortlist, calibration, and final report outputs now surface loaded MHz next to throughput. Agent repo-repair tasks, Claude-Code smoke, NUMA worker throughput, perplexity/review/EvalPlus quality paths also record clock evidence.
- Standalone `quality` now defaults to F16 KV rather than hard-coded Q4 KV, avoiding the known KNL Q4-KV SIGILL. Agent temporary repositories stay under the suite-local root instead of system `/tmp`.
- Add clock regression tests, Python syntax coverage for the clock helpers, and dynamic usage-version output.

# v5.4.5

- Fix Python 3.8 parse failure in `scripts/build.py` caused by a nested f-string; validate the complete Python source tree against the Python 3.8 grammar.
- Make `selftest` fail closed on py_compile, shell syntax, unit-test, KNL patcher, or codegen-verifier failure.
- Honor `LLAMA_REF` on an existing clean llama.cpp checkout, repinning to the exact resolved commit; refuse to repin a dirty checkout.
- Detect Threadripper 1920X/1950X explicitly as Zen1 and retain four-channel generic CPU comparison semantics; fix the old bare `pro` substring test that mistook the word `Processor` for Threadripper PRO and reported 8 channels on X399.
- Avoid false cpuset warnings when `/proc/self/status` lists possible/offline CPU IDs beyond the online topology; compare effective affinity count instead.
- Preserve all v5.4.4 F16-neutral platform mapping, KV SIGILL isolation, KNL MCDRAM/Q3 prioritization, agent-quality gates, and sparse SMT probing.

# v5.4.4

- Make bootstrap prove both PP32 and a tiny F16-KV decode; PP-only success can hide a KNL decode/GEMV SIGILL.
- Use F16 KV for the neutral quick platform map and the isolated 1/2x,1x,2x,4x KNL thread probe; quantized KV no longer gates CPU/build viability.
- Add `kv-probe`: cheaply test F16, Q8_0, and Q4_0 decode on the known-good baseline and record throughput, exit code, and signal.
- Normalize wrapped shell exit codes such as 132 to SIGILL(4) in benchmark records.
- Add the actual `bootstrap` sweep-mode dispatch (the previous explicit shell stage relied on the later thread/prefix call to invoke bootstrap indirectly).
- Remove the remaining unsupported `-tb` use from `node-workers` llama-bench invocations.
- Keep quantized KV as a later finalist ablation; Q3/IQ3/Q4 weight, MCDRAM, repack, NUMA, agent-quality, and representative-quant policy are unchanged.

# v5.4.3

- Add a pre-sweep load bootstrap: prefer `knl-base-norepack`, then a conservative `knl-generic-avx2-norepack` fallback, before trying repacked KNL paths. A model-load crash becomes an ablation result instead of aborting calibration.
- Add the generic AVX2/no-repack KNL control to the build matrix and platform shortlist when it actually succeeds.
- Prefix smoke now reuses the known-load-safe model/build/policy first, can fall back across small calibration models/builds, and records server exit code/signal versus startup timeout.
- Current-run diagnostics are isolated by experiment tag; stale v5.4.1 `-tb` failures no longer pollute v5.4.2+ failure summaries.
- `platform_shortlist.py` consumes the latest quick run only and never promotes a mandatory control that already failed on the 350M platform map.
- `calibrate` writes a session start marker so the final calibration report cannot silently mix old benchmark rows.
- Thread sanity probing uses the load-safe bootstrap configuration and prints PP/TG statuses as well as rates.

# v5.4.2

- Remove invalid `-tb` from every llama-bench command; current llama-bench exposes only `-t`.
- Run prefix-cache protocol smoke on ordinary memory placement before MCDRAM-specific ablations; use F16 KV by default, try safe placement fallbacks, and cap smoke startup at 90 seconds.
- Persist stderr tails for failed timed benchmark subprocesses and include examples in failure summaries.
- Preserve v5.4.1 thread/quant/HBM/repack search policy.

# v5.4.1
- Main sweeps now use one worker per physical core available to each NUMA locality; no default thread-count cross-product.
- Added one cheap Granite350M thread/SMT probe: 1/2x,1x,2x physical cores, plus 4x on KNL (34/68/136/272 on Xeon Phi 7250).
- Thread-probe surprises are reported, not automatically propagated to expensive models.
- Dual-socket/socket-local policies now use per-locality physical-core counts, preventing silent oversubscription.
- KNL Qwen3.6 shortlist defaults to Q3_K_S + IQ3_S + Q4_K_M plus at most one genuinely different low-bit surprise; same-family edge Q3 is opt-in.
- Generic CPUs now use one representative per useful low-bit quant family; Q6/Q8/F16 controls are opt-in.

# v5.4.0

- Make KNL/MCDRAM capacity a first-class quant-selection axis. Live GGUF candidates are annotated with visible-HBM bytes, reserve/headroom, and `comfortable`/`nominal`/`near`/`oversize` fit classes.
- Qwen3.6-35B-A3B gets a small hypothesis-rich KNL shortlist: `UD-Q3_K_S` full-HBM conventional control, `UD-IQ3_S` comfortable alternate-format surprise, `UD-Q3_K_M` higher-fidelity edge fit, and `UD-Q4_K_M` partial-HBM quality control when present. Other model families use the same dynamic quality/fit/format/spill logic instead of same-bit spam.
- Add strict DDR, strict MCDRAM, and MCDRAM-preferred spill/tiering runtime policies on KNL Flat mode. Strict HBM is attempted only for raw-weight fits; spill controls remain measurable without pretending they are fully resident.
- Add KNL `base` and `combo` no-repack controls; generic CPUs retain native repack/no-repack, mmap/no-mmap, LTO/BLAS/legal-ISA and full-socket NUMA controls for fair cross-platform comparison.
- Add one-time platform calibration: Granite 350M maps the full build × memory/NUMA × thread space, then `platform-shortlist.json` carries the best configurations plus mandatory controls into the larger calibration and primary-model sweeps.
- Version platform shortlists and validate the complete required build set before calibration/overnight, so upgrading an existing v5.3.x tree incrementally builds newly introduced ablations instead of silently reusing an incomplete old matrix.
- Add `prefix-smoke` to prove persistent llama-server prefix reuse before expensive work and emit raw cache/timing/server-log diagnostics on failure.
- Normalize cache accounting across `timings.cache_n/prompt_n`, top-level `tokens_cached`, and `prompt_progress`, avoiding false cache-reuse failures when a llama-server response path exposes only one schema.
- Raise the default agent-quality fallback budget to four candidates per family so the four-member KNL HBM shortlist can promote through quality levels before rejecting a model family.
- Refresh KNL/Ollama Granite historical calibration anchors with the measured DDR-vs-MCDRAM A/B: TG 2.4–2.9x faster in MCDRAM while first-run PP changes only ~2–4%.
- Add/update tests for HBM-aware quant shortlisting, KNL Flat memory-tier policies, repack/load controls, platform shortlist behavior, prefix-smoke, and cross-platform context invariants.

# v5.3.5

- Replaced the generic calibration ladder with agent/coding-oriented fixed GGUFs: Granite 4.0 350M Q4_K_M, Qwen3 0.6B Q4_0, Granite 4.0 1B Q4_K_M, and Granite 4.1 3B Q4_K_M.
- Bumped model catalog revision so stale calibration registries are automatically invalidated and re-planned.
- `calibrate` now runs the Anthropic Messages/tool-use repository-repair gate in diagnostic-only mode after timing. Calibration quality failures are recorded without aborting the timing ladder; production sweeps remain hard-gated.
- Calibration reports are platform-neutral and include CPU class/generation plus agent diagnostic pass/latency results; historical Granite numbers are explicitly labeled as KNL/Ollama anchors.
- Agent-quality requests now use Qwen3-recommended non-greedy sampling (temperature 0.6/top-p 0.95/top-k 20) for Qwen families; Granite/Nemotron remain deterministic. This avoids false quality failures from a decoding mode Qwen explicitly warns against.
- Added cross-platform regression coverage for KNL, dual Xeon Platinum 8168, Milan NPS4, Genoa NPS4, and WRX80 Threadripper Pro. Tests enforce identical 8K/64K context semantics, KNL-only source patch selection, generic native/repack/LTO/ISA build coverage, physical-socket NUMA policies, and platform memory-channel hints.

# v5.3.4

- Made the primary deep-sweep winner **quality constrained for agentic coding**, not merely the fastest 8K branch. `full-sweep`, `overnight`, and `full` now run performance screening -> agent gate -> deep sweep.
- Added `agent-quality`: candidates must first clear configurable 8K responsiveness floors (default PP >=20 tok/s and TG >=3 tok/s), then pass 100% of the built-in deterministic repository-repair/tool-use fixtures with a median task wall time <=90 s. Slow branches are rejected before expensive quality runs.
- If the fastest quant fails correctness/tool use, higher-fidelity quants are promoted automatically. If no tested quant in a model family passes both quality and responsiveness, the family is excluded from the >8K sweep.
- The built-in gate drives the same Anthropic `/v1/messages` + tool-use protocol expected by Claude-Code-compatible servers. If the `claude` CLI is installed, a real `claude -p` repository-repair smoke is also a hard gate; absence is tolerated unless `LAB_REQUIRE_CLAUDE_CODE=1`.
- Test repositories make their test trees immutable to the model and expose only safe file/read/write/test tools; arbitrary shell execution is not exposed by the built-in agent loop.
- Added whole-socket NUMA policies. Hardware detection maps NUMA nodes back to physical CPU packages and the planner emits `socketN-local` policies, so EPYC NPS2/NPS4 and Intel SNC systems can compare a complete socket (all local memory controllers) against node-local and cross-socket placements.
- Added agent-gate regression tests and report sections for rejected quants, accepted per-family agent winners, correctness rate, task latency and PP/TG responsiveness thresholds.

# v5.3.3

- Added a calibration-only four-rung model ladder: Gemma 3 270M IT Q8_0, Qwen3.5 0.8B Q4_0, Gemma 3 1B IT Q4_K_M, and Granite 4.1 3B Q4_K_M. These fixed representative quants are excluded from the ordinary overnight model matrix unless explicitly requested.
- Added `calibrate`: runs the same build/NUMA/thread/batch/KV/prefix-cache machinery through 8K and one 16K winner continuation, so orchestration bugs and wall-time scaling are visible before the expensive 27-35B run.
- Calibration prefixes are recycled exactly like the real sweep; no measured depth point refills 0->N.
- Added `calibration/CALIBRATION.md` and JSON output. Granite calibration is compared with prior KNL/Ollama measurements (PP ~26-27 tok/s; TG 7.77 @ ~1K, 5.42 @ ~1.7K, 2.65 @ ~3.5K) and reports a simple 8K TG extrapolation.
- Added `estimate`: after a primary `models-plan`, combines the live fitting quant count with measured Granite harness costs and active-parameter scaling to produce a deliberately lower-bound overnight wall-time estimate.
- Model registries now record catalog revision and scope so a calibration-only registry can never be mistaken for the primary overnight registry.

# v5.3.2

- Reoriented the KNL search around short-context viability: the automatic GGUF admission floor is now 8,192 tokens instead of 65,536, so larger/higher-quality quants that are useful at short context are no longer excluded merely because they cannot hold 64K.
- Broad factorial quick screening remains at <=2K; finalist PP/TG depth screening now uses one persistent recycled prefix at 512 -> 2K -> 8K. No pre-winner benchmark request may exceed 8K.
- Select exactly one combined winner per model family across weight quant, KNL build/ablation, NUMA policy, threads, KV type, and tuned batch/ubatch using the 8K screen. The default score weights normalized TG 65% and PP 35%.
- Only that one family winner may run beyond 8K. The default deep continuation is 8K -> 16K -> 32K -> 64K, capped at 64K unless LAB_DEEP_CAP/LAB_DEEP_DEPTHS are explicitly changed.
- Deep PP reconstructs the already-selected 8K prefix once, then records only new suffix work; every measured depth sweep uses persistent llama-server prompt-cache reuse and verifies cache_n before accepting an interval.
- Existing v5.3.0/v5.3.1 automatic-model registries planned with the old 64K admission floor are invalidated automatically and re-planned at 8K.
- quality now prefers the per-family winning quant produced by the sweep instead of evaluating every downloaded quant when winners are available.
- Added regression tests proving the 8K pre-winner ceiling, exact suffix sizes during prefix recycling, 8K->16K deep continuation, and exactly one deep-swept quant/configuration per model family.

# v5.3.1

- Added a no-install, no-prompt `overnight` pipeline for unattended benchmark/ablation runs.
- Added stable experiment-tagged run keys, append-only resume semantics, and `overnight-state.json`; interrupted runs skip completed measurements on restart and reconstruct the deepest proven prompt cache before continuing.
- Added bounded quick/batch/server-start/deep-PP/TG timeouts and explicit slow/failure pruning.
- Replaced deep per-depth `llama-bench -d` TG launches with a persistent `llama-server` cached-prefix progression. PP is now the measured newly evaluated suffix for each interval; TG is sampled at each cached endpoint. Cache reuse is verified and the branch fails closed if it cannot be proven. Resume backfills restore the deepest proven prefix after any missing shallower TG sample.
- Deep context allocation falls back empirically by context tier for larger KV formats; non-fitting deeper points are retained as explicit skipped records.
- Preserved the best KNL baseline in deep finalists so speed pruning cannot erase the long-context baseline/control.
- Added `models-plan` for metadata-only automatic-model/disk planning without GGUF downloads.
- Hardened unattended model acquisition: a failed HF family is recorded/skipped, a failed quant download is skipped after retries with resumable `.part` state, and stalled transfers trip a low-speed timeout instead of hanging the night. Aggregate disk insufficiency still fails before payload downloads begin.
- Reports now include status and progressive PP interval ranges.

# v5.3.0

- Restore pushbutton automatic GGUF planning: no `MODEL_*` variables are required for the standard sweep.
- Query live Hugging Face tree metadata for the four established model families before payload downloads.
- Aggregate sharded GGUF byte sizes, exclude projector/imatrix/helper files, and RAM-fit every discovered weight quant at a 65,536-token hard floor using the lowest-memory configured KV type.
- Download/reuse all fitting model+quant candidates beneath `LAB_ROOT/models`; record inventory, fit plan, registry, source repo, quant, byte size, and estimated maximum planned context.
- Share one model registry across sweep, preflight, manifest, and quality paths; retain manual models as optional additive controls.
- Add offline unit tests for GGUF inventory grouping, shard accounting, fit rejection, and quant ordering.

# Changelog

## v5.2.2

- fixed q4_0 KNL unroll generation against current upstream loop-body `//` comments;
- the generated multiline macro now strips C++-style line comments before appending continuation backslashes, avoiding translation-phase line-splicing that could comment out subsequent macro statements;
- added a regression fixture with line comments inside the q4_0 AVX2 hot loop so the unroll-enabled compile is exercised directly;
- bumped the KNL patch revision to `2026-08-22.1`.

## v5.2.1

- fixed `find_function_chunk()` so patch targets are bounded by the function's matching braces instead of the next `void` definition;
- fixes current llama.cpp commit `d775b8967a46d8beb110d444aa3b8938179e0dd8`, where a global `#if defined (__AVX__) || defined (__AVX2__)` block immediately after q6_K was incorrectly absorbed into the q6_K chunk;
- added a regression fixture reproducing that post-function AVX2 directive, while preserving strict ambiguity checks inside the actual function;
- bumped the KNL patch revision to `2026-08-21.3` and corrected the CLI version banner.

## v5.0.0

- hardened Ubuntu/current-kernel dependency handling;
- exact-kernel `perf` install/probe;
- private pinned uv and local caches/venvs;
- KNL C+C++ compiler probes and distro-only compiler policy;
- optional distro memkind + KNL HBM build;
- sanitized build and runtime environments;
- local ccache with compiler-content checking;
- preflight benchmark-hygiene checks and strict mode;
- machine/toolchain/package/microcode/mitigation manifest;
- post-build ldd/readelf ABI audits with GLIBC/GLIBCXX requirements;
- optional perf counters;
- richer debug/report evidence;
- added hardening unit tests.

## v5.1.0

- fixed upstream-drift failure on current llama.cpp CMake option alignment;
- replaced whitespace-sensitive KNL patch anchors with semantic/regex anchors;
- KNL patch application is now transactional: all five edits validate before any source file is written;
- added `--dry-run` patch compatibility mode and top-level `patch-check` command;
- q4_0 AVX2 hot-loop patching now tolerates harmless comment/blank-line/formatting changes;
- dynamic backend insertion targets the semantic `haswell` variant regardless of alignment;
- KNL compiler path now uses `-march=knl -mtune=knl` directly;
- build failures report the exact upstream commit and patch log without a Python traceback;
- regression test verifies a deliberately broken final anchor leaves every upstream source file byte-identical.
## v5.2.0

- fixed current x86 q3_K/q4_K/q6_K ambiguity when AVX2 and scalar/fallback branches contain identical outer `for (int i = 0; i < nb; ++i)` loops;
- KNL hot-loop injection now parses the function's preprocessor structure and patches only the unique `__AVX2__` branch that KNL can execute;
- nested preprocessor blocks are depth-tracked so inner `#if/#else/#endif` directives cannot terminate the selected AVX2 branch early;
- added a semantic guard requiring the selected hot loop to reference both `x[i]` and `y[i]`;
- regression fixture now contains duplicate AVX2/fallback loops for q3_K, q4_K, and q6_K, reproducing the upstream condition that broke v5.1;
- retained all-or-nothing transactional patching: ambiguity or future drift modifies zero upstream files.

