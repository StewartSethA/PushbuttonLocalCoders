# Resource-aware backend selector

`pushbutton-select` is the front door for choosing a model, quant and serving framework under live GPU/VRAM and host-RAM constraints.

With an explicit model:

```bash
pushbutton-select qwen3.8:27b --gpu 0 --vram-limit 16G --ram-limit 32G
```

With no model, an interactive terminal shows the welcome/action menu **before**
hardware inspection or provisioning: choose model(s) and launch, inspect
hardware/models/backends, preview a plan, show help, or quit. Installed coder,
Claude and Hermes frontends enter it automatically with no model, or explicitly
with `--select`, preserving the frontend and working directory.

```bash
pushbutton-select
```

No-model non-TTY input prints usage/menu guidance and exits promptly without
reading stdin, inspecting hardware, or provisioning. EOF and Ctrl-C cancel;
invalid menu/model entries are retried. `--help` works without a GPU or toolchain.
Explicit frontend model commands retain their existing noninteractive behavior.

## Multiple models and automation

Enter space-separated model numbers/names in the menu. Coder frontends also ask
for worker count, physical context, and optional per-GPU VRAM/advisory host RAM
ceilings. Placement syntax remains `MODEL@gpu=0+1,vram=30G`.

```bash
pushbutton-select qwen3.8-flash-next qwen3.8:27b --agents 2 --plan-only
pushbutton-select 'qwen3.8:27b@gpu=0,vram=16G' \
  'qwen3.6:35b@gpu=1,vram=16G' --agents 2 --context 65536 --json
pushbutton-select --frontend hermes-local qwen3.8:27b qwen3.6:35b --print-only
```

`--plan-only` / `--print-only`, `--json`, and `--no-interactive` never launch.
JSON includes inventory, tables, the complete joint plan, `launch_argv` (an argv
array), and a planning error if unavailable. With no models it returns structured
guidance instead of choosing a default. `--gpu` / `--gpus` constrain coder workers;
for multiple workers prefer separate per-model pins. Leases are capped by free,
not total, VRAM. Claude/Hermes use their existing role planner (up to four
selectors, one server per unique model); they do not accept coder GPU pins,
leases, or `--agents`.

The existing joint planners validate the whole proposed placement before launch:
coder replicas occupy disjoint GPU sets, while repeated Claude/Hermes roles
share a server. The preview lists GPU sets, free lease capacity, planning
envelopes, context/client budget, role mapping where applicable, and spare GPUs.
The default client context is `min(200000, physical context)`; an explicitly
larger client context is rejected. Interactive launch requires explicit `y`/`yes`
confirmation before downloading/building/launching. Commands execute as argv
lists, never `eval`. The launcher rechecks placement against live availability.

The table reports baseline `FIT` / `BLOCK` (planning only), or `UNKNOWN` /
`UNAVAILABLE` for experimental eligibility; legacy specialized rows may show
`UNVERIFIED`. Architecture alone never proves experimental memory fit.
`LOCAL MEASURED` is green; `UPSTREAM REFERENCE` is never colored as locally
verified. Without matching evidence PP/TG is `UNKNOWN`, not a dense-model
bandwidth projection. Local observations supersede reference observations
without mixing the two distributions. llama.cpp evidence must include the quant,
so a Q4 result cannot silently become an IQ3 result.

The selector inspects current free and occupied VRAM with `nvidia-smi` through the existing hardware inventory. If a requested VRAM ceiling exceeds currently free memory it warns rather than pretending the card is empty. Host RAM limits are currently planning ceilings, not kernel/cgroup enforcement.

Specialized fixed/tuned engines are intentionally conservative. A backend with a
known artifact requirement above the lease is `BLOCK`; a backend known only on a
different card configuration is not asserted to fit. Baseline profiles
automatically down-select using all eligible GPU capacity, not only one card.
Host RAM minima are unknown unless separately validated; the displayed RAM
ceiling is advisory and does not enforce a kernel/cgroup limit.

## Resident models and ports

`coder-local` probes for a free loopback port for every new worker. The selector
does **not** automatically attach to resident endpoints: model-name resemblance
does not establish matching quant, context, roles, or lease. The confirmed launch
uses the normal frontend lifecycle, not a promised endpoint reuse path.

## Benchmark evidence and real-world spread

`benchmarks/aggregate.json` is the bundled reference database. Its September 14,
2026 snapshot contains three upstream 27B samples, **not Flash-Next results**.
`benchmarks/results/` initially contains no local measurements. The selector
checks the cached/current aggregate and local JSON reports, computing median and
p10/p90 for matching model/backend/artifact/GPU/context evidence. Reference
results remain labeled upstream; they are not predictions for a new context,
topology, power limit or workload.

## Flash-Next support boundary and validation

The existing baseline is upstream llama.cpp with
`unsloth/Qwen3.8-Flash-Next-GGUF`. At 262,144 context the six planner envelopes
are: `UD-Q4_K_XL` 136 GiB, `UD-IQ4_XS` 116 GiB, `UD-Q3_K_XL` 112 GiB,
`UD-IQ3_XXS` 104 GiB, `UD-Q2_K_XL` 99 GiB, `UD-IQ1_M` 94 GiB. These are
**conservative plans, not measured weight/runtime allocations**. Four 32-GiB
cards select the existing recommended `UD-IQ4_XS` profile; two independent
replicas require two disjoint fitting groups (the synthetic eight-V100 planner
test remains 4+4).

Registry metadata exposes two existing experimental adapters. They are offered
as **benchmark/validation commands only**, not integrated frontend backends:

| Adapter | Published hardware/profile | Current selector eligibility |
| --- | --- | --- |
| `sglang-v100` | Four V100 32 GiB, NVLink, v4 image, NVFP4 weights, E5M2 KV, MTP | SM70/card count/capacity checks; topology and actual free-memory fit still unknown; context up to 262144 |
| `vllm-flashnext-3090` | Four RTX 3090 24 GiB, W4A16, default `mtp` | SM86/card count/capacity checks; actual allocation/calibration/runtime unknown; default profile context at most 65536 |

Published results **as of October 2, 2026**, not local measurements:

- [SGLang V100 recipe](https://github.com/haohervchb/sglang-V100):
  target-only about 74 tok/s; MTP about 118–120 tok/s at 1K/25K prompts,
  with prefill about 3269/4693 tok/s. The separately documented 262144-context
  concurrency conditions must not be inferred from those speed rows. Its
  empty-cache startup was validated on a 256-GiB RAM host with approximately
  126 GiB of downloaded weights; 256 GiB is a tested host, **not a proven minimum**.
- [4×3090 recipe](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe):
  at 220 W/card, default `mtp` about 115–117 tok/s at 4K/32K prompts.
  `fp8deep` reports about 68 tok/s after a 260K prompt. The approximately
  198 tok/s short-code peak is **not sustained general throughput**.
  `mtp` supports 65536 context; `bf16` and `fp8mtp128k` support 131072;
  `fp8deep` supports 262144. FP8 profiles require the optional KV patch and
  calibrated `scales.json`. `pushbutton-bench` currently uses the adapter's
  default MTP profile and cannot select those other profiles.
- [Official release/support](https://github.com/QwenLM/Qwen3.8-Flash-Next):
  Flash-Next was released August 26, 2026 and documents llama.cpp support.

First preview without downloading anything, then deliberately benchmark on a
machine matching the prerequisites:

```bash
pushbutton-select qwen3.8-flash-next --agents 1 --plan-only
pushbutton-backend check sglang-v100 qwen3.8-flash-next --gpus 0,1,2,3
pushbutton-backend check vllm-flashnext-3090 qwen3.8-flash-next --gpus 0,1,2,3

# These commands DO download/build/launch. Run only after reviewing the recipe.
pushbutton-bench qwen3.8-flash-next --backend llama.cpp \
  --gpus 0,1,2,3 --context 262144 --suite standard
pushbutton-bench qwen3.8-flash-next --backend sglang-v100 \
  --gpus 0,1,2,3 --context 262144 --suite standard
pushbutton-bench qwen3.8-flash-next --backend vllm-flashnext-3090 \
  --gpus 0,1,2,3 --context 65536 --suite standard
```

`check` tests adapter eligibility, **not memory fit or completed GPU validation**.
`pushbutton-bench` writes real local reports to `benchmarks/results/`; review
artifact, hardware, context and cases before comparing rates. No benchmark JSON
or claims of completed GPU validation have been added here. No `--force`,
unified-memory spill, or silent CPU offloading is enabled by selection.
The newer 2×3090 CPU/expert/PLE-offload recipe is a future experimental
integration, not an existing Pushbutton adapter; it would need explicit loader,
swap/RAM, bandwidth/topology and correctness/performance validation.

To measure an already-running OpenAI-compatible endpoint at multiple context depths:

```bash
pushbutton-observe \
  --endpoint http://127.0.0.1:20181/v1 \
  --model qwen3.8:27b \
  --backend vllm-qwen38-3090 \
  --depths 1024,8192,25000,65536,100000
```

`pushbutton-observe` records prompt-depth, TTFT, prompt-processing throughput and token-generation throughput without retaining prompts or generated text.

## Opt-in telemetry

Telemetry is **off by default**. Enable it explicitly:

```bash
pushbutton-select --telemetry-opt-in --telemetry-upload-url https://YOUR-COLLECTOR.example/v1/pushbutton
```

or while observing:

```bash
pushbutton-observe --telemetry-opt-in --telemetry-upload-url https://YOUR-COLLECTOR.example/v1/pushbutton ...
```

Queued telemetry contains only compact model/backend/artifact, coarse hardware, context and performance fields. It excludes prompts, generated text, usernames, hostnames and arbitrary file paths. Payloads are gzip-compressed newline-delimited JSON, so uploads are small. If no upload URL is configured, telemetry remains local and nothing is transmitted. Disable it with `pushbutton-select --telemetry-off ...`.

The repository includes the client and queueing protocol; deploying a public collector endpoint is a separate operational step.

## Direct curl use

The existing direct constrained launch still works:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-coder-local.sh \
  | bash -s -- --system \
    'qwen3.8:27b@gpu=0,vram=16G' \
    'qwen3.6:35b@gpu=1,vram=16G' \
    --agents 2
```

To install and enter the selector directly with flags:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-coder-local.sh \
  | bash -s -- --system --select \
    qwen3.8:27b --gpu 0 --vram-limit 16G --ram-limit 32G
```

If the installer is run with no model or other arguments in an interactive
terminal, it enters the welcome menu. A noninteractive curl pipeline never
implicitly opens `/dev/tty`; `--select` without a TTY prints guidance rather than
blocking. To select interactively after installing, run `qwen-local --select`
from a terminal. Automated scripts should pass explicit model(s) or request
structured selector output.
