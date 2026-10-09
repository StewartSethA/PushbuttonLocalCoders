# `claude-local`: local multi-model Claude Code harness

`claude-local` runs the normal Claude Code CLI while routing every model request to local `llama-server` instances. It is designed for mixed NVIDIA workstations and servers where model quality, long context, and interactive latency matter more than choosing a GGUF by hand.

## Install

From a checkout of PushbuttonLocalCoders:

```bash
./claude-local install
```

This creates `~/.local/bin/claude-local`. For a system-wide command:

```bash
./claude-local install --system
```

This creates `/usr/local/bin/claude-local` with `sudo`. It is an executable entry point rather than a shell alias, so Bash, zsh, scripts, and other launchers get identical argument handling.

Heavy dependencies are provisioned lazily. The harness installs ordinary build dependencies, installs Claude Code if needed, keeps a working NVIDIA driver untouched, bootstraps a compatible CUDA toolkit privately when `nvcc` is missing/incompatible, and builds the required CUDA llama.cpp variant.

## Basic use

With no model arguments, let the planner choose a fast local default:

```bash
claude-local
```

Use one model for every Claude role:

```bash
claude-local qwen3.6:35b
```

Or choose several models by role:

```bash
claude-local nemotron-3.5-lightning qwen3.6:35b qwen3.8:27b glm-5.3-flash
```

The positional mapping is:

| Models supplied | Haiku | Sonnet (daily driver) | Opus | Fable |
|---|---|---|---|---|
| 1 | #1 | #1 | #1 | #1 |
| 2 | #1 | #2 | #2 | #2 |
| 3 | #1 | #2 | #3 | #3 |
| 4 | #1 | #2 | #3 | #4 |

### Lightweight Haiku with Flash Next as Fable

Yes: the fourth positional selector can be `qwen3.8-flash-next`. For example,
this requests **four distinct models**, in Haiku / Sonnet / Opus / Fable order:

```bash
claude-local qwen3-4b-instruct-2507 qwen3.8:27b qwen3.6:35b qwen3.8-flash-next \
  --local-startup-policy allow-cpu-only \
  --local-memory-metadata /path/to/verified-memory.json \
  --local-dry-run
```

This is a configuration example, **not a claim it fits two ordinary GPUs**.
Four distinct GPU primaries cannot occupy two GPUs under exclusive placement.
With calibrated CPU placements and sufficient non-swap RAM/CPU capacity, the
planner can keep Sonnet on GPU and place other roles on CPU. Every GPU primary
also needs a resident, same-model/same-quant CPU overflow replica.
Neither GPU sharing nor a different-model universal fallback is enabled.

Flash Next is not lightweight: its existing catalogue envelopes are
**94–136 GiB at 262,144 tokens** (single native-context envelope). Two default
slots at that context require **188–272 GiB** in the uncalibrated strict GPU
planner, plus GPU reserves and an additional host staging envelope.
Even its lowest tier does not fit two 24 GiB or two 32 GiB GPUs at one slot.
Shorter contexts do not shrink these indivisible envelopes. If Sonnet and a
dedicated classifier occupy the two GPUs, Flash Next must instead have a
verified CPU-only placement. Its RAM footprint and long-context prefill/decode
latency must be measured; a small active-parameter count does not remove the
need to hold all expert weights. No production Flash Next CPU capacity or
latency calibration is shipped here.

A possible **six-server** role-only layout (no separate classifier) is:

| Resident server | Device |
|---|---|
| Haiku: Qwen3 4B Instruct 2507 | CPU primary |
| Sonnet: Qwen3.8 27B | GPU 0 primary |
| Sonnet: identical quant/context | CPU overflow |
| Opus: Qwen3.6 35B | GPU 1 primary |
| Opus: identical quant/context | CPU overflow |
| Fable: Qwen3.8 Flash Next | CPU primary |

This is illustrative, not pinned: quality tiers, measured budgets and available
hardware determine which secondary role receives GPU 1. RAM admission sums
**all six** weights, per-slot caches and buffers (including GPU host allocations
and loading/staging), plus the 1024 MiB host reserve. Identical replicas are
budgeted separately; do not assume mmap sharing or count only the two GPU
models. Each server still defaults to two slots; thread quotas cover all six.
The synthetic unit-test calibrations demonstrate placement mechanics only,
not these models' actual capacities.

For a separate safety classifier, keep the same four positional selectors and
add the existing explicit routing options before ordinary Claude Code flags:

```bash
  --local-classifier-model qwen3-4b-instruct-2507 \
  --local-classifier-gpu 1 \
  --local-classifier-context 32768 \
  --local-classifier-request-model OBSERVED_DISTINCT_REQUEST_ID
```

This reserves GPU 1 for a **distinct** classifier primary, with its own CPU
overflow. Sonnet is preferred on GPU 0; Haiku, Opus and Fable are CPU primaries.
That is **seven resident servers**, not six: four role primaries, one classifier
primary and two overflow replicas. The classifier's weights may match Haiku,
but its server ID, slots and routes never merge with Haiku. Follow the request-ID
verification procedure below; Haiku selection alone never redirects safety
requests or changes permissions.

### Dedicated four-GPU Fable group

Use `--local-role-placement configs/flash-next-role-placement.json` for an
explicit production **llama.cpp** layout:

| Role | Reservation |
|---|---|
| Haiku | CPU only; measured memory metadata required |
| Sonnet and Opus (same model) | physical GPU 4 |
| Fable: Flash Next | physical GPUs 0,1,2,3, exclusively |
| Dedicated classifier | physical GPU 5, independent server and slots |

For example, after calibrating the CPU Haiku envelope:

```bash
./claude-local q3-4b q38 q38 q38next \
  --local-role-placement configs/flash-next-role-placement.json \
  --local-memory-metadata verified-memory.json \
  --local-context 131072 --local-client-context 120000 \
  --local-classifier-model q38 --local-classifier-gpu 5 \
  --local-classifier-request-model OBSERVED_DISTINCT_REQUEST_ID \
  --local-dry-run
```

Edit physical indices for your machine; repeat the exact command without
`--local-dry-run` only after reviewing admission. Slots remain **two by default**.
Four 32 GiB V100s can be considered for the existing IQ4_XS envelope at this
context/slot combination; this is not a measured performance or fit guarantee.
The catalogue does **not** establish a four-3090 GGUF fit, nor a four-V100 fit at
262144 **per slot** with two slots. Insufficient capacity fails closed; neither
context, quant quality floor nor slots are silently reduced.

The JSON must list all four roles with `backend: "llama.cpp"` and a distinct
physical `gpus` array (empty means explicitly CPU-only). Shared-model roles must
declare the same placement. A classifier entry must match the existing explicit
classifier GPU option. Reserved groups cannot overlap or borrow cards, and
Flash Next requires exactly four cards in this mode. Fixed placements do not
add automatic overflow replicas even with `allow-cpu-only`; CPU Haiku is an
explicit calibrated primary, not a fallback for the classifier or Fable.
The same joint host/cgroup RAM, GPU reserves, thread admission, startup warmup
and gateway slot limits apply. A missing CPU calibration is an error.
An explicitly CPU-pinned role is permitted even under `gpu-only`; that policy
still forbids automatic CPU overflow or migration of the GPU-pinned roles.

**Protocol boundary:** the Claude gateway forwards Anthropic messages, tools,
streams and count-token requests unchanged. The specialized vLLM/SGLang adapters
serve **OpenAI**, not a verified Anthropic interface. They are intentionally
rejected in role placement rather than pretending to be wired into Claude Code.
Evaluate them separately using the [four-GPU bake-off](benchmarks/README.md).
Promoting a winner into Claude serving requires a validated Anthropic adapter,
including tool-use, streaming and classifier isolation; none is claimed here.
Selecting lightweight Haiku does not select or validate a safety classifier.

### Qwen3 4B Instruct 2507 profiles and evidence

`qwen3-4b-instruct-2507` (aliases `q3-4b`, `qwen3-4b-2507`,
`qwen3:4b-instruct-2507`) is the **non-thinking Instruct-2507** checkpoint, not
the earlier thinking/hybrid Qwen3-4B. Its native context is **262,144**, not 32K,
so it can coexist with 262K Fable without introducing per-role context flags.
The classifier retains its existing independent context option.

| `hf_spec` | GGUF filename |
|---|---|
| `unsloth/Qwen3-4B-Instruct-2507-GGUF:Q5_K_M` | `Qwen3-4B-Instruct-2507-Q5_K_M.gguf` |
| `unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M` | `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |

Q5 is considered before Q4; quality numbers are ordering tiers, not measured
safety or coding accuracy. These profiles use upstream llama.cpp and the
**embedded GGUF chat template**, not the Qwen3.8-specific fixed template.
Validate tool-call/Anthropic-wire compatibility on the exact downloaded GGUF
and current llama-server before relying on it for agent work.

Sources: [official Qwen release/checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[GGUF repository](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF),
[Xinference catalogue](https://github.com/xorbitsai/inference/blob/603adda5c50ed3ca0ac1cfd57572f7817fd19fb4/doc/source/models/builtin/llm/qwen3-instruct.rst)
lists 262144 context, both quant selectors and llama.cpp support;
[pinned Q4 artifact manifest](https://github.com/AkshitIreddy/Interactive-LLM-Powered-NPCs/blob/146ea50af1cce11d12b8e0b640550923d7f96b0b/packaging/model-packs/qwen3-4b-instruct-2507-q4-k-m.json)
records the exact filename, 2,497,281,120-byte file, non-thinking mode and 262144
context; [Q5 filename evidence](https://github.com/Thireus/GGUF-Tool-Suite/blob/384c1f85b77b30f2b388328af278aea0900f6ca1/ppl_from_others.db)
links the exact Unsloth Q5 artifact. These corroborating catalogues were accessible
when direct Hugging Face DNS access was unavailable; they are not local hardware
measurements or safety benchmarks.

**No runtime memory envelope is inferred from that small weight file.**
Both profiles expose `required_mib: null` and require explicit calibrated
metadata for **GPU as well as CPU** placement. Long-context caches, two slots,
buffers and staging can dominate weights. Supply entries for the exact quant
and device modes needed by the chosen layout using the calibration format below.
Legacy frontends reuse the shared aliases/catalogue but skip profiles without
an envelope; they do not currently consume this strict calibration metadata.

The 4B model is an **evaluation candidate**, not a proven Bash safety gate.
Before enabling automatic permissions, evaluate held-out harmless and risky
command *descriptions* without executing them: false approvals, false refusals,
prompt-injection resistance, verdict-format compatibility and timeout/error
behavior under concurrent agent load. Check both GPU and CPU classifier routes
and keep failures closed. No measured safety advantage is asserted here; use
manual approval until your exact model/quant/build passes your safety criteria.

All ordinary Claude Code arguments are passed through:

```bash
claude-local qwen3.6:35b --resume
claude-local qwen3.8:27b --continue
claude-local qwen3.6:35b --permission-mode plan
claude-local qwen3.6:35b -- --resume SESSION_ID
```

Harness-specific flags must appear before the first ordinary Claude Code flag.

## Hardware planner

The launcher inventories free GPU VRAM, compute capability and PCIe links, plus
Linux `MemAvailable`, remaining visible cgroup v1/v2 memory limits (including
ancestors), CPU affinity/quota and NUMA CPU lists. Swap is **never** capacity.
Host capacity is the minimum of MemAvailable and cgroup remaining memory.
Complete layouts jointly reserve RAM, disjoint GPU sets and CPU thread budgets
for **every** server, including a separate same-model classifier replica.
The planner keeps the daily driver and classifier on GPUs preferentially;
secondary workers can move to CPU when permitted. NUMA information is reported,
not automatically pinned: local memory bandwidth and placement remain relevant.

The default physical context is **262,144 tokens**. The default Claude Code client budget is **200,000 tokens**, leaving physical headroom for large tool results and token-accounting differences before the backend ceiling.

Run a dry plan before downloading anything:

```bash
claude-local qwen3.6:35b qwen3.8:27b --local-dry-run
```

### Explicit CPU fallback policies

`--local-startup-policy` (or `CLAUDE_LOCAL_STARTUP_POLICY`) selects:

| Policy | Permitted complete layouts |
|---|---|
| `gpu-only` (default) | Every server fully GPU-offloaded |
| `allow-cpu-only` | GPU servers with pre-provisioned CPU-only overflow replicas, CPU-only servers where GPUs cannot fit the requested models, or an entirely CPU-hosted layout |

There are **no hybrid weights or partial CPU/GPU layers**. `allow-hybrid` is
rejected. CPU fallback is explicit and local, not unified-memory spill.
With overflow enabled, each GPU backend requires a calibrated CPU-only replica
of exactly the same `hf_spec` (model and quant), context and slot count. All
replicas are loaded and jointly RAM/thread-budgeted **before** the gateway
accepts requests. A missing calibration or insufficient joint budget cannot
produce an unbudgeted runtime allocation; a fitting CPU-only startup layout
may be selected instead. No alternate fallback model is supported.

The gateway admits at most the configured active slots per backend (default
two), shared across all aliases/families pointing to that backend. Requests use
the GPU when a slot is available, otherwise their explicit CPU-only route.
Both GPU and CPU saturation return HTTP 503 immediately; there is no unbounded
inference queue. Slots remain held through response completion or detected
stream failure/disconnect, and are released on every exit path. A downstream
disconnect while waiting for upstream output is detected on the next write or
backend timeout; the lease remains conservative until then.

GPU connection failures, empty SSE responses and HTTP 500/502/503/504 can retry
once on GPU and then use the configured CPU route, **only before any response
headers/bytes are committed**. CPU fallback makes one attempt; its HTTP errors
are forwarded and transport failures return 502. After commit, failures close
the stream without replay, rerouting or a second response. Admission covers
both messages and count-tokens endpoints. Direct calls to backend ports bypass
gateway admission and are unsupported during managed use.

The requested model identities, per-slot contexts and slot counts are never
silently reduced. Quant selection remains catalogue-based. Set
`--local-min-quality N` (`CLAUDE_LOCAL_MIN_QUALITY`, default 0) to reject catalogue
tiers below N; scores are ordinal tiers, **not** measured accuracy percentages.
Fallback is not a reason to use a less capable model or change safety routing.

CPU-only startup needs neither NVIDIA tools nor a CUDA toolkit. It builds a
separate `build-cpu` with `GGML_CUDA=OFF`; mixed layouts use the CUDA build,
but CPU servers explicitly receive `-ngl 0 --device none --no-kv-offload`.
Every server gets the planned layer count, cache types, batch sizes and
`-t`/`-tb` thread budgets. `--fit off` disables llama.cpp's automatic parameter
adjustment. Required options are checked against the actual binary's help;
older binaries/forks that lack them must be rebuilt or are rejected.

**No built-in CPU capacities are fabricated.** These model architectures
include dense, MoE and recurrent caches, so evenly dividing weights by layer
count or using an arbitrary weight/cache percentage is unsafe. CPU-only
placements require `--local-memory-metadata FILE` (or
`CLAUDE_LOCAL_MEMORY_METADATA`) containing explicit calibrated placements.
Without it, GPU-only uses available indivisible conservative catalogue envelopes:
no context-based reduction, rounded-up aggregate-context envelope multiples,
and a full additional host envelope for mapped weights/loading staging.
This intentionally may reject otherwise workable GPU machines. It does not
pretend the old envelopes are measured weights/cache/buffer components.
Profiles with no envelope (currently Qwen3 4B Instruct 2507) require calibration
even for GPU-only; the legacy frontend planners skip them rather than guess.

#### Calibration format

An existing JSON file must have `{"version": 1, "placements": [...]}`.
Each placement is model/quant-specific (`hf_spec` exactly matches `models`
output) and must provide all of:

```json
{
  "hf_spec": "unsloth/Qwen3.8-27B-GGUF:Q8_0",
  "source": "REPLACE with GGUF identity, llama.cpp commit, hardware and measured evidence",
  "mode": "cpu",
  "layer_count": 64,
  "ngl": 0,
  "weights_host_mib": 0,
  "weights_gpu_mib": 0,
  "cache_host_mib_per_token": 0,
  "cache_gpu_mib_per_token": 0,
  "buffer_host_mib": 0,
  "buffer_gpu_mib": 0,
  "max_context": 262144,
  "max_slots": 2,
  "max_threads": 8,
  "batch": 512,
  "ubatch": 256,
  "kv_k": "f16",
  "kv_v": "f16",
  "flash_attn": "off"
}
```

**This is a schema illustration, not a runnable calibration:** zero weights
are rejected and the layer count is illustrative. Obtain actual GGUF metadata
and measured conservative upper bounds for your exact quant/build/hardware.
The harness trusts explicitly supplied calibration, not the illustrative values.
Include mapped GGUF pages, transient loading/staging allocations, CPU repacking,
recurrent state, compute/workspace, allocator overhead and concurrent-slot peaks.
Fixed or non-linear cache/state belongs in buffer bounds; a per-token cache
coefficient is only valid over a verified context/slot range. Bind evidence to
exact model files and re-calibrate after model or llama.cpp changes.

For each host/device, budget is `weights + buffer + context * slots *
cache_mib_per_token`, rounded up. Host reserve is 1024 MiB; each GPU retains
512 MiB. These margins are not a guarantee against unrelated processes.
`max_context`, `max_slots` and `max_threads` bound validity of the calibration;
batch/ubatch and cache/FA settings are launched exactly. GPU placements must
offload all layers including output; CPU-only placements require `ngl=0`
and zero GPU weights/cache/buffer memory.
CPU cache types currently require f16: **CPU cache quantization and
GPU-weights/CPU-cache-only placement are deferred**. Calibrated
GPU entries currently support a single GPU per server; legacy
GPU-only envelopes can still span disjoint GPU sets.

For an actual calibrated CPU machine:

```bash
claude-local q38 --local-startup-policy allow-cpu-only \
  --local-memory-metadata /path/to/verified-memory.json \
  --local-min-quality 95 --local-dry-run
```

A distinct classifier may omit `--local-classifier-gpu` only under a fallback
policy. The planner then prefers GPU placement but may use a calibrated CPU
server. An explicit physical classifier GPU remains reserved for its primary
and is never reassigned to main models. Under `allow-cpu-only` its distinct,
same-model CPU-only overflow replica is also budgeted, even with a reserved
GPU. The classifier primary still requires that GPU to fit. Classifier
overflow never uses the main model/backend or a different safety model.
Observed classifier request IDs remain mandatory and cannot override a session
route; failed classifications are never synthesized and permission policy is
never changed.

### Startup validation and latency limits

Startup tries up to `--local-max-layouts N` (`CLAUDE_LOCAL_MAX_LAYOUTS`,
default 3, maximum 8) complete pre-budgeted alternatives, with the same models,
contexts, slots and quality floor. There are no per-server ad-hoc downgrades.
Failed layouts are fully stopped and reaped before another starts, even with
`--local-keep-servers`; the gateway and Claude launch only after acceptance.
Readiness is bounded by `CLAUDE_LOCAL_STARTUP_TIMEOUT` (default 1800 seconds
per layout). Once **all** servers are healthy, concurrent real one-token
inference warmups cover every server/slot, including the classifier, followed
by resident RAM, cgroup/host remaining memory and GPU allocation checks.
The loader's actual `offloaded N/M layers to GPU` report must confirm full
GPU placement; an unnoticed partial/CPU execution
cannot satisfy `gpu-only`. Missing/incompatible GPU load reports fail closed.
Live RAM/VRAM/CPU admission is rechecked after provisioning and before each
layout allocates memory, refreshing GPU usage baselines.
`CLAUDE_LOCAL_WARMUP_TIMEOUT` defaults to 120 seconds for the whole warmup;
both deadlines accept 1..3600 seconds. Builds/download provisioning is separate
from the backend readiness deadline. Strict search is bounded to 16 GPUs,
4096 candidates per server, 200,000 enumeration/search steps and 64 aggregate
warmup slots, including CPU-only replicas. When available, attempts reserve
one layout for each primary-placement stage: fully GPU, mixed CPU/GPU, then
fully CPU. No layout changes or memory transfers occur after gateway launch.

Warmup verifies inference and allocation, **not** full-context throughput,
model correctness, classifier prompt quality, or production latency. CPU
weights can be extremely slow, even for MoE models; long prefills, two concurrent
slots, shared memory bandwidth and NUMA effects can exceed interactive or
safety-classifier deadlines. There are no benchmark-derived latency promises.
Test representative real requests locally before relying on auto mode;
manual approval is safer than a classifier configuration that times out.
No live GPU/CPU model benchmarks are implied by planner regression tests.

## Built-in model selectors

```text
qwen3.6:35b
qwen3.8:27b
nemotron-3.5-lightning
glm-5.3-flash
```

Aliases such as `q36`, `q38`, `nemotron`, and `glm53` are accepted. `claude-local models` prints the registered repositories, quant tiers, KV formats, and memory envelopes.

The model catalogue is intentionally curated rather than accepting arbitrary names: each profile has a known repository, quant hierarchy, context/KV configuration, compatibility template, and conservative VRAM envelope. Users choose a model family; the harness chooses the quant.

For Qwen 3.6/3.8, the harness downloads the current fixed Qwen Jinja template and supplies it to llama-server for agent/tool compatibility.

`GLM-5.3-Flash` is temporarily special. Upstream llama.cpp support is still under review in September 2026, so that profile builds `unslothai/llama.cpp` branch `glm5next/upstream`. It applies the current correctness settings `NVIDIA_TF32_OVERRIDE=0` and Flash Attention off. Because GLM-5.3-Flash is very large even at low bit rates, the planner selects it only when the requested hardware set has enough VRAM.

## CUDA and heterogeneous GPUs

A working NVIDIA driver is reused. If NVIDIA hardware is present on Ubuntu but `nvidia-smi` is unavailable, the harness installs Ubuntu's recommended driver and requests a reboot only when the new driver cannot become active immediately.

The CUDA toolkit is independent of the driver. If suitable `nvcc` is absent, the harness installs a private toolkit under its state directory using micromamba. Legacy architectures such as V100/SM70 use CUDA 12.8; newer NVIDIA cards use CUDA 12.9. A mixed V100 + Ada plan therefore builds a binary containing both SM70 and SM89 without replacing the host driver.

CUDA unified-memory spill is disabled. The legacy `--local-allow-offload`
option is rejected: use an explicit startup policy and calibrated CPU-layer
placement instead of unbudgeted spill.

## Model cache

Downloads default to:

```text
~/.cache/pushbutton/llama
```

Change it per run:

```bash
claude-local --local-cache /mnt/models/llama qwen3.6:35b
```

or persist it:

```bash
export CLAUDE_LOCAL_CACHE=/mnt/models/llama
```

`llama-server -hf ...` performs download/shard resolution and reuses the cache on later runs.

## Claude Code integration and local fan-out

A tiny local Anthropic-wire gateway routes `/v1/messages` and `/v1/messages/count_tokens` to the appropriate llama.cpp server. It recognizes both explicit local model IDs and Claude family IDs (`haiku`, `sonnet`, `opus`, `fable`). Unknown internal model IDs fail closed to the local Sonnet daily driver; there is no cloud fallback.

The main session and generated subagents use opaque local model IDs rather than built-in Claude aliases. This makes role routing independent of Claude Code's alias resolution and keeps custom-gateway context accounting conservative.

Unless `--local-no-teams` is supplied, the harness enables Claude Code's agent-team support and injects four fully local subagents:

- `local-fast`: repository reconnaissance and triage on Haiku
- `local-coder`: parallel implementation on Sonnet
- `local-reviewer`: independent review on Opus
- `local-deep`: difficult debugging/architecture work on Fable

Claude Code remains the orchestrator. When it fans out, all agent API traffic returns to the same local gateway and therefore to the planned local GPUs.

### Auto-mode classifier timeouts

Auto permission mode adds inference requests to classify tool safety. A session,
its agents, and the classifier can queue behind the same one-slot backend.
Repeated timeouts do not establish that the server is down. A free second GPU
does not help unless an independent server runs there and requests reach it.

Choose one of these options; none silently authorizes a tool after classification
fails:

1. **Manual approval:** resume with `--permission-mode default` and explicitly
   approve Bash actions. `--permission-mode acceptEdits` is another option for
   editing workflows, but does not blanket-authorize Bash.
2. **Dedicated backend:** reserve a physical GPU and route the classifier's
   observed API model ID to an independent replica.
3. **Shared capacity:** increase inference slots, retaining sufficient context
   and VRAM for each slot. This reduces queueing but does not guarantee the
   classifier's deadline.

For manual approval:

```bash
claude-local qwen3.8:27b --resume --permission-mode default
```

For a dedicated classifier on physical GPU 1:

```bash
claude-local qwen3.8:27b \
  --local-classifier-model qwen3.8:27b \
  --local-classifier-gpu 1 \
  --local-classifier-request-model claude-sonnet-5 \
  --local-classifier-context 32768 \
  --local-verbose \
  --resume --permission-mode auto
```

**Verify the request ID for your Claude Code version first.** An Anthropic
[maintainer response](https://github.com/anthropics/claude-code/issues/69002#issuecomment-5310942863)
states that an arbitrary classifier-model override is not exposed; a
[2.1.280 gateway report](https://github.com/anthropics/claude-code/issues/96411)
observes `claude-sonnet-5`. This example is an explicit gateway model-ID mapping,
not a supported Claude classifier-selection setting. No undocumented selector
variable or prompt-text heuristic is used.

Routing is **model-wide**, not classifier-purpose detection: any request using
the configured ID reaches the classifier backend. The harness keeps session
and built-in subagent models on their opaque local IDs and rejects `--model`
when a dedicated classifier is configured. Custom agents must also avoid
classifier request IDs. Repeat `--local-classifier-request-model` for multiple
verified, distinct IDs if needed. IDs that collide with harness role IDs are
rejected. If your version sends classification using the same opaque ID as the
session, this mapping cannot isolate it safely; use manual approval or shared
capacity instead. Recheck routing after CLI upgrades.

The classifier GPU is excluded from the main planner before allocation. The
classifier uses the configured slot count and its own per-slot context; it remains a distinct
`local-classifier` server even when its model matches the session's model.
Repeated positional model arguments still share a server. The main model is
planned on the remaining GPUs (GPU 0 on a two-GPU host); unavailable or
insufficient classifier/main capacity aborts startup instead of sharing the
reserved GPU. Use `--local-dry-run` with the same harness options to inspect
placement without starting servers. Select a different registered
`--local-classifier-model` to use a smaller model, but validate its safety
classification quality and response compatibility before relying on it.

All `claude-local` backends default to **two inference slots**, including a
dedicated classifier. Use `--local-slots 1` or `CLAUDE_LOCAL_SLOTS=1` to restore
single-slot operation. An explicit `--local-slots` overrides the environment.

For shared capacity with a smaller per-slot context:

```bash
claude-local qwen3.8:27b \
  --local-slots 2 \
  --local-context 131072 \
  --local-client-context 100000 \
  --resume --permission-mode auto
```

`--local-context` is the context **per slot**; the server receives aggregate
context `context × slots`. The planner scales its existing memory envelope for
that aggregate context. These are estimates, not measured KV/cache guarantees:
verify runtime VRAM and avoid offload if responsiveness matters. Two slots retain
full context per request, rather than sharing a fixed total KV budget; a 16 GB
card may need a smaller context or the one-slot override to fit. The classifier
uses the same slot count with its separate context budget. `CLAUDE_LOCAL_SLOTS`
also sets the slot count for all backends. Default tool concurrency is the sum of planned main slots,
capped at four, excluding the classifier; explicitly set
`CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY` to override it.

With `--local-verbose`, gateway logs show requested model ID, resolved backend
alias, endpoint, and path without recording prompts. Startup/dry-run output
shows physical GPU placement, context per slot, and slot count. Compare idle
and busy requests and inspect llama-server logs/metrics for slot occupancy.
Check that classifier requests reach GPU 1 while session requests remain on
their main backend, then exercise repeated classification under agent load.
If the classifier fails, its HTTP error is returned unless the explicit
CPU-only policy permits its same-model pre-commit fallback. There is no cloud
fallback, synthetic safety verdict, or automatic permission-mode change.
A live GPU/Claude Code test is required to establish latency and safety
compatibility; gateway unit tests alone cannot establish either.

## Useful commands

```bash
claude-local doctor
claude-local models
claude-local qwen3.6:35b qwen3.8:27b --local-dry-run
claude-local --local-context 262144 --local-client-context 200000 qwen3.8:27b
claude-local --local-keep-servers qwen3.6:35b
CLAUDE_LOCAL_REBUILD=1 claude-local qwen3.6:35b
```

State, builds, templates, plans, and logs default to `~/.local/share/pushbutton/claude-local`. Override with `CLAUDE_LOCAL_STATE`.

## Scope

The first harness targets NVIDIA/Linux because that is where heterogeneous CUDA topology and automatic toolkit selection are implemented. Existing PushbuttonLocalCoders Ollama, Metal, ROCm, CPU, Docker, and network-node paths remain available and unchanged.
