# Pushbutton Local Coders

> **Pick local models. Pushbutton handles the hardware and frontend.**
>
> The NVIDIA/Linux path can provision CUDA/nvcc, build llama.cpp, choose a fitting
> GGUF quant, place independent workers across GPUs, and launch several coding
> frontends against the resulting local model servers.

---

## Install

Claude Code frontend:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-claude-local.sh \
  | bash -s -- --system
```

Replica-aware coder frontends (Qwen Code, OpenCode, DeepSeek Harness, mini-SWE-agent):

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-coder-local.sh \
  | bash -s -- --system
```

Hermes Desktop/Bot Mode frontend:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-hermes-local.sh \
  | bash -s -- --system
```

---

## Data Storage

| Environment variable | Default | Contents |
| --- | --- | --- |
| `PUSHBUTTON_DIR` | `~/.local/share/pushbutton` | Repository in `PushbuttonLocalCoders/` |
| `CLAUDE_LOCAL_STATE` | `~/.local/share/pushbutton/claude-local` | Logs, CUDA, llama.cpp builds, templates, Qwen state |
| `CLAUDE_LOCAL_CACHE` | `~/.cache/pushbutton/llama` | Model weights (space critical) |
| `PUSHBUTTON_CONFIG_DIR` | `~/.config/pushbutton-local` | `folders.json`, placement configuration |

Installers and first startup offer folder customization in a terminal. Piped
installs use defaults without prompting and still save `folders.json`. Environment
overrides take precedence over saved paths; `--local-cache DIR` takes highest cache
precedence. Startup choices are saved for later sessions. State, cache, and config
paths must not overlap.

```bash
claude-local --folders    # paths, used space, free space
qwen-local --system-info  # same report; no GPU or model download required
hermes-local --folders
```

Normal startup prints storage paths (`--quiet` suppresses that banner) and each
backend prints a safely quoted `tail -n 50 -F -- ...` log-follow command.
Before model downloads, exact Hugging Face file metadata is checked against
remaining disk space plus a fixed 10 GiB reserve, consistently applied by plan
preflight and both download modes. Unknown sizes or insufficient space
stop startup with status 2 rather than retrying a doomed download. VRAM planning
is separate from this disk-space check.

For a two-GPU setup with a separate model disk:

```bash
export CLAUDE_LOCAL_CACHE=/mnt/nvme/pushbutton-models
qwen-local 'qwen3.8:27b@gpu=0,vram=16G' 'qwen3.6:35b@gpu=1,vram=16G' --agents 2
```

For multiple users, each can set `CLAUDE_LOCAL_CACHE=/srv/models/pushbutton`
while keeping private state/config directories. Use a trusted group-writable
cache and serialize downloads; different users' startup locks do not coordinate.
Do not make state or config world-writable.

To move data, stop all backends, copy the entire cache (including blobs and
symlinks) to the new disk, set `CLAUDE_LOCAL_CACHE` or edit `folders.json`, and
verify with `--folders` before removing the old copy. See [STORAGE.md](STORAGE.md)
for the schema, migration steps, and troubleshooting. Third-party frontend
installations may retain their own data: Claude Code in `~/.claude` and Hermes
profiles/sessions in `~/.hermes`.
## Start here: help and model selection

Run an installed frontend with no model in a terminal (`qwen-local`,
`opencode-local`, `deepseek-local`, `mini-swe-local`, `claude-local`, or
`hermes-local`) to see a welcome menu: **choose models, inspect hardware/models,
preview a plan, help, or quit**. `--select` explicitly opens that menu.
Nothing is downloaded, built, or launched until you confirm the placement preview.
Piped/non-TTY no-model launches print help and exit without prompting; `--help`
does not require GPUs or the frontend toolchain. Explicit model commands keep
their existing unattended behavior.

```bash
qwen-local --select
pushbutton-select qwen3.8-flash-next qwen3.8:27b --agents 2 --plan-only
pushbutton-select 'qwen3.8:27b@gpu=0,vram=16G' \
  'qwen3.6:35b@gpu=1,vram=16G' --agents 2 --context 65536 --json
pushbutton-select --frontend claude-local qwen3.8:27b qwen3.6:35b --plan-only
```

Coder selections use independent workers and the existing joint disjoint-GPU
planner. Claude/Hermes keep their shared model-to-role semantics, not replicas.
The preview shows free/occupied/total VRAM, per-GPU leases, quant envelopes,
context, and spare GPUs. A smaller selected physical context also lowers the
default client budget; an explicitly larger client budget is rejected.
See [selector and Flash-Next validation details](docs/BACKEND_SELECTOR.md).

### Per-model slots and token budgets

Model specs accept capacity overrides alongside GPU placement:

```bash
qwen-local 'qwen3.8:27b@gpu=0,slots=2,context=65536,output=4096' \
  'qwen3.6:35b@gpu=1,slots=1,context=32768,output=4096' --agents 2
claude-local 'qwen3.8:27b@slots=2,context=196608,output=8192,compact=120000' \
  'qwen3.6:35b@slots=1,context=131072,output=8192,compact=100000'
pushbutton --slots 2 --context 65536 --frontend none 'qwen3.8:27b'
```

`context` is the **total input plus output limit per slot**, not the sum across
slots. Slots share one model's weights; `--agents` creates independent replicas.
Repeating model specs with different overrides configures individual coder
replicas. Claude/Hermes roles sharing a server must agree on its settings.

Optional spec keys are `client_context`, `compact`, `safety`, `quant`, `kv_k`, `kv_v`,
`min_tps`, and `admission`, in addition to `slots`, `context`, `output`, `gpu`,
and `vram`. Weight quantization and K/V-cache precision are separate choices.
Explicit choices that cannot fit are rejected rather than silently reducing
context or slots. Unspecified quantization is selected from supported profiles.
Coder placement configuration accepts the same per-model capacity defaults:

```json
{"models":{"qwen3.8:27b":{"gpus":[0],"vram_limit":"16G","slots":2,"context":65536,"output":4096}}}
```

Use `--placement-config` to select that file. Explicit model-spec fields override
configuration defaults and global fallback settings.
Spec token counts accept integer values or binary `K`/`M` suffixes. `compact` accepts
an absolute token count or a fraction/percentage of the resolved input budget;
its trigger must remain strictly below that budget. `safety` is a positive
token reserve.

The plan reports each server's hard context, input limit, output reserve,
compaction trigger, slots, memory envelope, and headroom. Memory estimates use
the profiles' conservative context-dependent envelope, **not an architecture-
exact KV allocation or a measured throughput guarantee**. Admission remains
calibration-based; allocating slots does not automatically prove they are fast.

Input budgets reserve output and a safety margin. Compaction triggers leave
additional headroom for the next turn. Frontend adapters advertise resolved
budgets per model where supported; a shared session uses the smallest compatible
budget. Current upstream framework mappings are:

| Frontend | Budget/compaction mapping |
| --- | --- |
| Qwen Code | Per-model `contextWindowSize` and `samplingParams.max_tokens`; session `context.autoCompactThreshold` uses the earliest safe model ratio. Its built-in reserves may compact earlier. |
| Hermes | Per-role `model.context_length`; `compression.threshold_tokens` applies the absolute trigger, avoiding small-window ratio floors. |
| OpenCode | Per-model `limit.context`, `limit.input`, and `limit.output`; session `compaction.reserved` leaves enough room for every model's trigger. |
| Claude Code | Shared-role budgets are bounded by verified per-slot capacity; `CLAUDE_CODE_AUTO_COMPACT_WINDOW` and `CLAUDE_CODE_MAX_OUTPUT_TOKENS` are configured explicitly in isolated settings. The supported compact window must be at least 100,000 tokens; smaller configurations fail with guidance rather than clamping upward. |
| mini-SWE / DeepSeek Harness | Requests use guarded endpoints; mini-SWE receives per-worker output limits. No verified framework autocompact override is assumed for these adapters. |

These mappings follow current upstream
[Qwen compression](https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/services/chatCompressionService.ts),
[Hermes compression](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/context-compression-and-caching.md),
and [OpenCode overflow handling](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/overflow.ts).
Older frontend versions may have different semantics or ignore these settings.
Framework-specific compaction behavior is not assumed to enforce a hard limit;
the serving guard remains authoritative.
Direct frontend guards default to admission at C1; explicit `admission` may raise
that bound up to the allocated slots, without proving a throughput SLA.
The broker may promote default admission after calibration, while an explicit
admission cap remains binding. `min_tps` is checked against broker calibration
evidence; direct frontend launches warn when its throughput is unproven.
Large tool results can cross a threshold in one turn: summarize or limit them
before retrying an oversized request. History is never silently discarded by
the request guard.

The serving guards count fully rendered requests with backend tokenization where
available, including tool definitions and template overhead. Unsupported counting
paths must not be mistaken for exact token counts; backend context enforcement
and disabled llama.cpp context shifting provide the final safety boundary.
Changing to a smaller-context model requires its own budget check and may require
compacting first.
Unconfigured `pushbutton` launches retain measured backend selection and automatic
slot reservation; explicit placement, slots, quant, or cache overrides use the
llama.cpp capacity planner. Unsupported specialized token-counting paths fail
closed with an actionable error instead of silently changing the chosen backend.

### Qwen3.8 Flash-Next: existing support, not locally measured speed

Flash-Next is already in the llama.cpp planner with six profiles. At 262,144
context their **conservative planning envelopes** are 136/116/112/104/99/94 GiB;
these are not measured allocations. The existing four-32-GiB-card recommendation
is `UD-IQ4_XS` (116 GiB). Multiple independent copies need disjoint GPU groups.

`sglang-v100` (4× V100 32 GiB, NVLink recipe) and `vllm-flashnext-3090`
(4× RTX 3090 24 GiB) are existing **experimental benchmark adapters**, not
drop-in coder/Claude/Hermes launch backends. Architecture eligibility does not
prove memory fit or runtime correctness. The 3090 adapter's default `mtp` profile
is limited to 65,536 context; deeper upstream profiles need separate validation
and, for FP8 KV, calibration. There are **no repository-local Flash-Next
measurements** in this change, so selector PP/TG remains unknown without matching
evidence. Published ~118–120 tok/s V100 MTP and ~115–117 tok/s 3090 MTP results
are upstream claims under specific prompt/profile conditions, not local promises.

No CPU/expert offloading or unified-memory spill is silently enabled. Supporting
a new offload recipe would require a separate adapter, loader/RAM/topology checks,
and measured correctness/performance validation.

---

## Frontends: change the command, keep the model idea

### Claude Code

```bash
claude-local qwen3.8:27b
claude-local qwen3.8:27b --resume
claude-local nemotron-3.5-lightning qwen3.6:35b qwen3.8:27b --resume
```

`claude-local` maps one to four positional models onto Haiku/Sonnet/Opus/Fable,
starts the needed local backends, injects local subagents, and passes normal
Claude Code flags through, with validation of context-policy overrides.

It verifies live llama.cpp **per-slot capacity**, derives explicit Claude
auto-compaction/output budgets, and checks the full native-tokenized request
before inference. Auto-compaction alone cannot prevent every oversized tool
result or resumed transcript. See [context budgeting and recovery](CLAUDE_LOCAL.md#context-budgeting-and-auto-compaction)
for version requirements, isolated settings, and safe overrides.

### Hermes Bot Mode

```bash
hermes-local qwen3.8:27b
hermes-local nemotron-3.5-lightning qwen3.6:35b qwen3.8:27b
```

Hermes creates durable Pushbutton manager/fast/coder/reviewer/deep Bot profiles.
Use the Desktop UI to delegate work between the Bots; their memory/profile state
persists while the local model endpoints are refreshed on each launch.

### Qwen Code

```bash
qwen-local qwen3.8:27b --agents 1
qwen-local qwen3.8-flash-next --agents 2
```

Qwen Code uses native Agent Team/subagent model assignments. With one model and
`--agents N`, Pushbutton creates N independent model replicas when hardware permits.

### OpenCode

```bash
opencode-local qwen3.8:27b --agents 1
opencode-local qwen3.8-flash-next --agents 2
```

OpenCode gets a primary Pushbutton orchestrator and independent subagents bound
to the planned local endpoints.

### DeepSeek Harness

```bash
deepseek-local qwen3.8:27b --agents 1
deepseek-local qwen3.8-flash-next --agents 2
```

The DeepSeek Harness frontend uses the Pushbutton replica pool through a local
OpenAI-compatible router.

### mini-SWE-agent

Interactive single trajectory:

```bash
mini-swe-local qwen3.8:27b --agents 1
```

True parallel independent trajectories/race:

```bash
mini-swe-local qwen3.8-flash-next --agents 2 \
  --swarm "Fix the failing tests, implement the repair, and verify it."
```

mini-SWE is not a nested-subagent frontend; `--swarm` deliberately launches one
independent mini-SWE trajectory per planned backend.

---

## Replica-aware multi-GPU examples

### 8x V100 32 GB: recommended two-agent high-end layout

```bash
qwen-local qwen3.8-flash-next --agents 2
```

Target placement: 4 V100s + 4 V100s.

The same hardware plan can be used with another supported coder frontend by
changing the command name:

```bash
opencode-local qwen3.8-flash-next --agents 2
deepseek-local qwen3.8-flash-next --agents 2
```

### 8x V100 32 GB: recommended four-agent heterogeneous layout

```bash
qwen-local \
  qwen3.8-flash-next \
  qwen3.8:27b \
  qwen3.6:35b \
  nemotron-3.5-lightning \
  --agents 4
```

Target placement: 4 + 1 + 1 + 1 GPUs, deliberately preserving one V100 for
other work. The planner treats tiny quant-quality differences as less important
than preserving a useful spare GPU when quality is otherwise close.

### 8x V100 32 GB: two different large models

DeepSeek + Flash-Next:

```bash
qwen-local deepseek-v4-flash qwen3.8-flash-next --agents 2
```

GLM + Flash-Next:

```bash
qwen-local glm-5.3-flash qwen3.8-flash-next --agents 2
```

These are large-model alternatives, not additions to the four-agent mixed
layout: each large model can consume roughly four 32 GB V100s at the selected
full-context profile.

### Explicit GPU pins and per-model VRAM leases

Placement constraints can be attached directly to each model selector:

```bash
qwen-local \
  'qwen3.8:27b@gpu=0,vram=16G' \
  'qwen3.6:35b@gpu=1,vram=16G' \
  --agents 2
```

On a 2x RTX 3090 24 GB system this pins the first model to physical GPU 0 and
the second to GPU 1 while exposing only 16 GiB/card to each model planner. At
256K context the current catalogue chooses `IQ3_XXS` for Qwen3.8-27B and
`UD-IQ3_XXS` for Qwen3.6-35B. The remaining roughly 8 GiB/card stays available
to other work such as training.

Preview the selection without building llama.cpp or downloading weights:

```bash
qwen-local \
  'qwen3.8:27b@gpu=0,vram=16G' \
  'qwen3.6:35b@gpu=1,vram=16G' \
  --agents 2 --plan-only
```

For an exact multi-GPU placement use `+` between physical GPU indices:

```bash
qwen-local 'qwen3.8-flash-next@gpu=0+1+2+3,vram=30G' --agents 1
```

`vram=` is a per-selected-GPU lease. CUDA unified-memory spill remains disabled.
After a capped llama.cpp server loads, Pushbutton also reads that process's
actual allocation from `nvidia-smi`; if it exceeds the lease, the server is
terminated rather than silently taking the reserved memory.

Persistent defaults live at `~/.config/pushbutton-local/placement.json` by
default (override with `--placement-config FILE` or `PUSHBUTTON_PLACEMENT_CONFIG`):

```json
{
  "models": {
    "qwen3.8:27b": {"gpus": [0], "vram_limit": "16G"},
    "qwen3.6:35b": {"gpus": [1], "vram_limit": "16G"}
  }
}
```

With that file present the equivalent launch is simply:

```bash
qwen-local qwen3.8:27b qwen3.6:35b --agents 2
```

Inline `@gpu=` / `@vram=` values override persistent defaults for that launch.

---

## Current model families

The hardware-aware local planner currently knows these selectors:

| Selector | Intended use |
|---|---|
| `ornith-1.5:9b` | Small dense coder; GGUF joint placement |
| `ornith-1.5:35b-a3b` | MoE coder; GGUF joint placement |
| `qwen3.8-flash-next` | Highest-end local coding agent; multi-GPU |
| `qwen3.8:27b` | Strong dense coder; excellent single 24/32 GB GPU target |
| `qwen3.6:35b` | Fast MoE coder; excellent V100/3090 target |
| `nemotron-3.5-lightning` | Fast alternate/orchestrator model |
| `deepseek-v4-flash` | Large high-quality agent model; multi-GPU |
| `glm-5.3-flash` | Large alternate agent model; multi-GPU/forked llama.cpp path |

Aliases such as `q38`, `q36`, `nemotron`, `deepseek`, and `glm` are also accepted.
Pushbutton chooses the actual quant based on live free VRAM and the requested
worker layout.

Ornith aliases `ornith`, `ornith-9b`, and `ornith-35b` are also accepted.
Its GGUF profiles use conservative memory estimates and the embedded chat
template. FIT does not validate GPU execution or speed; vendor capability
benchmarks remain upstream hints, not local measurements. Ornith 397B remains
catalog-only because it has no concrete GGUF planning profiles.

---

## Build/state and model-cache locations

The locations are independently configurable:

- `PUSHBUTTON_DIR` — repository/bootstrap install root
- `CLAUDE_LOCAL_STATE` — builds, private CUDA/GCC, llama.cpp source/builds, logs,
  plans, frontend state
- `CLAUDE_LOCAL_CACHE` — downloaded GGUF/model cache

Example:

```bash
export PUSHBUTTON_DIR=/mnt/fast/pushbutton
export CLAUDE_LOCAL_STATE=/mnt/fast/pushbutton-state
export CLAUDE_LOCAL_CACHE=/mnt/models/pushbutton-llama
```

Then run any frontend normally:

```bash
qwen-local qwen3.8:27b --agents 1
```

Installed `claude-local` additionally accepts:

```bash
claude-local qwen3.8:27b \
  --local-state /mnt/fast/pushbutton-state \
  --local-cache /mnt/models/pushbutton-llama
```

`--local-cache` is also accepted by the replica-aware coder launcher. For a
single convention that works across all coder frontends, the environment
variables are recommended.

---

## Web search / MCP

### Claude Code local models

The installed `claude-local` command creates an isolated MCP configuration with
Exa's hosted MCP endpoint on first use and preserves it on subsequent launches. Local Qwen, Nemotron,
DeepSeek and GLM models therefore get provider-neutral web search and page fetch
inside Claude Code without relying on Anthropic's hosted WebSearch service.

Built-in `WebSearch` and `WebFetch` are disabled for local sessions; the model is
instructed to use the exact connected MCP tool names. Startup discloses the
selected server names and the persistent config path (normally
`~/.local/share/pushbutton/claude-local/web-mcp.json`). Check `/mcp` for a connected
`pushbutton-web` server and use its advertised search/fetch tools.

**Change the backend:** edit the disclosed JSON and restart. The launcher does
not overwrite it. Alternatively, use `--local-web-config /absolute/path/web.json`
before ordinary Claude flags, or set `CLAUDE_LOCAL_WEB_CONFIG` persistently in
your shell environment. Supply a Claude-compatible `mcpServers` configuration
for Exa, Tavily, or a SearXNG MCP adapter; a plain SearXNG HTTP endpoint is not an
MCP server. Keep provider credentials out of source control and prefer the
provider's environment/credential mechanism.

If an old transcript keeps calling `web_search`, start a fresh session with a
handoff summary. The gateway rejects unsupported web calls immediately with MCP
recovery guidance instead of replaying slow inference or suggesting a temporary
outage. Other tool schemas remain strictly validated.

Claude may still ask permission for the MCP search tool. Choose **“Yes, and don't
ask again”** only for the specific read-only search tool and directory you trust.
This consent is separate from backend selection and can be reviewed with
`/permissions`; the launcher does not automatically approve MCP tools. Local
Claude settings/session storage is isolated under the local state directory, so
changing that directory may require consent again.

Disable the automatic web MCP layer for a session with:

```bash
claude-local qwen3.8:27b --local-no-web
```

This does not enable hosted WebSearch/WebFetch. You may still supply custom MCP
tools with Claude's `--mcp-config` flag. For Anthropic's built-in web services,
use a supported Anthropic API deployment outside the local launcher; it never
falls back to the cloud.

### Qwen Code local models

The installed `qwen-local` command automatically loads a Qwen Code system-default
configuration for Exa MCP and restricts it to the read-only/default search and
page-fetch tools (`web_search_exa`, `web_fetch_exa`).

This matters because Qwen Code's built-in server-side web search is not enabled
for arbitrary local model endpoints; MCP is the portable path.

Hermes has its own native web-search/extraction configuration. OpenCode likewise
has native web search/fetch support; those frontends do not require the Qwen or
Claude MCP shim.

---

## Local Claude timeout / streaming policy

Local inference has very different latency from Anthropic's hosted API. A large
prompt on one RTX 3090 or V100 can spend substantial time in prompt processing,
and several Claude subagents can otherwise queue behind a single `llama-server
-np 1` backend.

The installed `claude-local` wrapper therefore defaults to:

- a 30-minute Claude API request budget
- a 15-minute stream-idle budget
- only two API retries instead of repeatedly replaying an expensive local request
- local subagent stall budget of 30 minutes
- read-only/tool/subagent concurrency capped to the number of visible GPUs (max 4)
- one safe gateway retry only before an SSE response is committed downstream

These can all be overridden with the corresponding Claude Code environment
variables (`API_TIMEOUT_MS`, `CLAUDE_STREAM_IDLE_TIMEOUT_MS`,
`CLAUDE_CODE_MAX_RETRIES`, `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY`, etc.).

If you use Claude Code's `auto` permission mode on a slow single-GPU local model,
remember that auto mode itself uses model-classified background safety checks.
`default` or `acceptEdits` avoids adding that classifier traffic when you do not
need it.

For a single RTX 3090, a smaller context is often a better responsiveness/reliability
tradeoff than allocating 262K merely because it fits:

```bash
claude-local qwen3.8:27b \
  --local-context 131072
```

The client budget is derived automatically from verified backend capacity.
With a 131,072-token slot, defaults are a 114,688-token custom-model window,
106,496-token compaction window and 8,192-token output budget, plus reserves.
Claude Code 2.1.221+ and native llama.cpp Anthropic counting are required.
Settings/sessions are isolated under `$CLAUDE_LOCAL_STATE/claude-config`;
ordinary user/project settings (including permission customizations) are not loaded.

`147023 > 131072` is a context-budget error, **not a network error**.
The old client-budget ≤ requested-context check was insufficient.
Lowering a threshold does not retroactively fix an oversized saved transcript:
even `/compact` may fail. Start a **new session** with a saved handoff summary,
without `--resume`/`--continue`; only use a temporary larger backend when its
model/hardware fit has been verified. Do not blindly increase context or VRAM.

---

## What Pushbutton automates

On the current NVIDIA/Linux path it can:

- inventory each NVIDIA GPU and current free VRAM
- account for compute capability and PCIe link capability
- choose curated model quants and KV/cache profiles
- plan disjoint multi-model placements before launching anything
- preserve independent replicas for true agent concurrency
- build CUDA llama.cpp for the selected GPU architectures
- provision a private CUDA toolkit/nvcc and compatible GCC when needed
- share the on-disk GGUF cache across sessions/frontends
- select free ports and isolate frontend/runtime state
- stream model-load/download/server status to the terminal
- inject local Claude subagents, Qwen Agent Team definitions, OpenCode subagents,
  DeepSeek replica routing, or mini-SWE swarm workers
- provide web search/fetch through MCP or the frontend's native tools

CUDA unified-memory spill remains disabled by default; the planner should choose
something that actually fits rather than silently turning VRAM pressure into a
very slow CPU/RAM fallback.

---

## CPU-only and multi-CPU hosts (Xeon Phi, AVX-512, VNNI, AMX, EPYC, ARM)

When no working NVIDIA GPU is found, `pushbutton`, `claude-local` and `coder-local`
automatically deploy every model on the CPU (set `PUSHBUTTON_DISABLE_CPU_FALLBACK=1`
to fail instead). `lib/cpu_platform.py` handles the CPU-specific work:

- **Detection**: `/proc/cpuinfo` and sysfs NUMA nodes. It identifies the CPU family
  (KNL/KNM, Haswell, Skylake-SP, Cascade Lake, Cooper Lake, Ice Lake, Sapphire/Emerald/Granite
  Rapids, Zen 2–5, Neoverse, Apple), sockets, physical cores, NUMA layout, and
  MCDRAM/HBM in flat mode (CPU-less NUMA nodes) or cache mode.
- **Build**: a llama.cpp CPU build per ISA (`build-cpu-<tier>-<hash>`) that enables exactly
  the extensions the host reports: AVX2/FMA/F16C, AVX-512, AVX-512 VNNI (Cascade Lake+),
  AVX-512 BF16, AVX-VNNI, and AMX (Sapphire Rapids+). KNL/KNM lack AVX-512 BW/VL/DQ, so
  they use the AVX2 kernels with `-mtune=knl`. For binaries shared across different hosts,
  `PUSHBUTTON_CPU_BUILD=portable` builds every variant with runtime dispatch.
- **Quant choice**:
  - With MCDRAM/HBM, the best-quality quant whose weights and KV fit the on-package
    memory (flat mode), or whose weights fit the MCDRAM cache (cache mode).
  - Otherwise, the best-quality quant estimated to reach the decode target (`min_tps`,
    default `pushbutton_policy.MIN_DECODE_TOK_S`, 25 tok/s), falling back to the fastest high-quality quant.
- **Launch strategy**:
  - On a Phi in flat mode: `numactl --membind=<MCDRAM node>` with `--no-mmap`.
  - On multi-socket hosts: `--numa distribute`/`isolate`.
  - With several instances: one NUMA node group each.
  - Physical-core thread counts (with SMT batch threads on KNL), and optional
    `--no-mmap`/`--mlock`.
- **Measured over estimated**: once `pushbutton-cpu-bench` has measured a model on an
  identical host layout, the planner uses the fastest measured strategy.

```bash
python3 lib/cpu_platform.py detect                 # what was detected, build flags, capacities
python3 lib/cpu_platform.py plan qwen3.6:35b       # quant, strategy and tok/s estimate on this host
./pushbutton-cpu-bench qwen3.6:35b                 # benchmark every applicable strategy
./pushbutton-cpu-bench --summarize                 # regenerate benchmarks/cpu/RESULTS.md
```

Estimated load time and PP/TG at 0/4K/16K/32K context depth for 21 reference CPUs and every
CPU-planned model are in [benchmarks/cpu/ESTIMATES.md](benchmarks/cpu/ESTIMATES.md).
Measured strategy benchmarks are in [benchmarks/cpu/RESULTS.md](benchmarks/cpu/RESULTS.md).
Install `numactl` for NUMA/MCDRAM binding. Without it, the launchers start unbound and
print a warning.

---

## Telemetry

On first interactive run, `pushbutton` shows the full disclosure and disclaimer
(`pushbutton_metrics.CONSENT_TEXT`) and asks `Enable telemetry? [Y/n]`. Pressing Enter enables it.
Non-interactive first runs never enable telemetry silently. Users who opted in under an older
disclosure are asked again, and nothing is queued or uploaded until they answer. Change your
choice at any time with `--telemetry-on` / `--telemetry-off`; opting out also clears the queue.

When enabled, compact records are queued locally and uploaded **at most once every 5
minutes**. This limit is enforced across processes with a file lock, and failed attempts
also back off. Records contain:
- model weight load time;
- PP and TG tok/s at each context depth;
- model, quant and backend;
- hardware type: GPU names, or CPU model/family/ISA tier, MCDRAM mode and launch strategy.

Records never contain prompts, outputs, usernames, hostnames or paths.

Where the data goes:

| stage | where | contents |
|---|---|---|
| ingest | always-on relay ([telemetry/worker](telemetry/worker), Cloudflare Worker) or self-hosted `pushbutton-telemetry-collector` | adds receive time and a **keyed hash of the sender IP**; the raw IP is never stored |
| raw store | a **private** GitHub data repository | hashed records, readable only by maintainers |
| public | GitHub Pages benchmark dashboard built by `telemetry/aggregate.py` | anonymized aggregates only: medians per model/quant/hardware/strategy/depth, shown only when ≥3 contributors share a group |

The client sends to `upload_url` in `~/.config/pushbutton-local/telemetry.json` (set with
`--telemetry-url`), `PUSHBUTTON_TELEMETRY_UPLOAD_URL`, or `pushbutton_metrics.DEFAULT_UPLOAD_URL`
once the maintainers' relay is deployed. With no URL, records stay queued. Setup is in
[telemetry/README.md](telemetry/README.md).

---

## Diagnostics

```bash
claude-local doctor
claude-local models
```

Local logs live under:

```text
$CLAUDE_LOCAL_STATE/logs/
```

(default `~/.local/share/pushbutton/claude-local/logs/`).

If Claude reports a dropped/incomplete stream after updating, inspect actual
backend errors with:

```bash
grep -Ei 'error|fatal|failed|cuda|out of memory|oom' \
  "${CLAUDE_LOCAL_STATE:-$HOME/.local/share/pushbutton/claude-local}"/logs/*.log \
  | tail -100
```

A remaining incomplete stream accompanied by CUDA/OOM/fatal lines is a backend
failure rather than a Claude timeout. Reduce context or let a future automatic
replan choose a lower-memory profile; do not keep increasing client timeouts for
an unhealthy backend.

---

## General / legacy installer

The original all-in-one installer remains available for Ollama, Apple Silicon,
ROCm, CPU, Docker, benchmarking and network-node workflows:

```bash
curl -fsSL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install.sh | bash
```

The new replica-aware coder path is NVIDIA/Linux-first. On Linux hosts without an NVIDIA GPU,
it deploys on the CPU automatically (see above).

---

## Requirements

For the new local coder path:

- Linux
- working NVIDIA driver / `nvidia-smi`, or CPU-only (automatic fallback; `numactl`
  recommended on NUMA and Xeon Phi hosts)
- `bash`, `curl`, `git`, Python 3
- CUDA toolkit/nvcc do **not** have to be preinstalled; Pushbutton can provision
  a compatible private toolchain

Frontend-specific CLIs are installed automatically when practical.

---

## Download Speed

`claude-local`, `coder-local`, and `hermes-local` download model weights before
starting llama.cpp, with live download progress in the terminal. They use
**aria2c** automatically: up to four simultaneous files and three connections
per file, with resumable chunks. On Linux, a missing aria2c triggers a silent,
non-interactive apt/dnf installation attempt (bounded to 30 seconds plus cleanup).
No password prompts or manual installation are required. If installation or the
parallel download fails, the launcher falls back to the Hugging Face CLI,
automatically installing it in a private runtime virtual environment if needed.

Use `--no-parallel` to skip aria2c and its installation, for example:

```bash
claude-local qwen3.8:27b --no-parallel --local-cache /mnt/models
```

Parallel downloads can substantially improve throughput, but **5–10× is not
guaranteed**: network speed, Hugging Face rate limits, and storage matter.
Metadata and disk-space validation still apply in sequential mode. Network,
authentication, or disk errors are reported rather than starting a server with
incomplete weights. See [STORAGE.md](STORAGE.md) for storage and resume details.

## License

MIT
