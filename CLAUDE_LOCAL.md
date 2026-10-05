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

All ordinary Claude Code arguments are passed through:

```bash
claude-local qwen3.6:35b --resume
claude-local qwen3.8:27b --continue
claude-local qwen3.6:35b --permission-mode plan
claude-local qwen3.6:35b -- --resume SESSION_ID
```

Harness-specific flags must appear before the first ordinary Claude Code flag.

## Hardware planner

The planner inventories every visible NVIDIA GPU independently, including free VRAM, compute capability, PCIe generation, and link width. It then:

1. Allocates the Sonnet/daily-driver model first.
2. Prefers the smallest GPU count that can run a high-quality full-context quant, reducing PCIe traffic and preserving other GPUs for concurrent agents.
3. Selects the highest-quality registered quant within that device-count tier.
4. Gives each concurrently served unique model an exclusive GPU set by default; it does not intentionally overcommit VRAM.
5. Uses layer splitting when a model must span GPUs.
6. On a heterogeneous layer split, places the slowest PCIe endpoint at an edge rather than between faster cards.
7. Builds CUDA for all compute capabilities in the plan, so mixed Volta/Ada hosts can use one compatible build.

The default physical context is **262,144 tokens**. The default Claude Code client budget is **200,000 tokens**, leaving physical headroom for large tool results and token-accounting differences before the backend ceiling.

Run a dry plan before downloading anything:

```bash
claude-local qwen3.6:35b qwen3.8:27b --local-dry-run
```

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

Automatic CUDA unified-memory spill is disabled by default because it can destroy interactive decode performance. `--local-allow-offload` enables it as an emergency safety net, but the planner still does not deliberately choose an overcommitted plan.

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
If the classifier fails, its HTTP error is returned; there is no alternate
backend, cloud fallback, synthetic safety verdict, or automatic permission-mode
change. A live GPU/Claude Code test is required to establish latency and safety
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
