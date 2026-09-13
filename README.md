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

## Frontends: change the command, keep the model idea

### Claude Code

```bash
claude-local qwen3.8:27b
claude-local qwen3.8:27b --resume
claude-local nemotron-3.5-lightning qwen3.6:35b qwen3.8:27b --resume
```

`claude-local` maps one to four positional models onto Haiku/Sonnet/Opus/Fable,
starts the needed local backends, injects local subagents, and passes normal
Claude Code flags through unchanged.

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

### 8x V100 32 GB: two independent Flash-Next coders

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

### 8x V100 32 GB: four heterogeneous coders + one spare GPU

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

---

## Current model families

The hardware-aware local planner currently knows these selectors:

| Selector | Intended use |
|---|---|
| `qwen3.8-flash-next` | Highest-end local coding agent; multi-GPU |
| `qwen3.8:27b` | Strong dense coder; excellent single 24/32 GB GPU target |
| `qwen3.6:35b` | Fast MoE coder; excellent V100/3090 target |
| `nemotron-3.5-lightning` | Fast alternate/orchestrator model |
| `deepseek-v4-flash` | Large high-quality agent model; multi-GPU |
| `glm-5.3-flash` | Large alternate agent model; multi-GPU/forked llama.cpp path |

Aliases such as `q38`, `q36`, `nemotron`, `deepseek`, and `glm` are also accepted.
Pushbutton chooses the actual quant based on live free VRAM and the requested
worker layout.

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

The installed `claude-local` command automatically supplies an isolated MCP
configuration containing Exa's hosted MCP endpoint. Local Qwen, Nemotron,
DeepSeek and GLM models therefore get provider-neutral web search and page fetch
inside Claude Code without relying on Anthropic's hosted WebSearch service.

Disable the automatic web MCP layer for a session with:

```bash
claude-local qwen3.8:27b --local-no-web
```

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
  --local-context 131072 \
  --local-client-context 100000 \
  --resume
```

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

The new replica-aware coder path is currently NVIDIA/Linux-first.

---

## Requirements

For the new local coder path:

- Linux
- working NVIDIA driver / `nvidia-smi`
- `bash`, `curl`, `git`, Python 3
- CUDA toolkit/nvcc do **not** have to be preinstalled; Pushbutton can provision
  a compatible private toolchain

Frontend-specific CLIs are installed automatically when practical.

---

## License

MIT
