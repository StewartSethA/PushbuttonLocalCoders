# Pushbutton Local Coders

> **Pick local models. Pushbutton handles the hardware.**
>
> Run local coding agents through Claude Code, Hermes Bot Mode, Qwen Code,
> OpenCode, DeepSeek Harness, or mini-SWE-agent without hand-picking GGUF quants,
> CUDA toolkits, GPU splits, or ports.

## Install

Claude Code frontend:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-claude-local.sh \
  | bash -s -- --system
```

Hermes Bot Mode frontend:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-hermes-local.sh \
  | bash -s -- --system
```

Replica-aware coder frontends (Qwen Code, OpenCode, DeepSeek Harness, mini-SWE):

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-coder-local.sh \
  | bash -s -- --system
```

The model cache, CUDA toolkits, llama.cpp builds, and hardware state are shared
under the same Pushbutton state/cache directories.

## Models

Current built-in model selectors include:

```text
qwen3.8-flash-next
qwen3.8:27b
qwen3.6:35b
nemotron-3.5-lightning
deepseek-v4-flash
glm-5.3-flash
```

You choose the model family. Pushbutton chooses the concrete GGUF profile that
fits the live hardware and free VRAM.

## Same hardware, different frontend

For the replica-aware coding CLIs, change only the command name:

```bash
qwen-local     qwen3.8-flash-next --agents 2
opencode-local qwen3.8-flash-next --agents 2
deepseek-local qwen3.8-flash-next --agents 2
```

All three request two independent model replicas. On an otherwise-free 8x V100
32 GB machine, the planner targets two 4-GPU Flash-Next workers rather than one
8-GPU server with serialized agents.

mini-SWE-agent is intentionally different because upstream mini-SWE is a single
autonomous trajectory rather than a nested-subagent frontend. Use parallel swarm
mode to run one independent trajectory per replica:

```bash
mini-swe-local qwen3.8-flash-next --agents 2 \
  --swarm "Fix the failing tests, find the root cause, implement a minimal repair, and verify it."
```

For an interactive single mini-SWE trajectory:

```bash
mini-swe-local qwen3.8-flash-next
```

Extra replicas do not accelerate one interactive mini-SWE trajectory; `--swarm`
is what makes them concurrent.

## 8x V100 examples

### Two strong independent Flash-Next coders: 4 + 4 GPUs

```bash
qwen-local qwen3.8-flash-next --agents 2
```

Equivalent frontend changes:

```bash
opencode-local qwen3.8-flash-next --agents 2
deepseek-local qwen3.8-flash-next --agents 2
mini-swe-local qwen3.8-flash-next --agents 2 --swarm "<task>"
```

### Four heterogeneous coders: 4 + 1 + 1 + 1 GPUs, one V100 spare

```bash
qwen-local \
  qwen3.8-flash-next \
  qwen3.8:27b \
  qwen3.6:35b \
  nemotron-3.5-lightning \
  --agents 4
```

The intended topology on 8x V100 32 GB is:

```text
worker 1: Qwen3.8-Flash-Next   -> 4 GPUs
worker 2: Qwen3.8 27B          -> 1 GPU
worker 3: Qwen3.6 35B-A3B      -> 1 GPU
worker 4: Nemotron 3.5         -> 1 GPU
spare                           -> 1 GPU
```

Use the same model list with `opencode-local` or `deepseek-local` to change the
coding frontend while preserving the hardware plan.

## Frontend behavior

| Command | Frontend | Multi-agent behavior |
|---|---|---|
| `claude-local` | Claude Code | Claude subagents / role models |
| `hermes-local` | Hermes Desktop / CLI | durable Bot profiles + manager/orchestrator |
| `qwen-local` | Qwen Code | native Agent Team + per-agent local model endpoints |
| `opencode-local` | OpenCode | primary orchestrator + per-subagent model endpoints |
| `deepseek-local` | DeepSeek Harness | replica pool behind the DeepSeek TUI |
| `mini-swe-local` | mini-SWE-agent | independent parallel trajectories in `--swarm` mode |

Qwen Code and OpenCode are the cleanest native matches for heterogeneous local
subagents because both can bind different workers to different model endpoints.
DeepSeek Harness currently consumes a replica pool through the local OpenAI
router. mini-SWE is best treated as a race/ensemble of autonomous trajectories,
not as a nested-subagent UI.

## Claude Code

One local model for all Claude roles:

```bash
claude-local qwen3.6:35b
```

Resume the current project's Claude Code session:

```bash
claude-local qwen3.6:35b --resume
```

Four positional role models (Haiku, Sonnet, Opus, Fable):

```bash
claude-local \
  nemotron-3.5-lightning \
  qwen3.6:35b \
  qwen3.8:27b \
  qwen3.8-flash-next \
  --resume
```

Claude Code receives local `local-fast`, `local-coder`, `local-reviewer`, and
`local-deep` subagents. `claude-local` has no cloud fallback.

## Hermes Bot Mode

Launch Hermes Desktop Bot Mode:

```bash
hermes-local qwen3.6:35b
```

Or use the terminal UI:

```bash
hermes-local qwen3.6:35b --tui
hermes-local qwen3.6:35b --tui --resume
```

Hermes creates durable profiles such as:

```text
pushbutton-manager
pushbutton-fast
pushbutton-coder
pushbutton-reviewer
pushbutton-deep
```

The manager delegates to the specialist Bots. Their memories/profiles persist;
Pushbutton updates only the local endpoint/model wiring when the hardware plan
changes.

## Qwen Code

```bash
# One coder
qwen-local qwen3.8:27b

# Two true independent replicas
qwen-local qwen3.8-flash-next --agents 2

# Four heterogeneous workers
qwen-local qwen3.8-flash-next qwen3.8:27b qwen3.6:35b nemotron-3.5-lightning --agents 4
```

Qwen Code Agent Team is enabled and each worker gets its own local endpoint.

## OpenCode

```bash
opencode-local qwen3.8-flash-next --agents 2
```

Pushbutton writes a local OpenCode provider/agent configuration with a primary
orchestrator and separate subagents backed by the allocated local workers.

## DeepSeek Harness

```bash
deepseek-local deepseek-v4-flash --agents 1
```

Or put two Flash-Next replicas behind the Harness frontend:

```bash
deepseek-local qwen3.8-flash-next --agents 2
```

Pushbutton installs the official `@deepseek-ai/dsh` Harness and its TUI profile,
then exposes the local replicas through a private OpenAI-compatible router.

## mini-SWE-agent

Interactive:

```bash
mini-swe-local qwen3.8:27b
```

Parallel independent repair attempts:

```bash
mini-swe-local qwen3.8-flash-next --agents 2 --swarm "Fix issue #123 and run the full test suite"
```

Each swarm worker writes its own run log under the Pushbutton frontend state.

## Replica planner semantics

For the replica-aware coder frontends:

```text
one model + --agents N  => N independent replicas of that model
N model names           => one independent worker/backend per model
N models + --agents N   => explicit heterogeneous N-worker team
```

Workers receive disjoint GPU sets. The planner minimizes GPU count per worker
while choosing the best profile that fits its assigned free VRAM. This is what
allows 8x V100 to become either 2x4-GPU large-model workers or a 4+1+1+1 team.

Each llama.cpp worker currently uses `-np 1`; concurrency comes from independent
replicas rather than making several agents serialize through one inference slot.

## Web search

Web search is a frontend/tool capability, not part of the model weights.
Recommended defaults are:

```text
search/discovery: Exa
page extraction: Firecrawl
private search:   SearXNG + self-hosted Firecrawl
```

Hermes has native web search/extraction. OpenCode also exposes web search/fetch.
For local-model Claude Code and Qwen Code, use an MCP search provider so web
research remains independent of the model/API vendor.

See `WEB_SEARCH.md` for details.

## Useful diagnostics

```bash
claude-local doctor
claude-local models
claude-local qwen3.8:27b --local-dry-run

# Replica planner synthetic/self test
coder-local --self-test
```

All model downloads and llama.cpp startup output are intentionally visible in the
foreground terminal while services are starting.

## Architecture

```text
claude-local                 Claude Code / Anthropic-wire frontend
hermes-local                 Hermes Bot Mode frontend
coder-local                  replica-aware OpenAI-compatible worker launcher
qwen-local                   Qwen Code frontend
opencode-local               OpenCode frontend
deepseek-local               DeepSeek Harness frontend
mini-swe-local               mini-SWE parallel trajectory frontend
lib/claude_local_plan.py     role-oriented Claude/Hermes planner
lib/coder_local_plan.py      replica-aware coding-worker planner
lib/claude_local_gateway.py  Anthropic-wire local router
lib/coder_local_lb.py        OpenAI-compatible replica router
```

The newer hardware-aware launchers are currently NVIDIA/Linux-first. The legacy
`install.sh` remains available for Ollama, Apple Silicon, ROCm, CPU, Docker,
benchmarking, and network-node workflows.

## Legacy/general installer

```bash
curl -fsSL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install.sh | bash
```

## License

MIT
