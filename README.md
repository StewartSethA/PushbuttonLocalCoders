# Pushbutton Local Coders

> **Pick local models. Pushbutton handles the hardware.**
> `claude-local` turns one or more local models into a hardware-aware Claude Code
> team: it selects GPUs, quants, context/cache settings and ports, provisions
> CUDA/nvcc and llama.cpp when needed, and keeps the session local.

---

## 🚀 Recommended: `claude-local`

### Test this PR now

Until PR #9 is merged, this one-liner installs/updates the PR branch and launches
Qwen3.6-35B-A3B using the best fitting local profile for the machine:

```bash
curl -fsSL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/refs/heads/feature/claude-local-harness/install-claude-local.sh \
  | PUSHBUTTON_REF=feature/claude-local-harness bash -s -- qwen3.6:35b
```

To inspect the hardware/model plan without downloading or starting a model:

```bash
curl -fsSL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/refs/heads/feature/claude-local-harness/install-claude-local.sh \
  | PUSHBUTTON_REF=feature/claude-local-harness bash -s -- \
    qwen3.6:35b qwen3.8:27b --local-dry-run
```

### After merge

Install the command once:

```bash
curl -fsSL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-claude-local.sh | bash
```

Then use it from any project directory:

```bash
# Let Pushbutton choose a fast local default
claude-local

# One model for all Claude roles
claude-local qwen3.6:35b

# Keep normal Claude Code flags
claude-local qwen3.6:35b --resume

# Four explicitly selected local role models:
# Haiku, Sonnet (daily driver), Opus, Fable
claude-local \
  nemotron-3.5-lightning \
  qwen3.6:35b \
  qwen3.8:27b \
  glm-5.3-flash \
  --resume
```

Model order is positional:

| Models supplied | Claude role mapping |
|---|---|
| 1 | Haiku = Sonnet = Opus = Fable |
| 2 | Haiku = #1; Sonnet/Opus/Fable = #2 |
| 3 | Haiku = #1; Sonnet = #2; Opus/Fable = #3 |
| 4 | Haiku, Sonnet, Opus, Fable respectively |

Current built-in model families are `qwen3.6:35b`, `qwen3.8:27b`,
`nemotron-3.5-lightning`, and `glm-5.3-flash`. Users choose model families;
Pushbutton chooses the concrete GGUF quant and placement.

### What `claude-local` automates

- inventories each NVIDIA GPU independently, including **currently free VRAM**, compute capability and negotiated PCIe generation/width
- for one model, prefers the **freest viable GPU**, breaking ties by PCIe link speed
- for several models, **plans the complete disjoint GPU assignment before launching anything**, avoiding greedy placements that strand a later model
- prefers one-GPU residency and preserves spare GPUs for parallel agents; uses llama.cpp layer splitting only when needed
- selects curated model quants, Q4/Q4 KV where appropriate, and a **262,144-token physical context** by default
- gives Claude Code a conservative 200,000-token client budget so auto-compaction occurs before the physical backend ceiling
- installs build dependencies, Claude Code and a private compatible CUDA toolkit/nvcc when the host lacks a suitable one
- builds CUDA llama.cpp for the actual compute capabilities in the selected GPU plan (for example SM70 V100 + SM89 Ada)
- downloads/caches the selected GGUF automatically
- applies the fixed Qwen agent/tool Jinja template for Qwen3.6/3.8
- auto-selects free backend and gateway ports so multiple running instances can coexist
- exposes local Anthropic-compatible routing for Claude Code and injects `local-fast`, `local-coder`, `local-reviewer`, and `local-deep` subagents
- has **no cloud fallback** in the `claude-local` path; CUDA unified-memory spill is disabled unless explicitly requested

Useful commands:

```bash
claude-local doctor
claude-local models
claude-local qwen3.6:35b qwen3.8:27b --local-dry-run
claude-local install            # ~/.local/bin/claude-local
claude-local install --system   # /usr/local/bin/claude-local
```

The new hardware-aware `claude-local` path is currently NVIDIA/Linux-first.
The existing installer below retains the repository's Ollama, Apple Silicon,
ROCm, CPU, Docker, benchmarking and network-node workflows.

---

## General installer / existing workflows

The original all-in-one installer remains available:

```bash
curl -fsSL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install.sh | bash
```

Or clone and run locally:

```bash
git clone https://github.com/StewartSethA/PushbuttonLocalCoders.git
cd PushbuttonLocalCoders
bash install.sh          # "just get me running" mode
```

---

## Modes

| Flag | Description |
|------|-------------|
| *(default)* `--quick` | Install Ollama + Claude Code, auto-select and pull the best coder model for your hardware, then drop you into an interactive Claude Code session bridged to that local model |
| `--explore` | Run rapid model/quant ablations to find the optimal setup for this machine |
| `--agent` | Wrap a project directory in a sandboxed Docker agent (dangerously-skip-permissions enabled) |
| `--team` | Launch a full orchestrator + developer agent team via Docker Compose |
| `--monitor` | Live TUI showing GPU, CPU, VRAM and RAM utilisation |
| `--benchmark` | Run a short Ollama benchmark, compare actual vs guessed tok/s, and save the runtime profile |
| `--nodes` | Monitor networked boxes (GPU/CPU/VRAM/RAM over SSH) |
| `--build-llamacpp` | Build llama.cpp with GPU acceleration (CUDA / Metal / ROCm) and CPU fallback |
| `--orchestrator` | Start a local multi-agent orchestrator + developer processes |
| `--submit-benchmarks` | Prepare a system benchmark contribution file (and optionally a PR) with true PP/TG data |

### Examples

```bash
# Install everything, then jump straight into Claude Code on the local model
bash install.sh

# Explore optimal model/quant for your hardware
bash install.sh --explore

# Run a single coding agent on your project
bash install.sh --agent --project ~/my-project --task "Add unit tests"

# Multi-agent team (orchestrator + 2 developers)
bash install.sh --team --project ~/my-project --devs 2 --task "Refactor and optimise"

# Live hardware monitor
bash install.sh --monitor

# Benchmark the current Ollama runtime (GPU by default)
bash install.sh --benchmark

# Force the post-setup benchmark in quick mode
bash install.sh --quick --run-benchmark

# Benchmark Ollama in CPU-only mode
bash install.sh --benchmark --framework ollama-cpu

# Monitor networked GPU boxes
bash install.sh --nodes

# Scan configured nodes for reachable Ollama instances and their models
bash install.sh --nodes scan

# Build llama.cpp with GPU support
bash install.sh --build-llamacpp
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(optional)* | Anthropic key for Claude cloud features |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `DEVELOPER_MODEL` | *auto-selected* | Override developer model (Ollama tag) |
| `ORCHESTRATOR_MODEL` | *auto-selected* | Override orchestrator model |
| `PUSHBUTTON_CLAUDE_GATEWAY_PORT` | `4000` | LiteLLM bridge port used to connect Claude Code to the local Ollama model |
| `PUSHBUTTON_PROMPT_FOR_ANTHROPIC_KEY` | `0` | Set to `1` to prompt for an Anthropic API key during setup |
| `PUSHBUTTON_DIR` | `~/.local/share/pushbutton` | Install directory when run via curl |
| `RUNTIME_ENV_FILE` | `~/.config/pushbutton/runtime.env` | Saved local runtime profile from the latest benchmark |

---

## Architecture

```
install.sh                  ← Existing all-in-one entry point
install-claude-local.sh     ← Curl-safe claude-local bootstrap
claude-local                ← Hardware-aware local Claude Code launcher
lib/
  claude_local_plan.py      ← GPU/model/quant/context placement scheduler
  claude_local_gateway.py   ← Local Anthropic-wire role router
  detect_hardware.sh        ← GPU/CPU/VRAM/RAM detection (Linux, Mac, Windows)
  select_model.sh           ← Legacy/general model + quant selection
  install_ollama.sh         ← Ollama install + service management + model pull
  install_claude.sh         ← Claude Code install + LiteLLM/Ollama bridge setup
  install_llamacpp.sh       ← llama.cpp build (CUDA / Metal / ROCm / CPU)
  ablation.sh               ← Rapid model/quant benchmarking with progress bar
  benchmark.sh              ← Quick Ollama speed benchmark + runtime persistence
  tui.sh                    ← TUI helpers: progress bars, spinners, monitor
  network_nodes.sh          ← SSH-based remote GPU/CPU/RAM monitor + Ollama scan
  orchestrator.sh           ← Local multi-agent orchestrator
  docker_agent.sh           ← Docker sandbox wrapper
agents/
  Dockerfile.agent          ← Minimal agent container image
  agent_entrypoint.sh       ← Container entrypoint (task → Ollama)
  docker-compose.yml        ← Multi-agent Compose file (ollama + orchestrator + developer)
configs/                    ← User config files (nodes.txt, claude.env, etc.)
```

---

## Model Selection Logic

The general installer's `select_model.sh` keeps its catalogue intentionally modern:

| Model | Purpose | Native Context | Notes |
|-------|---------|----------------|-------|
| Qwen 3.6 35B-A3B | Primary coder | 262,144 | Highest-quality default when it fits |
| Qwen 3.8 27B | Primary coder / fallback | 262,144 | Better fit for smaller single-GPU boxes |
| Nemotron 3.5 Lightning 30B-A3B | Orchestrator / alternate coder | 262,144 | Fast modern option for routing and coding |

Before any model pull, the general installer:

- prompts for confirmation
- shows estimated disk pull and active runtime memory
- shows effective max context for the proposed quant + KV quant
- lets you choose model quant, KV quant, primary coder, and additional coders from terminal dropdowns
- records estimated prompt-processing (PP) and token-generation (TG) tok/s, then compares them with measured values after the benchmark run

Override at any time with `--model <tag>` or the `DEVELOPER_MODEL` env var.

### Benchmark contribution flow

Benchmarks are recorded under `~/.config/pushbutton/benchmarks/` and can be staged for contribution back to this repository:

```bash
bash install.sh --submit-benchmarks
```

That command writes a PR-ready report under `benchmarks/system/` and, when `gh` is installed, can optionally create a benchmark contribution PR automatically.

---

## Quick Benchmark

After `--quick` setup, PushbuttonLocalCoders can immediately prompt to run a short
benchmark on the selected Ollama runtime. The first pass uses a short 128-token
prompt and automatically sizes the generation budget from the guessed total speed
so the run stays under roughly 10 seconds on starter settings.

The benchmark:
- shows an estimated PP/TG progress bar while it runs
- compares actual vs guessed prompt-processing and token-generation speeds
- saves the chosen local runtime profile to `~/.config/pushbutton/runtime.env`
- can optionally run a longer context sweep capped to a projected runtime below 30 minutes

That saved runtime file is meant to be sourced later by local tooling such as
Claude Code wrappers or other scripts that need the currently selected local
model/framework combination.

---

## Multi-Agent Docker Sandbox

```bash
# Single agent with unrestricted file access ("dangerously skip permissions")
bash install.sh --agent --project ~/my-project --task "Add comprehensive tests"

# Full team: Ollama service + orchestrator + developer in Docker Compose
ANTHROPIC_API_KEY=sk-... bash install.sh --team --project ~/my-project
```

The Docker Compose stack:
- **ollama** — GPU-accelerated inference server (NVIDIA passthrough included)
- **orchestrator** — plans and delegates tasks (lighter, faster model)
- **developer** — implements code changes (best-fit coder model)

All containers share the project directory as `/workspace` and
`DANGEROUSLY_SKIP_PERMISSIONS=true` is set so agents can write freely.

---

## Network Node Monitor

Add remote GPU/CPU boxes to `~/.config/pushbutton/nodes.txt` (one IP per line),
then watch their utilisation in real time or scan them for reachable Ollama models:

```bash
bash lib/network_nodes.sh add 192.168.1.10
bash lib/network_nodes.sh scan
bash lib/network_nodes.sh live
```

Requires passwordless SSH to the remote hosts as `$USER` (or set `SSH_USER`).

---

## Requirements

For `claude-local`:

- Linux with an NVIDIA GPU and working NVIDIA driver
- `bash`, `curl`, and `git` (the bootstrap installs missing supported dependencies where practical)
- CUDA toolkit/nvcc do **not** need to be preinstalled; the launcher provisions a compatible private toolkit when needed

For the repository's other workflows:

- **bash** ≥ 4.0
- **curl** or **wget**
- **git**
- Linux, macOS (10.15+), or Windows (WSL2 recommended)
- For GPU builds: CUDA toolkit (NVIDIA), ROCm (AMD), or Xcode (Apple Silicon)
- For Docker modes: Docker Engine ≥ 20 / Docker Compose v2

---

## Licence

MIT
