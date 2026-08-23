# Pushbutton Local Coders

> **Pushbutton bootstrap for a powerful local AI coding assistant.**
> One `curl` command installs Ollama, Claude Code, and the best local coder model
> your hardware can run — with GPU-optimised llama.cpp builds, a multi-agent
> Docker sandbox, hardware ablation, a quick speed benchmark, and a live TUI monitor.

---

## Quick Start

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

## Actions

| Flag | Description |
|------|-------------|
| *(default)* `--quick` | Install Ollama + Claude Code, auto-select and pull the best coder model for your hardware, then drop you into an interactive Claude Code session bridged to that local model |
| `--explore` | Run rapid model/quant ablations to find the optimal setup for this machine |
| `--agent` | Wrap a project directory in a sandboxed Docker agent (dangerously-skip-permissions enabled) |
| `--team` | Launch a full orchestrator + developer agent team via Docker Compose |
| `--monitor` | Live TUI showing GPU, CPU, VRAM and RAM utilisation |
| `--benchmark` | Run a short Ollama benchmark, compare actual vs guessed tok/s, and save the runtime profile |
| `--nodes` | Monitor networked boxes (GPU/CPU/VRAM/RAM over SSH) |
| `--build-llamacpp` | Build llama.cpp with automatic platform acceleration (auto-provisions local CUDA nvcc/libs on NVIDIA when needed, Metal on Apple, ROCm on AMD) |
| `--orchestrator` | Start a local multi-agent orchestrator + developer processes |
| `--prompt-shell <claude\|opencode\|hermes>` | Select prompt shell (default: `claude`) |
| `--network-mode <host\|bridge\|none>` | Configure agent sandbox network access |
| `--request "<target request>"` | Resolve explicit target request (example: `"V100 16GB 4-bit qwen3.8:27b model"`) |
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
install.sh                  ← Single entry point (curl-installable)
lib/
  detect_hardware.sh        ← Hardware capability profile (GPU/CPU/VRAM/RAM/CUDA/Apple/CPU caps)
  select_model.sh           ← Model + quant selection based on inference memory
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

## Pushbutton capability flow

Pushbutton now follows a unified capability-driven flow with no model-specific or CUDA-specific install mode:

1. Auto-discover hardware capabilities.
2. Auto-recommend local models and quants.
3. Optionally add cloud providers/models.
4. Auto-populate local+cloud agent inventory pinned to hardware targets.
5. Launch selected prompt shell (Claude Code by default).

On NVIDIA systems, local CUDA nvcc/libraries are automatically provisioned when system CUDA is missing or unsuitable.
On Apple Silicon, Apple-specific optimizations are applied automatically.
CPU optimization packs are offered when CPU fallback models are selected.

## Model Selection Logic

`select_model.sh` now keeps the catalogue intentionally modern:

| Model | Purpose | Native Context | Notes |
|-------|---------|----------------|-------|
| Qwen 3.6 35B-A3B | Primary coder | 262,144 | Highest-quality default when it fits |
| Qwen 3.8 27B | Primary coder / fallback | 262,144 | Better fit for smaller single-GPU boxes |
| Nemotron 3.5 Lightning 30B-A3B | Orchestrator / alternate coder | 262,144 | Fast modern option for routing and coding |

Before any model pull, the installer now:

- prompts for confirmation
- shows estimated disk pull and active runtime memory
- shows effective max context for the proposed quant + KV quant
- lets you choose model quant, KV quant, primary coder, and additional coders from terminal dropdowns
- records estimated prompt-processing (PP) and text-generation (TG) tok/s, then compares them with measured values after the benchmark run

Override at any time with `--model <tag>` or the `DEVELOPER_MODEL` env var.  
Add cloud providers with `PUSHBUTTON_CLOUD_PROVIDERS` (`name|url|auth|token;...`) and cloud models with `PUSHBUTTON_CLOUD_MODELS` (`model1,model2`).

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

- **bash** ≥ 4.0
- **curl** or **wget**
- **git**
- Linux, macOS (10.15+), or Windows (WSL2 recommended)
- For GPU builds: CUDA toolkit (NVIDIA), ROCm (AMD), or Xcode (Apple Silicon)
- For Docker modes: Docker Engine ≥ 20 / Docker Compose v2

---

## Licence

MIT
