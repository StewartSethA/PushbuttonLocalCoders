# PushbuttonLocalCoders

> **Pushbutton bootstrap for a powerful local AI coding assistant.**
> One `curl` command installs Ollama, Claude CLI, and the best local coder model
> your hardware can run — with GPU-optimised llama.cpp builds, a multi-agent
> Docker sandbox, hardware ablation, and a live TUI monitor.

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

## Modes

| Flag | Description |
|------|-------------|
| *(default)* `--quick` | Install Ollama + Claude CLI after an interactive modern-model plan review |
| `--explore` | Run PP/TG benchmarks for the modern model/quant combinations that fit this machine |
| `--agent` | Wrap a project directory in a sandboxed Docker agent (dangerously-skip-permissions enabled) |
| `--team` | Launch a full orchestrator + developer agent team via Docker Compose |
| `--monitor` | Live TUI showing GPU, CPU, VRAM and RAM utilisation |
| `--nodes` | Monitor networked boxes (GPU/CPU/VRAM/RAM over SSH) |
| `--build-llamacpp` | Build llama.cpp with GPU acceleration (CUDA / Metal / ROCm) and CPU fallback |
| `--orchestrator` | Start a local multi-agent orchestrator + developer processes |
| `--submit-benchmarks` | Prepare a system benchmark contribution file (and optionally a PR) with true PP/TG data |

### Examples

```bash
# Explore optimal model/quant for your hardware
bash install.sh --explore

# Run a single coding agent on your project
bash install.sh --agent --project ~/my-project --task "Add unit tests"

# Multi-agent team (orchestrator + 2 developers)
bash install.sh --team --project ~/my-project --devs 2 --task "Refactor and optimise"

# Live hardware monitor
bash install.sh --monitor

# Monitor networked GPU boxes
bash install.sh --nodes

# Build llama.cpp with GPU support
bash install.sh --build-llamacpp
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(prompt)* | Anthropic key for Claude cloud features |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `DEVELOPER_MODEL` | *auto-selected* | Override developer model (Ollama tag) |
| `ORCHESTRATOR_MODEL` | *auto-selected* | Override orchestrator model |
| `PUSHBUTTON_DIR` | `~/.local/share/pushbutton` | Install directory when run via curl |

---

## Architecture

```
install.sh                  ← Single entry point (curl-installable)
lib/
  detect_hardware.sh        ← GPU/CPU/VRAM/RAM detection (Linux, Mac, Windows)
  select_model.sh           ← Model + quant selection based on inference memory
  install_ollama.sh         ← Ollama install + service management + model pull
  install_claude.sh         ← Claude CLI install + API key config
  install_llamacpp.sh       ← llama.cpp build (CUDA / Metal / ROCm / CPU)
  ablation.sh               ← Rapid model/quant benchmarking with progress bar
  tui.sh                    ← TUI helpers: progress bars, spinners, monitor
  network_nodes.sh          ← SSH-based remote GPU/CPU/RAM monitor
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

### Benchmark contribution flow

Benchmarks are recorded under `~/.config/pushbutton/benchmarks/` and can be staged for contribution back to this repository:

```bash
bash install.sh --submit-benchmarks
```

That command writes a PR-ready report under `benchmarks/system/` and, when `gh` is installed, can optionally create a benchmark contribution PR automatically.

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
then watch their utilisation in real time:

```bash
bash lib/network_nodes.sh add 192.168.1.10
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
