# Pushbutton Hermes Bot Mode

`hermes-local` runs Hermes Desktop Bot Mode on the same hardware-aware local llama.cpp stack used by `claude-local`.

It shares:

- CUDA/toolchain state under `~/.local/share/pushbutton/claude-local`
- llama.cpp builds under the same state directory
- GGUF downloads under `~/.cache/pushbutton/llama`
- the same model catalogue, quant selection, context sizing, GPU placement, and V100/A100/RTX hardware planning

Hermes itself keeps its normal durable profiles, sessions, memories, skills, and routines under `~/.hermes/`.

## Install

After merge:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-hermes-local.sh \
  | bash -s -- --system qwen3.6:35b
```

During development:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/refs/heads/feature/hermes-bot-frontend/install-hermes-local.sh \
  | PUSHBUTTON_REF=feature/hermes-bot-frontend bash -s -- \
    --system qwen3.6:35b
```

Subsequent launches:

```bash
hermes-local qwen3.6:35b
```

The default frontend is Hermes Desktop. Bot Mode is built into current Hermes Desktop and enabled by default.

## Models

The positional model mapping intentionally matches `claude-local`:

```text
1 model : Haiku = Sonnet = Opus = Fable
2 models: Haiku = #1, Sonnet/Opus/Fable = #2
3 models: Haiku = #1, Sonnet = #2, Opus/Fable = #3
4 models: Haiku, Sonnet, Opus, Fable respectively
```

Examples:

```bash
hermes-local qwen3.6:35b

hermes-local nemotron-3.5-lightning qwen3.6:35b

hermes-local \
  nemotron-3.5-lightning \
  qwen3.6:35b \
  qwen3.8:27b
```

Supported built-ins are the same catalogue as `claude-local`:

- `qwen3.6:35b`
- `qwen3.8:27b`
- `nemotron-3.5-lightning`
- `glm-5.3-flash`

## Bot roster

The first launch creates durable Hermes profiles. With the default prefix they are:

- `pushbutton-manager` — project-manager / orchestrator; Sonnet-role model
- `pushbutton-fast` — reconnaissance / triage; Haiku-role model
- `pushbutton-coder` — implementation; Sonnet-role model
- `pushbutton-reviewer` — correctness / regression review; Opus-role model
- `pushbutton-deep` — architecture / difficult debugging; Fable-role model

The launcher updates only the model endpoint/configuration required to reach the current local llama.cpp servers. Hermes memory, sessions, skills, routines, and user-edited `SOUL.md` files remain durable between launches.

Profile descriptions are populated so Hermes Kanban/decomposition can route work by role. The manager profile is configured as the Kanban orchestrator and defaults unmatched work to the coder profile.

## Frontends

Desktop Bot Mode is the default:

```bash
hermes-local qwen3.6:35b
```

Hermes TUI:

```bash
hermes-local qwen3.6:35b --tui
hermes-local qwen3.6:35b --tui --resume
```

Classic Hermes CLI:

```bash
hermes-local qwen3.6:35b --cli
hermes-local qwen3.6:35b --cli --resume
```

## Notes

`hermes-local` deliberately keeps the default model route local. It does not silently sign into ChatGPT, Claude, Nous Portal, OpenRouter, or any other cloud provider.

You can still configure a Hermes profile manually to use Codex OAuth or another cloud provider after the local Bot roster exists. That is separate from the Pushbutton local model mapping.

The current llama.cpp backends use `-np 1`. Distinct model servers can run concurrently, but multiple Hermes Bots sharing one model backend may serialize inference. A future scheduler improvement can replicate a model or raise parallel slots automatically when spare GPUs and KV-cache headroom make that worthwhile.
