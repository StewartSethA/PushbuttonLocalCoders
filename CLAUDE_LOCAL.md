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

Ordinary Claude Code arguments are passed through, subject to the context-policy checks below:

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

The default **requested** physical context is **262,144 tokens**. After loading each backend, the launcher verifies its actual per-slot capacity and derives Claude's budgets. Requested `-c`, model-native context, assumed client window, and proactive compaction window are different quantities.

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

The main session and generated subagents default to opaque local model IDs rather than built-in Claude aliases. This makes role routing independent of Claude Code's alias resolution. A recognized Claude `--model` override can still change Claude's assumed model window; the explicit compaction window and per-request gateway check remain necessary.

Unless `--local-no-teams` is supplied, the harness enables Claude Code's agent-team support and injects four fully local subagents:

- `local-fast`: repository reconnaissance and triage on Haiku
- `local-coder`: parallel implementation on Sonnet
- `local-reviewer`: independent review on Opus
- `local-deep`: difficult debugging/architecture work on Fable

Claude Code remains the orchestrator. When it fans out, all agent API traffic returns to the same local gateway and therefore to the planned local GPUs.

The local gateway and model backends cannot run Anthropic's server-side auto-mode
classifier. `claude-local` therefore sets `CLAUDE_CODE_AUTO_MODE_SERVER=0` for
the launched Claude Code process by default. Auto mode remains available and uses
Claude Code's own classifier requests; this avoids the eligibility warning but
does not provide the new no-charge server-side checks. To override the default,
set `CLAUDE_CODE_AUTO_MODE_SERVER=1` before launching. This setting is scoped to
`claude-local` and is not written to persistent Claude Code configuration.

Client-side tool-use concurrency defaults to the minimum configured slot count
across the routed local models; explicit `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY`
is respected. This is a pressure limit, not a strict backend admission control.
When Auto mode is selected and any routed model has one slot, the launcher warns
that classifier requests may queue and time out, then offers to continue,
continue without local agent teams, use `default` (interactive permission
prompts), use `acceptEdits` (edits auto-approved; other permission checks
remain), or cancel before setup. Neither alternative disables permission checks.
Non-interactive Auto launches stop before setup when any routed model has one
slot unless `--local-allow-single-slot-auto` explicitly acknowledges the risk.
Explicit `default` or `acceptEdits` modes avoid Auto's classifier request.
Auto-mode detection uses the isolated settings file at
`$CLAUDE_LOCAL_STATE/claude-config/settings.json` and managed settings; project
settings are intentionally excluded.

## Context budgeting and auto-compaction

With the default policy, claude-local does **not** select an auto-compaction
window below Claude Code's supported 100,000-token minimum. It derives the
client and compaction windows from verified route capacity, reserves room for
output and the next compacted turn, and rejects smaller unsafe configurations.
For example, a 131,072-token route defaults to a 106,496-token compaction
window. A caller can request a lower window only if Claude supports it and the
launcher accepts it; under this policy values below 100,000 are rejected.

The old `route admission limit reached; retry later` 429 is emitted by
Pushbutton's **local gateway/capacity proxy**, not by Claude's telemetry or an
external API. It meant a local generation slot stayed occupied beyond a
60-second admission wait. Admission now queues until the local route has room,
so temporary concurrency no longer produces that 429. Requests still undergo
the same full-payload token/capacity checks before inference.

Use `--no-telemetry` to opt Claude Code and its local frontend integrations out
of telemetry, error reporting, OpenTelemetry and tracking. This preserves local
model requests and explicitly configured MCP/web tools.

The earlier `CLIENT_CTX <= CTX` check was **not sufficient**: it compared configured numbers without verifying the server slot or controlling version-dependent Claude compaction behavior. The reported `147023 > 131072` error proves a request/window mismatch of **15,951 tokens**, not a network failure. It does not establish the installed Claude version, environment, or why that server selected 131,072.

The launcher now:

- Reads each live backend's `/props.default_generation_settings.n_ctx` and verifies the requested slot count after loading. In current official llama.cpp this is the **effective slot capacity**, including slot/native caps, not total KV allocation; do not divide it again. The server command uses `-c (CONTEXT_PER_SLOT * SLOTS) -np SLOTS --jinja --no-context-shift`. Slots default to one and may be specified globally with `--slots` or per model with `MODEL@slots=N,context=N`. Current upstream fitting leaves explicit `-c` unchanged, but caps/alignment and different cached builds still require verification.
- Reports requested/effective mismatches before launching Claude and derives a global policy from the **smallest Haiku/Sonnet/Opus/Fable route**, including subagents and compaction traffic. Unsafe explicit budgets are rejected rather than silently lowered. Missing runtime metadata or native token-count support stops launch with update/rebuild guidance.
- Sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS` for unresolved custom IDs, `CLAUDE_CODE_AUTO_COMPACT_WINDOW` explicitly, and `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (default **8,192**, avoiding an unknown-model output default of 32K).
- Reserves **8,192** tokens for prompt/formatting/token-accounting margin and another **8,192** of proactive compaction headroom. If capacity is `C` and configured output is `O`, the default custom-model assumed window is `min(200000, C-O-8192)`, further bounded by per-model client limits; the compaction window is `min(client window, C-O-16384, 1000000, per-model compact limits)`. The window is not the exact trigger: Claude applies its own output/compaction reserves within it.

For an effective capacity of **131,072**, with an explicit model output cap of **8,192** and compact limit of **106,496**, but no client override:

| Setting | Tokens |
|---|---:|
| Verified backend capacity | 131,072 |
| Custom-model assumed window | 114,688 |
| Proactive auto-compaction window | 106,496 |
| Maximum output | 8,192 |
| Prompt safety reserve / extra compaction headroom | 8,192 / 8,192 |

```bash
claude-local 'qwen3.8:27b@context=131072,output=8192,compact=106496'

# Safe lower overrides are retained:
CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000 \
CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096 \
  claude-local 'qwen3.8:27b@context=131072,output=8192,compact=100000' --local-client-context 110000
```

`CLAUDE_LOCAL_CLIENT_CONTEXT`, `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, and the supported `--autocompact` integer can also lower the policy. Use **plain integers**, not `100k`, percentages, or `off`. Conflicting safe lower values select the smaller window. Values that cannot fit output and reserves fail with the allowable range.
The generic planner's default compact trigger is 80% of its input budget. For
smaller Claude contexts that may be below the documented 100,000-token minimum;
specify fitting `output` and `compact` values as above rather than expecting an
unsafe upward adjustment. Per-model specs retain their token-suffix/fraction
syntax; the plain-integer restriction here applies to Claude environment/CLI
window overrides.

### Compatibility and isolated settings

Require **Claude Code 2.1.221+**, whose CLI reference documents `--autocompact`. Startup prints the installed version, checks required flags in `--help`, and checks the actual native executable/npm bundle for the documented context/output/compaction feature markers. This is a conservative compatibility check, **not a behavioral proof** for every Claude release. Opaque wrappers/older artifacts fail explicitly; update the official installation (`claude update`) rather than guessing support from a version number alone.

The documented explicit auto-compaction range is **100,000–1,000,000** and Claude can cap it to its assumed model window. We reject smaller windows instead of allowing silent upward clamping beyond a local limit. With default reserves/output, capacity must be at least **124,576**; reducing output can permit a somewhat smaller capacity, but a sub-100K backend is unsupported by this policy. Do not disable compaction or blindly increase context/VRAM to get around this.

Claude user/session data is isolated in **`$CLAUDE_LOCAL_STATE/claude-config`** (default **`~/.local/share/pushbutton/claude-local/claude-config`**), outside the repository. Saved user preferences and permissions in **`claude-config/settings.json`** are loaded on every launch with `--setting-sources user`, so they survive repository pulls and installer updates. Changes saved through Claude's `/config` are reused on subsequent launches; the harness does not overwrite the file. Project/local settings sources remain excluded, and the harness explicitly enforces auto-compaction and denies built-in `WebSearch`/`WebFetch` through CLI settings. Saved `env` overrides of provider/model routing or context safeguards are rejected with the settings path and corrective guidance. Use the harness environment/options for safe context changes instead.

The harness never rewrites `~/.claude` or `~/.claude.json`. Existing settings and sessions from the default Claude config directory are not automatically imported; copy desired non-provider customizations into the isolated `settings.json`. Keep the same state directory (or migrate its contents) when updating; changing `CLAUDE_LOCAL_STATE` selects different settings/session storage.

`DISABLE_AUTO_COMPACT`, `DISABLE_COMPACT`, the legacy percentage override, provider-bypass variables, `[1m]` frontend arguments, and caller-supplied `--settings`/`--setting-sources` are rejected with diagnostics. Linux `/etc/claude-code/managed-settings.json` and alphabetically merged `managed-settings.d/*.json` conflicts are diagnosed, not rewritten; unverifiable dynamic `policyHelper` configuration is rejected. Managed organizational policy has higher precedence and cannot be bypassed by this harness. Current docs say a shell-exported custom `ANTHROPIC_BASE_URL` skips new server-managed settings fetches, but startup still warns that cached/remote policy cannot be independently verified. Resolve any conflict with your administrator. Changing `/config` during a session can still disable compaction; the gateway check remains the local request boundary.

### Accurate admission, not history manipulation

Before every inference request, the gateway passes the **full outgoing Anthropic payload** to the routed backend's native `/v1/messages/count_tokens`: system content, message history, tool schemas/results, supported thinking content, and chat-template/special-token overhead. Current official llama.cpp uses the same Anthropic-to-OpenAI conversion and template parser for counting and inference. `/tokenize` on concatenated text or `/apply-template` on an unconverted Anthropic body is **not** an accurate substitute.

Counting means the content **supported and rendered by llama.cpp**, not parity with Anthropic's cloud tokenizer or support for every Anthropic content block. Multimodal counting uses upstream placeholder handling; these launchers disable the projector. Unsupported content may be rejected/ignored by the upstream adapter. Use a current compatible build, including for the GLM fork; the launcher probes native counting and refuses an unavailable counter. Rebuild cached servers with `CLAUDE_LOCAL_REBUILD=1` if needed.

The gateway requires:

```text
native input tokens + request's actual max_tokens + 8192 <= route's verified capacity
```

Violations return HTTP **400**, Anthropic `invalid_request_error`, and recognizable `prompt is too long` / `exceeds the available context size` wording **before inference**. A failed tokenizer preflight never proceeds with an estimate. No history/tool pair is truncated, no summary is fabricated, and requested output is never silently reduced. 400s are not retried; transient inference retries remain restricted to before response commitment. Partial SSE output is never replayed.

Preflight adds template rendering/tokenization, not a second inference. A bounded 30-second cache reuses exact-payload counts (including Claude's count request when identical); it stores only digests/counts, not prompts. Startup diagnostics show version, requested/effective context, client/compaction windows and reserves, without logging prompts or credentials.

**Auto-compaction alone cannot guarantee safety** for an arbitrary huge tool result, system/tool definition, tokenizer mismatch, or already oversized transcript. Such requests are rejected intact.

### Recovering repeated auto-compaction thrashing

The warning that context refilled within three turns of compaction, three times
in a row, describes conversation growth, not proof of GPU memory exhaustion.
Large tool results or an ineffective compacted summary can cause it. The
launcher appends bounded-content guidance to Claude's system prompt: read small
line ranges, constrain searches, and inspect targeted log excerpts rather than
dumping generated files or dependency directories. This is model guidance, not
a hard tool-output limit; the gateway still rejects oversized requests intact.

1. Preserve a short handoff (goal, completed changes, decisions, relevant paths,
   remaining work, verification), then use `/clear` in the interactive session
   or launch without `--continue`/`--resume`. The harness never clears history
   automatically. If the model cannot process the current history, recover the
   handoff from notes rather than asking it to read that history again.
2. Compare the Qwen models **sequentially**, using the same short handoff and
   task in separate fresh sessions:
   ```bash
   claude-local qwen3.8:27b
   # Exit the first session before launching the second.
   claude-local qwen3.6:35b
   ```
   Record which tools and output sizes precede the warning. Single-model
   placement/quantization can differ from the two-model run, so record those too;
   this comparison is diagnostic, not a controlled model-quality benchmark.
3. Inspect the startup placement and budget diagnostics: model/GPU assignment,
   slots, requested and effective per-slot capacity, client assumed window, and
   auto-compaction window. `--local-dry-run` shows a planned placement only;
   `doctor` does not prove a live backend's capacity. Two 16GB V100s do not
   automatically provide one shared context window. For
   `claude-local qwen3.8:27b qwen3.6:35b`, Haiku uses the first model and
   Sonnet/Opus/Fable use the second; the shared budget uses the smallest routed
   backend, not the sum of GPU memory or context capacities.
4. Only after verifying live capacity, consider a lower fitting compaction
   window using the budget options above. Windows below 100,000 tokens are
   rejected. Do not disable compaction or blindly increase context; lowering a
   window cannot repair an already oversized transcript.

### Recovering an already oversized session

Lowering the threshold does **not** retroactively compact a saved transcript. Resuming/continuing a transcript that already exceeds the smaller backend may fail even for `/compact`, since compaction itself needs to read that history.

1. Save a concise handoff outside the conversation: goal, completed changes, decisions, relevant paths, remaining work and verification. Recover it from your notes/transcript without sending the oversized history back to the model.
2. Start an explicit **new session**, without `--resume` or `--continue`, and give it only the handoff:
   ```bash
   claude-local 'qwen3.8:27b@context=131072,output=8192,compact=106496'
   ```
3. Only if a larger context is **proven to fit both the model and available hardware**, temporarily use that backend to resume and produce a handoff/compact. Verify its actual slot capacity first; increasing `-c` blindly is not the fix.

References: [Claude model configuration](https://code.claude.com/docs/en/model-config#correct-the-window-for-a-gateway-or-custom-model-id), [environment variables](https://code.claude.com/docs/en/env-vars), [CLI reference](https://code.claude.com/docs/en/cli-reference), [errors](https://code.claude.com/docs/en/errors), and [official llama.cpp implementation](https://github.com/ggml-org/llama.cpp/blob/bed0a856606ee4a24a164066f73d2379447033f5/tools/server/server-context.cpp) (`get_res_props`, `n_ctx_slot`, `handle_count_tokens`). At implementation time the Claude docs host was unavailable in the sandbox; current mirrored official page contents were cross-checked with the upstream changelog. No real Claude/GPU inference was run; tests use mocked Claude and HTTP backends.

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
