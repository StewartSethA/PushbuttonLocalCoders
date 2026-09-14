# Resource-aware backend selector

`pushbutton-select` is the front door for choosing a model, quant and serving framework under live GPU/VRAM and host-RAM constraints.

With an explicit model:

```bash
pushbutton-select qwen3.8:27b --gpu 0 --vram-limit 16G --ram-limit 32G
```

With no model, an interactive terminal first asks for resource ceilings, then the model family, then presents the ranked model/backend/quant table:

```bash
pushbutton-select
```

The table reports `FIT`, `BLOCK`, or `UNVERIFIED`, framework, quant/artifact, planning VRAM and RAM, quality estimate, PP tok/s, TG tok/s, and evidence status. `MEASURED` is shown in green on a color terminal; `ESTIMATED` is red. Measured figures come from exact-enough local reports first and then the repo aggregate. llama.cpp measurements are never reused across quants unless the artifact is tagged, so a Q4 result cannot silently become an IQ3 result.

The selector inspects current free and occupied VRAM with `nvidia-smi` through the existing hardware inventory. If a requested VRAM ceiling exceeds currently free memory it warns rather than pretending the card is empty. Host RAM limits are currently planning ceilings, not kernel/cgroup enforcement.

Specialized fixed/tuned engines are intentionally conservative. A backend with a proven hard minimum above the lease is `BLOCK`; a backend known only on a larger memory budget is `UNVERIFIED`. Quant-selectable llama.cpp profiles can genuinely down-select to the best profile that fits.

## Resident models and ports

Before proposing a new server, the selector checks Pushbutton's recorded `coder-backends.*.tsv` endpoint files and verifies that the endpoint still answers `/v1/models`. A matching resident model is highlighted as `REUSE` in placement preview. New placement previews choose a currently free loopback port rather than assuming a fixed port.

`coder-local` already probes for a free port for every new worker. The selector's resident-endpoint support is currently a preview/recommendation layer; a future launcher integration can attach a frontend directly to the resident endpoint instead of starting a new process.

## Benchmark evidence and real-world spread

`benchmarks/aggregate.json` is the bundled low-bandwidth reference database. The selector also checks a cached copy of the current repo aggregate and local JSON reports under `benchmarks/results/`. For matching model/backend/GPU/context evidence it computes median and p10/p90 distributions for PP, TG and TTFT. Local data therefore supersedes the bundled/upstream reference as measurements accumulate.

To measure an already-running OpenAI-compatible endpoint at multiple context depths:

```bash
pushbutton-observe \
  --endpoint http://127.0.0.1:20181/v1 \
  --model qwen3.8:27b \
  --backend vllm-qwen38-3090 \
  --depths 1024,8192,25000,65536,100000
```

`pushbutton-observe` records prompt-depth, TTFT, prompt-processing throughput and token-generation throughput without retaining prompts or generated text.

## Opt-in telemetry

Telemetry is **off by default**. Enable it explicitly:

```bash
pushbutton-select --telemetry-opt-in --telemetry-upload-url https://YOUR-COLLECTOR.example/v1/pushbutton
```

or while observing:

```bash
pushbutton-observe --telemetry-opt-in --telemetry-upload-url https://YOUR-COLLECTOR.example/v1/pushbutton ...
```

Queued telemetry contains only compact model/backend/artifact, coarse hardware, context and performance fields. It excludes prompts, generated text, usernames, hostnames and arbitrary file paths. Payloads are gzip-compressed newline-delimited JSON, so uploads are small. If no upload URL is configured, telemetry remains local and nothing is transmitted. Disable it with `pushbutton-select --telemetry-off ...`.

The repository includes the client and queueing protocol; deploying a public collector endpoint is a separate operational step.

## Direct curl use

The existing direct constrained launch still works:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-coder-local.sh \
  | bash -s -- --system \
    'qwen3.8:27b@gpu=0,vram=16G' \
    'qwen3.6:35b@gpu=1,vram=16G' \
    --agents 2
```

To install and enter the selector directly with flags:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-coder-local.sh \
  | bash -s -- --system --select \
    qwen3.8:27b --gpu 0 --vram-limit 16G --ram-limit 32G
```

If the installer is run with no model or other arguments in an interactive terminal, it enters `pushbutton-select` automatically. In a non-interactive pipeline it retains the safe help behavior.
