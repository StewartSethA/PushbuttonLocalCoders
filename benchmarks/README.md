# Pushbutton backend bake-off

This directory collects reproducible measurements from real PushbuttonLocalCoders hardware. The goal is to choose defaults from measured results rather than vendor or fork claims.

## Run one backend

```bash
pushbutton-bench qwen3.8:27b --backend llamampere --suite standard --label 3090-lab
```

First use is intentionally pushbutton: the backend adapter prepares its own source/container, downloads its model artifacts into the shared Pushbutton model cache, starts the OpenAI-compatible server, waits for readiness, runs the suite, stops the server, and writes a report under `benchmarks/results/`.

## Compare every compatible backend on the current machine

```bash
pushbutton-bench qwen3.8:27b --backend all --suite standard --label 3090-lab
```

Unsupported GPU/backend combinations are skipped rather than forced.

For Flash-Next on a four-V100 set:

```bash
pushbutton-bench qwen3.8-flash-next \
  --backend all \
  --gpus 0,1,2,3 \
  --context 262144 \
  --suite standard \
  --label v100-sxm2-tp4
```

## Suites

- `quick`: 1K-scale prompt, 256 output tokens, C1. Use this to validate an installation.
- `standard`: 1K/C1, 25K/C1, and 1K/C4. This is the preferred comparison suite.
- `long`: the standard suite plus a 100K-scale C1 request when the requested context can hold it.

The harness records the server-reported prompt/completion token counts whenever the backend provides them. If a compatibility layer omits streaming usage, completion-token count is explicitly marked as estimated rather than silently pretending it was measured.

## Contribute results

Run with `--stage` from the installed/repository checkout:

```bash
pushbutton-bench qwen3.8:27b \
  --backend all \
  --suite standard \
  --label 3090-350w \
  --stage
```

This stages only the generated JSON report(s) and regenerated `benchmarks/RESULTS.md`. Review them, then commit and push normally:

```bash
git diff --cached
git commit -m 'bench: add RTX 3090 Qwen3.8 backend results'
git push
```

Do not edit benchmark numbers by hand. If a run was invalid, delete the JSON and rerun.

## Fair-comparison notes

A backend can use the quantization and speculative-decoding scheme it is designed around. That is intentional: this project is comparing deployable systems, not isolated kernels. Reports therefore preserve backend revision, served model, context, hardware and exact suite so quality/speed/context tradeoffs remain visible.

For publication-quality A/B work, use the same GPU clocks/power limit, keep unrelated GPU workloads off the selected cards, run multiple repetitions, and note any non-default environment variables in the pull request description.
