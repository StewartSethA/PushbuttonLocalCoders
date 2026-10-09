# Pushbutton backend bake-off

This directory collects reproducible measurements from real PushbuttonLocalCoders hardware. The goal is to choose defaults from measured results rather than vendor or fork claims.

## Run one backend

```bash
pushbutton-bench qwen3.8:27b --backend llamampere --suite standard --label 3090-lab
```

The backend adapter prepares its source/container, downloads model artifacts,
starts an OpenAI-compatible server, runs the suite and writes a report under
`benchmarks/results/`. Specialized Flash-Next engines additionally require
**previously calibrated capacity**; prepare/calibrate them first and benchmark
with `--no-prepare` so updating a recipe/image cannot invalidate its calibration.
The opaque `vllm-qwen38-3090` image has no verified concurrency control and is
limited to `--concurrency 1`; unsupported comparisons are skipped explicitly.

## Compare every compatible backend on the current machine

```bash
pushbutton-bench qwen3.8:27b --backend all --suite standard --label 3090-lab
```

Unsupported GPU/backend combinations are skipped rather than forced.

For Flash-Next on a four-V100 set, after preparing and calibrating SGLang:

```bash
pushbutton-bench qwen3.8-flash-next \
  --backend all \
  --gpus 0,1,2,3 \
  --context 262144 \
  --concurrency 2 \
  --capacity verified-flash-capacity.json --no-prepare \
  --suite standard \
  --label v100-sxm2-tp4
```

## Suites

- `quick`: 1K-scale prompts, 256 output tokens at C1 and requested Cn.
- `standard`: 1K/C1, 25K/C1, 1K/requested Cn, plus a context-scaled prompt at
  C1 and Cn. Default Cn is four; use `--concurrency 2` for the normal two-slot
  deployment.
- `long`: standard plus a 100K-scale C1 request if it fits.
- `--prompt-words N --output-tokens M`: replace suite cases with the explicitly
  requested prompt scale at C1 and Cn.

The harness records the server-reported prompt/completion token counts whenever the backend provides them. If a compatibility layer omits streaming usage, completion-token count is explicitly marked as estimated rather than silently pretending it was measured.
Both stream readers count `reasoning_content` as generated output. TTFT is the
first nonempty content **or reasoning** delta, not time to the final answer.
Decode and aggregate throughput use server completion usage (including reasoning
according to the server's accounting), or a labeled word-based estimate across
both fields. `chars` remains answer-content characters; `reasoning_chars` is
separate. These measure generation, not useful-answer throughput; reasoning-only
budget exhaustion is valid generation, not proof of answer quality. Stream
chunks can contain multiple tokens, so decode timing remains an approximation.
Older reports may have content-only timing.

Live launches reject occupied loopback ports. Readiness rechecks the foreground
child after HTTP success; native adapters and both specialized Flash engines
(including SGLang Docker) must advertise a fresh per-launch model alias rather
than merely return healthy. Opaque prebuilt Docker adapters cannot advertise
this alias and use the free-port/child-liveness checks only.
Context is **per request/slot**: llama.cpp receives `context * concurrency` total
context and the matching parallel slot count. SGLang receives the requested
`--max-running-requests`, not the old fixed value one. Its multi-request lane
disables unverified batch CUDA graphs. Prompt scale is in **words**, not an
exact token occupancy guarantee; context-scaled probes use a conservative word
scale and reports preserve actual tokenizer usage. Token overflow is a failed
run, not a silently truncated successful case. Empty/error streams produce no
synthetic throughput. Reports include launch command, capacity metadata,
profile, context and configured concurrency, and distinct context/concurrency
runs do not overwrite one another.

## Flash-Next evaluation order and capacity gate

1. On **four RTX 3090s**, first evaluate the existing
   [W4A16/MTP recipe](https://github.com/loktar00/qwen38-flash-next-vllm-3090-recipe).
   The adapter launches its equivalent **foreground vLLM process** so readiness
   and teardown work; upstream `serve.sh` backgrounds a server and exits, so
   invoking it as a managed backend would not work. GPU order is preserved for
   adjacent NVLink pairs. MTP is limited to **65536** context by this recipe;
   `--profile bf16` supports up to **131072** without MTP. Requested context and
   concurrency are passed explicitly. FP8 extended-context profiles need
   additional patches/calibrated scales and are **not integrated**.
2. On **four V100s**, compare `sglang-v100` with the existing upstream
   `llama.cpp` Flash-Next **UD-IQ4_XS** adapter using identical selected GPUs,
   context, prompt scale, output cap and C1/Cn probes.
3. Keep Sonnet and the classifier on **other** GPUs and Haiku on its calibrated
   CPU reservation. Stop any existing Fable process before using its four-card
   group for the standalone bake-off; do not benchmark on a live production
   Fable allocation. Two-card specialized recipes are rejected even with
   `--force` or dry-run.

Preparation (builds/downloads only, not a capacity claim):

```bash
python3 ./pushbutton-backend prepare vllm-flashnext-3090 flash-next --gpus 0,1,2,3
python3 ./pushbutton-backend fingerprint vllm-flashnext-3090 flash-next --gpus 0,1,2,3
# On the V100 host, prepare/fingerprint sglang-v100 instead.
```

Live specialized `check`/`serve` require `--capacity FILE`. This is **separate
from the llama.cpp CPU/GPU memory metadata**. Its schema is version 1 with a
`placements` array. Each entry identifies:

- `backend`, canonical `model: "qwen3.8-flash-next"` and `profile` (`mtp` or the
  supported vLLM `bf16` lane);
- `artifact`: exactly the adapter fingerprint's image tag or pinned W4A16
  checkpoint/revision;
- `runtime_revision`: the fingerprint's installed recipe commit (vLLM), or
  Docker image ID (SGLang); mismatches require recalibration;
- `source`: real measurement provenance, including software/hardware and
  combined max-context/max-concurrency load;
- integer `host_mib` (weights, staging and buffers), `gpu_mib` (four per-device
  peak envelopes in selected GPU order), `cpu_threads`, `max_context` and
  `max_concurrency`.

No production capacity numbers are shipped or inferred from model file size.
Host admission checks available non-swap/cgroup memory plus the reserve and CPU
quota. Each GPU must fit its measured envelope plus 512 MiB; envelopes cannot
be below the configured engine memory fraction. The W4A16 recipe alone requires
approximately **97 GiB of host PLE weights**, before staging/buffers, so a
smaller host envelope is rejected. SGLang Docker is also constrained to the
declared host RAM/CPU budget with no swap. vLLM uses the admitted host capacity
and thread setting; this standalone adapter does not establish a new host
cgroup. Rechecks follow provisioning. `--force` cannot bypass these memory gates.
Calibration must include the **joint** context and concurrency peak; C1
measurements cannot authorize a larger Cn lane.

Example evaluation commands, using your measured capacity file:

```bash
python3 ./pushbutton-bench flash-next --backend vllm-flashnext-3090 \
  --gpus 0,1,2,3 --context 65536 --concurrency 2 \
  --capacity verified-flash-capacity.json --no-prepare --label 3090-w4a16-mtp

python3 ./pushbutton-bench flash-next --backend sglang-v100 \
  --gpus 0,1,2,3 --context 131072 --concurrency 2 \
  --capacity verified-flash-capacity.json --no-prepare --label v100-sglang
python3 ./pushbutton-bench flash-next --backend llama.cpp \
  --gpus 0,1,2,3 --context 131072 --concurrency 2 \
  --no-prepare --label v100-iq4-xs
```

Add `--dry-run` to inspect commands and cases without downloads, server
processes or metrics. Dry-run capacity is always labeled **unverified**, not
proof of hardware fit. On hosts without this hardware, only dry-run and mocked
regression validation are possible. There is **no measured winner** yet.
These OpenAI adapters are not automatically connected to the Anthropic Claude
gateway; production dedicated-role serving remains the supported llama.cpp
path documented in [CLAUDE_LOCAL.md](../CLAUDE_LOCAL.md).

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
