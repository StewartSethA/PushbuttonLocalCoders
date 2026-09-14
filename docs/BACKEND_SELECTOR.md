# VRAM-aware backend selector

`pushbutton-select` compares the curated llama.cpp quant ladder with specialized backend recipes for a requested model, GPU and VRAM lease.

```bash
pushbutton-select qwen3.8:27b --gpu 0 --vram-limit 16G
```

The table shows:

- feasibility under the requested lease (`FIT`, `BLOCK`, or `UNVERIFIED`)
- backend/runtime
- quant or fixed serving artifact
- known minimum/planning VRAM
- planner-relative quality estimate when one is defensible
- prompt-processing (PP) and token-generation (TG) rates
- whether performance data comes from a local Pushbutton report or an upstream reference

Local JSON reports under `benchmarks/results/` supersede upstream reference speeds automatically.

A specialized backend is **not** declared compatible with a reduced VRAM lease merely because it runs on the same physical GPU. Fixed artifacts and runtime workspaces may consume most of a 24 GB RTX 3090. For example, the current NInfer Qwen3.8 artifact is 16.96 GiB before runtime allocations, so a 16 GiB lease is blocked rather than attempted.

The generic llama.cpp lane is different: its curated quant ladder is genuinely selectable under the lease. At 256K context, a 16 GiB Qwen3.8-27B lease currently selects the 15,000 MiB `IQ3_XXS` planning profile.

## Direct curl launch

The installer passes remaining arguments directly to `qwen-local`, so placement constraints can be used on the first command:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-coder-local.sh \
  | bash -s -- --system \
    'qwen3.8:27b@gpu=0,vram=16G' \
    'qwen3.6:35b@gpu=1,vram=16G' \
    --agents 2
```

Preview only, with no compile or model download:

```bash
curl -fL https://raw.githubusercontent.com/StewartSethA/PushbuttonLocalCoders/main/install-coder-local.sh \
  | bash -s -- --system \
    'qwen3.8:27b@gpu=0,vram=16G' \
    'qwen3.6:35b@gpu=1,vram=16G' \
    --agents 2 --plan-only
```

After installation, inspect specialized alternatives with:

```bash
pushbutton-select qwen3.8:27b --gpu 0 --vram-limit 24G
```

Then benchmark a candidate on your exact hardware:

```bash
pushbutton-bench qwen3.8:27b --backend vllm-qwen38-3090 --gpus 0 --suite standard --label 3090-local
```

As reports accumulate, selector speed columns will prefer your local measurements over the upstream references.
