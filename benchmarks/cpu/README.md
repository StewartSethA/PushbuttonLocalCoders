# CPU benchmarks

| file | contents |
|---|---|
| [ESTIMATES.md](ESTIMATES.md) | Estimated load time and PP/TG tok/s at each context depth for every reference CPU × CPU-planned model (`python3 lib/cpu_platform.py estimates --markdown`; CI checks it is in sync). |
| [RESULTS.md](RESULTS.md) | Measured results for each launch strategy (`./pushbutton-cpu-bench --summarize`). |
| `results/*.json` | One file per host × model × quant × strategy (schema `pushbutton.cpu-bench.v1`). |

`./pushbutton-cpu-bench MODEL [MODEL ...]` does the following on the current host:

1. Picks the quant the planner would use. On a Xeon Phi or HBM part, this is the
   best-quality quant that fits MCDRAM/HBM.
2. Builds llama.cpp for the host ISA (`llama-server` + `llama-bench`) and downloads the GGUF.
3. Runs each applicable strategy (`mcdram-bind`, `mcdram-preferred`, `mcdram-cache`,
   `hbm-bind`, `numa-distribute`, `numa-isolate`, `numa-partition`, `physical-cores`,
   `all-threads`, `no-mmap`) and measures:
   - weight load time (`llama-server` start → `/health`);
   - PP with the batch thread count and TG with the generation thread count, using
     `llama-bench -d` at each context depth (default 0, 4096, 16384, 32768).
4. Writes a result JSON with the roofline estimate next to each measurement and
   regenerates `RESULTS.md`.
5. When telemetry is enabled, queues each run (uploads at most every 5 minutes).

Useful options: `--strategies mcdram-bind,physical-cores`, `--depths 0,8192`, `--quant Q4_K_M`,
`--gguf PATH --label NAME` (benchmark a local file), `--dry-run` (print strategies and
estimates only).

Commit new `results/*.json` files together with the regenerated `RESULTS.md`. The planner
uses a measurement only on a host with an identical layout (`cpu.signature`: model name,
sockets, cores, NUMA/MCDRAM layout and mode).
