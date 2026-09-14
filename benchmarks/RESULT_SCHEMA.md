# Benchmark report schema v1

Each `benchmarks/results/*.json` report contains:

- `schema_version`: currently `1`.
- `timestamp_utc`: ISO-8601 UTC timestamp.
- `label`: contributor-supplied machine/test label. Hostnames are not collected.
- `hardware_id`: short SHA-256-derived identifier from the visible GPU model list; it is not a machine serial number.
- `hardware`: OS/kernel, RAM when available, and `nvidia-smi` GPU name, total VRAM, compute capability, driver and configured power limit.
- `backend`: backend id, upstream source URL, supported-model metadata, selected GPUs, and source Git revision when the adapter has a source checkout.
- `model`: canonical Pushbutton model selector.
- `served_model`: model id returned by the running OpenAI-compatible endpoint.
- `suite`: `quick`, `standard`, or `long`.
- `context`: server context requested by the harness.
- `cases`: normalized benchmark cases.

Each case records:

- requested prompt scale (`words`) and maximum output tokens (`output`),
- concurrency and repetitions,
- per-request TTFT, elapsed time, server-reported prompt/completion token counts when available, and decode tokens/s,
- per-wave aggregate output throughput,
- medians used by `benchmarks/RESULTS.md`.

`completion_tokens_estimated=true` means the server did not return streaming usage and the harness estimated the completion count from text. Such rows remain useful for smoke tests but should not be used to choose a default backend when an otherwise comparable report has authoritative token accounting.

The raw JSON is authoritative. `benchmarks/RESULTS.md` is a generated index and may be recreated with:

```bash
pushbutton-bench --summarize
```
