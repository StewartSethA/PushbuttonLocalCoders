# Telemetry records

`records/YYYY-MM-DD.ndjson` holds the telemetry received by
[`pushbutton-telemetry-collector`](../pushbutton-telemetry-collector), one JSON object per line.

## Record fields

The collector adds two fields to every record:

- `received_utc`: the time the server received the record.
- `sender_ip`: the uploader's IP address, taken from the TCP peer, or from
  `X-Forwarded-For` only with `--trust-forwarded-for`.

All other fields are whitelisted by `lib/pushbutton_metrics._sanitize_observation`:

| category | fields |
|---|---|
| model and backend | `model`, `artifact` (quant), `backend` |
| load and throughput | `load_time_s`; `depths`: `[{depth, pp_tps, tg_tps, pp_tps_estimate, tg_tps_estimate}]`; per-request `pp`, `tg`, `ttft`, `prompt_tokens` |
| hardware | GPU names/memory, or `cpu_model`, `cpu_family`, `cpu_tier`, `cpu_sockets`, `cpu_cores`, `fast_mem_kind`, `fast_mem_mode` |
| CPU launch strategy | `strategy`, `threads`, `threads_batch`, `memory_tier` |

Prompts, generated text, usernames, hostnames and local paths are never accepted.

## Running the collector

```bash
./pushbutton-telemetry-collector --port 8787 --git-commit --git-push   # behind TLS, e.g. a reverse proxy
# clients:
pushbutton --telemetry-on --telemetry-url https://YOUR-HOST/v1/telemetry
```

The collector enforces these limits:

- One accepted upload per sender IP every 5 minutes; otherwise it returns HTTP 429 with
  `Retry-After`. Clients enforce the same interval themselves.
- 256 KiB compressed and 4 MiB decompressed per upload, and at most 2000 records per upload.

`--git-commit` commits new records every `--commit-interval` seconds. `--git-push` also
pushes, using the operator's git credentials. Clients never receive repository
credentials.

## Privacy

IP addresses are personal data in many jurisdictions (for example under GDPR). Records in
this directory are public once pushed. Operators should publish a privacy notice and
consider truncating or hashing `sender_ip` before pushing.
