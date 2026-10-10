# Telemetry destination

GitHub can host the **storage** (a private repository for raw records) and the **public
dashboard** (GitHub Pages). It cannot host the **always-on ingest endpoint**:

- Pages serves static files only.
- Actions jobs are not servers.
- Every GitHub write API needs a token, which must never ship inside the client.

So a tiny relay that holds the token sits in front:

```
pushbutton client ──POST gzip NDJSON──▶ relay (Cloudflare Worker, always on, free tier)
   ≤ 1 upload / 5 min                    │  whitelist fields, stamp received_utc,
                                         │  sender_hash = HMAC-SHA256(IP_SALT, ip)[:16]
                                         ▼  (raw IP never stored)
                     PRIVATE repo  <owner>/pushbutton-telemetry-data   records/YYYY/MM/DD/*.ndjson
                                         │  read-only token, every 6 h
                                         ▼
             .github/workflows/telemetry-dashboard.yml → telemetry/aggregate.py
                                         │  anonymize: drop hashes/timestamps, k ≥ 3 contributors
                                         ▼
                      GitHub Pages (public): index.html + summary.json
```

| piece | file | runs on |
|---|---|---|
| relay | [`worker/src/index.js`](worker/src/index.js), [`worker/wrangler.toml`](worker/wrangler.toml) | Cloudflare Workers (100k requests/day free) |
| self-hosted alternative | [`../pushbutton-telemetry-collector`](../pushbutton-telemetry-collector) | any small VM behind TLS |
| aggregator | [`aggregate.py`](aggregate.py) | GitHub Actions |
| dashboard | [`dashboard/index.html`](dashboard/index.html) | GitHub Pages |

## Setup (once)

1. **Private data repository.** Create `pushbutton-telemetry-data` as **private**.
   - Create two fine-grained tokens, each scoped to that repository only:
     - **relay**: Contents read and write.
     - **dashboard**: Contents read-only.
2. **Relay.** In `telemetry/worker`, set `DATA_REPO` in `wrangler.toml`, then:
   ```bash
   npx wrangler secret put GITHUB_TOKEN    # the relay token
   npx wrangler secret put IP_SALT         # e.g. `openssl rand -hex 32`; keep it secret
   npx wrangler kv namespace create RATE_KV   # optional: paste the id into wrangler.toml
   npx wrangler deploy
   ```
   Put the resulting `https://pushbutton-telemetry.<account>.workers.dev/v1/telemetry` into
   `DEFAULT_UPLOAD_URL` in `lib/pushbutton_metrics.py`.
3. **Dashboard.** In this public repository:
   - Settings → Pages → Source: **GitHub Actions**.
   - Add the variable `TELEMETRY_DATA_REPO=<owner>/pushbutton-telemetry-data`.
   - Add the secret `TELEMETRY_DATA_TOKEN` (the dashboard token).
   - Run the **telemetry dashboard** workflow. It also runs every 6 hours.

## What is stored, and what is public

**Raw records (private repository).**
- Whitelisted fields only: `pushbutton_metrics.SANITIZED_FIELDS`, the same list as the
  worker's `SANITIZED_FIELDS`; a unit test keeps the two in sync.
  - model, quant, backend
  - hardware type
  - CPU launch strategy
  - load time
  - PP/TG per context depth
- Plus `received_utc` and `sender_hash`.
- `sender_hash` is a keyed HMAC of the IP. Without `IP_SALT` it cannot be reversed or matched
  against a list of IPs, and rotating the salt unlinks all future records from past ones.

**Public dashboard.** Built by `aggregate.py`:
- Keeps only model, quant, backend, a normalized hardware class, the strategy and a
  context-depth bucket.
- Publishes median and p25–p75 of PP/TG tok/s and load time.
- Never publishes hashes, IPs, timestamps or individual records.
- k-anonymity: a group is published only when at least 3 distinct senders
  (`PUBLIC_MIN_CONTRIBUTORS`) contributed. Smaller groups are counted as withheld.
- Strings that don't look like model or hardware names become `other`.
- The page renders data with DOM text nodes only.
- The workflow fails if `sender_*` or `received_utc` ever reach the artifact.

## Limits

- One accepted upload per sender every 5 minutes (HTTP 429 + `Retry-After`). Clients enforce
  the same interval.
- Rejected payloads do not consume the slot.
- 256 KiB compressed / 4 MiB decompressed per upload; at most 2000 records per upload.

## Self-hosted collector

```bash
git clone git@github.com:<owner>/pushbutton-telemetry-data.git ~/telemetry-data
./pushbutton-telemetry-collector --port 8787 --records ~/telemetry-data/records --git-commit --git-push
```

- The collector writes the same `sender_hash` format as the relay.
- The salt is read from `$PUSHBUTTON_TELEMETRY_SALT` or a generated 0600 file
  (`--salt-file`).

## Tests

```bash
python3 -m unittest tests/test_telemetry.py
(cd telemetry/worker && npm test)
```
