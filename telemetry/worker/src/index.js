// Pushbutton telemetry ingest relay (Cloudflare Worker, always on, free tier).
//
// POST /v1/telemetry  gzip or plain NDJSON, as sent by lib/pushbutton_metrics.upload_pending.
// Each accepted row is whitelisted (same fields as pushbutton_metrics.SANITIZED_FIELDS),
// stamped with the receive time and a keyed hash of the sender IP, and committed as one
// new file to the PRIVATE data repository via the GitHub Contents API. The raw IP is only
// used to compute the hash and is never stored (not in GitHub, not in KV).
//
// Bindings (see wrangler.toml and telemetry/README.md):
//   secrets: GITHUB_TOKEN (fine-grained, Contents: read+write on DATA_REPO only), IP_SALT
//   vars:    DATA_REPO ("owner/private-repo"), DATA_BRANCH (default "main")
//   kv:      RATE_KV (optional; per-sender 5-minute upload limit)

export const MAX_BODY = 256 * 1024;
export const MAX_DECOMPRESSED = 4 * 1024 * 1024;
export const MAX_RECORDS = 2000;
export const RATE_LIMIT_S = 300;

export const SANITIZED_FIELDS = [
  "schema_version", "timestamp_utc", "model", "backend", "artifact", "backend_commit",
  "context", "gpu_models", "gpu_memory_mib", "ram_gib", "prompt_tokens", "completion_tokens",
  "prompt_tokens_estimated", "completion_tokens_estimated", "pp", "tg", "ttft", "elapsed_s",
  "concurrency", "source", "method", "status_code",
  "event", "device", "cpu_model", "cpu_family", "cpu_tier", "cpu_sockets", "cpu_cores",
  "fast_mem_kind", "fast_mem_mode", "strategy", "threads", "threads_batch", "memory_tier",
  "load_time_s",
];
export const STRING_FIELDS = [
  "model", "backend", "artifact", "device", "cpu_model", "cpu_family", "cpu_tier", "strategy",
  "memory_tier", "event", "source", "method", "fast_mem_kind", "fast_mem_mode", "backend_commit",
];
export const DEPTH_FIELDS = ["depth", "pp_tps", "tg_tps", "pp_tps_estimate", "tg_tps_estimate"];

const isNum = (x) => typeof x === "number" && Number.isFinite(x);

function json(status, obj, headers = {}) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

export function sanitize(row) {
  if (!row || typeof row !== "object" || Array.isArray(row)) return null;
  const out = {};
  for (const k of SANITIZED_FIELDS) {
    const v = row[k];
    if (v === null || v === undefined) continue;
    if (STRING_FIELDS.includes(k)) out[k] = String(v).slice(0, 160);
    else if (isNum(v) || typeof v === "boolean") out[k] = v;
    else if (Array.isArray(v)) {
      out[k] = v
        .slice(0, 16)
        .filter((x) => isNum(x) || typeof x === "string")
        .map((x) => (typeof x === "string" ? x.slice(0, 160) : x));
    } else if (typeof v === "string") out[k] = v.slice(0, 160);
  }
  if (Array.isArray(row.depths)) {
    out.depths = row.depths
      .filter((d) => d && typeof d === "object")
      .map((d) => Object.fromEntries(DEPTH_FIELDS.filter((k) => isNum(d[k])).map((k) => [k, d[k]])))
      .filter((d) => "depth" in d)
      .slice(0, 32);
  }
  return Object.keys(out).length ? out : null;
}

export async function senderHash(ip, salt) {
  // Must match pushbutton-telemetry-collector sender_hash(): HMAC-SHA256 hex, first 16 chars.
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey("raw", enc.encode(salt), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = new Uint8Array(await crypto.subtle.sign("HMAC", key, enc.encode(ip)));
  return Array.from(sig, (b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

async function readCapped(stream, cap) {
  const reader = stream.getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > cap) {
      await reader.cancel();
      throw new Error("body too large");
    }
    chunks.push(value);
  }
  const out = new Uint8Array(size);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out;
}

function httpError(message, status) {
  return Object.assign(new Error(message), { status });
}

export async function decodeBody(request) {
  const declared = Number(request.headers.get("content-length") || "0");
  if (declared > MAX_BODY || !request.body) throw httpError("body too large or empty", 413);
  let raw;
  try {
    raw = await readCapped(request.body, MAX_BODY);
  } catch {
    throw httpError("body too large", 413);
  }
  if (!raw.byteLength) throw httpError("empty body", 413);
  let bytes = raw;
  if ((request.headers.get("content-encoding") || "").toLowerCase() === "gzip") {
    const ds = new Blob([raw]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = await readCapped(ds, MAX_DECOMPRESSED);
  }
  const rows = [];
  for (const line of new TextDecoder().decode(bytes).split("\n")) {
    if (!line.trim()) continue;
    const clean = sanitize(JSON.parse(line));
    if (clean) rows.push(clean);
    if (rows.length > MAX_RECORDS) throw new Error("too many records");
  }
  return rows;
}

function base64(text) {
  const bytes = new TextEncoder().encode(text);
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

async function commitRecords(env, path, ndjson) {
  const res = await fetch(`https://api.github.com/repos/${env.DATA_REPO}/contents/${path}`, {
    method: "PUT",
    headers: {
      authorization: "Bearer " + env.GITHUB_TOKEN,
      accept: "application/vnd.github+json",
      "x-github-api-version": "2022-11-28",
      "user-agent": "pushbutton-telemetry-relay",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      message: "telemetry: append records",
      content: base64(ndjson),
      branch: env.DATA_BRANCH || "main",
    }),
  });
  if (!res.ok) throw new Error(`GitHub ${res.status}`);
}

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);
    if (request.method === "GET" && (pathname === "/health" || pathname === "/v1/health")) {
      return json(200, { ok: true, min_interval_s: RATE_LIMIT_S });
    }
    if (request.method !== "POST" || (pathname !== "/" && pathname !== "/v1/telemetry")) {
      return json(404, { error: "not found" });
    }
    if (!env.GITHUB_TOKEN || !env.DATA_REPO || !env.IP_SALT) return json(503, { error: "relay not configured" });

    let rows;
    try {
      rows = await decodeBody(request);
    } catch (e) {
      return json(e.status || 400, { error: e.status ? e.message : "invalid NDJSON payload" });
    }
    if (!rows.length) return json(400, { error: "no valid records" });

    // Raw IP: used for this hash only, never stored or logged.
    const sender = await senderHash(request.headers.get("cf-connecting-ip") || "unknown", env.IP_SALT);
    const now = Date.now();
    const rlKey = `rl:${sender}`;
    if (env.RATE_KV) {
      const last = Number(await env.RATE_KV.get(rlKey));
      if (last && now - last < RATE_LIMIT_S * 1000) {
        const wait = Math.ceil((RATE_LIMIT_S * 1000 - (now - last)) / 1000);
        return json(429, { error: "at most one upload per 5 minutes", retry_after_s: wait }, { "retry-after": String(wait) });
      }
      await env.RATE_KV.put(rlKey, String(now), { expirationTtl: RATE_LIMIT_S });
    }

    const stamp = new Date(now).toISOString().replace(/\.\d{3}Z$/, "Z");
    const ndjson = rows.map((r) => JSON.stringify({ ...r, received_utc: stamp, sender_hash: sender })).join("\n") + "\n";
    const rand = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) => b.toString(16).padStart(2, "0")).join("");
    const day = stamp.slice(0, 10).replace(/-/g, "/");
    const path = `records/${day}/${stamp.slice(11, 19).replace(/:/g, "")}-${sender}-${rand}.ndjson`;
    try {
      await commitRecords(env, path, ndjson);
    } catch {
      if (env.RATE_KV) await env.RATE_KV.delete(rlKey);
      return json(502, { error: "storage unavailable, retry later" });
    }
    return json(200, { accepted: rows.length });
  },
};
