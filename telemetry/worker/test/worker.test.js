import { test } from "node:test";
import assert from "node:assert/strict";
import { gzipSync } from "node:zlib";
import worker, { senderHash, sanitize, MAX_BODY } from "../src/index.js";

class FakeKV {
  constructor() { this.m = new Map(); }
  async get(k) { return this.m.get(k) ?? null; }
  async put(k, v) { this.m.set(k, v); }
  async delete(k) { this.m.delete(k); }
}

function setup(status = 201) {
  const env = { GITHUB_TOKEN: "test-token", DATA_REPO: "o/private", IP_SALT: "salt", RATE_KV: new FakeKV() };
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url, init });
    return new Response("{}", { status });
  };
  return { env, calls };
}

const post = (body, headers = {}) =>
  new Request("https://relay.example/v1/telemetry", {
    method: "POST",
    body,
    headers: { "cf-connecting-ip": "203.0.113.7", ...headers },
  });

const ROW = JSON.stringify({
  event: "performance", model: "ornith-1.5:9b", artifact: "Q8_0", load_time_s: 4.2,
  depths: [{ depth: 0, pp_tps: 40, tg_tps: 10, junk: "x" }], prompt: "SECRET", hostname: "box",
});

test("hash matches the Python collector (HMAC-SHA256, 16 hex)", async () => {
  // python3 -c "import hmac,hashlib;print(hmac.new(b'salt',b'203.0.113.7',hashlib.sha256).hexdigest()[:16])"
  assert.equal(await senderHash("203.0.113.7", "salt"), "549fe5bec61ee86f");
});

test("sanitize drops non-whitelisted fields", () => {
  const s = sanitize(JSON.parse(ROW));
  assert.equal(s.prompt, undefined);
  assert.equal(s.hostname, undefined);
  assert.deepEqual(s.depths, [{ depth: 0, pp_tps: 40, tg_tps: 10 }]);
});

test("commits hashed records to the private repo, never the raw IP, then rate limits", async () => {
  const { env, calls } = setup();
  const res = await worker.fetch(post(gzipSync(ROW + "\n"), { "content-encoding": "gzip" }), env);
  assert.equal(res.status, 200);
  assert.equal((await res.json()).accepted, 1);
  assert.equal(calls.length, 1);
  assert.match(calls[0].url, /^https:\/\/api\.github\.com\/repos\/o\/private\/contents\/records\/\d{4}\/\d{2}\/\d{2}\//);
  const body = JSON.parse(calls[0].init.body);
  const text = Buffer.from(body.content, "base64").toString();
  const rec = JSON.parse(text);
  assert.equal(rec.sender_hash, await senderHash("203.0.113.7", "salt"));
  assert.match(rec.received_utc, /Z$/);
  assert.ok(!text.includes("203.0.113.7"));
  assert.ok(!text.includes("SECRET"));
  assert.ok(!calls[0].url.includes("203.0.113.7"));
  const again = await worker.fetch(post(ROW), env);
  assert.equal(again.status, 429);
  assert.ok(Number(again.headers.get("retry-after")) > 0);
  assert.equal(calls.length, 1);
});

test("rejected payloads do not consume the slot; storage failure releases it", async () => {
  const { env, calls } = setup(500);
  assert.equal((await worker.fetch(post("not json"), env)).status, 400);
  assert.equal((await worker.fetch(post("x".repeat(MAX_BODY + 1)), env)).status, 413);
  assert.equal((await worker.fetch(post(ROW), env)).status, 502);
  globalThis.fetch = async (url, init) => { calls.push({ url, init }); return new Response("{}", { status: 201 }); };
  assert.equal((await worker.fetch(post(ROW), env)).status, 200);
});

test("unconfigured relay refuses uploads", async () => {
  setup();
  const res = await worker.fetch(post(ROW), {});
  assert.equal(res.status, 503);
});
