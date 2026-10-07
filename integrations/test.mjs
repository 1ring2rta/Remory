import test from "node:test";
import assert from "node:assert/strict";
import { RemoryClient } from "./client.mjs";
import { compactResponse, readCompactionItem } from "./codex/bridge.mjs";
import { afterCompact, generateWithCheckpoint } from "./claude-code/bridge.mjs";
import { installRemoryCompaction } from "./pi/extension.mjs";

const handle = "rm_" + "a".repeat(48);

test("HTTP client carries native IDs, session, authentication, and cancellation", async () => {
  const abort = new AbortController();
  const client = new RemoryClient({ url: "http://localhost:8421", apiKey: "key", sessionId: "session",
    fetchImpl: async (url, options) => {
      assert.equal(url, "http://localhost:8421/v1/compact");
      assert.equal(options.headers["X-Remory-Session"], "session");
      assert.equal(options.headers.Authorization, "Bearer key");
      assert.equal(options.signal, abort.signal);
      assert.deepEqual(JSON.parse(options.body), { prefix_ids: [1], history_ids: [2], summary_ids: [3], previous: null });
      return new Response(JSON.stringify({ handle }), { status: 200 });
    } });
  assert.equal((await client.compact({ prefixIds: [1], historyIds: [2], summaryIds: [3] }, abort.signal)).handle, handle);
});

test("Codex carrier round-trip rejects foreign ciphertext", async () => {
  const response = await compactResponse({ compact: async () => ({ handle }) }, {});
  assert.equal(response.object, "response.compaction");
  assert.equal(readCompactionItem(response.output[0]), handle);
  assert.throws(() => readCompactionItem({ type: "compaction", encrypted_content: "foreign" }));
  assert.equal(readCompactionItem({ type: "message" }), null);
});

test("Pi commits details only on successful encoding and cancels on failure", async () => {
  let handler;
  const pi = { on: (name, fn) => { assert.equal(name, "session_before_compact"); handler = fn; } };
  let fail = false;
  installRemoryCompaction(pi, { clientForSession: () => ({ compact: async () => {
    if (fail) throw new Error("backend failed");
    return { handle };
  } }), buildRequest: async () => ({ summary: "summary", prefixIds: [1], historyIds: [2], summaryIds: [3] }) });
  const event = { preparation: { firstKeptEntryId: "entry-5", tokensBefore: 9000 }, signal: new AbortController().signal };
  const ctx = { ui: { notify: () => {} } };
  const success = await handler(event, ctx);
  assert.equal(success.compaction.details.remory.handle, handle);
  assert.equal(success.compaction.firstKeptEntryId, "entry-5");
  fail = true;
  assert.deepEqual(await handler(event, ctx), { cancel: true });
});

test("Claude bridge keeps the captured source and uses the new summary", async () => {
  const client = { compact: async (body) => {
    assert.deepEqual(body, { prefixIds: [1], historyIds: [2], previous: handle, summaryIds: [3] });
    return { handle };
  }, generate: async (body) => body };
  const checkpoint = await afterCompact(client, { capture: { prefixIds: [1], historyIds: [2], previous: handle },
    compactSummary: "new", encodeSummary: (s) => { assert.equal(s, "new"); return [3]; } });
  assert.equal(checkpoint.summary, "new");
  assert.equal((await generateWithCheckpoint(client, { checkpoint, continuationIds: [4], maxNewTokens: 10 })).handle, handle);
  await assert.rejects(() => afterCompact(client, { capture: null }));
});
