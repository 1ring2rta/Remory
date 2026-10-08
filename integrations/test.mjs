import test from "node:test";
import assert from "node:assert/strict";
import { RemoryClient } from "./client.mjs";
import { compactResponse, readCompactionItem, generateFromItems } from "./codex/bridge.mjs";

const handle = "rm_" + "a".repeat(48);

test("HTTP client carries native IDs, session, and cancellation", async () => {
  const abort = new AbortController();
  const client = new RemoryClient({ url: "http://localhost:8421", sessionId: "session",
    fetchImpl: async (url, options) => {
      assert.equal(url, "http://localhost:8421/v1/compact");
      assert.deepEqual(options.headers, { "Content-Type": "application/json", "X-Remory-Session": "session" });
      assert.equal(options.signal, abort.signal);
      assert.deepEqual(JSON.parse(options.body), { prefix_ids: [1], history_ids: [2], summary_ids: [3], previous: null });
      return new Response(JSON.stringify({ handle }), { status: 200 });
    } });
  assert.equal((await client.compact({ prefixIds: [1], historyIds: [2], summaryIds: [3] }, abort.signal)).handle, handle);
});

test("Codex restores the latest checkpoint and renders only its continuation", async () => {
  const response = await compactResponse({ compact: async () => ({ handle }) }, {});
  const message = { type: "message", role: "user", content: "next" };
  const render = async (items, { continuation }) => {
    assert.equal(continuation, true);
    assert.deepEqual(items, [message]);
    return [7, 8];
  };
  const client = { generate: async args => args };
  const result = await generateFromItems(client,
    [{ type: "message", content: "old" }, ...response.output, message], { render });
  assert.equal(result.handle, handle);
  assert.deepEqual(result.continuationIds, [7, 8]);
  const before = await generateFromItems(client, [message], {
    render: async (items, { continuation }) => {
      assert.equal(continuation, false);
      assert.deepEqual(items, [message]);
      return [1, 2, 3];
    },
  });
  assert.deepEqual(before.inputIds, [1, 2, 3]);
  await assert.rejects(generateFromItems(client,
    [{ type: "compaction", encrypted_content: "foreign" }], { render }), /foreign/);
});

test("generation uses SGLang's native endpoint before and after compaction", async () => {
  const bodies = [];
  const client = new RemoryClient({ url: "http://localhost:8421", sessionId: "session",
    fetchImpl: async (url, options) => {
      assert.equal(url, "http://localhost:8421/generate");
      bodies.push(JSON.parse(options.body));
      return new Response(JSON.stringify({ text: "ok", output_ids: [2], meta_info: {} }));
    } });
  await client.generate({ inputIds: [1, 2], maxNewTokens: 10 });
  await client.generate({ handle, continuationIds: [3], maxNewTokens: 20 });
  assert.deepEqual(bodies[0], { input_ids: [1, 2], sampling_params: { temperature: 0, max_new_tokens: 10 } });
  assert.deepEqual(bodies[1], { input_ids: [3], sampling_params: { temperature: 0, max_new_tokens: 20 }, remory: { handle } });
});

test("Codex carrier round-trip rejects foreign ciphertext", async () => {
  const response = await compactResponse({ compact: async () => ({ handle }) }, {});
  assert.equal(response.object, "response.compaction");
  assert.equal(readCompactionItem(response.output[0]), handle);
  assert.throws(() => readCompactionItem({ type: "compaction", encrypted_content: "foreign" }));
  assert.equal(readCompactionItem({ type: "message" }), null);
});
