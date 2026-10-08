// Run after examples/quickstart.py --write-input /tmp/remory-input.json.
// This exercises both provider seams with already rendered native token IDs.
import { readFile } from "node:fs/promises";
import { RemoryClient } from "../client.mjs";
import { compactResponse, generateFromItems } from "./bridge.mjs";

const input = process.argv[2];
if (!input) throw new Error("Usage: node integrations/codex/example.mjs native-input.json");
const data = JSON.parse(await readFile(input, "utf8"));
const client = new RemoryClient({ url: process.env.REMORY_URL || "http://127.0.0.1:8421",
  sessionId: process.env.REMORY_SESSION || "codex-example" });

const response = await compactResponse(client, { prefixIds: data.prefix_ids,
  historyIds: data.history_ids, summaryIds: data.summary_ids, previous: data.previous || null });
// Persist this compaction item with the Codex transcript only after success.
console.log(JSON.stringify(response));
const result = await generateFromItems(client, [...response.output,
  { type: "message", role: "user", content: "What should we do next?" }], {
  // The quickstart rendered this exact continuation with the native Qwen template.
  // A provider supplies its own renderer for arbitrary Responses input items.
  render: async () => data.continuation_ids, maxNewTokens: 128,
});
console.log(result.text);
