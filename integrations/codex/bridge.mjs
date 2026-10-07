/** Call from your Responses provider, not from Codex's tool/MCP layer. */
export async function compactResponse(client, request, { id, signal } = {}) {
  const checkpoint = await client.compact(request, signal);
  return {
    id: id || `cmp_${checkpoint.handle.slice(3)}`,
    object: "response.compaction",
    output: [{ id: `cmp_${checkpoint.handle.slice(3)}`, type: "compaction",
      encrypted_content: `remory.v1.${checkpoint.handle}` }],
  };
}

export function readCompactionItem(item) {
  if (item?.type !== "compaction") return null;
  const match = /^remory\.v1\.(rm_[0-9a-f]{48})$/.exec(item.encrypted_content || "");
  if (!match) throw new Error("foreign compaction item; cannot restore Remory memory");
  return match[1];
}

/** Return native actor output; the Responses provider formats messages/tool calls.
 * render(items, { continuation }) uses the actor's tokenizer and chat template.
 * With a checkpoint, render only the tail: the server restores prefix + summary.
 */
export async function generateFromItems(client, items, {
  render, maxNewTokens = 1024, sampling = null, signal,
}) {
  const index = items.findLastIndex(item => item.type === "compaction");
  if (index < 0) {
    return client.generate({ inputIds: await render(items, { continuation: false }),
      maxNewTokens, sampling }, signal);
  }
  const handle = readCompactionItem(items[index]);
  return client.generate({ handle,
    continuationIds: await render(items.slice(index + 1), { continuation: true }),
    maxNewTokens, sampling }, signal);
}
// encrypted_content is the Responses protocol field name. The value above is
// an opaque local database reference, NOT OpenAI ciphertext. Only a Remory-aware
// provider should consume it. Preserve the other output items and native tools.
