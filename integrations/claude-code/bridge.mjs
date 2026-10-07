/** Gateway-side bridge for Claude Code's compaction lifecycle.
 * capture must be the exact tokenized source from the residual provider before
 * compaction. Do not retokenize the transcript using a different chat template.
 */
export async function afterCompact(client, { capture, compactSummary, encodeSummary, signal }) {
  if (!capture) throw new Error("missing pre-compaction source capture");
  const result = await client.compact({ prefixIds: capture.prefixIds,
    historyIds: capture.historyIds, previous: capture.previous || null,
    summaryIds: await encodeSummary(compactSummary) }, signal);
  return { handle: result.handle, summary: compactSummary };
}

export async function generateWithCheckpoint(client, { checkpoint, continuationIds,
  maxNewTokens, sampling, signal }) {
  return client.generate({ handle: checkpoint.handle, continuationIds, maxNewTokens, sampling }, signal);
}
// Native PostCompact is observational: it cannot replace Claude Code's summary.
// The gateway must persist this descriptor keyed by the stable session identity
// and inject residual memory on generation. A hook alone cannot do that.
