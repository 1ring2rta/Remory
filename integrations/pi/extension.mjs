/** Factory for Pi's session_before_compact hook.
 * Supply buildRequest using the SAME tokenizer/renderer as your residual model
 * provider. Persist remory details with the compaction entry, not a global var.
 * The provider must read that entry on every request, including resume/branch.
 */
export function installRemoryCompaction(pi, { clientForSession, buildRequest }) {
  pi.on("session_before_compact", async (event, ctx) => {
    try {
      // buildRequest produces the native summary and exact source span; summary
      // generation remains the harness's policy, not an opaque text rewrite.
      const { summary, prefixIds, historyIds, summaryIds, previous = null } =
        await buildRequest(event, ctx);
      const client = clientForSession(ctx);
      const result = await client.compact({ prefixIds, historyIds, summaryIds, previous }, event.signal);
      if (event.signal.aborted) return { cancel: true };
      return { compaction: {
        summary,
        firstKeptEntryId: event.preparation.firstKeptEntryId,
        tokensBefore: event.preparation.tokensBefore,
        details: { remory: { handle: result.handle, summary } },
      } };
    } catch (error) {
      // Throwing from an extension may let other handlers continue. Explicitly
      // cancel so Pi cannot replace source history with a text-only fallback.
      ctx.ui.notify(`Remory compaction cancelled: ${error.message}`, "error");
      return { cancel: true };
    }
  });
}
