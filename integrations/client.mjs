/** SGLang generation and Remory compaction for Node >= 18. */
export class RemoryClient {
  constructor({ url, sessionId, fetchImpl = fetch }) {
    this.url = url.replace(/\/$/, "");
    this.fetch = fetchImpl;
    this.headers = { "Content-Type": "application/json",
      "X-Remory-Session": sessionId };
  }
  async request(path, body, signal) {
    const response = await this.fetch(this.url + path, {
      method: "POST", headers: this.headers, body: JSON.stringify(body), signal,
    });
    if (!response.ok) throw new Error(`Remory ${response.status}: ${await response.text()}`);
    return response.json();
  }
  compact({ prefixIds, historyIds, summaryIds, previous = null }, signal) {
    return this.request("/v1/compact", { prefix_ids: prefixIds, history_ids: historyIds,
      summary_ids: summaryIds, previous }, signal);
  }
  async generate({ handle = null, inputIds = null, continuationIds = [], maxNewTokens = 1024,
    sampling = null }, signal) {
    if (handle && inputIds !== null) throw new Error("Use continuationIds with a memory handle");
    if (!handle && continuationIds.length) throw new Error("continuationIds requires a memory handle");
    const result = await this.request("/generate", { input_ids: handle ? continuationIds : inputIds,
      sampling_params: { temperature: 0, ...sampling, max_new_tokens: maxNewTokens },
      ...(handle ? { remory: { handle } } : {}) }, signal);
    if (["abort", "error"].includes(result.meta_info?.finish_reason?.type)) {
      throw new Error("SGLang aborted generation");
    }
    return result;
  }
}
