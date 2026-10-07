/** Token-level sidecar client for Node >= 18 (also works with browser fetch). */
export class RemoryClient {
  constructor({ url, apiKey, sessionId, fetchImpl = fetch }) {
    this.url = url.replace(/\/$/, "");
    this.fetch = fetchImpl;
    this.headers = { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}`,
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
  prepare({ handle, continuationIds = [] }, signal) {
    return this.request("/v1/prepare", { handle, continuation_ids: continuationIds }, signal);
  }
  generate({ handle = null, inputIds = null, continuationIds = [], maxNewTokens = 1024,
    sampling = null }, signal) {
    return this.request("/v1/generate", { handle, input_ids: inputIds,
      continuation_ids: continuationIds, max_new_tokens: maxNewTokens, sampling }, signal);
  }
}
