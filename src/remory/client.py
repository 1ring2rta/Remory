"""Small synchronous HTTP client, independent of any agent framework."""
import httpx


class Client:
    def __init__(self, url: str, *, session_id: str, api_key: str, timeout: float = 600,
                 transport=None):
        self.http = httpx.Client(base_url=url.rstrip("/") + "/", timeout=timeout,
                                 headers={"Authorization": f"Bearer {api_key}",
                                          "X-Remory-Session": session_id}, transport=transport)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.http.close()

    def _post(self, route, body):
        response = self.http.post(route, json=body)
        response.raise_for_status()
        return response.json()

    def compact(self, *, prefix_ids, history_ids, summary_ids, previous=None):
        return self._post("v1/compact", dict(prefix_ids=list(prefix_ids), history_ids=list(history_ids),
                                           summary_ids=list(summary_ids), previous=previous))

    def generate(self, *, max_new_tokens=1024, handle=None, input_ids=None,
                 continuation_ids=(), sampling=None):
        return self._post("v1/generate", dict(handle=handle, input_ids=input_ids,
                                            continuation_ids=list(continuation_ids),
                                            max_new_tokens=max_new_tokens, sampling=sampling))

    def delete(self, handle):
        response = self.http.delete("v1/memories/" + handle)
        response.raise_for_status()
        return response.json()
