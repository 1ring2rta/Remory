"""Residual encoding through the native SGLang /generate endpoint."""
from pathlib import Path

import httpx

from .sglang import SGLangBackend
from ..types import Contract


class SGLangHTTPBackend(SGLangBackend):
    def __init__(self, url, config, actor, *, context_limit, timeout=600, transport=None):
        self.contract = Contract.from_config(config, identity=
            f"sglang:{Path(actor).resolve()}:{config['target_revision']}")
        self.context_limit = context_limit - 1
        self.http = httpx.Client(base_url=url, timeout=timeout, transport=transport, trust_env=False)

    def _request(self, source, sampling, payload=None):
        response = self.http.post("/generate", json=self.build_request(source, sampling, payload))
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"SGLang request failed: {result['error']}")
        return result

    def close(self):
        self.http.close()
