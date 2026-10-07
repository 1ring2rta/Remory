"""Token-level sidecar API. Tool execution and chat-template rendering stay in the harness."""
from dataclasses import asdict
import secrets
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from .engine import Remory


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompactRequest(Request):
    prefix_ids: list[StrictInt]
    history_ids: list[StrictInt]
    summary_ids: list[StrictInt]
    previous: str | None = None


class GenerateRequest(Request):
    handle: str | None = None
    input_ids: list[StrictInt] | None = None
    continuation_ids: list[StrictInt] = Field(default_factory=list)
    max_new_tokens: StrictInt = 1024
    sampling: dict | None = None


def create_app(engine: Remory, *, api_key: str) -> FastAPI:
    if not api_key:
        raise ValueError("a nonempty API key is required")
    app = FastAPI(title="Remory", version="0.1.0")

    def authorize(authorization: Annotated[str | None, Header()] = None,
                  x_remory_session: Annotated[str | None, Header()] = None):
        if not secrets.compare_digest(authorization or "", "Bearer " + api_key):
            raise HTTPException(401, "invalid API key")
        try:
            engine.store._key(x_remory_session)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return x_remory_session

    def call(fn, **kwargs):
        try:
            return fn(**kwargs)
        except KeyError as error:
            raise HTTPException(404, "memory not found for this session") from error
        except (ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        except (RuntimeError, httpx.HTTPError) as error:
            raise HTTPException(502, "residual backend failed; retain original history") from error

    @app.get("/health")
    def health():
        return {"status": "ok", "schema": "remory.v1"}

    @app.post("/v1/compact")
    def compact(request: CompactRequest, owner: str = Depends(authorize)):
        memory = call(engine.compact, owner=owner, **request.model_dump())
        return {"handle": memory.handle, "summary_ids": list(memory.summary_ids),
                "receipt": memory.receipt}

    @app.post("/v1/generate")
    def generate(request: GenerateRequest, owner: str = Depends(authorize)):
        return asdict(call(engine.generate, owner=owner, **request.model_dump()))

    @app.delete("/v1/memories/{handle}")
    def delete(handle: str, owner: str = Depends(authorize)):
        if not call(engine.store.delete, owner=owner, handle=handle):
            raise HTTPException(404, "memory not found for this session")
        return {"deleted": True}

    return app
