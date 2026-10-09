"""The backend boundary is token IDs + sparse embedding overrides, never text markers."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Protocol, Sequence

import numpy as np


def token_ids(values: Sequence[int], name: str, *, empty: bool = False) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or (not values and not empty):
        raise ValueError(f"{name} must be a {'possibly empty' if empty else 'nonempty'} token list")
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError(f"{name} must contain nonnegative integers")
    return tuple(values)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Contract:
    hidden_size: int
    block_tokens: int
    ratio: int
    memory_budget: int
    max_depth: int
    placeholder_id: int
    summary_before: tuple[int, ...]
    summary_after: tuple[int, ...]
    memory_before: tuple[int, ...]
    memory_after: tuple[int, ...]
    identity: str

    def __post_init__(self):
        for name in ("hidden_size", "block_tokens", "ratio", "memory_budget", "max_depth"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"invalid {name}")
        if self.ratio < 2 or self.block_tokens % self.ratio:
            raise ValueError("block_tokens must be divisible by ratio >= 2")
        if self.memory_budget < self.block_tokens // self.ratio:
            raise ValueError("memory budget is smaller than one node")
        token_ids([self.placeholder_id], "placeholder_id")
        for name in ("summary_before", "summary_after", "memory_before", "memory_after"):
            token_ids(getattr(self, name), name, empty=True)
        if not self.identity:
            raise ValueError("backend identity must include the model/checkpoint identity")

    @classmethod
    def from_config(cls, config: dict, *, identity: str) -> Contract:
        if config.get("schema") == "remory_glm53_local_evaluation_v1":
            c, s = config["compressor"], config["token_contract"]
            return cls(c["target_hidden_size"], 1024, c["compression_ratio"], 4096,
                       8, 154820, tuple(s["student_summary_before_ids"]),
                       tuple(s["student_summary_after_ids"]), tuple(s["student_memory_before_ids"]),
                       tuple(s["student_memory_after_ids"]), digest({"config": config, "backend": identity}))
        c, r, s = config["compressor"], config["residual"], config["summary_contract"]
        return cls(c["target_hidden_size"], r["block_tokens"], r["compression_ratio"],
                   r["memory_budget"], c["max_depth"], s["end_token_id"],
                   tuple(s["student_summary_before_ids"]), tuple(s["student_summary_after_ids"]),
                   tuple(s["student_memory_before_ids"]), tuple(s["student_memory_after_ids"]),
                   digest({"config": config, "backend": identity}))


@dataclass
class Prepared:
    input_ids: tuple[int, ...]
    memory_positions: tuple[int, ...] = ()
    memory: np.ndarray | None = None
    slot_depths: tuple[int, ...] = ()


@dataclass
class Generation:
    text: str
    output_ids: list[int] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: Any = None


class Backend(Protocol):
    """Implement these two operations to support another inference engine.

    encode('summary'): final normalized actor states at [start:end].
    encode('leaves'/'recursive_leaves'/'parent'): compressor output rows.
    Source actor features MUST be computed under the complete causal prefix.
    generate MUST scatter memory into input embeddings before actor prefill.
    """
    contract: Contract
    context_limit: int

    def encode(self, source: Prepared, *, start: int, end: int, operation: str,
               summary: np.ndarray | None = None, input_depth: int = 0,
               block_depths: Sequence[int] = ()) -> np.ndarray: ...

    def generate(self, source: Prepared, *, max_new_tokens: int,
                 sampling: dict | None = None) -> Generation: ...
