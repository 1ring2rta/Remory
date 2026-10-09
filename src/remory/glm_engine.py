"""GLM compaction: retain the old frontier and encode only newly removed history."""
import numpy as np

from .engine import Remory
from .types import Prepared, digest, token_ids


class GlmRemory(Remory):
    def compact(self, *, owner, prefix_ids, history_ids, summary_ids, previous=None):
        self.store._key(owner)
        prefix = token_ids(prefix_ids, "prefix_ids")
        history = token_ids(history_ids, "history_ids", empty=previous is not None)
        summary = token_ids(summary_ids, "summary_ids")
        c = self.contract
        slots = c.block_tokens // c.ratio
        nodes, rows, source_offset = [], [], 0
        if previous:
            old = self.store.get(owner, previous)
            self._check_memory(old)
            if old.prefix_ids != prefix:
                raise ValueError("original system/tools/task prefix changed across compaction")
            nodes = [dict(n) for n in old.receipt["frontier_nodes"]]
            rows = list(np.split(old.embeddings, len(nodes)))
            source_offset = old.receipt["source_tokens"]
        old_slots = len(rows) * slots
        if history:
            # The evaluated GLM encoder never reads the summary. Existing soft
            # memory supplies the causal prefix for newly observed raw history.
            source = Prepared(prefix + (c.placeholder_id,) * old_slots + history,
                tuple(range(len(prefix), len(prefix) + old_slots)),
                np.concatenate(rows) if rows else None)
            self._check_length(len(source.input_ids))
            encoded = self._matrix(self.backend.encode(source,
                start=len(prefix) + old_slots, end=len(source.input_ids), operation="leaves"),
                ((len(history) + c.block_tokens - 1) // c.block_tokens) * slots)
            for i, start in enumerate(range(0, len(history), c.block_tokens)):
                rows.append(encoded[i * slots:(i + 1) * slots])
                nodes.append({"depth": 1, "start": source_offset + start,
                              "stop": source_offset + min(start + c.block_tokens, len(history))})
        if not rows:
            raise ValueError("compaction requires history or existing memory")
        while len(rows) * slots > c.memory_budget:
            for i in range(len(nodes) - c.ratio + 1):
                children = nodes[i:i + c.ratio]
                if len({n["depth"] for n in children}) != 1:
                    continue
                if any(a["stop"] != b["start"] for a, b in zip(children, children[1:])):
                    raise RuntimeError("GLM memory lost chronological coverage")
                source = Prepared(prefix + (c.placeholder_id,) * c.block_tokens,
                    tuple(range(len(prefix), len(prefix) + c.block_tokens)),
                    np.concatenate(rows[i:i + c.ratio]))
                self._check_length(len(source.input_ids))
                parent = self._matrix(self.backend.encode(source, start=len(prefix),
                    end=len(source.input_ids), operation="parent"), slots)
                rows[i:i + c.ratio] = [parent]
                nodes[i:i + c.ratio] = [{"depth": children[0]["depth"] + 1,
                    "start": children[0]["start"], "stop": children[-1]["stop"]}]
                break
            else:
                raise ValueError("cannot meet the memory budget without dropping history")
        memory = np.concatenate(rows)
        self._check_length(len(prefix) + len(c.summary_before) + len(summary)
            + len(c.summary_after) + len(c.memory_before) + len(memory) + len(c.memory_after))
        receipt = dict(schema="remory.glm.compaction.v1", source_tokens=source_offset + len(history),
            new_source_tokens=len(history), summary_tokens=len(summary), soft_tokens=len(memory),
            source_memory_slots=old_slots, previous=previous, frontier_nodes=nodes,
            prefix_sha256=digest(prefix), summary_sha256=digest(summary), history_sha256=digest(history))
        return self.store.put(owner=owner, identity=c.identity, prefix_ids=prefix,
            summary_ids=summary, embeddings=memory,
            slot_depths=tuple(n["depth"] for n in nodes for _ in range(slots)), receipt=receipt)
