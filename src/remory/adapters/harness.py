"""Compaction and generation, leaving tools and transcript ownership in the harness."""
from dataclasses import asdict, dataclass
import re
from ..types import digest


@dataclass(frozen=True)
class Checkpoint:
    handle: str
    summary: str
    prefix_sha256: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or not re.fullmatch(r"rm_[0-9a-f]{48}", value.get("handle", "")):
            raise ValueError("invalid Remory checkpoint descriptor")
        if not isinstance(value.get("summary"), str):
            raise ValueError("missing human-readable summary")
        if not re.fullmatch(r"[0-9a-f]{64}", value.get("prefix_sha256", "")):
            raise ValueError("missing prefix identity")
        return cls(value["handle"], value["summary"], value["prefix_sha256"])


class ChatTemplate:
    """Render the ACTOR's native template; preserve structured tool calls/results.

    Keep prefix_messages identical throughout a branch. They normally contain
    the system/tools and original user task. History/continuations contain only
    subsequent complete native messages. Tools are passed separately to the
    template. Unsupported modalities must be rejected by the caller.
    """
    def __init__(self, tokenizer, *, tools=None, template_kwargs=None):
        self.tokenizer, self.tools = tokenizer, tools
        self.template_kwargs = dict(template_kwargs or {})

    def _render(self, messages, *, generation=False):
        for m in messages:
            if m.get("content") is not None and not isinstance(m["content"], str):
                raise ValueError("normalize text blocks before rendering; images are unsupported")
        return list(self.tokenizer.apply_chat_template(messages, tools=self.tools,
            tokenize=True, add_generation_prompt=generation, **self.template_kwargs))

    def split(self, prefix_messages, messages, *, generation=False):
        prefix = self._render(prefix_messages)
        full = self._render(prefix_messages + messages, generation=generation)
        if full[:len(prefix)] != prefix:
            raise ValueError("chat template changed the immutable prefix; choose a stable message boundary")
        return prefix, full[len(prefix):]

    def summary(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))


class Harness:
    def __init__(self, client, codec: ChatTemplate):
        self.client, self.codec = client, codec

    def on_compact(self, *, prefix_messages, removed_messages, summary: str,
                   previous: Checkpoint | None = None) -> Checkpoint:
        """Call BEFORE deleting messages; persist the returned descriptor with the summary.

        With previous, removed_messages is only the new span since that checkpoint.
        Generate the summary over the active context first, then supply it here.
        No client state is changed on a failed call.
        """
        prefix, history = self.codec.split(prefix_messages, removed_messages)
        result = self.client.compact(prefix_ids=prefix, history_ids=history,
            summary_ids=self.codec.summary(summary), previous=previous.handle if previous else None)
        return Checkpoint(result["handle"], summary, digest(prefix))

    def _tail(self, checkpoint, prefix_messages, messages):
        prefix, tail = self.codec.split(prefix_messages, messages, generation=True)
        if digest(prefix) != checkpoint.prefix_sha256:
            raise ValueError("system/tools/task prefix changed since compaction")
        return tail

    def generate(self, *, checkpoint: Checkpoint, prefix_messages, messages,
                 max_new_tokens=1024, sampling=None):
        """For a provider delegating token generation to the Remory backend."""
        tail = self._tail(checkpoint, prefix_messages, messages)
        return self.client.generate(handle=checkpoint.handle, continuation_ids=tail,
                                    max_new_tokens=max_new_tokens, sampling=sampling)
