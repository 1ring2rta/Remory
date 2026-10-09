"""Compaction and generation, leaving tools and transcript ownership in the harness."""
from dataclasses import asdict, dataclass
import json
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
            tokenize=True, return_dict=False, add_generation_prompt=generation,
            **self.template_kwargs))

    def split(self, prefix_messages, messages, *, generation=False):
        prefix = self._render(prefix_messages)
        full = self._render(prefix_messages + messages, generation=generation)
        if full[:len(prefix)] != prefix:
            raise ValueError("chat template changed the immutable prefix; choose a stable message boundary")
        return prefix, full[len(prefix):]

    def summary(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def final_text(self, text):
        return text.strip()


class GlmChatTemplate(ChatTemplate):
    """Native GLM thinking boundary; only the completed answer becomes a summary."""
    def __init__(self, tokenizer, *, tools=None, reasoning_effort="max"):
        super().__init__(tokenizer, tools=tools,
            template_kwargs={"reasoning_effort": reasoning_effort, "clear_thinking": False})

    def final_text(self, text):
        if text.count("</think>") != 1:
            raise RuntimeError("GLM did not finish its reasoning; retain the previous context")
        final = text.split("</think>", 1)[1].strip()
        for end in ("<|endoftext|>", "<|user|>", "<|observation|>"):
            final = final.removesuffix(end).rstrip()
        if "<think>" in final:
            raise RuntimeError("invalid GLM final answer boundary")
        return final


class Harness:
    def __init__(self, client, codec: ChatTemplate):
        self.client, self.codec = client, codec

    def compact(self, *, prefix_messages, removed_messages, summary_prompt,
                summary_schema=None, max_summary_tokens=1024,
                previous: Checkpoint | None = None) -> Checkpoint:
        """Replace compact: generate the summary, then encode residual memory.

        Both steps use the deployed SGLang model. Publish the new checkpoint
        only after they succeed; previous remains usable if either step fails.
        """
        messages = removed_messages + [{"role": "user", "content": summary_prompt}]
        prefix, tail = self.codec.split(prefix_messages, messages, generation=True)
        if previous and digest(prefix) != previous.prefix_sha256:
            raise ValueError("system/tools/task prefix changed since compaction")
        source = ({"handle": previous.handle, "continuation_ids": tail} if previous
                  else {"input_ids": prefix + tail})
        sampling = {"json_schema": json.dumps(summary_schema)} if summary_schema else None
        result = self.client.generate(**source, max_new_tokens=max_summary_tokens, sampling=sampling)
        reason = result.get("meta_info", {}).get("finish_reason")
        if reason == "length" or isinstance(reason, dict) and reason.get("type") in {"length", "abort", "error"}:
            raise RuntimeError("summary did not complete; keep the previous context")
        summary = getattr(self.codec, "final_text", str.strip)(result["text"])
        if not summary:
            raise RuntimeError("summary is empty; keep the previous context")
        if summary_schema:
            json.loads(summary)
        return self.on_compact(prefix_messages=prefix_messages, removed_messages=removed_messages,
                               summary=summary, previous=previous)

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
