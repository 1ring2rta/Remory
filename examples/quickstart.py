"""Generate, compact, and continue through one SGLang server."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from remory.adapters.harness import ChatTemplate, Harness
from remory.client import Client
from remory.models.load import resolve_checkpoint, resolve_actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8421")
    parser.add_argument("--remory-checkpoint", "--checkpoint", dest="checkpoint", default="mocoV3/Remory-Qwen3.8-27B")
    parser.add_argument("--model-path", "--actor", dest="actor", help="same actor model ID or local snapshot used by the server")
    parser.add_argument("--write-input", type=Path, help="also save the tokenized input for the Codex example")
    args = parser.parse_args()
    _, config = resolve_checkpoint(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(resolve_actor(config, args.actor))
    codec = ChatTemplate(tokenizer, template_kwargs={"enable_thinking": False})

    prefix = [{"role": "system", "content": "You are a helpful assistant. Answer briefly."},
              {"role": "user", "content": "Help me prepare the release of a project named Remory."}]
    history = [{"role": "assistant", "content": "What remains before release?"},
               {"role": "user", "content": "The tests passed. The next step is to publish the README."},
               {"role": "assistant", "content": "Understood. I will keep that next step in mind."}]
    prefix_ids, history_ids = codec.split(prefix, history)
    question = [{"role": "user", "content": "What should we do next?"}]
    _, tail = codec.split(prefix, question, generation=True)
    with Client(args.url, session_id="quickstart") as client:
        _, full_tail = codec.split(prefix, history + question, generation=True)
        result = client.generate(input_ids=prefix_ids + full_tail, max_new_tokens=128)
        print("Before compaction:", result["text"], flush=True)

        harness = Harness(client, codec)
        contract = config["summary_contract"]
        checkpoint = harness.compact(prefix_messages=prefix, removed_messages=history,
            summary_prompt=contract["prompt"], summary_schema=contract["output_schema"])
        try:
            print("Generated summary:", checkpoint.summary, flush=True)
            result = harness.generate(checkpoint=checkpoint, prefix_messages=prefix,
                                      messages=question, max_new_tokens=128)
            print("After compaction:", result["text"], flush=True)
            if args.write_input:
                request = dict(prefix_ids=prefix_ids, history_ids=history_ids,
                    summary_ids=codec.summary(checkpoint.summary), continuation_ids=tail)
                args.write_input.write_text(json.dumps(request) + "\n")
        finally:
            client.delete(checkpoint.handle)


if __name__ == "__main__":
    main()
