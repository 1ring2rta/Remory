"""Run one compaction and continue from its memory with the deployed Qwen actor."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from remory.adapters.harness import ChatTemplate
from remory.client import Client
from remory.models.load import resolve_checkpoint, resolve_actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8421")
    parser.add_argument("--checkpoint", default="mocoV3/Remory-Qwen3.8-27B")
    parser.add_argument("--actor", help="same local actor snapshot used by the server")
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
    summary = json.dumps({
        "current_progress": ["Tests passed."],
        "key_decisions": [],
        "important_context_constraints_preferences": ["Answer briefly."],
        "next_steps": ["Publish the README."],
        "critical_data_examples_references": ["Project: Remory."],
    })
    prefix_ids, history_ids = codec.split(prefix, history)
    _, tail = codec.split(prefix, [{"role": "user", "content": "What should we do next?"}],
                          generation=True)
    request = dict(prefix_ids=prefix_ids, history_ids=history_ids,
                   summary_ids=codec.summary(summary), continuation_ids=tail)
    if args.write_input:
        args.write_input.write_text(json.dumps(request) + "\n")
    with Client(args.url, session_id="quickstart") as client:
        checkpoint = client.compact(prefix_ids=prefix_ids, history_ids=history_ids,
                                    summary_ids=request["summary_ids"])
        print(f"Compacted {len(history_ids)} history tokens into "
              f"{checkpoint['receipt']['soft_tokens']} residual tokens plus the summary.")
        result = client.generate(handle=checkpoint["handle"], continuation_ids=tail,
                                 max_new_tokens=128)
        print(result["text"])
        client.delete(checkpoint["handle"])


if __name__ == "__main__":
    main()
