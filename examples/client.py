"""Run: python examples/client.py native_compaction.json --session my-session

Input JSON: prefix_ids, history_ids, summary_ids, continuation_ids and optionally
previous. Generate these with the actor's native tokenizer/template in your harness.
"""
import argparse
import json
import os
from pathlib import Path

from remory.client import Client

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input")
parser.add_argument("--session", required=True)
parser.add_argument("--url", default="http://127.0.0.1:8421")
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()
data = json.loads(Path(args.input).read_text())
with Client(args.url, session_id=args.session, api_key=os.environ["REMORY_API_KEY"]) as client:
    result = client.compact(prefix_ids=data["prefix_ids"], history_ids=data["history_ids"],
                           summary_ids=data["summary_ids"], previous=data.get("previous"))
    print(json.dumps(result, ensure_ascii=False))
    generated = client.generate(handle=result["handle"], continuation_ids=data["continuation_ids"],
                                 max_new_tokens=args.max_new_tokens)
    print(generated["text"])
