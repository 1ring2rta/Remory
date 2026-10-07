import argparse
import json
import os

from .backends.sglang import from_config_file
from .engine import Remory
from .store import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="Remory residual inference sidecar")
    parser.add_argument("command", choices=["serve", "doctor"])
    parser.add_argument("--backend", choices=["sglang", "transformers"], default="sglang")
    parser.add_argument("--backend-url")
    parser.add_argument("--config", help="config.json matching the worker checkpoint")
    parser.add_argument("--server-checkpoint", help="checkpoint path as seen by worker")
    parser.add_argument("--server-model", help="actor path/ID as seen by worker")
    parser.add_argument("--checkpoint", default="mocoV3/Remory-Qwen3.8-27B")
    parser.add_argument("--actor", help="optional local actor snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--store", default="remory.sqlite")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8421)
    args = parser.parse_args()
    if args.command == "serve" and not os.environ.get("REMORY_API_KEY"):
        parser.error("set REMORY_API_KEY for the sidecar (a single trusted operator's sessions)")
    if args.backend == "sglang":
        if not all([args.config, args.backend_url, args.server_checkpoint, args.server_model]):
            parser.error("sglang requires --config, --backend-url, --server-checkpoint, --server-model")
        backend = from_config_file(args.config, base_url=args.backend_url,
                               server_checkpoint=args.server_checkpoint, server_model=args.server_model,
                               api_key=os.environ.get("REMORY_BACKEND_API_KEY"))
    else:
        from .backends.transformers import TransformersBackend
        backend = TransformersBackend.from_pretrained(args.checkpoint, actor_path=args.actor,
            device=args.device, context_limit=args.context_limit)
    try:
        if args.command == "doctor":
            print(json.dumps(backend.attest() if hasattr(backend, "attest") else
                {"identity": backend.contract.identity, "context_limit": backend.context_limit}, indent=2))
            return
        import uvicorn
        from .server import create_app
        engine = Remory(backend, MemoryStore(args.store))
        uvicorn.run(create_app(engine, api_key=os.environ["REMORY_API_KEY"]),
                    host=args.host, port=args.port)
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
