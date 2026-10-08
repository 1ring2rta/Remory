import argparse
import json

from .engine import Remory
from .store import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="Serve compaction with residual memory")
    parser.add_argument("command", choices=["serve", "doctor"])
    parser.add_argument("--backend", choices=["sglang", "transformers"], default="sglang")
    parser.add_argument("--checkpoint", default="mocoV3/Remory-Qwen3.8-27B")
    parser.add_argument("--actor", help="optional local actor snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu", type=int, default=0, help="SGLang GPU index in CUDA_VISIBLE_DEVICES")
    parser.add_argument("--tp", "--tp-size", "--tensor-parallel-size", dest="tp_size", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=0.8)
    parser.add_argument("--context-limit", type=int, help="context length (default: model maximum)")
    parser.add_argument("--store", default="remory.sqlite")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8421)
    args = parser.parse_args()
    if args.backend == "sglang":
        from .launch_server import configure_runtime
        configure_runtime()
    if args.backend == "sglang" and args.command == "serve":
        from .sglang_server import launch
        launch(args)
        return
    if args.backend == "sglang":
        from .backends.sglang import SGLangBackend
        backend = SGLangBackend.from_pretrained(args.checkpoint, actor_path=args.actor,
            gpu=args.gpu, tp_size=args.tp_size, context_limit=args.context_limit,
            memory_fraction=args.memory_fraction)
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
        uvicorn.run(create_app(engine), host=args.host, port=args.port)
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
