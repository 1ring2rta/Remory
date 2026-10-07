"""Own SGLang's processes and event loop, isolated from the HTTP server."""
import os
import signal
import traceback


def run_worker(pipe, settings):
    os.setsid()
    def stop(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    engine = None
    try:
        from .sglang_setup import verify_installation
        verify_installation()
        from sglang import Engine
        from .sglang_hook import HOOK
        print("[Remory] Loading the actor and residual memory; first startup compiles GPU kernels.",
              flush=True)
        engine = Engine(
            model_path=settings["actor"], dtype="bfloat16", tp_size=1,
            base_gpu_id=settings["gpu"], context_length=settings["context_limit"],
            max_total_tokens=settings["context_limit"], max_running_requests=1,
            mem_fraction_static=settings["memory_fraction"], chunked_prefill_size=-1,
            disable_radix_cache=True, mamba_scheduler_strategy="no_buffer",
            disable_cuda_graph=True, disable_piecewise_cuda_graph=True,
            disable_overlap_schedule=True, disable_custom_all_reduce=True,
            enforce_disable_flashinfer_allreduce_fusion=True,
            attention_backend="flashinfer", enable_return_hidden_states=True,
            enable_custom_logit_processor=True, random_seed=42, log_level="warning",
            forward_hooks=[{"name": "remory", "target_modules": ["logits_processor"],
                            "hook_factory": HOOK,
                            "config": {"checkpoint": settings["checkpoint"]}}],
        )
        info = engine.get_server_info()
        pipe.send({"ok": True, "result": {k: info[k] for k in
            ("context_length", "max_running_requests", "disable_radix_cache")}})
        while True:
            request = pipe.recv()
            try:
                result = engine.generate(**request)
                pipe.send({"ok": True, "result": result})
            except Exception as exc:
                traceback.print_exc()
                pipe.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    except (EOFError, SystemExit):
        pass
    except BaseException as exc:
        traceback.print_exc()
        try:
            pipe.send({"ok": False, "error": f"SGLang startup failed: {type(exc).__name__}: {exc}"})
        except (BrokenPipeError, OSError):
            pass
    finally:
        if engine is not None:
            engine.shutdown()
        pipe.close()
