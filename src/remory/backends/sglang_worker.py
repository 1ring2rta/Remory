"""Own SGLang's processes and event loop, isolated from the HTTP server."""
import os
import signal
import traceback


def engine_settings(settings):
    from .sglang_hook import HOOK
    tp_size = settings.get("tp_size", 1)
    if type(tp_size) is not int or tp_size < 1:
        raise ValueError("tensor parallel size must be a positive integer")
    if settings.get("model_family") == "glm":
        return dict(
            model_path=settings["actor"], served_model_name="GLM-5.3-Flash",
            base_gpu_id=settings["gpu"], tp_size=tp_size, ep_size=tp_size,
            context_length=settings["context_limit"], mem_fraction_static=settings["memory_fraction"],
            dtype="bfloat16", quantization="fp8", trust_remote_code=False,
            reasoning_parser="glm45", tool_call_parser="glm47",
            attention_backend="dsa", dsa_prefill_backend="trtllm", dsa_decode_backend="trtllm",
            linear_attn_backend="triton", kv_cache_dtype="fp8_e4m3",
            moe_runner_backend="flashinfer_trtllm", chunked_prefill_size=8192,
            max_prefill_tokens=16384, max_running_requests=1, max_mamba_cache_size=4,
            disable_prefill_cuda_graph=True, disable_overlap_schedule=True,
            cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
            forward_hooks=[{"name": "remory_glm", "target_modules": ["logits_processor"],
                "hook_factory": "remory.backends.glm_hook:make_attestation_hook",
                "config": {"checkpoint": settings["checkpoint"],
                    "compressor_sha256": settings["weights_sha256"],
                    "feature_view": "decoder_block_attn_hc_collapsed_input",
                    "target_layer_ids": [0, 11, 22, 33, 44], "abi": "glm53_stage2_memory_v1"}}],
            watchdog_timeout=1200, log_level="warning",
        )
    return dict(
        model_path=settings["actor"], dtype="bfloat16", tp_size=tp_size,
        base_gpu_id=settings["gpu"], context_length=settings["context_limit"],
        max_total_tokens=settings["context_limit"], max_running_requests=1,
        mem_fraction_static=settings["memory_fraction"], chunked_prefill_size=-1,
        disable_radix_cache=True, mamba_radix_cache_strategy="no_buffer",
        cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
        disable_overlap_schedule=True, disable_custom_all_reduce=True,
        enforce_disable_flashinfer_allreduce_fusion=True,
        attention_backend="triton", enable_return_hidden_states=True,
        enable_custom_logit_processor=True, random_seed=42, log_level="warning",
        forward_hooks=[{"name": "remory", "target_modules": ["logits_processor"],
                        "hook_factory": HOOK,
                        "config": {"checkpoint": settings["checkpoint"]}}],
    )


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
        print("[Remory] Loading the actor and residual memory; first startup compiles GPU kernels.",
              flush=True)
        engine = Engine(**engine_settings(settings))
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
