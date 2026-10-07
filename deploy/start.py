"""Clone, run ./deploy.sh, then call the Remory API. No research repo required."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import subprocess
import sys
import tempfile
import venv

from toolchain import install_toolchain, prepare_headers

ROOT = Path(__file__).resolve().parents[1]
KERNEL130 = ("https://github.com/sgl-project/whl/releases/download/v0.4.1/"
    "sglang_kernel-0.4.1+cu130-cp310-abi3-manylinux2014_x86_64.whl"
    "#sha256=9164b8fc2c1652a52156f21d1a960116670c29033f686946840ef5b13558c5c0")


def run(argv, **kwargs):
    subprocess.run(list(map(str, argv)), check=True, **kwargs)


def cuda_build(requested):
    lib = ctypes.CDLL("libcuda.so.1")
    def call(name, *args):
        if getattr(lib, name)(*args):
            raise RuntimeError(f"{name} failed; run on a machine with a visible NVIDIA GPU")
    call("cuInit", 0)
    count, driver = ctypes.c_int(), ctypes.c_int()
    call("cuDeviceGetCount", ctypes.byref(count))
    call("cuDriverGetVersion", ctypes.byref(driver))
    if count.value == 0:
        raise RuntimeError("no NVIDIA GPUs are visible")
    capabilities = []
    for index in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        call("cuDeviceGet", ctypes.byref(device), index)
        call("cuDeviceComputeCapability", ctypes.byref(major), ctypes.byref(minor), device)
        capabilities.append((major.value, minor.value))
    needs130 = max(capabilities) >= (10, 3)
    build = ("cu130" if needs130 else "cu128") if requested == "auto" else requested
    if needs130 and build != "cu130":
        raise RuntimeError("SM 10.3+ requires --cuda cu130")
    if driver.value < (13000 if build == "cu130" else 12080):
        raise RuntimeError(f"NVIDIA driver is too old for {build}")
    return build


def setup_module():
    spec = importlib.util.spec_from_file_location("sglang_setup", ROOT / "src/remory/backends/sglang_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(state, build):
    state.mkdir(parents=True, exist_ok=True)
    setup = setup_module()
    recipe = setup.recipe()
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()
        + (ROOT / "pyproject.toml").read_bytes() + (ROOT / "deploy/constraints.txt").read_bytes()
        + (ROOT / "deploy/cuda-toolchains.json").read_bytes()
        + build.encode()).hexdigest()
    marker = state / "installation.json"
    python = state / "venv/bin/python"
    source = state / "sglang"
    if marker.exists():
        receipt = json.loads(marker.read_text())
        if receipt["fingerprint"] != fingerprint:
            raise RuntimeError("deployment recipe changed; use a new --runtime-dir for this revision")
        setup.verify_installation(source / "python/sglang")
        return python, Path(receipt["cuda_home"])
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(state / "venv")
    if not source.exists():
        with tempfile.TemporaryDirectory(prefix=".sglang-", dir=state) as tmp:
            staging = Path(tmp) / "source"
            run(["git", "init", "-q", staging])
            run(["git", "-C", staging, "remote", "add", "origin", recipe["repository"]])
            run(["git", "-C", staging, "fetch", "--depth", "1", "origin", recipe["commit"]])
            run(["git", "-C", staging, "checkout", "--detach", "FETCH_HEAD"])
            staging.rename(source)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != recipe["commit"]:
        raise RuntimeError("unexpected SGLang source revision")
    setup.patch_installation(source / "python/sglang")
    env = {**os.environ, "PIP_CONFIG_FILE": os.devnull, "PIP_EXTRA_INDEX_URL": "",
           "SETUPTOOLS_SCM_PRETEND_VERSION": recipe["version"]}
    pip = [python, "-m", "pip", "install"]
    print(f"[Remory] Installing the pinned {build} runtime", flush=True)
    # Select CUDA wheels here; resolve their dependencies from PyPI below.
    run([*pip, "--no-deps", f"torch==2.9.1+{build}", f"torchvision==0.24.1+{build}",
         f"torchaudio==2.9.1+{build}", "--index-url", f"https://download.pytorch.org/whl/{build}"], env=env)
    run([*pip, "--index-url", "https://pypi.org/simple", "-c", ROOT / "deploy/constraints.txt",
         "-e", source / "python", KERNEL130 if build == "cu130" else "sglang-kernel==0.4.1",
         "-e", str(ROOT) + "[server,transformers]", "dill==0.3.9"], env=env)
    cuda = install_toolchain(state, build)
    prepare_headers(python, cuda, build)
    marker.write_text(json.dumps({"fingerprint": fingerprint, "sglang": recipe["commit"],
                                 "cuda_home": str(cuda), "cuda_build": build}, indent=2) + "\n")
    return python, cuda


def main():
    parser = argparse.ArgumentParser(description="Install and start Remory with its own SGLang worker.",
        epilog="Remaining options go to remory serve, e.g. --gpu 0 --context-limit 32768 --port 8421.")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / ".remory")
    parser.add_argument("--cuda", choices=["auto", "cu128", "cu130"], default="auto")
    parser.add_argument("--install-only", action="store_true")
    args, extra = parser.parse_known_args()
    if sys.version_info < (3, 11) or platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("deployment requires Linux x86_64 and Python 3.11+")
    for binary in ("git", "c++"):
        if not shutil.which(binary):
            parser.error(f"install {binary} before deploying")
    state = args.runtime_dir.expanduser().resolve()
    python, cuda = install(state, cuda_build(args.cuda))
    if args.install_only:
        print(f"Runtime ready: {python}")
        return
    key = state / "api-key"
    if not key.exists():
        with os.fdopen(os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            f.write(secrets.token_urlsafe(32))
    env = {**os.environ, "REMORY_API_KEY": os.environ.get("REMORY_API_KEY") or key.read_text().strip(),
           "CUDA_HOME": str(cuda), "CUDA_PATH": str(cuda),
           "TRITON_PTXAS_PATH": str(cuda / "bin/ptxas"),
           "FLASHINFER_WORKSPACE_BASE": str(state / "kernel-cache"),
           "PATH": str(cuda / "bin") + os.pathsep + os.environ.get("PATH", os.defpath),
           "TOKENIZERS_PARALLELISM": "false", "SGLANG_DISABLE_CUDNN_CHECK": "1"}
    key_source = "REMORY_API_KEY" if os.environ.get("REMORY_API_KEY") else str(key)
    print(f"[Remory] API key: {key_source}; state: {state / 'memory.sqlite'}", flush=True)
    os.execve(python, [str(python), "-m", "remory.cli", "serve", "--backend", "sglang",
        "--store", str(state / "memory.sqlite"), *extra], env)


if __name__ == "__main__":
    main()
