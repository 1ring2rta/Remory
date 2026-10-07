"""Private, checksum-verified CUDA compiler components; no system installation."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def install_toolchain(state, build):
    recipe = json.loads(Path(__file__).with_name("cuda-toolchains.json").read_text())[build]
    destination = state / ("cuda-" + recipe["release"])
    if (destination / "recipe.json").exists():
        if json.loads((destination / "recipe.json").read_text()) != recipe:
            raise RuntimeError("private CUDA toolchain recipe changed")
        return destination
    with tempfile.TemporaryDirectory(prefix=".cuda-", dir=state) as tmp:
        stage = Path(tmp)
        merged = stage / "merged"
        merged.mkdir()
        for name, entry in recipe["components"].items():
            print(f"[Remory] CUDA component: {name} {entry['version']}", flush=True)
            archive = state / "downloads" / Path(entry["relative_path"]).name
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                part = archive.with_suffix(archive.suffix + ".partial")
                with urlopen("https://developer.download.nvidia.com/compute/cuda/redist/"
                             + entry["relative_path"], timeout=60) as response, part.open("wb") as f:
                    shutil.copyfileobj(response, f)
                if sha(part) != entry["sha256"]:
                    raise RuntimeError(f"CUDA download checksum mismatch: {name}")
                part.replace(archive)
            if sha(archive) != entry["sha256"]:
                raise RuntimeError(f"cached CUDA archive changed: {name}")
            unpacked = stage / name
            with tarfile.open(archive) as stream:
                stream.extractall(unpacked, filter="data")
            roots = list(unpacked.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise RuntimeError("unexpected NVIDIA archive layout")
            shutil.copytree(roots[0], merged, dirs_exist_ok=True, symlinks=True)
        if not (merged / "lib64").exists():
            (merged / "lib64").symlink_to("lib", target_is_directory=True)
        (merged / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        merged.rename(destination)
    return destination


def prepare_headers(python, cuda, build):
    # PyTorch supplies cuRAND/NVRTC; FlashInfer's JIT also needs their headers
    # under CUDA_HOME. Keep links inside this private toolchain.
    code = r'''
from importlib.metadata import distribution
from pathlib import Path
import sys,sysconfig
home,build=Path(sys.argv[1]),sys.argv[2]
name,include,library = (("nvidia-curand","nvidia/cu13/include/","nvidia/cu13/lib/libnvrtc.so.13")
    if build=="cu130" else ("nvidia-curand-cu12","nvidia/curand/include/","nvidia/cuda_nvrtc/lib/libnvrtc.so.12"))
d=distribution(name)
for f in d.files:
    if str(f).startswith(include) and str(f).endswith(".h"):
        target=home/"include"/Path(str(f)).relative_to(include)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): target.symlink_to(Path(d.locate_file(f)).resolve())
site=Path(sysconfig.get_path("purelib")); library=site/library
if not library.is_file(): raise RuntimeError("NVRTC is missing")
(site/"remory_nvrtc.pth").write_text("import ctypes; ctypes.CDLL("+repr(str(library))+", mode=ctypes.RTLD_GLOBAL)\n")
'''
    subprocess.run([str(python), "-c", code, str(cuda), build], check=True)
