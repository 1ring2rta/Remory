"""Apply and verify the small, version-pinned SGLang integration."""
import hashlib
import importlib.util
import json
from pathlib import Path


def recipe():
    return json.loads(Path(__file__).with_name("sglang_recipe.json").read_text())


def patch_installation(package):
    package = Path(package)
    pending = []
    for name, change in recipe()["files"].items():
        path = package / name
        content = path.read_text()
        digest = hashlib.sha256(content.encode()).hexdigest()
        if digest == change["after"]:
            continue
        if digest != change["before"] or content.count(change["old"]) != 1:
            raise RuntimeError(f"SGLang source differs from the pinned revision: {name}")
        updated = content.replace(change["old"], change["new"])
        if hashlib.sha256(updated.encode()).hexdigest() != change["after"]:
            raise RuntimeError(f"invalid SGLang patch: {name}")
        pending.append((path, updated))
    for path, content in pending:
        path.write_text(content)


def verify_installation(package=None):
    if package is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None:
            raise RuntimeError("SGLang is not installed; run python deploy/install.py")
        package = Path(spec.origin).parent
    for name, change in recipe()["files"].items():
        path = Path(package) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != change["after"]:
            raise RuntimeError(f"SGLang integration is missing or incompatible: {name}; run python deploy/install.py")
    return {"commit": recipe()["commit"], "patched_files": len(recipe()["files"])}
