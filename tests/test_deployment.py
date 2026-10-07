import hashlib

import pytest

from remory.backends import sglang_setup


def test_patch_is_idempotent_and_rejects_unknown_sources_before_writing(tmp_path, monkeypatch):
    changes = {}
    for name in ("first.py", "second.py"):
        old, new = "upstream\n", "patched\n"
        (tmp_path/name).write_text(old)
        changes[name] = {"before": hashlib.sha256(old.encode()).hexdigest(),
                         "after": hashlib.sha256(new.encode()).hexdigest(), "old": old, "new": new}
    monkeypatch.setattr(sglang_setup, "recipe", lambda: {"commit": "pinned", "files": changes})
    (tmp_path/"second.py").write_text("different upstream\n")
    with pytest.raises(RuntimeError, match="differs"):
        sglang_setup.patch_installation(tmp_path)
    assert (tmp_path/"first.py").read_text() == "upstream\n"
    (tmp_path/"second.py").write_text("upstream\n")
    sglang_setup.patch_installation(tmp_path)
    sglang_setup.patch_installation(tmp_path)
    assert sglang_setup.verify_installation(tmp_path)["patched_files"] == 2
    (tmp_path/"first.py").write_text("changed after install")
    with pytest.raises(RuntimeError, match="incompatible"):
        sglang_setup.verify_installation(tmp_path)
