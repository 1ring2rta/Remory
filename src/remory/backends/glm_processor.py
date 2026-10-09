"""Load the actual GLM vision processor; never accept tokenizer fallback.

SGLang's serving environment has Transformers 5.12.1, whose AutoProcessor
silently falls back to TokenizersBackend for this new model. Only the three
GLM processor components are copied from the existing 5.17.0 training runtime;
the shared Python environment and weights remain unchanged.
"""
from __future__ import annotations
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent


def register_components():
    import transformers
    import transformers.models
    name = "transformers.models.glm5_next"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / "glm_processor_sources/glm5_next")]
        package.__package__ = name
        sys.modules[name] = package
        setattr(transformers.models, "glm5_next", package)
    classes = []
    for module_name, class_name in (
        ("image_processing_glm5_next", "Glm5NextImageProcessor"),
        ("video_processing_glm5_next", "Glm5NextVideoProcessor"),
        ("processing_glm5_next", "Glm5NextProcessor"),
    ):
        module = importlib.import_module(name + "." + module_name)
        cls = getattr(module, class_name)
        setattr(sys.modules[name], class_name, cls)
        setattr(transformers, class_name, cls)
        classes.append(cls)
    return classes


def assert_processor(processor):
    required = ("replace_image_token", "_get_num_multimodal_tokens")
    if type(processor).__name__ != "Glm5NextProcessor" or any(not callable(getattr(processor, k, None)) for k in required):
        raise RuntimeError("GLM requires a real Glm5NextProcessor with image expansion; tokenizer fallback is forbidden")
    if type(getattr(processor, "image_processor", None)).__name__ != "Glm5NextImageProcessor":
        raise RuntimeError("GLM requires the native Glm5NextImageProcessor")
    return processor


def load_processor(model_path, tokenizer):
    image_class, video_class, processor_class = register_components()
    root = Path(model_path).resolve()
    config = json.loads((root / "processor_config.json").read_text())
    if config.get("processor_class") != "Glm5NextProcessor":
        raise RuntimeError("Unexpected GLM processor config")
    image_config = dict(config["image_processor"])
    video_config = dict(config["video_processor"])
    if image_config.pop("image_processor_type") != "Glm5NextImageProcessor" or video_config.pop("video_processor_type") != "Glm5NextVideoProcessor":
        raise RuntimeError("Unexpected GLM vision component config")
    processor = processor_class(image_processor=image_class(**image_config), tokenizer=tokenizer,
        video_processor=video_class(**video_config), chat_template=(root / "chat_template.jinja").read_text())
    return assert_processor(processor)


def processor_attestation(processor):
    assert_processor(processor)
    def component(value):
        module = importlib.import_module(type(value).__module__)
        path = Path(module.__file__).resolve()
        return {"class": type(value).__name__, "module": type(value).__module__, "source": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"processor": component(processor), "image_processor": component(processor.image_processor),
            "video_processor": component(processor.video_processor), "tokenizer_class": type(processor.tokenizer).__name__,
            "native_image_expansion": True, "source_package": "transformers-5.17.0 isolated components"}
