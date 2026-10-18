from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPRESSIVE_TTS_MODELS = {
    "cosyvoice2-tts/0.5b.json": {
        "adapter_package": "vox-cosyvoice",
        "min_vram_gb": 8,
    },
    "dia-tts/1.6b.json": {
        "adapter_package": "vox-dia",
        "min_vram_gb": 12,
    },
    "orpheus-tts/medium-3b.json": {
        "adapter_package": "vox-orpheus",
        "min_vram_gb": 10,
    },
    "indextts-tts/2.json": {
        "adapter_package": "vox-indextts",
        "min_vram_gb": 10,
    },
    "step-audio-editx/3b-awq.json": {
        "adapter_package": "vox-step-audio-editx==0.1.9",
        "min_vram_gb": 10,
    },
}


def _load_model(relative_path: str) -> dict:
    path = ROOT / "library" / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def test_expressive_tts_entries_declare_honest_runtime_requirements():
    for relative_path, expected in EXPRESSIVE_TTS_MODELS.items():
        model = _load_model(relative_path)
        required = model["runtime"]["required"]

        assert model["type"] == "tts"
        assert model["format"] == "pytorch"
        assert model["adapter_package"] == expected["adapter_package"]
        assert required["python_modules"] == ["torch"]
        assert required["accelerators"] == ["cuda"]
        assert required["systems"] == ["linux"]
        assert required["machines"] == ["x86_64"]
        assert required["min_vram_gb"] == expected["min_vram_gb"]

        notes = " ".join(required.get("notes", ()))
        assert "CPU" in notes
        assert "Spark/ARM NVIDIA" in notes


def test_index_points_to_existing_model_files():
    index = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))

    for entry in index:
        path = ROOT / "library" / str(entry["name"]) / f"{entry['tag']}.json"
        assert path.is_file(), f"index entry does not exist: {path}"


def test_step_audio_editx_declares_external_audio_tokenizer_artifact():
    model = _load_model("step-audio-editx/3b-awq.json")

    assert model["source"] == "stepfun-ai/Step-Audio-EditX-AWQ-4bit"
    assert model["adapter"] == "step-audio-editx-tts-vllm"
    assert model["artifacts"] == [
        {
            "source": "stepfun-ai/Step-Audio-Tokenizer",
            "prefix": "audio_tokenizer",
        }
    ]


def test_smart_turn_defaults_to_cpu_and_supports_forced_gpu():
    model = _load_model("smart-turn/v3.2.json")
    variants = {variant["id"]: variant for variant in model["variants"]}

    assert model["type"] == "turn"
    assert model["license"] == "BSD-2-Clause"
    assert variants["cpu"]["priority"] > variants["gpu"]["priority"]
    assert variants["cpu"]["parameters"]["provider"] == "cpu"
    assert variants["gpu"]["requires"]["accelerators"] == ["onnx_cuda"]
    assert variants["gpu"]["parameters"]["provider"] == "gpu"


def test_magpie_declares_cpu_runtime_and_companion_codec():
    model = _load_model("magpie-tts/v2607-f16.json")
    index = json.loads((ROOT / "index.json").read_text(encoding="utf-8"))
    index_entry = next(
        entry for entry in index if entry["name"] == "magpie-tts" and entry["tag"] == "v2607-f16"
    )

    assert model["source"] == "nvidia/magpie_tts_multilingual_357m"
    assert model["adapter"] == "magpie-tts-gguf"
    assert model["format"] == "gguf"
    assert model["adapter_package"] == "vox-magpie==0.1.0"
    assert model["parameters"] == {"sample_rate": 22050, "default_voice": "john"}
    assert model["files"] == ["magpie_tts_multilingual_357m.v2607.f16.gguf"]
    assert model["artifacts"] == [
        {
            "source": "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
            "prefix": "codec",
            "files": ["nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"],
        }
    ]
    required = model["runtime"]["required"]
    assert required["accelerators"] == ["cpu"]
    assert required["systems"] == ["linux"]
    assert required["machines"] == ["x86_64"]
    assert required["min_ram_gb"] == 4
    assert index_entry == {
        "name": "magpie-tts",
        "tag": "v2607-f16",
        "type": "tts",
        "description": model["description"],
        "adapter_package": "vox-magpie==0.1.0",
    }
