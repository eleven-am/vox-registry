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
