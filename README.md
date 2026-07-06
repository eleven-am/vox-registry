# Vox Model Registry

Model registry for [Vox](https://github.com/eleven-am/vox) — Ollama for Speech.

## Adding a model

1. Create a JSON file under `library/<model-name>/<tag>.json`
2. Submit a pull request
3. Once merged, `vox pull <model-name>:<tag>` will find your model

## Schema

Each model JSON file contains:

| Field | Required | Description |
|-------|----------|-------------|
| `source` | yes | HuggingFace repo ID |
| `architecture` | yes | Model architecture name |
| `type` | yes | `stt` or `tts` |
| `adapter` | yes | Adapter entry point name |
| `format` | yes | `onnx`, `ct2`, `pytorch`, `gguf` |
| `adapter_package` | yes | pip package name (e.g. `vox-kokoro`) |
| `description` | no | Human-readable description |
| `license` | no | License identifier |
| `parameters` | no | Default model parameters |
| `files` | no | Specific files to download (all if omitted) |
| `runtime` | no | Pull-time runtime requirements and notes |

## Runtime requirements

Use `runtime.required` when a model has real hardware, platform, memory, or
Python runtime constraints. Vox uses this metadata during `vox pull` before the
adapter package is imported, so registry entries must describe requirements
honestly instead of relying on adapter load failures.

Supported requirement fields:

| Field | Description |
|-------|-------------|
| `python_modules` | Python imports that must be available in the Vox runtime, such as `torch` |
| `accelerators` | Required accelerator family, such as `cuda`, `mps`, `onnx_cuda`, or `cpu` |
| `systems` | Supported operating systems, such as `linux`, `darwin`, or `windows` |
| `machines` | Supported CPU architectures, such as `x86_64`, `arm64`, or `aarch64` |
| `min_compute_capability` | Minimum NVIDIA compute capability, for example `80` for `sm_80` |
| `min_cuda_version` | Minimum CUDA runtime version |
| `min_vram_gb` | Minimum GPU memory in GiB |
| `min_ram_gb` | Minimum system memory in GiB |
| `notes` | Human-readable explanation of unsupported paths or special constraints |

Expressive GPU-heavy TTS entries must include runtime requirements and clear
notes for unsupported CPU/ONNX or Spark/ARM NVIDIA paths.

## Example

`library/kokoro/v1.0.json`:

```json
{
  "source": "hexgrad/Kokoro-82M-v1.0-ONNX",
  "architecture": "kokoro",
  "type": "tts",
  "adapter": "kokoro",
  "format": "onnx",
  "description": "Kokoro 82M ONNX — fast, lightweight TTS with preset voices",
  "license": "Apache-2.0",
  "parameters": {
    "sample_rate": 24000,
    "default_voice": "af_heart"
  },
  "files": ["model.onnx", "voices.bin"],
  "adapter_package": "vox-kokoro"
}
```
