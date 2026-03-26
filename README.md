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
