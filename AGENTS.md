# vox-registry rules (for contributors and reviewers)

This repo is the **single source of truth** for the vox model catalog. The vox
runtime is remote-only: it ships no bundled catalog and resolves every model
from here. A wrong entry here breaks `vox pull` for everyone, so accuracy matters.

## Layout

- `index.json` — a flat list of summaries, one object per `(name, tag)`.
- `library/<name>/<tag>.json` — the full entry for that model tag.

The two must stay in sync: every `library` entry has an `index.json` summary and
vice versa. A PR that adds a library file without an index summary (or the
reverse) is incomplete.

## Naming rules

- Public model **names are logical** — never put a backend suffix in a NAME:
  no `-onnx`, `-torch`, `-ct2`, `-nemo`, `-vllm`, `-mlx`, `-gguf`. The runtime
  picks the backend from the detected hardware, so backend-suffixed names are a
  hard cutoff and would 404.
- Keep the **task suffix** `-stt` / `-tts` when a family ships both (e.g.
  `voxtral-stt` and `voxtral-tts`, `speecht5-stt` and `speecht5-tts`,
  `qwen3-stt` and `qwen3-tts`). A bare `voxtral` would be ambiguous.
- The concrete backend identity lives in the entry's `adapter` **field** (e.g.
  `"adapter": "whisper-stt-ct2"`), which is the adapter name the vox package
  registers. The `adapter` field keeps its concrete name; the model NAME does not.

## index.json summary schema

Each item: `{ "name", "tag", "type" ("stt"|"tts"), "description", "adapter_package" }`.
For variant entries, take `adapter_package` from the first variant.

## library entry schema (`library/<name>/<tag>.json`)

Required: `type`, `adapter`, `format` (`onnx`|`ct2`|`pytorch`|`gguf`), `source`
(the exact Hugging Face repo id), `adapter_package` (the `vox-*` pip package).
Optional: `description`, `license`, `parameters`, `files`, `runtime_source`.

Plus, when applicable, one of two structures:

### `variants` — pull-time hardware selection

Use when one model has **different downloads** for different hardware (different
weights/format/adapter). The runtime picks one at pull time and downloads only it.

```json
"variants": [
  { "id": "torch", "aliases": ["cuda"], "priority": 100,
    "requires": {"python_modules": ["torch"], "accelerators": ["cuda"]},
    "source": "...", "adapter": "kokoro-tts-torch", "format": "pytorch",
    "files": [...], "adapter_package": "vox-kokoro" },
  { "id": "onnx", "aliases": ["cpu"], "priority": 0, "fallback": true,
    "requires": {"python_modules": ["onnxruntime"]},
    "source": "...", "adapter": "kokoro-tts-onnx", "format": "onnx",
    "adapter_package": "vox-kokoro" }
]
```

- Exactly one variant should be the `fallback` (its `requires` are satisfiable in
  any environment, e.g. an ONNX/CPU build).
- Among runnable variants, higher `priority` wins.
- Examples in this repo: `kokoro-tts` (torch on CUDA, onnx on CPU),
  `parakeet-stt` (nemo on CUDA, onnx on CPU).

### `backends` — load-time runtime selection

Use when one **download** can run through **two runtimes** (same weights,
different inference code). The adapter picks at load time.

```json
"backends": {
  "preferred": [
    { "name": "faster-qwen3-tts",
      "requires": {"python_modules": ["torch", "faster_qwen3_tts"],
                   "accelerators": ["cuda"], "min_versions": {"torch": "2.5.1"}} }
  ],
  "fallback": { "name": "qwen-tts", "requires": {"python_modules": ["torch"]} }
}
```

- Currently only `qwen3-tts` uses this.

### variant vs backend — the rule

- **Different download → `variants`.** (Most "faster X" cases: they need their own
  converted checkpoint. e.g. faster-whisper is a separate CTranslate2 download, so
  it would be a variant, not a backend.)
- **Same download, different runtime → `backends`.** Rare.

### `requires` schema (used by both variants and backends)

```json
{ "python_modules": ["torch"], "accelerators": ["cuda"|"mps"|"onnx_cuda"|"cpu"],
  "min_versions": {"torch": "2.5.1"}, "systems": ["linux"], "machines": ["x86_64"],
  "min_compute_capability": 80, "min_cuda_version": "12.4",
  "min_vram_gb": 8, "min_ram_gb": 16 }
```

All fields optional. Ordered constraints (versions, compute capability, VRAM, RAM)
fail when the value can't be detected (conservative), so only add them when the
model genuinely needs them — an unnecessary `min_compute_capability` can push a
capable GPU box onto the fallback.

## Data accuracy (reviewers must check these)

- **`license`**: use the model's **actual** license. A standard SPDX id
  (`apache-2.0`, `mit`, `cc-by-nc-sa-4.0`, ...) or `other` for a custom /
  non-standard license. **Do not default to Apache-2.0.** Non-commercial licenses
  (e.g. CC-BY-NC-SA) change deployment compliance, so mislabeling is a real bug.
- **`source`**: must be the exact HF repo holding the weights **for that entry or
  variant** (turbo/base checkpoints often live in separate repos).
- **`adapter` / `adapter_package`**: must match a real vox adapter and its pip
  package.

## Checklist for a new/changed model

1. `library/<name>/<tag>.json` exists with the full entry.
2. A matching `index.json` summary exists.
3. JSON is valid; the name is logical (no backend suffix).
4. `license` and `source` verified against the model's Hugging Face page.
5. If multiple downloads → `variants` with a fallback; if one download, two
   runtimes → `backends`.
