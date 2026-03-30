# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy==2.0.2",
#     "requests",
#     "torch==2.5.1",
#     "tqdm==4.67.1",
# ]
# ///
"""
Run this file via:
uv run scripts/fetch_voices.py

See voices in
https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md

Download locations (always under the repository root, not the shell cwd):

**Export / training checkpoints** (used by ``scripts/export.py`` and
``scripts/export_split.py`` defaults):

  - ``<repo>/checkpoints/config.json``
  - ``<repo>/checkpoints/<checkpoint_path>`` (e.g. ``kokoro-v1_0.pth``)

**Package + release artifacts**

  - Same ``config.json`` is also written to ``<repo>/src/kokoro_onnx/config.json``.
  - Voice bundle: ``<repo>/<npz_path>`` (e.g. ``voices-v1.0.bin``).

For each entry in ``config``, ``voice_url``, ``api_url``, ``config_url``,
``checkpoint_url``, and ``checkpoint_path`` must refer to the **same** Hugging
Face model repo. ``voice_url`` uses a ``{name}`` placeholder (``str.format``),
filled from ``api_url`` (see ``get_voice_names``).
"""

import io
from pathlib import Path

import numpy as np
import requests
import torch
from tqdm import tqdm

# Repository root (parent of ``scripts/``), so outputs do not depend on cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

config = {
    # "Kokoro-82M-v1.1-zh": {
    #     "voice_url": "https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/voices/{name}.pt",
    #     "api_url": "https://huggingface.co/api/models/hexgrad/Kokoro-82M-v1.1-zh/tree/main/voices",
    #     "config_url": "https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/raw/main/config.json",
    #     "checkpoint_url": "https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth",
    #     "checkpoint_path": "kokoro-v1_1-zh.pth",
    #     "npz_path": "voices-v1.1-zh.bin",
    # },
    "Kokoro-82M": {
        "voice_url": "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{name}.pt",
        "api_url": "https://huggingface.co/api/models/hexgrad/Kokoro-82M/tree/main/voices",
        "config_url": "https://huggingface.co/hexgrad/Kokoro-82M/raw/main/config.json",
        "checkpoint_url": "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth",
        "checkpoint_path": "kokoro-v1_0.pth",
        "npz_path": "voices-v1.0.bin",
    },
}
# Extract voice names


def get_voice_names(api_url):
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json()
    names = [voice["path"][7:-3] for voice in data]
    return names


def download_model_config(config_url: str) -> None:
    resp = requests.get(config_url)
    resp.raise_for_status()
    data = resp.content
    destinations = (
        CHECKPOINTS_DIR / "config.json",
        REPO_ROOT / "src/kokoro_onnx/config.json",
    )
    for dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"Wrote {dest} (from {config_url})")


def download_large_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        chunk = 1024 * 1024
        with open(dest, "wb") as f, tqdm(
            total=total if total else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            for part in resp.iter_content(chunk_size=chunk):
                if not part:
                    continue
                f.write(part)
                bar.update(len(part))
    print(f"Wrote {dest} (from {url})")


def download_checkpoint(checkpoint_url: str, checkpoint_path: str) -> None:
    dest = (
        Path(checkpoint_path)
        if Path(checkpoint_path).is_absolute()
        else CHECKPOINTS_DIR / checkpoint_path
    )
    download_large_file(checkpoint_url, dest)


def download_voices(voice_url: str, names: list[str], npz_path: str):
    count = len(names)
    out = Path(npz_path) if Path(npz_path).is_absolute() else REPO_ROOT / npz_path
    out.parent.mkdir(parents=True, exist_ok=True)

    # Extract voice files
    print(f"Found {count} voices")
    voices = {}
    for name in tqdm(names):
        url = voice_url.format(name=name)
        print(f"Downloading {name}")
        r = requests.get(url)
        r.raise_for_status()  # Ensure the request was successful
        content = io.BytesIO(r.content)
        data: np.ndarray = torch.load(content, weights_only=True).numpy()
        voices[name] = data

    # Save all voices to a single .npz file
    with open(out, "wb") as f:
        np.savez(f, **voices)

    mb_size = out.stat().st_size // 1000 // 1000
    print(f"Created {out} ({mb_size}MB)")


def main():
    for model_name, model_config in config.items():
        print(f"Downloading {model_name}")
        voice_url = model_config["voice_url"]
        api_url = model_config["api_url"]
        config_url = model_config["config_url"]
        checkpoint_url = model_config["checkpoint_url"]
        checkpoint_path = model_config["checkpoint_path"]
        npz_path = model_config["npz_path"]
        download_model_config(config_url)
        download_checkpoint(checkpoint_url, checkpoint_path)
        voice_names = get_voice_names(api_url)
        download_voices(voice_url, voice_names, npz_path)


main()
