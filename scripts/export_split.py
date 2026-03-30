# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "kokoro==0.8.4",
#     "onnx==1.17.0",
#     "onnxruntime==1.20.1",
#     "transformers>=4.43.0,<4.48.0",
# ]
# ///

"""
Export Kokoro as two ONNX graphs: encoder (text/style -> acoustic) + decoder (vocoder).

Use this when a single ``kokoro.onnx`` fails RKNN conversion (unsupported ops,
memory limits, or dynamic-shape issues). Run ``scripts/convert_rknn.py`` on
each ONNX, or run the decoder with ONNX Runtime / PyTorch on CPU only.

Pipeline on device: encoder RKNN -> decoder (RKNN or CPU).

**ONNX export:** Default is **traced** ``torch.onnx.export`` (``dynamo=False``). That
requires **transformers < 4.48** — newer releases change Albert/PL-BERT masking in
ways that break tracing, and **Dynamo** export hits unsupported introspection inside
Transformers decorators. This script's PEP 723 block pins ``transformers`` accordingly
when you use ``uv run``. With a plain ``python`` venv, run
``pip install 'transformers>=4.43,<4.48'`` or use ``uv run``.

Optional ``--dynamo-onnx`` tries ``dynamo=True`` (install ``onnxscript``); it often
still fails on Kokoro + current library stacks.
"""

from __future__ import annotations

import argparse
import inspect
import os
import re

import onnx
import torch
import torch.nn as nn
from kokoro import KModel

# Traced ONNX breaks on Albert when transformers uses the newer flex/SDPA mask path.
_MAX_TRANSFORMERS_FOR_TRACE = (4, 48, 0)


def _transformers_version_tuple() -> tuple[int, int, int]:
    import transformers

    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", transformers.__version__.strip())
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.match(r"^(\d+)\.(\d+)", transformers.__version__.strip())
    if m:
        return int(m.group(1)), int(m.group(2)), 0
    return 0, 0, 0


def _require_transformers_for_traced_onnx() -> None:
    tv = _transformers_version_tuple()
    if tv >= _MAX_TRANSFORMERS_FOR_TRACE:
        raise SystemExit(
            f"transformers {'.'.join(map(str, tv))} is too new for traced ONNX export "
            f"(need < {_MAX_TRANSFORMERS_FOR_TRACE[0]}.{_MAX_TRANSFORMERS_FOR_TRACE[1]}). "
            "Run: pip install 'transformers>=4.43,<4.48'\n"
            "Or use: uv run scripts/export_split.py\n"
            "Experimental alternative: --dynamo-onnx (requires onnxscript; often fails)."
        )


def _force_plbert_eager_attention_for_onnx(kmodel: KModel) -> None:
    """
    Newer Transformers uses SDPA + masking helpers that break under ``torch.jit.trace``
    (legacy ``torch.onnx.export``). PL-BERT is ``CustomAlbert`` → ``AlbertModel``.
    """
    bert = getattr(kmodel, "bert", None)
    if bert is None:
        return
    cfg = getattr(bert, "config", None)
    if cfg is not None:
        for name in ("attn_implementation", "_attn_implementation", "_attn_implementation_internal"):
            if hasattr(cfg, name):
                try:
                    setattr(cfg, name, "eager")
                except (AttributeError, TypeError):
                    pass
    setter = getattr(bert, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter("eager")
        except Exception:
            pass


def _torch_onnx_export(*, legacy_trace: bool, **kwargs) -> None:
    sig = inspect.signature(torch.onnx.export)
    has_dynamo_kw = "dynamo" in sig.parameters

    def _trace_export() -> None:
        if has_dynamo_kw:
            torch.onnx.export(**kwargs, dynamo=False)
        else:
            torch.onnx.export(**kwargs)

    if legacy_trace or not has_dynamo_kw:
        _trace_export()
        return

    dynamo_kwargs = {k: v for k, v in kwargs.items() if k != "verbose"}
    try:
        torch.onnx.export(**dynamo_kwargs, dynamo=True)
    except Exception as e:
        raise RuntimeError(
            "torch.onnx.export(dynamo=True) failed (expected for many Kokoro + Transformers versions). "
            "Install onnxscript if missing: pip install onnxscript\n"
            "For a reliable path, omit --dynamo-onnx and use transformers>=4.43,<4.48 with traced export."
        ) from e


class KokoroEncoderONNX(nn.Module):
    """
    KModel.forward_with_tokens up to (excluding) ``self.decoder``.
    Outputs match ``Decoder.forward(asr, F0_pred, N_pred, ref_s[:, :128])``.
    """

    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel

    def forward(
        self,
        input_ids: torch.LongTensor,
        ref_s: torch.FloatTensor,
        speed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        km = self.kmodel
        dev = input_ids.device

        input_lengths = torch.full(
            (input_ids.shape[0],),
            input_ids.shape[-1],
            device=dev,
            dtype=torch.long,
        )

        text_mask = (
            torch.arange(input_lengths.max(), device=dev)
            .unsqueeze(0)
            .expand(input_lengths.shape[0], -1)
            .type_as(input_lengths)
        )
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(dev)
        bert_dur = km.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = km.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = km.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = km.predictor.lstm(d)
        duration = km.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=dev), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=dev, dtype=d.dtype
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=dev)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(dev)

        en = d.transpose(-1, -2) @ pred_aln_trg
        f0_pred, n_pred = km.predictor.F0Ntrain(en, s)
        t_en = km.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        return asr, f0_pred, n_pred, pred_dur


class KokoroDecoderONNX(nn.Module):
    """ISTFT-net vocoder only."""

    def __init__(self, kmodel: KModel):
        super().__init__()
        self.decoder = kmodel.decoder

    def forward(
        self,
        asr: torch.Tensor,
        f0_pred: torch.Tensor,
        n_pred: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(asr, f0_pred, n_pred, style).squeeze()


def export_encoder(model: KokoroEncoderONNX, out_dir: str, *, legacy_trace: bool) -> str:
    path = os.path.join(out_dir, "kokoro_encoder.onnx")
    input_ids = torch.randint(1, 100, (48,)).numpy()
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    style = torch.randn(1, 256)
    speed = torch.tensor([1], dtype=torch.int32).reshape(1)

    _torch_onnx_export(
        legacy_trace=legacy_trace,
        model=model,
        args=(input_ids, style, speed),
        f=path,
        export_params=True,
        input_names=["input_ids", "style", "speed"],
        output_names=["asr", "f0", "n", "pred_dur"],
        opset_version=17,
        dynamic_axes={
            "input_ids": {1: "input_ids_len"},
            "asr": {2: "mel_frames"},
            "f0": {1: "mel_frames"},
            "n": {1: "mel_frames"},
        },
        # RKNN toolkit 2.3.x can crash in fold_constant on folded Kokoro graphs.
        do_constant_folding=False,
    )
    onnx.checker.check_model(onnx.load(path))
    print(f"export {path} ok")
    return path


def export_decoder(model: KokoroDecoderONNX, out_dir: str, km: KModel, *, legacy_trace: bool) -> str:
    path = os.path.join(out_dir, "kokoro_decoder.onnx")
    # Shapes from a short dummy forward through encoder (same device as km)
    dev = next(km.parameters()).device
    input_ids = torch.LongTensor([[0, 5, 7, 9, 0]]).to(dev)
    ref_s = torch.randn(1, 256, device=dev)
    speed = torch.tensor([1], dtype=torch.int32, device=dev).reshape(1)
    enc = KokoroEncoderONNX(km).to(dev).eval()
    with torch.no_grad():
        asr, f0, n, _ = enc(input_ids, ref_s, speed)
    style = ref_s[:, :128]

    _torch_onnx_export(
        legacy_trace=legacy_trace,
        model=model,
        args=(asr, f0, n, style),
        f=path,
        export_params=True,
        input_names=["asr", "f0", "n", "style"],
        output_names=["waveform"],
        opset_version=17,
        dynamic_axes={
            "asr": {2: "mel_frames"},
            "f0": {1: "mel_frames"},
            "n": {1: "mel_frames"},
            "waveform": {0: "num_samples"},
        },
        do_constant_folding=False,
    )
    onnx.checker.check_model(onnx.load(path))
    print(f"export {path} ok")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Kokoro encoder+decoder ONNX")
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="checkpoints/config.json",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        default="checkpoints/kokoro-v1_0.pth",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="onnx_split",
    )
    parser.add_argument(
        "--dynamo-onnx",
        action="store_true",
        help="Experimental: torch.onnx.export(dynamo=True); needs onnxscript, often fails on Transformers.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    legacy_trace = not args.dynamo_onnx
    if legacy_trace:
        _require_transformers_for_traced_onnx()

    kmodel = KModel(
        config=args.config_file,
        model=args.checkpoint_path,
        disable_complex=True,
    ).eval()
    if legacy_trace:
        _force_plbert_eager_attention_for_onnx(kmodel)
    export_encoder(
        KokoroEncoderONNX(kmodel), args.output_dir, legacy_trace=legacy_trace
    )
    export_decoder(
        KokoroDecoderONNX(kmodel),
        args.output_dir,
        kmodel,
        legacy_trace=legacy_trace,
    )
