# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
#
# Install RKNN-Toolkit2 separately (see Rockchip docs). Wheels are often
# Linux x86_64 only: https://github.com/airockchip/rknn-toolkit2
#
# Kokoro ONNX uses dynamic axes (e.g. input_ids length). RKNN needs either
# fixed shapes or dynamic_input in config, for example:
#
#   python scripts/convert_rknn.py -i onnx/kokoro.onnx -o rknn/kokoro.rknn -p rk3588 --seq-len 256
#   python scripts/convert_rknn.py -i onnx_split/kokoro_encoder.onnx -o out.rknn -p rk3588 --seq-len 256
#   python scripts/convert_rknn.py -i onnx_split/kokoro_decoder.onnx -o out.rknn -p rk3588 --decoder-preset --mel-len 200

"""
Convert Kokoro ONNX (from scripts/export.py / export_split.py) to RKNN.

Notes
-----
- RKNN-Toolkit2 is not a normal PyPI dependency; install the wheel from Rockchip.
- ONNX models with symbolic dimensions (``input_ids_len``, ``mel_frames``, …)
  require either ``--seq-len`` / ``--decoder-preset`` / ``--input-size-list``,
  or ``--dynamic-input`` if your toolkit version supports it in ``rknn.config``.
- INT8 quantization needs a calibration dataset path for ``rknn.build``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Any, Optional


def _apply_rknn_config(rknn, platform: str, *, dynamic_input: bool) -> None:
    """Call rknn.config; dynamic flag uses Rockchip's ``dyanmic_input`` typo when present."""
    params = inspect.signature(rknn.config).parameters
    kwargs: dict = {"target_platform": platform}
    if dynamic_input:
        if "dyanmic_input" in params:
            kwargs["dyanmic_input"] = True
        elif "dynamic_input" in params:
            kwargs["dynamic_input"] = True
        else:
            print(
                "Warning: --dynamic-input was set but rknn.config has no "
                "dynamic_input / dyanmic_input parameter; use --seq-len or "
                "--input-size-list for fixed shapes.",
                file=sys.stderr,
            )
    rknn.config(**kwargs)


def _resolve_input_size_list(args: argparse.Namespace) -> Optional[list[Any]]:
    if args.input_size_list:
        data = json.loads(args.input_size_list)
        if not isinstance(data, list):
            raise SystemExit("--input-size-list must be a JSON array of shape lists")
        return data
    if args.decoder_preset:
        t = args.mel_len
        # Kokoro-82M: hidden_dim 512 for asr; f0/n are [B, T]; style [B, 128]
        return [[1, 512, t], [1, t], [1, t], [1, 128]]
    if args.seq_len is not None:
        l_ = args.seq_len
        return [[1, l_], [1, 256], [1]]
    return None


def _hint_dynamic_onnx() -> None:
    print(
        "\nKokoro ONNX uses dynamic input lengths. RKNN needs concrete shapes or "
        "dynamic_input support. Examples:\n"
        "  --seq-len 256\n"
        "    → input_size_list [[1,256],[1,256],[1]] for kokoro.onnx / kokoro_encoder.onnx\n"
        "  --decoder-preset [--mel-len 200]\n"
        "    → four inputs for kokoro_decoder.onnx\n"
        "  --input-size-list '[[1,128],[1,256],[1]]'\n"
        "    → explicit JSON (match your ONNX inputs in order)\n"
        "  --dynamic-input\n"
        "    → enable dynamic inputs in rknn.config if your toolkit supports it\n",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Kokoro ONNX to .rknn (RKNN-Toolkit2)",
    )
    parser.add_argument(
        "--onnx",
        "-i",
        type=str,
        required=True,
        help="Path to kokoro.onnx (or encoder/decoder split ONNX)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        required=True,
        help="Output path, e.g. kokoro.rknn",
    )
    parser.add_argument(
        "--platform",
        "-p",
        type=str,
        required=True,
        help="target_platform for rknn.config(), e.g. rk3588, rk3566, rv1106",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization in rknn.build()",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Calibration dataset path (required when --quantize for most toolkits)",
    )
    parser.add_argument(
        "--input-size-list",
        type=str,
        default=None,
        help=(
            "Explicit input shapes as JSON, e.g. '[[1,256],[1,256],[1]]' for "
            "[input_ids, style, speed]. Overrides --seq-len / --decoder-preset."
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        metavar="L",
        help="Fixed sequence length for 3-input Kokoro ONNX (builds [[1,L],[1,256],[1]]).",
    )
    parser.add_argument(
        "--decoder-preset",
        action="store_true",
        help="Use fixed shapes for 4-input kokoro_decoder.onnx (see --mel-len).",
    )
    parser.add_argument(
        "--mel-len",
        type=int,
        default=200,
        metavar="T",
        help="Time/mel length for --decoder-preset (default 200).",
    )
    parser.add_argument(
        "--dynamic-input",
        action="store_true",
        help="Set dynamic_input / dyanmic_input in rknn.config when supported.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.decoder_preset and args.seq_len is not None:
        print("Use only one of --decoder-preset or --seq-len.", file=sys.stderr)
        return 1

    try:
        input_size_list = _resolve_input_size_list(args)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON for --input-size-list: {e}", file=sys.stderr)
        return 1

    try:
        from rknn.api import RKNN
    except ImportError:
        print(
            "RKNN-Toolkit2 is not installed. Install the wheel from Rockchip:\n"
            "  https://github.com/airockchip/rknn-toolkit2\n"
            "Then ensure `from rknn.api import RKNN` works in this environment.",
            file=sys.stderr,
        )
        return 1

    rknn = RKNN(verbose=args.verbose)
    _apply_rknn_config(rknn, args.platform, dynamic_input=args.dynamic_input)

    load_kwargs: dict = {"model": args.onnx}
    if input_size_list is not None:
        load_kwargs["input_size_list"] = input_size_list

    try:
        ret = rknn.load_onnx(**load_kwargs)
    except ValueError as e:
        err = str(e).lower()
        if "input_ids_len" in str(e) or "input shape" in err or "not support" in err:
            _hint_dynamic_onnx()
        print(f"load_onnx failed: {e}", file=sys.stderr)
        rknn.release()
        return 1

    if ret != 0:
        print(f"load_onnx failed, return code {ret}", file=sys.stderr)
        rknn.release()
        return 1

    if args.quantize and not args.dataset:
        print(
            "Warning: --quantize without --dataset may fail; "
            "provide a calibration dataset path.",
            file=sys.stderr,
        )

    ret = rknn.build(
        do_quantization=args.quantize,
        dataset=args.dataset or None,
    )
    if ret != 0:
        print(f"build failed, return code {ret}", file=sys.stderr)
        rknn.release()
        return 1

    ret = rknn.export_rknn(args.out)
    if ret != 0:
        print(f"export_rknn failed, return code {ret}", file=sys.stderr)
        rknn.release()
        return 1

    rknn.release()
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
