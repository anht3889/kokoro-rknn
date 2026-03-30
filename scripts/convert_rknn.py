# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
#
# Install RKNN-Toolkit2 separately (see Rockchip docs). Wheels are often
# Linux x86_64 only: https://github.com/airockchip/rknn-toolkit2
#
# Example:
#   uv run scripts/convert_rknn.py --onnx onnx/kokoro.onnx --out rknn/kokoro.rknn --platform rk3588

"""
Convert Kokoro ONNX (from scripts/export.py) to RKNN for Rockchip NPUs.

Notes
-----
- RKNN-Toolkit2 is not declared as a normal project dependency; install the
  wheel matching your OS/Python from Rockchip's release packages.
- Dynamic shapes: many RKNN builds require fixed ``input_size_list``. If
  conversion fails, re-export ONNX with fixed sequence length or pass
  ``--input-size-list`` (see ``rknn.load_onnx`` docs for your toolkit version).
- For TTS, INT8 quantization needs a calibration dataset in the format your
  toolkit expects (often a text file of numpy paths per input).
"""

from __future__ import annotations

import argparse
import sys


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
            "Optional explicit input shapes as JSON list of lists, e.g. "
            '\'[[1,128],[1,256],[1]]\' for [input_ids, style, speed]. '
            "Omit to let the toolkit infer from ONNX (may fail for dynamic axes)."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

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

    input_size_list = None
    if args.input_size_list:
        import json

        input_size_list = json.loads(args.input_size_list)
        if not isinstance(input_size_list, list):
            print("--input-size-list must be a JSON array", file=sys.stderr)
            return 1

    rknn = RKNN(verbose=args.verbose)

    # Non-vision models: only target_platform is typically required.
    rknn.config(target_platform=args.platform)

    load_kwargs: dict = {"model": args.onnx}
    if input_size_list is not None:
        load_kwargs["input_size_list"] = input_size_list

    ret = rknn.load_onnx(**load_kwargs)
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
