# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "onnx>=1.14.0",
# ]
# ///
#
# Install RKNN-Toolkit2 separately (see Rockchip docs). Wheels are often
# Linux x86_64 only: https://github.com/airockchip/rknn-toolkit2
#
# Kokoro ONNX uses dynamic axes (e.g. input_ids length). This script can pick
# fixed shapes automatically if `onnx` is installed, or you pass flags:
#
#   python scripts/convert_rknn.py -i onnx/kokoro.onnx -o rknn/kokoro.rknn -p rk3588 --seq-len 256
#   python scripts/convert_rknn.py -i onnx_split/kokoro_decoder.onnx -o out.rknn -p rk3588 --decoder-preset

"""
Convert Kokoro ONNX (from scripts/export.py / export_split.py) to RKNN.

Notes
-----
- RKNN-Toolkit2 is not a normal PyPI dependency; install the wheel from Rockchip.
- Many ``load_onnx`` builds need both ``input_size_list`` **and** input names
  (``inputs=[...]``) when fixing dynamic axes.
- If ``onnx`` is installed, symbolic dimensions are detected and defaults are
  applied (3 inputs → ``--seq-len 256``, 4 inputs → ``--decoder-preset``).
- INT8 quantization needs a calibration dataset path for ``rknn.build``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from typing import Any, Optional

KOKORO_3_INPUTS = ["input_ids", "style", "speed"]
KOKORO_4_INPUTS = ["asr", "f0", "n", "style"]


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


def _onnx_input_meta(onnx_path: str) -> tuple[bool, list[str]]:
    """Return (has_symbolic_dim, input_names_in_order)."""
    import onnx

    model = onnx.load(onnx_path)
    names: list[str] = []
    symbolic = False
    for inp in model.graph.input:
        names.append(inp.name)
        shape = inp.type.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param:
                symbolic = True
                break
    return symbolic, names


def _apply_auto_shapes(args: argparse.Namespace) -> None:
    """If user gave no shape flags, default from ONNX (needs ``onnx`` package)."""
    if (
        args.input_size_list
        or args.seq_len is not None
        or args.decoder_preset
        or args.dynamic_input
    ):
        return
    try:
        import onnx  # noqa: F401
    except ImportError:
        return
    symbolic, names = _onnx_input_meta(args.onnx)
    if not symbolic:
        return
    n = len(names)
    if n == 3:
        print(
            "Note: ONNX has dynamic axes; using fixed shapes with --seq-len 256 "
            "(override with e.g. --seq-len 128).",
            file=sys.stderr,
        )
        args.seq_len = 256
    elif n == 4:
        print(
            "Note: ONNX has dynamic axes; using --decoder-preset --mel-len 200 "
            "(override with --mel-len).",
            file=sys.stderr,
        )
        args.decoder_preset = True


def _resolve_input_size_list(args: argparse.Namespace) -> Optional[list[Any]]:
    if args.input_size_list:
        data = json.loads(args.input_size_list)
        if not isinstance(data, list):
            raise SystemExit("--input-size-list must be a JSON array of shape lists")
        return data
    if args.decoder_preset:
        t = args.mel_len
        return [[1, 512, t], [1, t], [1, t], [1, 128]]
    if args.seq_len is not None:
        l_ = args.seq_len
        return [[1, l_], [1, 256], [1]]
    return None


def _resolve_load_onnx_inputs(args: argparse.Namespace, onnx_path: str) -> Optional[list[str]]:
    if args.decoder_preset:
        return list(KOKORO_4_INPUTS)
    if args.seq_len is not None:
        return list(KOKORO_3_INPUTS)
    if args.input_size_list:
        try:
            _, names = _onnx_input_meta(onnx_path)
            return names
        except Exception:
            return None
    return None


def _hint_dynamic_onnx() -> None:
    print(
        "\nKokoro ONNX uses dynamic input lengths. RKNN needs concrete shapes or "
        "dynamic_input in config. Examples:\n"
        "  --seq-len 256\n"
        "    → shapes [[1,256],[1,256],[1]] + inputs for kokoro.onnx / kokoro_encoder.onnx\n"
        "  --decoder-preset [--mel-len 200]\n"
        "    → four inputs for kokoro_decoder.onnx\n"
        "  --input-size-list '[[1,128],[1,256],[1]]'\n"
        "    → explicit JSON (names taken from the ONNX file if possible)\n"
        "  --dynamic-input\n"
        "    → rknn.config(dyanmic_input=True) when supported\n"
        "  pip install onnx\n"
        "    → enables automatic --seq-len 256 / decoder-preset for symbolic models\n",
        file=sys.stderr,
    )


def _call_load_onnx(rknn, **kwargs: Any) -> Any:
    """Only pass kwargs supported by this toolkit's ``load_onnx``."""
    sig = inspect.signature(rknn.load_onnx)
    allowed = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return rknn.load_onnx(**filtered)


def _require_shapes_or_dynamic(args: argparse.Namespace) -> Optional[int]:
    """
    If ONNX is symbolic and user still has no fix, return exit code 1.
    Returns None if OK to proceed.
    """
    if args.dynamic_input:
        return None
    input_sizes = _resolve_input_size_list(args)
    if input_sizes is not None:
        return None
    try:
        symbolic, _ = _onnx_input_meta(args.onnx)
    except ImportError:
        print(
            "Install `onnx` (pip install onnx) so this script can detect dynamic "
            "shapes and set defaults, or pass e.g. --seq-len 256 explicitly.",
            file=sys.stderr,
        )
        _hint_dynamic_onnx()
        return 1
    except Exception as e:
        print(f"Could not read ONNX inputs: {e}", file=sys.stderr)
        _hint_dynamic_onnx()
        return 1
    if symbolic:
        print(
            "This ONNX uses symbolic input shapes. Pass --seq-len, --decoder-preset, "
            "--input-size-list, or --dynamic-input.",
            file=sys.stderr,
        )
        _hint_dynamic_onnx()
        return 1
    return None


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
            "Explicit input shapes as JSON, e.g. '[[1,256],[1,256],[1]]'. "
            "Overrides --seq-len / --decoder-preset."
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        metavar="L",
        help="Fixed sequence length for 3-input Kokoro ONNX (implies named inputs).",
    )
    parser.add_argument(
        "--decoder-preset",
        action="store_true",
        help="Fixed shapes for 4-input kokoro_decoder.onnx (see --mel-len).",
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

    _apply_auto_shapes(args)

    try:
        input_size_list = _resolve_input_size_list(args)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON for --input-size-list: {e}", file=sys.stderr)
        return 1

    early = _require_shapes_or_dynamic(args)
    if early is not None:
        return early

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

    load_kwargs: dict[str, Any] = {"model": args.onnx}
    if input_size_list is not None:
        load_kwargs["input_size_list"] = input_size_list
    input_names = _resolve_load_onnx_inputs(args, args.onnx)
    if input_names is not None:
        load_kwargs["inputs"] = input_names

    try:
        ret = _call_load_onnx(rknn, **load_kwargs)
    except TypeError:
        load_kwargs.pop("inputs", None)
        try:
            ret = _call_load_onnx(rknn, **load_kwargs)
        except ValueError as e2:
            err = str(e2).lower()
            if "input_ids_len" in str(e2) or "input shape" in err or "not support" in err:
                _hint_dynamic_onnx()
            print(f"load_onnx failed: {e2}", file=sys.stderr)
            rknn.release()
            return 1
        except Exception as e2:
            print(f"load_onnx failed: {e2}", file=sys.stderr)
            rknn.release()
            return 1
    except ValueError as e:
        err = str(e).lower()
        if "input_ids_len" in str(e) or "input shape" in err or "not support" in err:
            _hint_dynamic_onnx()
        print(f"load_onnx failed: {e}", file=sys.stderr)
        rknn.release()
        return 1
    except Exception as e:
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
