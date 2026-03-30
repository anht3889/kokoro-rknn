# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "kokoro==0.8.4",
#     "onnx==1.17.0",
#     "onnxruntime==1.20.1",
#     "onnxscript>=0.6.0",
#     "sounddevice==0.5.1",
# ]
#
# ///

"""
From https://github.com/hexgrad/kokoro/blob/3f9dd88d6f739b98a86aea608e238621f5b40add/examples/export.py

mkdir checkpoints
wget https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json -O checkpoints/config.json
wget https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth -O checkpoints/kokoro-v1_1-zh.pth
uv run examples/export.py
uv run examples/export.py --config_file checkpoints/config.json --checkpoint_path checkpoints/kokoro-v1_1-zh.pth

Default ONNX export uses ``torch.onnx.export(..., dynamo=True)`` (needs onnxscript) because
current Transformers + PL-BERT breaks the legacy traced exporter. Use ``--legacy-onnx-trace``
only with older transformers (e.g. 4.4x).
"""

import argparse
import inspect
import os

import onnx
import onnxruntime as ort
import sounddevice as sd
import torch
from kokoro import KModel, KPipeline
from kokoro.model import KModelForONNX


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


def _torch_onnx_export(*, legacy_trace: bool, **kwargs):
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
            "torch.onnx.export(dynamo=True) failed. Recent Transformers + PL-BERT do not work "
            "with the legacy traced ONNX exporter. Install onnxscript (pip install onnxscript). "
            "To force tracing anyway, use --legacy-onnx-trace with an older transformers (e.g. 4.4x)."
        ) from e


def export_onnx(model, output, *, legacy_trace: bool):
    onnx_file = output + "/" + "kokoro.onnx"

    input_ids = torch.randint(1, 100, (48,)).numpy()
    input_ids = torch.LongTensor([[0, *input_ids, 0]])
    style = torch.randn(1, 256)
    speed = torch.randint(1, 10, (1,)).int()

    _torch_onnx_export(
        legacy_trace=legacy_trace,
        model=model,
        args=(input_ids, style, speed),
        f=onnx_file,
        export_params=True,
        verbose=True,
        input_names=["input_ids", "style", "speed"],
        output_names=["waveform", "duration"],
        opset_version=17,
        dynamic_axes={
            "input_ids": {1: "input_ids_len"},
            "waveform": {0: "num_samples"},
        },
        do_constant_folding=True,
    )

    print("export kokoro.onnx ok!")

    onnx_model = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_model)
    print("onnx check ok!")


def load_input_ids(pipeline, text):
    if pipeline.lang_code in "ab":
        _, tokens = pipeline.g2p(text)
        for gs, ps, tks in pipeline.en_tokenize(tokens):
            if not ps:
                continue
    else:
        ps, _ = pipeline.g2p(text)

    if len(ps) > 510:
        ps = ps[:510]

    input_ids = list(
        filter(lambda i: i is not None, map(lambda p: pipeline.model.vocab.get(p), ps))
    )
    print(f"text: {text} -> phonemes: {ps} -> input_ids: {input_ids}")
    input_ids = torch.LongTensor([[0, *input_ids, 0]]).to(pipeline.model.device)
    return ps, input_ids


def load_voice(pipeline, voice, phonemes):
    pack = pipeline.load_voice(voice).to("cpu")
    return pack[len(phonemes) - 1]


def load_sample(model):
    pipeline = KPipeline(lang_code="a", model=model.kmodel, device="cpu")
    text = """
    In today's fast-paced tech world, building software applications has never been easier — thanks to AI-powered coding assistants.'
    """
    text = """
    The sky above the port was the color of television, tuned to a dead channel.
    """
    voice = "checkpoints/voices/af_heart.pt"

    pipeline = KPipeline(lang_code="z", model=model.kmodel, device="cpu")
    text = """
    2月15日晚，猫眼专业版数据显示，截至发稿，《哪吒之魔童闹海》（或称《哪吒2》）今日票房已达7.8亿元，累计票房（含预售）超过114亿元。
    """
    voice = "checkpoints/voices/zf_xiaoxiao.pt"

    phonemes, input_ids = load_input_ids(pipeline, text)
    style = load_voice(pipeline, voice, phonemes)
    speed = torch.IntTensor([1])

    return input_ids, style, speed


def inference_onnx(model, output):
    onnx_file = output + "/" + "kokoro.onnx"
    session = ort.InferenceSession(onnx_file)

    input_ids, style, speed = load_sample(model)

    outputs = session.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "style": style.numpy(),
            "speed": speed.numpy(),
        },
    )

    output = torch.from_numpy(outputs[0])
    print(f"output: {output.shape}")
    print(output)

    audio = output.numpy()
    sd.play(audio, 24000)
    sd.wait()


def check_model(model):
    input_ids, style, speed = load_sample(model)
    output, duration = model(input_ids, style, speed)

    print(f"output: {output.shape}")
    print(f"duration: {duration.shape}")
    print(output)

    audio = output.numpy()
    sd.play(audio, 24000)
    sd.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Export kokoro Model to ONNX", add_help=True)
    parser.add_argument(
        "--inference", "-t", help="test kokoro.onnx model", action="store_true"
    )
    parser.add_argument("--check", "-m", help="check kokoro model", action="store_true")
    parser.add_argument(
        "--config_file",
        "-c",
        type=str,
        default="checkpoints/config.json",
        help="path to config file",
    )
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        default="checkpoints/kokoro-v1_0.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="onnx", help="output directory"
    )
    parser.add_argument(
        "--legacy-onnx-trace",
        action="store_true",
        help="Use legacy torch.jit-traced ONNX export (fails on current Transformers + PL-BERT).",
    )

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    output_dir = args.output_dir

    # make dir
    os.makedirs(output_dir, exist_ok=True)

    kmodel = KModel(config=config_file, model=checkpoint_path, disable_complex=True)
    if args.legacy_onnx_trace:
        _force_plbert_eager_attention_for_onnx(kmodel)
    model = KModelForONNX(kmodel).eval()

    if args.inference:
        inference_onnx(model, output_dir)
    elif args.check:
        check_model(model)
    else:
        export_onnx(model, output_dir, legacy_trace=args.legacy_onnx_trace)
