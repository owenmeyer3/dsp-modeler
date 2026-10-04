#!/usr/bin/env python3
"""
Export a trained ConditionedLSTM's state_dict (.pt) into the JSON format
RTNeural's torch_helpers loader expects, and upload it to S3.

RTNeural's C++ loader (RTNeural::torch_helpers::loadLSTM / loadDense) reads
PyTorch's *native* state_dict keys directly -- e.g. "lstm.weight_ih_l0",
"dense.weight" -- so no reformatting, transposing, or renaming is needed.
This mirrors RTNeural's own verified example (python/lstm_torch.py) and its
CI test (tests/functional/torch_lstm_test.cpp), which checks the C++ output
matches Python to within 1e-6.

Usage:
    python export_rtneural_model.py \
        --model-path /home/ubuntu/dsp-modeler/black_box/model/models/transform_model/2026-09-13_13-07/model_best.pt \
        --s3-path s3://your-bucket/models/rat_lstm_2026-09-13.json \
        --input-size 4 --hidden-size 40 --num-layers 2
"""
import io
import json
import re

import boto3
import torch

s3 = boto3.client("s3")

class TorchStateDictEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super().default(obj)


def parse_s3_path(s3_path: str):
    if not s3_path.startswith("s3://"):
        raise ValueError(f"--s3-path must start with s3://, got: {s3_path}")
    bucket, _, key = s3_path[len("s3://"):].partition("/")
    if not bucket or not key:
        raise ValueError(f"Could not parse bucket/key from: {s3_path}")
    return bucket, key


def describe_architecture(state_dict: dict):
    """Infer (input_size, hidden_size, num_layers) from tensor shapes, purely
    as a sanity check printed to stdout -- the actual C++ ModelT template
    args must be set to match these when you write the loading code."""
    layer_ids = sorted({
        int(m.group(1))
        for k in state_dict
        if (m := re.match(r"lstm\.weight_ih_l(\d+)$", k))
    })
    num_layers = len(layer_ids)
    hidden_size = state_dict["lstm.weight_hh_l0"].shape[1]
    input_size = state_dict["lstm.weight_ih_l0"].shape[1]
    output_size = state_dict["dense.weight"].shape[0]
    return input_size, hidden_size, num_layers, output_size


def export(
        model_path:str, # Local path to a .pt state_dict (e.g. model_best.pt)
        s3_path:str # s3://bucket/key.json destination

):
    state_dict = torch.load(model_path, map_location="cpu")

    # Get metadata
    input_size, hidden_size, num_layers, output_size = describe_architecture(state_dict)
    print(f"Detected architecture: input_size={input_size}, hidden_size={hidden_size}, "
          f"num_layers={num_layers}, output_size={output_size}")
    print("Make sure your C++ RTNeural::ModelT<...> template args match these exactly.")

    # Save metadata
    metadata={"input_size":input_size, "hidden_size":hidden_size, "num_layers":num_layers, "output_size":output_size}
    payload = json.dumps(metadata).encode("utf-8")
    bucket, key = parse_s3_path(f"{s3_path}/metadata.json")
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")

    # Save model data
    buffer = io.StringIO()
    json.dump(state_dict, buffer, cls=TorchStateDictEncoder)
    payload = buffer.getvalue().encode("utf-8")
    bucket, key = parse_s3_path(f"{s3_path}/model.json")
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")

    print(f"Uploaded {len(payload) / 1024:.1f} KB to s3://{bucket}/{key}")


if __name__ == "__main__":
    export(
        "/home/ubuntu/dsp-modeler/black_box/model/models/transform_model/2026-09-14_01-11/model_best.pt",
        "s3://omm-test-bucket/neiro/projects/nrat/models/2026-09-14_01-11"
    )
