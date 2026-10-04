import numpy as np
import json, boto3


s3 = boto3.client("s3")

def parse_s3_path(s3_path: str):
    if not s3_path.startswith("s3://"):
        raise ValueError(f"--s3-path must start with s3://, got: {s3_path}")
    bucket, _, key = s3_path[len("s3://"):].partition("/")
    if not bucket or not key:
        raise ValueError(f"Could not parse bucket/key from: {s3_path}")
    return bucket, key

def export(npz_path: str, s3_path: str):
    data = np.load(npz_path)
    coeffs = data["coeffs"].tolist()  # 11 floats

    bucket, key = parse_s3_path(s3_path)

    

    # Save metadata
    payload = json.dumps({"coeffs": coeffs}).encode("utf-8")
    bucket, key = parse_s3_path(f"{s3_path}/model.json")
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")

    print(f"Uploaded gain model coeffs ({len(coeffs)} values) to s3://{bucket}/{key}")


if __name__ == "__main__":
    export(
        "/home/ubuntu/dsp-modeler/black_box/model/models/gain_model/2026-09-12_14-40/gain_model.npz",
        "s3://omm-test-bucket/neiro/projects/nrat/models/gain/2026-09-12_14-40",
    )

    export(
        "/home/ubuntu/dsp-modeler/black_box/model/models/noise_model/2026-09-12_14-29/noise_model.npz",
        "s3://omm-test-bucket/neiro/projects/nrat/models/noise/2026-09-12_14-29",
    )