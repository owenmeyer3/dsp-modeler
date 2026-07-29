"""
manifest_transform.py

Rewrites the dry_file/wet_file paths in a manifest.jsonl by replacing an
origin prefix with a destination prefix. Works for any combination of
local paths and s3:// URIs -- it's just a string prefix swap, so which
side is "local" and which is "s3" doesn't matter.

The manifest.jsonl itself (--input / --output) can also independently be
local or s3:// -- e.g. read a manifest from S3 and write the rewritten
copy back to local disk, or vice versa.

Usage:
    python manifest_transform.py \
        --input /Users/owenmeyer/dsp-modeler/data/outputs/manifest.jsonl \
        --output s3://omm-test-bucket/dsp-modeler/data/outputs/s3_manifest.jsonl \
        --origin-prefix /Users/owenmeyer/dsp-modeler/black-box/ \
        --dest-prefix s3://omm-test-bucket/dsp-modeler/
"""
import argparse
import json

import boto3


def _parse_s3_uri(uri):
    bucket, key = uri[len('s3://'):].split('/', 1)
    return bucket, key


def read_manifest_lines(path):
    if path.startswith('s3://'):
        bucket, key = _parse_s3_uri(path)
        body = boto3.client('s3').get_object(Bucket=bucket, Key=key)['Body'].read()
        return body.decode().splitlines()
    with open(path, 'r') as f:
        return f.readlines()


def write_manifest_lines(path, lines):
    content = '\n'.join(lines) + '\n'
    if path.startswith('s3://'):
        bucket, key = _parse_s3_uri(path)
        boto3.client('s3').put_object(Bucket=bucket, Key=key, Body=content.encode())
    else:
        with open(path, 'w') as f:
            f.write(content)


def transform_path(path, origin_prefix, dest_prefix):
    if path.startswith(origin_prefix):
        return dest_prefix + path[len(origin_prefix):]
    return path


def transform_manifest(input_path, output_path, origin_prefix, dest_prefix):
    out_lines = []
    for line in read_manifest_lines(input_path):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        record['dry_file'] = transform_path(record['dry_file'], origin_prefix, dest_prefix)
        record['wet_file'] = transform_path(record['wet_file'], origin_prefix, dest_prefix)
        out_lines.append(json.dumps(record))
    write_manifest_lines(output_path, out_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Path to the source manifest.jsonl (local path or s3:// URI)')
    parser.add_argument('--output', required=True, help='Path to write the transformed manifest.jsonl (local path or s3:// URI)')
    parser.add_argument('--origin-prefix', required=True, help='Prefix to replace (local path or s3:// URI)')
    parser.add_argument('--dest-prefix', required=True, help='Replacement prefix (local path or s3:// URI)')
    args = parser.parse_args()

    transform_manifest(args.input, args.output, args.origin_prefix, args.dest_prefix)
    print(f"Wrote {args.output}")


def repath_manifest(
        manifest_origin,#='/Users/owenmeyer/dsp-modeler/data/outputs/manifest.jsonl',
        manifest_destination,#='/Users/owenmeyer/dsp-modeler/data/outputs/s3_manifest.jsonl',
        origin_prefix,#='/Users/owenmeyer/dsp-modeler/black-box/',
        destination_prefix#='s3://omm-test-bucket/dsp-modeler/'
):
    out_lines = []
    for line in read_manifest_lines(manifest_origin):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        record['dry_file'] = destination_prefix + record['dry_file'][len(origin_prefix):] if record['dry_file'].startswith(origin_prefix) else record['dry_file']
        record['wet_file'] = destination_prefix + record['wet_file'][len(origin_prefix):] if record['wet_file'].startswith(origin_prefix) else record['wet_file']
        out_lines.append(json.dumps(record))
    write_manifest_lines(manifest_destination, out_lines)
    print(f"Wrote {manifest_destination}")