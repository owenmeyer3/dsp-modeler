import json, boto3
# {
#     "id": "95acc994-d2f5-40a8-8ea1-fe873eccf4a5",
#     "dry_file": "/Users/owenmeyer/dsp-modeler/black-box/data/input/input.wav",
#     "wet_file": "/Users/owenmeyer/dsp-modeler/black-box/data/outputs/95acc994-d2f5-40a8-8ea1-fe873eccf4a5.wav",
#     "sample_rate": 96000,
#     "duration_sec": 244.0,
#     "peak_dbfs": -4.63,
#     "clipped_samples": 0,
#     "estimated_latency_samples": 5394,
#     "captured_at": "2026-07-27T17:42:24",
#     "params": {
#         "d": 4.0,
#         "f": 4.0,
#         "v": 7.0
#     }
# }

def flatten_record(record):
    flat = {k: v for k, v in record.items() if k != "params"}
    for k, v in record.get("params", {}).items():
        flat[f"params_{k}"] = v
    return flat

def get_jsonl_records(flatten=False):
    with open('/Users/owenmeyer/dsp-modeler/black-box/data/outputs/manifest.jsonl', 'r') as f:
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rows.append(flatten_record(record) if flatten else record)
        return rows

def print_table(rows, columns=None):
    if not rows:
        print("No rows to display.")
        return
    if not columns:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    widths = {
        col: max(len(col), max(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-+-".join("-" * widths[col] for col in columns))
    for row in rows:
        line = " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        print(line)

def get_records_by_params(params):
    records = get_jsonl_records()
    found_records = []
    for r in records:
        if r['params'] == params:
            found_records.append(r)
    assert not len(found_records) == 0, "No records found"
    assert not len(found_records) > 1, f"multiple records found records found:{[r['id'] for r in found_records]}"
    return found_records[0]

def get_all_dict_params():
    a=[]
    for d in [1.0,2.0,3.0,4.0,5.0,6.0,7.0]:
        r1 = {'d':d}
        for f in [1.0,2.0,3.0,4.0,5.0,6.0,7.0]:
            r2 = r1|{'f':f}
            for v in [1.0,2.0,3.0,4.0,5.0,6.0,7.0]:
                r3 = r2|{'v':v}
                a.append(r3)
    return a

def repath_manifest(
        manifest_origin,#='/Users/owenmeyer/dsp-modeler/data/outputs/manifest.jsonl',
        manifest_destination,#='/Users/owenmeyer/dsp-modeler/data/outputs/s3_manifest.jsonl',
        origin_prefix,#='/Users/owenmeyer/dsp-modeler/black-box/',
        destination_prefix#='s3://omm-test-bucket/dsp-modeler/'
):
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


def get_existing_params():
    dps = get_all_dict_params()
    rps=[]
    nps=[]
    for dp in dps:
        try:
            d=get_records_by_params(dp)
            rps.append(dp)
        except:
            nps.append(dp)
    return [rps, nps]

#print_table(get_jsonl_records(), ['wet_file', 'params_d', 'params_f', 'params_v'])

#print(get_records_by_params({'d': 1.0, 'f': 7.0, 'v': 7.0}))

#print(get_all_dict_params())

# e, n =get_existing_params()

# #print(n)
# print_table(n)


repath_manifest(
        manifest_origin='/Users/owenmeyer/dsp-modeler/data/outputs/manifest.jsonl',
        manifest_destination='/Users/owenmeyer/dsp-modeler/data/outputs/manifest-2.jsonl',
        origin_prefix='/Users/owenmeyer/dsp-modeler/black-box/data/',
        destination_prefix='/Users/owenmeyer/dsp-modeler/data/'
)