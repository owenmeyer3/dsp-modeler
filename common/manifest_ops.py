import json
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

e, n =get_existing_params()

#print(n)
print_table(n)
