from common.delay_ops import measure_delay
from common.cfg import get_config
from common.utils import load_wav
import json

if __name__ == '__main__':

    wet_dir = '/home/ubuntu/dsp-modeler/data/outputs'
    dry_file = '/home/ubuntu/dsp-modeler/data/input/input.wav'
    chunk_seconds=0.1
    train_manifest='/home/ubuntu/dsp-modeler/black_box/data/train/manifest copy.jsonl'

    config = get_config()

    dry_full, sr = load_wav(dry_file)

    # preprocess
    n_trim = int(config['SILENT_LEADIN_SECONDS'] * sr)
    dry_trim = dry_full[n_trim:]
    chunk_len = int(chunk_seconds * sr)

    wet_records = []
    with open(train_manifest, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            wet_records.append(record)

    for wr in wet_records: 
        wet_path = wet_dir + '/' + wr['id'] + '.wav'
        wet_full, w_sr = load_wav(wet_path)
        assert w_sr == sr, f"{wet_path} has a different sample rate than {dry_file}"
        wet_trim = wet_full[n_trim:]

        delay_samples, sr = measure_delay(
            wet_trim,
            dry_trim,
            sr,
            n_onsets=20,
            search_seconds=1.0,
            cluster_window_seconds=0.02,
            verbose=False,
        )