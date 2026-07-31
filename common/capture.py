import datetime, json, os, sys, uuid
import numpy as np
import sounddevice as sd
import soundfile as sf
from common.alignment import estimate_shift
from common.cfg import get_config
from common.utils import load_wav, write_wav

config = get_config()

def validate_device_channel(device_name, channel, input=True):
    device_dicts = sd.query_devices()
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
    device_dict = None
    for d in device_dicts:
        if d['name'] == device_name:
            device_dict = d
            break
    assert device_dict, f'No device found: {device_name}'
    if input:
        assert channel <= device_dict['max_input_channels']
    else:
        assert channel <= device_dict['max_output_channels']

def capture_output(
    dry_data,
    sample_rate,
    output_device,
    input_device,
    output_channel,
    input_channel,
    gain=1.0,
    tail_seconds=0.0
):
    tail_samples = int(tail_seconds * sample_rate)
    play_buf = np.zeros((len(dry_data) + tail_samples, 1), dtype=np.float64)
    play_buf[: len(dry_data), 0] = dry_data * gain

    sd.default.samplerate = sample_rate
    sd.default.channels = (1, 1)
    sd.default.device = (input_device, output_device)

    recorded = sd.playrec(
        play_buf,
        samplerate=sample_rate,
        input_mapping=[input_channel],
        output_mapping=[output_channel],
        blocking=True,
    )
    sd.wait()
    wet_data = recorded[:, 0]
    return wet_data


def save_capture(
    dry_file,
    dry_data,
    wet_data,
    sample_rate,
    out_dir,
    params={'d': 4.0, 'f': 4.0, 'v': 6.0}
):
    id = str(uuid.uuid4())
    wet_file = f"{out_dir}/{id}.wav"
    manifest_path = f"{out_dir}/manifest.jsonl"

    print(f'saving wet: {wet_file}')
    write_wav(wet_file, wet_data, sample_rate)
    print('saved wet')

    peak = float(np.max(np.abs(wet_data)))
    clipped = int(np.sum(np.abs(wet_data) >= 0.999))

    print('estimating latency')
    n_trim = int(config['SILENT_LEADIN_SECONDS'] * sample_rate) # keep silent area out of first note detection for latency finding, in case some early artifact triggers first-note recognition
    latency = estimate_shift(dry_data[n_trim:], wet_data[n_trim:], sample_rate)
    print(f'estimated latency: {latency} samples ({latency / sample_rate * 1000:.3f} ms)')

    manifest_entry = {
        "id":id,
        "dry_file": dry_file,
        "wet_file": wet_file,
        "sample_rate": sample_rate,
        "duration_sec": round(len(wet_data) / sample_rate, 3),
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
        "clipped_samples": clipped,
        "estimated_latency_samples": latency,
        "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "params":params
    }
    print(f'saving manifest: {manifest_path}')
    with open(manifest_path, "a") as f:
        f.write(json.dumps(manifest_entry) + "\n")
    print('saved manifest')

    if clipped > 0:
        print(f"  WARNING: {clipped} clipped samples in recording — check pedal/interface gain staging.")

def capture_wet(
    input_device,
    input_channel,
    output_device,
    output_channel,
    params,
    dry_file,
    out_dir
):
    print('validating')
    validate_device_channel(input_device, input_channel, input=True)
    validate_device_channel(output_device, output_channel, input=False)

    print('load dry')
    dry_data, sample_rate = load_wav(dry_file)
    print(f'found file {dry_file} with SR={sample_rate}')

    print(f'capturing {params}')
    wet_data = capture_output(
        dry_data,
        sample_rate,
        output_device,
        input_device,
        output_channel,
        input_channel
    )

    print('saving')
    save_capture(
        dry_file,
        dry_data,
        wet_data,
        sample_rate,
        out_dir,
        params=params
    )

if __name__ == "__main__":

    capture_wet(
        input_device='UMC204HD 192k',
        input_channel=1,
        output_device='UMC204HD 192k',
        output_channel = 1,
        params={'d': 7.0, 'f':  7.0, 'v': 7.0},
        dry_file='/home/ubuntu/dsp-modeler/black-box/data/input/input.wav',
        out_dir='/home/ubuntudsp-modeler/black-box/data/fake'
    )