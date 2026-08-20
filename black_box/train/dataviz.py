from data_objects import Track, DataSet
from model_objects import GainModel, ConditionedLSTM
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from common.utils import load_wav
from common.delay_ops import measure_delay, apply_shift

def fit(train_dataset):
    ds, fs, vs, gs = [], [], [], []
    for track in train_dataset:
        norm_params = track[0][0].normalize_params(param_configs)
        d, f, v = norm_params['d'], norm_params['f'], norm_params['v']
        ds.append(d)
        fs.append(f)
        vs.append(v)
        gs.append(track.compute_wet_gain())

    ds, fs, vs, gs = np.array(ds), np.array(fs), np.array(vs), np.array(gs)
    X = np.column_stack([np.ones_like(ds), ds, fs, vs, ds*vs, ds*fs, fs*vs, ds*fs*vs, vs**2, ds**2, vs**3])
    y = np.log(gs)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs

def predict_global_gain(track, coeffs):
    norm_params = track[0][0].normalize_params(param_configs)
    print(norm_params)
    d, f, v = norm_params['d'], norm_params['f'], norm_params['v']
    x = np.array([1, d, f, v, d*v, d*f, f*v, d*f*v, v**2, d**2, v**3])
    return np.exp(x @ coeffs)

def validate(validation_dataset: DataSet, coeffs):
    for i, track in enumerate(validation_dataset):
        # print(f'Track {i}')
        predicted_gain = predict_global_gain(track, coeffs)
        actual_gain = track.compute_wet_gain()
        print(f'gain {predicted_gain} vs. {actual_gain}')

def cross_validate(full_dataset):
    tracks = full_dataset.tracks
    for i in range(len(full_dataset.tracks)):
       print(f'V Track {i}')
       train_tracks = [t for i,t in enumerate(tracks) if i!=3]
       validation_tracks = [tracks[i]]
       coeffs = fit(train_tracks)
       validate(validation_tracks, coeffs)

def print_track_section(waves:list[dict], start_s:int, end_s:int, zoom_start_s=0, zoom_len=0.001,title="Title", out_path=''):

    sr=waves[0]['track'].sample_rate    
    start_samples = int(start_s*sr)
    end_samples = int(end_s*sr)
    zoom_start_samples = int(zoom_start_s*sr)
    zoom_len_samples = int(zoom_len*sr)

    fig, axes = plt.subplots(len(waves) + 1, 1, figsize=(12, 2.2 * len(waves) + 3))


    t_full = np.arange(end_samples-start_samples) / sr + start_s
    t_zoom = np.arange(zoom_len_samples) / sr * 1000 + zoom_start_s*1000  # ms

    for ax, wave in zip(axes[:-1], waves):
        name = wave['name']
        chunk=None
        if wave['channel'] == 'dry': chunk = wave['track'].get_dry()[start_samples:end_samples]
        else: chunk = wave['track'].get_wet()[start_samples:end_samples]
        print(chunk.shape)
        ax.plot(t_full, chunk, linewidth=0.5)
        ax.plot(t_full, chunk, linewidth=0.5)
        ax.set_ylabel("Amplitude")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Time (s)")



    ax = axes[-1]
    for wave in waves:
        name = wave['name']
        chunk=None
        if wave['channel'] == 'dry': chunk = wave['track'].get_dry()[zoom_start_samples:zoom_start_samples+zoom_len_samples]
        elif wave['channel'] == 'wet': chunk = wave['track'].get_wet()[zoom_start_samples:zoom_start_samples+zoom_len_samples]
        else:
            track = wave['track']
            track.denoise_wet_data()
            chunk = wave['track'].get_wet()[zoom_start_samples:zoom_start_samples+zoom_len_samples]
        ax.plot(t_zoom, chunk, label=name, linewidth=1.0)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def print_wav_data_section(waves:list[dict], start_s:int, end_s:int, zoom_start_s=0, zoom_len=0.001,title="Title", out_path=''):

    dry = waves[0]['data']
    sr = waves[0]['sample_rate']
    wet = waves[1]['data']

    delay_samples, sr = measure_delay(wet, dry, sr, cluster_window_seconds=0.01, verbose=True)
    print(f"delay_samples {delay_samples}")
    dry_aligned, wet_aligned = apply_shift(dry, wet, delay_samples)

    waves = [
        {
            'name':'dry', 
            'data':dry_aligned, 
            "sample_rate":sr
        },
        {
            'name':'wet', 
            'data':wet_aligned, 
            "sample_rate":sr
        }
    ]

    sr=waves[0]["sample_rate"]   
    start_samples = int(start_s*sr)
    end_samples = int(end_s*sr)
    zoom_start_samples = int(zoom_start_s*sr)
    zoom_len_samples = int(zoom_len*sr)

    fig, axes = plt.subplots(len(waves) + 1, 1, figsize=(12, 2.2 * len(waves) + 3))


    t_full = np.arange(end_samples-start_samples) / sr + start_s
    t_zoom = np.arange(zoom_len_samples) / sr * 1000 + zoom_start_s*1000  # ms

    print(t_full)

    for ax, wave in zip(axes[:-1], waves):
        name = wave['name']
        chunk = wave['data'][start_samples:end_samples]
        ax.plot(t_full, chunk, linewidth=0.5)
        ax.plot(t_full, chunk, linewidth=0.5)
        ax.set_ylabel("Amplitude")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Time (s)")



    ax = axes[-1]
    for wave in waves:
        name = wave['name']
        chunk = wave['data'][zoom_start_samples:zoom_start_samples+zoom_len_samples]
        ax.plot(t_zoom, chunk, label=name, linewidth=1.0)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)





if __name__ == '__main__':
    param_names=["d", "f", "v"]
    param_configs={
        'd':{'min':1, 'max':7, 'dtype':torch.float32},
        'f':{'min':1, 'max':7, 'dtype':torch.float32},
        'v':{'min':1, 'max':7, 'dtype':torch.float32},
    }
    chunk_seconds=0.03
    denoise_wet=False
    silent_lead_in_seconds=8

    gain_model = GainModel(param_configs)
    gain_model.load('/home/ubuntu/dsp-modeler/black_box/model/models/gain_model/2026-08-18_20-17/gain_model.npz')

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = ConditionedLSTM(input_size=4, hidden_size=20).to(device)
    model.load_state_dict(torch.load('/home/ubuntu/dsp-modeler/black_box/model/models/transform_model/2026-08-20_15-01/model_best.pt', map_location=device))
    model.eval()

    example_dataset = DataSet(
        '/home/ubuntu/dsp-modeler/black_box/data/examples/manifest_5_5_7.jsonl', 
        '/home/ubuntu/dsp-modeler/data/input/input.wav', 
        '/home/ubuntu/dsp-modeler/data/outputs', 
        chunk_seconds, 
        param_names, 
        param_configs, 
        silent_lead_in_seconds=silent_lead_in_seconds, 
        denoise_wet = denoise_wet
    )
    t = deepcopy(example_dataset.tracks[0])

    t_dn = deepcopy(t)
    t_dn.compute_noise_profile(silent_lead_in_seconds=8)
    t_dn.denoise_wet_data()

    # t_g_m = deepcopy(t)
    # t_g_m.compute_model_gain(gain_model)
    # t_g_m.add_model_gain()

    # t_g_r = deepcopy(t)
    # t_g_r.add_constant_gain(t_g_r.compute_rms_gain())

    # t_dn_g_r = deepcopy(t_dn)
    # t_dn_g_r.add_constant_gain(t_dn_g_r.compute_rms_gain())

    t_p = deepcopy(t)
    t_p = Track.from_data(96000, chunk_seconds, t_p.get_params(), dry_data=t_p.get_dry(), wet_data=None)
    t_p = model.predict_track(t_p, device, param_names, param_configs, chunk_seconds, sr=96000, out_path='/home/ubuntu/dsp-modeler/data/predictions/p.wav')

    waves = [
        {'name':'dry', 'track':t, 'channel':'dry'},
        {'name':'wet', 'track':t, 'channel':'wet'},
        # {'name':'wet_dn', 'track':t2, 'channel':'wet'},
        # {'name':'wet_mdl_gn', 'track':t3, 'channel':'wet'},
        # {'name':'wet_rms_gn', 'track':t4, 'channel':'wet'},
        # {'name':'t_dn_g_r', 'track':t_dn_g_r, 'channel':'wet'},
        {'name':'preds', 'track':t_p, 'channel':'wet'},
    ]


    print_track_section(waves, 1, 5, zoom_start_s=11.95, zoom_len=0.15,title="Title", out_path='/home/ubuntu/dsp-modeler/black_box/train/viz/test.png')

    # waves = [
    #     {
    #         'name':'dry', 
    #         'data':load_wav("/home/ubuntu/dsp-modeler/data/input/input.wav")[0], 
    #         "sample_rate":96000
    #     },
    #     {
    #         'name':'wet', 
    #         'data':load_wav("/home/ubuntu/dsp-modeler/data/outputs/8bd1a579-014f-43d8-bf26-5c6fcd16f2f6.wav")[0], 
    #         "sample_rate":96000
    #     }
    # ]

    # print_wav_data_section(waves, 4, 15, zoom_start_s=9.9, zoom_len=0.2,title="Title", out_path='/home/ubuntu/dsp-modeler/black_box/train/viz/test.png')




