from data_objects import Track, DataSet
from model_objects import ConditionedLSTM, LSGainModel
import torch
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from common.utils import load_wav
from common.delay_ops import measure_delay, apply_shift
from eval.spectral_compare_db import plot_average_spectrum


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
    silent_lead_in_seconds=8

    #####################################################
    # 
    #####################################################

    # example_dataset = DataSet(
    #     '/home/ubuntu/dsp-modeler/data/outputs/manifest_dv3_plus.jsonl', 
    #     '/home/ubuntu/dsp-modeler/data/input/input.wav', 
    #     '/home/ubuntu/dsp-modeler/data/outputs', 
    #     chunk_seconds, 
    #     param_names, 
    #     param_configs, 
    #     silent_lead_in_seconds=silent_lead_in_seconds, 
    #     dbg_tracks = [1]
    # )
    # print(example_dataset.tracks)
    # for trk in example_dataset:
    #     print(f"TRACK {trk.id} {trk.get_params()}")

    # trk = deepcopy(example_dataset[0])

    # gain_model = LSGainModel(param_configs)
    # gain_model.load('/home/ubuntu/dsp-modeler/black_box/model/models/ls_gain_model/2026-08-22_00-24/gain_model.npz')
    # example_dataset.calculate_noise_profiles()
    # example_dataset.denoise_wet_data()
    # example_dataset.compute_model_gain(gain_model)
    # example_dataset.add_model_gain()

    # trk_mod = deepcopy(example_dataset[0])

    # waves = [
    #     {'name':'dry',     'track':trk,     'channel':'dry'},
    #     {'name':'wet',     'track':trk,     'channel':'wet'},
    #     {'name':'wet_mod', 'track':trk_mod, 'channel':'wet'}
    # ]
    # print_track_section(waves, 1, 5, zoom_start_s=2, zoom_len=0.15,title="Silence", out_path='/home/ubuntu/dsp-modeler/black_box/train/viz/test_sln.png')
    # print_track_section(waves, 7, 17, zoom_start_s=11.95, zoom_len=0.15,title="Sound", out_path='/home/ubuntu/dsp-modeler/black_box/train/viz/test.png')

    #####################################################
    # Single track from dataset - predict and compare
    #####################################################
    gain_model = LSGainModel(param_configs)
    gain_model.load('/home/ubuntu/dsp-modeler/black_box/model/models/ls_gain_model/2026-08-22_00-24/gain_model.npz')

    example_dataset = DataSet(
        '/home/ubuntu/dsp-modeler/data/outputs/manifest_dv3_plus.jsonl', 
        '/home/ubuntu/dsp-modeler/data/input/input.wav', 
        '/home/ubuntu/dsp-modeler/data/outputs', 
        chunk_seconds, 
        param_names, 
        param_configs, 
        silent_lead_in_seconds=silent_lead_in_seconds, 
        dbg_tracks = [37]
    )
    print(example_dataset.tracks)
    for trk in example_dataset:
        print(f"TRACK {trk.id} {trk.get_params()}")
    trk = deepcopy(example_dataset[0])
    trk_dry=trk.get_dry()
    trk_wet=trk.get_wet()

    mod_trk = deepcopy(trk)
    mod_trk.compute_noise_profile()
    mod_trk.denoise_wet_data()

    print(f'trk {mod_trk.get_wet().shape}')

    params=trk.get_params()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = ConditionedLSTM(input_size=4, hidden_size=20).to(device)
    model.load_state_dict(torch.load('/home/ubuntu/dsp-modeler/black_box/model/models/transform_model/2026-08-24_13-02/model_best.pt', map_location=device))

    trk_f = Track.from_data(96000, chunk_seconds, params, dry_data=trk_dry, wet_data=None)
    trk_p = model.predict_track(trk_f, device, param_names, param_configs, chunk_seconds)#, out_path='/home/ubuntu/dsp-modeler/data/predictions/p.wav')
    trk_p.compute_model_gain(gain_model) # gain for these params
    trk_p.inverse_model_gain()

    waves = [
        #{'name':'dry',     'track':trk,     'channel':'dry'},
        {'name':'wet_dn',  'track':mod_trk,     'channel':'wet'},
        {'name':'trk_p',   'track':trk_p, 'channel':'wet'}
    ]
    print_track_section(waves, 1, 5, zoom_start_s=2, zoom_len=0.15,title="Silence", out_path='/home/ubuntu/dsp-modeler/black_box/train/viz/test_sln.png')
    print_track_section(waves, 7, 17, zoom_start_s=11.95, zoom_len=0.15,title="Sound", out_path='/home/ubuntu/dsp-modeler/black_box/train/viz/test.png')
    print(f'trk {mod_trk.get_wet().shape}')
    print(f'trk_p {trk_p.get_wet().shape}')
    serieses = {'trk': mod_trk.get_wet(), 'trk_p': trk_p.get_wet()}

    plot_average_spectrum(serieses, 96000, f'/home/ubuntu/dsp-modeler/black_box/train/viz/eval_db_spectrum_sln.png', start_seconds=1)
    plot_average_spectrum(serieses, 96000, f'/home/ubuntu/dsp-modeler/black_box/train/viz/eval_db_spectrum.png', start_seconds=12.5)