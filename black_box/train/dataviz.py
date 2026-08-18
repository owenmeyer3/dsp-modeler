from train import Track, DataSet
import torch
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    param_configs={
        'd':{'min':1, 'max':7, 'dtype':torch.float32},
        'f':{'min':1, 'max':7, 'dtype':torch.float32},
        'v':{'min':1, 'max':7, 'dtype':torch.float32},
    }

    full_dataset = DataSet(
        '/home/ubuntu/dsp-modeler/data/outputs/odds-50.jsonl', 
        '/home/ubuntu/dsp-modeler/data/input/input.wav', 
        '/home/ubuntu/dsp-modeler/data/outputs', 
        0.03, 
        ["d", "f", "v"], 
        {'d':{'min':1, 'max':7, 'dtype':torch.float32},'f':{'min':1, 'max':7, 'dtype':torch.float32},'v':{'min':1, 'max':7, 'dtype':torch.float32}}, 
        silent_lead_in_seconds=8, 
        trim_noise = True
    )

    cross_validate(full_dataset)



