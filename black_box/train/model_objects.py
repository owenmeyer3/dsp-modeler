import torch, datetime, os
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree
from data_objects import DataSet, Track, Chunk, parse_to_subarrays
from common.utils import write_wav

class ConditionedLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=20, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x, states=None):
        out, states = self.lstm(x, states)
        out = self.dense(out)
        return out, states

    def predict_track(self, track, device, param_names, param_configs, chunk_seconds, sr=False, out_path=False):
        preds_data = []
        hidden = None
        with torch.no_grad():
            for i, chunk in enumerate(track):
                features_tensor = chunk.get_features_tensor(device, param_names, param_configs)
                pred, hidden = self(features_tensor, hidden)
                preds_data.append(pred)

            preds = np.concatenate(preds_data)

        preds = np.clip(preds, -1.0, 1.0)
        print(f"preds {preds.shape}")

        if out_path and sr:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            write_wav(out_path, preds, sr)

        return Track.from_data(sr, chunk_seconds, track.get_params(), dry_data=track.get_dry(), wet_data=preds)

            
# def esr_loss(pred, target, eps=1e-8, min_energy=1e-4):
#     energy = torch.sum(target ** 2)
#     return torch.sum((pred - target) ** 2) / (torch.maximum(energy, torch.tensor(min_energy)) + eps)


# def dc_loss(pred, target, eps=1e-8):
#     """Penalizes any DC offset difference (mean-level mismatch) between
#     prediction and target -- ESR alone doesn't strongly constrain this."""
#     target_var = torch.var(target) + eps
#     return (torch.mean(pred) - torch.mean(target)) ** 2 / target_var

# def pos_neg_balance_loss(pred, target, eps=1e-8):
#     target_var = torch.var(target) + eps
#     pred_pos_rms = torch.sqrt((pred.clamp(min=0) ** 2).mean() + eps)
#     pred_neg_rms = torch.sqrt((pred.clamp(max=0) ** 2).mean() + eps)
#     target_pos_rms = torch.sqrt((target.clamp(min=0) ** 2).mean() + eps)
#     target_neg_rms = torch.sqrt((target.clamp(max=0) ** 2).mean() + eps)
#     return ((pred_pos_rms - target_pos_rms) ** 2 + (pred_neg_rms - target_neg_rms) ** 2) / target_var

# def combined_loss(pred, target, esr_weight = 1.0, dc_weight=0.5, pos_neg_weight=0.0):
#     esr = esr_loss(pred, target)
#     dc = dc_loss(pred, target)
#     pos_neg_balance = pos_neg_balance_loss(pred, target)

#     return [
#         esr + dc_weight * dc + pos_neg_weight * pos_neg_balance,
#         esr,
#         dc,
#         pos_neg_balance
#     ]

def esr_loss(pred, target, eps=1e-8, min_energy=1e-4, batch_size=None):
    energy = torch.sum(target ** 2)
    return torch.sum((pred - target) ** 2) / (torch.maximum(energy, torch.tensor(min_energy)) + eps)


def dc_loss(pred, target, eps=1e-8):
    """Penalizes any DC offset difference (mean-level mismatch) between
    prediction and target -- ESR alone doesn't strongly constrain this."""
    target_var = torch.var(target) + eps
    return (torch.mean(pred) - torch.mean(target)) ** 2 / target_var

def pos_neg_balance_loss(pred, target, eps=1e-8):
    target_var = torch.var(target) + eps
    pred_pos_rms = torch.sqrt((pred.clamp(min=0) ** 2).mean() + eps)
    pred_neg_rms = torch.sqrt((pred.clamp(max=0) ** 2).mean() + eps)
    target_pos_rms = torch.sqrt((target.clamp(min=0) ** 2).mean() + eps)
    target_neg_rms = torch.sqrt((target.clamp(max=0) ** 2).mean() + eps)
    return ((pred_pos_rms - target_pos_rms) ** 2 + (pred_neg_rms - target_neg_rms) ** 2) / target_var

def combined_loss(pred_batch, target_batch, batch_size, esr_weight = 1.0, dc_weight=0.5, pos_neg_weight=0.0):

    esr_losses, dc_losses, pos_neg_balance_losses = [], [], []

    for i in range(len(pred_batch)):
        seg_pred=pred_batch[i] # (56600, 1)
        seg_target=target_batch[i] # (56600, 1)
        esr_losses.append(esr_loss(seg_pred, seg_target))
        dc_losses.append(dc_loss(seg_pred, seg_target))
        pos_neg_balance_losses.append(pos_neg_balance_loss(seg_pred, seg_target))
    esr = torch.stack(esr_losses).mean()
    dc = torch.stack(dc_losses).mean()
    pos_neg_balance = torch.stack(pos_neg_balance_losses).mean()

    return [
        esr + dc_weight * dc + pos_neg_weight * pos_neg_balance,
        esr,
        dc,
        pos_neg_balance
    ]

    # print(esr_losses)

    # # Split batch into segments and calculate losses on segments separately
    # if batch_size:
    #     print(pred)
    #     print(pred.shape) # ([30, 56600, 1])
    #     batch_length = len(pred)
    #     assert batch_length % segment_size == 0, f'predictions length must be a multiple of segment_size (batch_length={batch_length}, segment_size={segment_size})'
        
    #     esr_losses, dc_losses, pos_neg_balance_losses = [], [], []
    #     for i in range(0, batch_length, segment_size):
    #         seg_pred = pred[i:i+segment_size]
    #         seg_target = target[i:i+segment_size]

    #         esr_losses.append(esr_loss(seg_pred, seg_target))
    #         dc_losses.append(dc_loss(seg_pred, seg_target))
    #         pos_neg_balance_losses.append(pos_neg_balance_loss(seg_pred, seg_target))
        
    #     esr = torch.stack(esr_losses).mean()
    #     dc = torch.stack(dc_losses).mean()
    #     pos_neg_balance = torch.stack(pos_neg_balance_losses).mean()

    # # Calculate losses on batch in entirety
    # else:
    #     esr = esr_loss(pred, target)
    #     dc = dc_loss(pred, target)
    #     pos_neg_balance = pos_neg_balance_loss(pred, target)

    # return [
    #     esr + dc_weight * dc + pos_neg_weight * pos_neg_balance,
    #     esr,
    #     dc,
    #     pos_neg_balance
    # ]
##########################################################################################################################################
##########################################################################################################################################

# This model currently ingests unscaled parameters. Given all knobs have the same range, 
# this doesnt't bias toward any knob. Worth addressing for other models. 
class TrackDataModel(object):
    def __init__(self, k=8, bandwidth=0.3):
        self.tree=None
        self.noise_profiles=None # (# tracks, # n_freq_bins)
        self.gain_vals=None # (# tracks, 1)
        self.param_vals=None
        self.bandwidth=bandwidth
        self.k=k
        super().__init__()
    
    def train(self, train_dataset:DataSet):
        self.param_vals = [] # (# tracks, # params)
        self.noise_profiles = []
        self.gain_vals = []
        for track in train_dataset:
            segment_0 = track[0]
            chunk_0 = segment_0[0]
            norm_params = chunk_0.normalize_params(train_dataset.param_configs)
            pv = [norm_params[pn] for pn in norm_params]
            self.param_vals.append(pv)
            self.noise_profiles.append(segment_0.noise_profile.flatten())
            self.gain_vals.append(track.compute_wet_gain())

        self.param_vals = np.array(self.param_vals)
        self.noise_profiles = np.array(self.noise_profiles)
        self.gain_vals = np.array(self.gain_vals)
        self.tree = cKDTree(self.param_vals)

    def validate(self, validation_dataset:DataSet):
        for i, track in enumerate(validation_dataset):
            segment_0 = track[0]
            chunk_0 = segment_0[0]
            norm_params = chunk_0.normalize_params(validation_dataset.param_configs)
            query_params = [norm_params[pn] for pn in norm_params]
            predicted_noise_profile, predicted_gain = self.predict(query_params)
            actual_noise_profile = segment_0.noise_profile
            actual_gain = track.compute_wet_gain()

            print(f'Track {i}')
            print(f'noise_profile {predicted_noise_profile} vs. {actual_noise_profile}')
            print(f'gain {predicted_gain} vs. {actual_gain}')


    def predict(self, query_params):
        distances, indices = self.tree.query(query_params, k=self.k)
        weights = np.exp(-(distances**2) / (2 * self.bandwidth**2))
        weights /= weights.sum()

        neighbor_noise_profiles = self.noise_profiles[indices]  # (k, n_freq_bins)
        predicted_noise_profile = (weights[:, None] * neighbor_noise_profiles).sum(axis=0)

        neighbor_gains = self.gain_vals[indices]               # (k,)
        predicted_gain = (weights * neighbor_gains).sum()

        return predicted_noise_profile, predicted_gain
    
    def save(self, model_output_dir):
        model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
        os.makedirs(model_v_output_dir, exist_ok=True)
        np.savez(
            f'{model_v_output_dir}/track_model.npz',
            param_vals=self.param_vals,
            noise_profiles=self.noise_profiles,
            gain_vals=self.gain_vals
        )

    def load(self, model_path):
        data = np.load(model_path)
        self.param_vals = data['param_vals']
        self.noise_profiles = data['noise_profiles']
        self.gain_vals = data['gain_vals']
        self.tree = cKDTree(self.param_vals)  # cheap to rebuild from scratch every time

##########################################################################################################################################
##########################################################################################################################################

class TrackDataModel2(object):
    def __init__(self, k=8, bandwidth=0.5):
        self.tree=None
        self.noise_profiles=None # (# tracks, # n_freq_bins)
        self.gain_vals=None # (# tracks, 1)
        self.param_vals=None
        self.bandwidth=bandwidth
        self.k=k
        super().__init__()
    
    def train(self, train_dataset:DataSet):
        self.param_vals = [] # (# tracks, # params)
        self.noise_profiles = []
        self.gain_vals = []
        for track in train_dataset:
            segment_0 = track[0]
            chunk_0 = segment_0[0]
            norm_params = chunk_0.normalize_params(train_dataset.param_configs)
            pv = [norm_params[pn] for pn in norm_params]
            self.param_vals.append(pv)
            self.noise_profiles.append(segment_0.noise_profile.flatten())
            self.gain_vals.append(track.compute_wet_gain())

        self.param_vals = np.array(self.param_vals)
        self.noise_profiles = np.array(self.noise_profiles)
        self.gain_vals = np.array(self.gain_vals)
        self.tree = cKDTree(self.param_vals)

    def validate(self, validation_dataset:DataSet):
        for i, track in enumerate(validation_dataset):
            segment_0 = track[0]
            chunk_0 = segment_0[0]
            norm_params = chunk_0.normalize_params(validation_dataset.param_configs)
            query_params = [norm_params[pn] for pn in norm_params]
            predicted_noise_profile, predicted_gain = self.predict(query_params)
            actual_noise_profile = segment_0.noise_profile
            actual_gain = track.compute_wet_gain()

            print(f'Track {i}')
            print(f'noise_profile {predicted_noise_profile} vs. {actual_noise_profile}')
            print(f'gain {predicted_gain} vs. {actual_gain}')


    def predict(self, query_params):
        distances, indices = self.tree.query(query_params, k=self.k)
        weights = np.exp(-(distances**2) / (2 * self.bandwidth**2))
        # weights /= weights.sum()

        X = np.hstack([np.ones((self.k, 1)), self.param_vals[indices]]) # (k, 4) -- [bias, d, f, v]
        sw = np.sqrt(weights)
        Xw = X * sw[:, None]
        query_row = np.concatenate([[1.0], query_params])

        # Gain (scalar target)
        # yw_gain = self.gain_vals[indices] * sw
        yw_gain = np.log(self.gain_vals[indices]) * sw
        gain_coeffs, *_ = np.linalg.lstsq(Xw, yw_gain, rcond=None)          # weighted least squares
        predicted_gain = query_row @ gain_coeffs

        # Noise profile (vector target -- fit every bin in one call)
        # yw_noise_profile = self.noise_profiles[indices] * sw[:, None]           # (k, n_freq_bins)
        yw_noise_profile = np.log(self.noise_profiles[indices]) * sw[:, None]           # (k, n_freq_bins)
        profile_coeffs, *_ = np.linalg.lstsq(Xw, yw_noise_profile, rcond=None)  # (4, n_freq_bins)
        predicted_noise_profile = query_row @ profile_coeffs               # (n_freq_bins,)

        print(f"predicted_gain: {predicted_gain}")
        print(f"predicted_noise_profile: {predicted_noise_profile}")

        return predicted_noise_profile, predicted_gain

    
    def save(self, model_output_dir):
        model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
        os.makedirs(model_v_output_dir, exist_ok=True)
        np.savez(
            f'{model_v_output_dir}/track_model.npz',
            param_vals=self.param_vals,
            noise_profiles=self.noise_profiles,
            gain_vals=self.gain_vals
        )

    def load(self, model_path):
        data = np.load(model_path)
        self.param_vals = data['param_vals']
        self.noise_profiles = data['noise_profiles']
        self.gain_vals = data['gain_vals']
        self.tree = cKDTree(self.param_vals)  # cheap to rebuild from scratch every time


class GainModel():
    def __init__(
        self, 
        param_configs={'d':{'min':1, 'max':7, 'dtype':torch.float32},'f':{'min':1, 'max':7, 'dtype':torch.float32},'v':{'min':1, 'max':7, 'dtype':torch.float32}}
    ):
        self.coeffs = None
        self.param_configs=param_configs

    def fit(self, train_dataset):
        ds, fs, vs, gs = [], [], [], []
        for track in train_dataset:
            chunk_0=track[0]
            norm_params = chunk_0.normalize_params(self.param_configs)
            d, f, v = norm_params['d'], norm_params['f'], norm_params['v']
            ds.append(d)
            fs.append(f)
            vs.append(v)
            gs.append(track.compute_wet_gain())

        ds, fs, vs, gs = np.array(ds), np.array(fs), np.array(vs), np.array(gs)
        X = np.column_stack([np.ones_like(ds), ds, fs, vs, ds*vs, ds*fs, fs*vs, ds*fs*vs, vs**2, ds**2, vs**3])
        y = np.log(gs)
        self.coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self.coeffs

    def predict(self, track):
        chunk_0=track[0]
        norm_params = chunk_0.normalize_params(self.param_configs)
        d, f, v = norm_params['d'], norm_params['f'], norm_params['v']
        x = np.array([1, d, f, v, d*v, d*f, f*v, d*f*v, v**2, d**2, v**3])
        return float(np.exp(x @ self.coeffs)) # float64 to 32

    def validate(self, validation_dataset: DataSet): # validation_dataset is just 1 index array w/ cross val
        for i, track in enumerate(validation_dataset):
            predicted_gain = self.predict(track)
            actual_gain = track.compute_wet_gain()
            print(f'Track {i}: gain {predicted_gain} / {actual_gain}')

    def cross_validate(self, full_dataset):
        tracks = full_dataset.tracks
        for i in range(len(full_dataset.tracks)):
            train_tracks = [t for i,t in enumerate(tracks) if i!=3]
            validation_tracks = [tracks[i]]
            self.fit(train_tracks)
            self.validate(validation_tracks)

    def save(self, model_output_dir):
        model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
        os.makedirs(model_v_output_dir, exist_ok=True)
        np.savez(
            f'{model_v_output_dir}/gain_model.npz',
            coeffs=self.coeffs,
        )

    def load(self, model_path):
        data = np.load(model_path)
        self.coeffs = data['coeffs']

# Fit with X-Validation
# {'d': -1.0, 'f': 0.33333333333333326, 'v': -1.0}
# Track 0: gain 8.649732859179214e-06 / 3.7479399907169864e-05
# {'d': -1.0, 'f': 1.0, 'v': -1.0}
# Track 0: gain 1.7039548639884965e-05 / 3.754878343897872e-05
# {'d': -1.0, 'f': -1.0, 'v': -0.33333333333333337}
# Track 0: gain 0.0009189353338475999 / 0.0007683449657633901
# {'d': -1.0, 'f': -1.0, 'v': 0.33333333333333326}
# Track 0: gain 0.00433171043788962 / 0.003593521425500512
# {'d': -1.0, 'f': -1.0, 'v': 1.0}
# Track 0: gain 0.023550714205278163 / 0.02020937390625477
# {'d': -1.0, 'f': -0.33333333333333337, 'v': -0.33333333333333337}
# Track 0: gain 0.001400074824981685 / 0.0006669199792668223
# {'d': -1.0, 'f': -0.33333333333333337, 'v': 0.33333333333333326}
# Track 0: gain 0.0051043050017298445 / 0.0034218034707009792
# {'d': -1.0, 'f': -0.33333333333333337, 'v': 1.0}
# Track 0: gain 0.02146308698915286 / 0.021725960075855255
# {'d': -1.0, 'f': 0.33333333333333326, 'v': -0.33333333333333337}
# Track 0: gain 0.002133131073913615 / 0.0006220475770533085
# {'d': -1.0, 'f': 0.33333333333333326, 'v': 0.33333333333333326}
# Track 0: gain 0.006014697871489681 / 0.0036298418417572975
# {'d': -1.0, 'f': 0.33333333333333326, 'v': 1.0}
# Track 0: gain 0.019560515196634654 / 0.020080916583538055
# {'d': -0.33333333333333337, 'f': 0.33333333333333326, 'v': -1.0}
# Track 0: gain 5.336071785918129e-05 / 4.27193044743035e-05
# {'d': -0.33333333333333337, 'f': 1.0, 'v': -1.0}
# Track 0: gain 7.339257981792472e-05 / 3.8929130823817104e-05
# {'d': -0.33333333333333337, 'f': -1.0, 'v': -0.33333333333333337}
# Track 0: gain 0.011317430943067261 / 0.02219834364950657
# {'d': -0.33333333333333337, 'f': -0.33333333333333337, 'v': -0.33333333333333337}
# Track 0: gain 0.013541202397733555 / 0.02205759845674038
# {'d': -0.33333333333333337, 'f': 0.33333333333333326, 'v': -0.33333333333333337}
# Track 0: gain 0.016201924562102895 / 0.01872614026069641
# {'d': -0.33333333333333337, 'f': 1.0, 'v': -0.33333333333333337}
# Track 0: gain 0.01938545424592496 / 0.013829538598656654
# {'d': -0.33333333333333337, 'f': -1.0, 'v': 0.33333333333333326}
# Track 0: gain 0.051917820280654704 / 0.08983675390481949
# {'d': -0.33333333333333337, 'f': -0.33333333333333337, 'v': 0.33333333333333326}
# Track 0: gain 0.05403867762923445 / 0.08924134075641632
# {'d': -0.33333333333333337, 'f': 0.33333333333333326, 'v': 0.33333333333333326}
# Track 0: gain 0.05624617258834376 / 0.08658488094806671
# {'d': -0.33333333333333337, 'f': 1.0, 'v': 0.33333333333333326}
# Track 0: gain 0.058543844328386406 / 0.066166453063488
# {'d': 0.33333333333333326, 'f': -1.0, 'v': 1.0}
# Track 0: gain 1.8730951985646536 / 1.6996015310287476
# {'d': 0.33333333333333326, 'f': -0.33333333333333337, 'v': -1.0}
# Track 0: gain 0.00020039573944143513 / 0.00012174757284810767
# {'d': 0.33333333333333326, 'f': -0.33333333333333337, 'v': -0.33333333333333337}
# Track 0: gain 0.07656259641638705 / 0.09815417230129242
# {'d': 0.33333333333333326, 'f': -0.33333333333333337, 'v': 0.33333333333333326}
# Track 0: gain 0.3344461872092744 / 0.3041910231113434
# {'d': 0.33333333333333326, 'f': -0.33333333333333337, 'v': 1.0}
# Track 0: gain 1.6850238150789933 / 1.565757155418396
# {'d': 0.33333333333333326, 'f': 0.33333333333333326, 'v': -1.0}
# Track 0: gain 0.00019243943483504525 / 0.00012295677151996642
# {'d': 0.33333333333333326, 'f': 0.33333333333333326, 'v': -0.33333333333333337}
# Track 0: gain 0.07193978073307143 / 0.0805082842707634
# {'d': 0.33333333333333326, 'f': 0.33333333333333326, 'v': 0.33333333333333326}
# Track 0: gain 0.30748616077674523 / 0.2743311822414398
# {'d': 0.33333333333333326, 'f': 0.33333333333333326, 'v': 1.0}
# Track 0: gain 1.5158360661855923 / 1.6604552268981934
# {'d': 0.33333333333333326, 'f': 1.0, 'v': -1.0}
# Track 0: gain 0.0001847990190951854 / 8.524286386091262e-05
# {'d': 0.33333333333333326, 'f': 1.0, 'v': -0.33333333333333337}
# Track 0: gain 0.06759608861455349 / 0.05452679097652435
# {'d': 0.33333333333333326, 'f': 1.0, 'v': 0.33333333333333326}
# Track 0: gain 0.28269940781253605 / 0.2105213701725006
# {'d': 0.33333333333333326, 'f': 1.0, 'v': 1.0}
# Track 0: gain 1.3636359076867361 / 1.1440902948379517
# {'d': -0.33333333333333337, 'f': 0.33333333333333326, 'v': 1.0}
# Track 0: gain 0.225210970636803 / 0.4172809422016144
# {'d': -0.33333333333333337, 'f': 1.0, 'v': 1.0}
# Track 0: gain 0.20391852940574695 / 0.30897602438926697
# {'d': 0.33333333333333326, 'f': -1.0, 'v': -1.0}
# Track 0: gain 0.00020868099316911082 / 0.0001377320004394278
# {'d': 1.0, 'f': -1.0, 'v': 1.0}
# Track 0: gain 7.466514205387028 / 5.98728084564209
# {'d': 1.0, 'f': -0.33333333333333337, 'v': -1.0}
# Track 0: gain 0.000605116834378131 / 0.0005916827358305454
# {'d': 1.0, 'f': -0.33333333333333337, 'v': -0.33333333333333337}
# Track 0: gain 0.2530635039571305 / 0.2622905671596527
# {'d': 1.0, 'f': -0.33333333333333337, 'v': 0.33333333333333326}
# Track 0: gain 1.210044177057715 / 1.0680551528930664
# {'d': 1.0, 'f': -0.33333333333333337, 'v': 1.0}
# Track 0: gain 6.673338166472642 / 5.569765090942383
# {'d': 1.0, 'f': 0.33333333333333326, 'v': -1.0}
# Track 0: gain 0.0004057139331707778 / 0.0005536163807846606
# {'d': 1.0, 'f': 0.33333333333333326, 'v': -0.33333333333333337}
# Track 0: gain 0.1867347112535317 / 0.3487813174724579
# {'d': 1.0, 'f': 0.33333333333333326, 'v': 0.33333333333333326}
# Track 0: gain 0.9826788602083226 / 0.9187756776809692
# {'d': 1.0, 'f': 0.33333333333333326, 'v': 1.0}
# Track 0: gain 5.964422092972106 / 5.935342788696289
# {'d': 1.0, 'f': 1.0, 'v': -1.0}
# Track 0: gain 0.00027201985834365914 / 0.000425469595938921
# {'d': 1.0, 'f': 1.0, 'v': -0.33333333333333337}
# Track 0: gain 0.1377909174641273 / 0.20100203156471252
# {'d': 1.0, 'f': 1.0, 'v': 0.33333333333333326}
# Track 0: gain 0.7980351136008726 / 0.7955979108810425
# {'d': 1.0, 'f': 1.0, 'v': 1.0}
# Track 0: gain 5.330814955828538 / 4.125934600830078