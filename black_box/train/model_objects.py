import torch, datetime, os, copy
import torch.nn as nn
import numpy as np
import torch.optim as optim
from data_objects import DataSet, Track
from common.utils import write_wav
from scipy.stats import skew

def get_symmetry(x):
    x = x.flatten()
    pos = x[x > 0]
    neg = x[x < 0]
    pos_rms = np.sqrt(np.mean(pos**2)) if len(pos) else 0
    neg_rms = np.sqrt(np.mean(neg**2)) if len(neg) else 0
    p99_9 = np.percentile(x, 99.9)
    p0_1 = np.percentile(x, 0.1)
    skewness = skew(x)
    pn_rms = pos_rms/neg_rms
    p999Overp001 = p99_9/abs(p0_1)
    return [skewness, pn_rms, p999Overp001]

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


class ConditionedLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=20, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x, states=None):
        out, states = self.lstm(x, states)
        out = self.dense(out)
        return out, states

    def predict_track(self, track, device, param_names, param_configs, chunk_seconds, out_path=False):
        preds_data = []
        hidden = None
        sr = track.sample_rate
        with torch.no_grad():
            for i, chunk in enumerate(track):
                features_tensor = chunk.get_features_tensor(device, param_names, param_configs)
                pred, hidden = self(features_tensor, hidden)
                preds_data.append(pred)

            preds = np.concatenate(preds_data)

        # preds = np.clip(preds, -1.0, 1.0)
        preds = preds.squeeze(-1) # reduce to single track
        print(f"preds {preds.shape}")

        if out_path and sr:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            write_wav(out_path, preds, sr)

        return Track.from_data(sr, chunk_seconds, track.get_params(), dry_data=track.get_dry(), wet_data=preds)

    # self, input_size=4, hidden_size=20, num_layers=1
    
    def train_manifest(
        self,
        train_dataset,
        validation_dataset,
        learning_rate=5e-4,
        epochs=10,
        warmup_samples=1000, # only applied to the first chunk -- state is cold there; every later chunk inherits an already-"settled" hidden state
        param_names=["d", "f", "v"],
        param_configs={
            'd':{'min':1, 'max':7, 'dtype':torch.float32},
            'f':{'min':1, 'max':7, 'dtype':torch.float32},
            'v':{'min':1, 'max':7, 'dtype':torch.float32},
        },
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
        model_output_dir='',
        lr_patience = 6,
        lr_factor = 0.5,
        batch_size=30,
        verbose_time=False,
        verbose_performance = False
    ):
        # Make out path
        start = datetime.datetime.now()
        model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
        os.makedirs(model_v_output_dir, exist_ok=True)

        # Model info
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        prepped = datetime.datetime.now()
        if verbose_time: print(f"Prep took {prepped - start}")

        # Modeling
        best_loss, best_state, bad_epochs_count = float('inf'), None, 0
        print("      |                 LOSS                   |           PREDICTIONS         |         TARGET             ")
        print("EPOCH |    total     esr        dc     pos neg |     mean      std      skew   |    mean       std      skew")


        train_denoised = True

        for epoch in range(epochs):
            e_time = datetime.datetime.now()

            # Training
            self.train()
            
            for batch in train_dataset.batches_of_random(): # batched segments of no specified track or ts
                features_tensors = batch.get_features_tensor(device, param_names, param_configs)
                target_tensors = batch.get_target_tensor(device)

                # gains = batch.get_gains_tensor(device, param_names)
                # features_tensors = features_tensors * gains

                pred_tensors, _ = self(features_tensors, None) # torch.Size([30, 57600, 1])

                # pred_tensors = pred_tensors / gains

                pred_for_loss = pred_tensors[:, warmup_samples:, :]
                target_for_loss = target_tensors[:, warmup_samples:, :]
                loss, _, _, _ = combined_loss(pred_for_loss, target_for_loss, batch_size, dc_weight=0.5, pos_neg_weight=0.2)

                # Backprop
                optimizer.zero_grad() # clear old gradients
                loss.backward() # compute fresh gradients for this accumulated window only
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # cap their magnitude
                optimizer.step() # apply them to the weights

            t_time =  datetime.datetime.now()
            if verbose_time: print(f"Train time: {t_time - e_time}")

            self.eval()
            num_tracks = len(validation_dataset.tracks)
            with torch.no_grad():
            # Predictions
                eval_preds = [[] for _ in range(num_tracks)]
                eval_targets = [[] for _ in range(num_tracks)]

                # Per Track group
                batch_groups = validation_dataset.make_window_batches(batch_size=batch_size)
                for batches in batch_groups:
                    eval_states = None
                    for i, batch in enumerate(batches):

                        features_tensors = batch.get_features_tensor(device, param_names, param_configs)
                        # gains = batch.get_gains_tensor(device, param_names)
                        target_tensors = batch.get_target_tensor(device)

                        # features_tensors = features_tensors * gains
                        pred_tensors, eval_states = self(features_tensors, eval_states)
                        # pred_tensors = pred_tensors / gains

                        # Save pred, tgt in memory structure
                        for i, track in enumerate(batch):
                            eval_preds[i].append(pred_tensors[i:i+1])
                            eval_targets[i].append(target_tensors[i:i+1])

                p_time =  datetime.datetime.now()
                if verbose_time: print(f"Prediction time: {p_time - t_time}")


            # Validation
                eval_losss = eval_esrs = eval_dcs = eval_pns = pSkewnesss = pPn_rmss = pP999Overp001s = tSkewnesss = tPn_rmss = tP999Overp001s = pMeans = pStds = tMeans = tStds = 0.0
                for p in range(num_tracks):
                    eval_pred_p = torch.cat(eval_preds[p], dim=1)
                    eval_target_p = torch.cat(eval_targets[p], dim=1)
                    eval_loss, eval_esr, eval_dc, eval_pn = combined_loss(eval_pred_p, eval_target_p, batch_size, dc_weight=0.5, pos_neg_weight=0.2)
                    eval_losss += eval_loss.item()
                    eval_esrs += eval_esr.item()
                    eval_dcs += eval_dc.item()
                    eval_pns += eval_pn.item()
                    pSkewness, pPn_rms, pP999Overp001 = get_symmetry(eval_pred_p.detach().cpu().numpy())
                    pSkewnesss += pSkewness
                    pPn_rmss += pPn_rms
                    pP999Overp001s += pP999Overp001
                    tSkewness, tPn_rms, tP999Overp001 = get_symmetry(eval_target_p.detach().cpu().numpy())
                    tSkewnesss += tSkewness
                    tPn_rmss += tPn_rms
                    tP999Overp001s += tP999Overp001
                    pMeans += eval_pred_p.mean().item()
                    pStds += eval_pred_p.std().item()
                    tMeans += eval_target_p.mean().item()
                    tStds += eval_target_p.std().item()
                eval_loss = eval_losss/num_tracks
                eval_esr = eval_esrs/num_tracks
                eval_dc = eval_dcs/num_tracks
                eval_pn = eval_pns/num_tracks
                pMean = pMeans/num_tracks
                pStd = pStds/num_tracks
                pSkewness = pSkewnesss/num_tracks
                tMean = tMeans/num_tracks
                tStd = tStds/num_tracks
                tSkewness = tSkewnesss/num_tracks
                print(f"{(epoch+1):02d}/{epochs:02d} |  {eval_loss:+07.3f}   {eval_esr:+07.3f} / {eval_dc:+07.3f} / {eval_pn:+07.3f} |  {pMean:+0.5f} / {pStd:+0.5f} / {pSkewness:+04.2f}  |  {tMean:+0.5f} / {tStd:+0.5f} / {tSkewness:+04.2f}")

            v_time =  datetime.datetime.now()
            if verbose_time: print(f"Validation time: {v_time - p_time}")


            # Reset model if diverging and lower learning rate
            if eval_loss < best_loss:
                bad_epochs_count = 0
                best_loss = eval_loss
                best_state = copy.deepcopy(self.state_dict())
            else:
                bad_epochs_count += 1
                if bad_epochs_count > lr_patience:
                    # Restore best model weights
                    self.load_state_dict(best_state)
                    # change optimizer lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_factor
                    bad_epochs_count = 0
                    print(f"Plateaued: Restored best weights (L = {best_loss:+9.3f}) with LR {optimizer.param_groups[0]['lr']:.2e}")


        # Get best model and save
        self.load_state_dict(best_state)
        torch.save(best_state, f'{model_v_output_dir}/model_best.pt')
            
##########################################################################################################################################
##########################################################################################################################################

class LSGainModel():
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
            gs.append(track.gain)

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
            actual_gain = track.compute_rms_gain()
            print(f'ABS ERR {(predicted_gain - actual_gain):+06.3f} PERC_ERR {((predicted_gain-actual_gain) / actual_gain):+06.3f} for {track.get_params()}')

    def cross_validate(self, full_dataset):
        tracks = full_dataset.tracks
        for i in range(len(full_dataset.tracks)):
            print(f'Track {i}')
            train_tracks = [t for j,t in enumerate(tracks) if j!=i]
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

##########################################################################################################################################
##########################################################################################################################################


class LSNoiseModel():
    def __init__(
        self,
        param_configs={'d':{'min':1, 'max':7, 'dtype':torch.float32},'f':{'min':1, 'max':7, 'dtype':torch.float32},'v':{'min':1, 'max':7, 'dtype':torch.float32}},
        silent_lead_in_seconds=8
    ):
        self.coeffs = None
        self.param_configs = param_configs
        self.silent_lead_in_seconds=silent_lead_in_seconds

        

    def fit(self, train_dataset):
        ds, fs, vs, ns = [], [], [], []
        for track in train_dataset:
            chunk_0 = track[0]
            norm_params = chunk_0.normalize_params(self.param_configs)
            d, f, v = norm_params['d'], norm_params['f'], norm_params['v']
            ds.append(d)
            fs.append(f)
            vs.append(v)
            ns.append(track.noise_profile.flatten())  # (n_freq_bins,)

        ds, fs, vs = np.array(ds), np.array(fs), np.array(vs)
        Y = np.stack(ns)  # (n_tracks, n_freq_bins)
        X = np.column_stack([np.ones_like(ds), ds, fs, vs, ds*vs, ds*fs, fs*vs, ds*fs*vs, vs**2, ds**2, vs**3])
        y = np.log(Y + 1e-12)
        self.coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)  # (n_features, n_freq_bins)
        return self.coeffs

    def predict(self, track):
        chunk_0 = track[0]
        norm_params = chunk_0.normalize_params(self.param_configs)
        d, f, v = norm_params['d'], norm_params['f'], norm_params['v']
        x = np.array([1, d, f, v, d*v, d*f, f*v, d*f*v, v**2, d**2, v**3])
        return np.exp(x @ self.coeffs).reshape(-1, 1)  # (n_freq_bins, 1), matches Track.noise_profile shape

    def validate(self, validation_dataset: DataSet):
        for i, track in enumerate(validation_dataset):
            predicted_profile = self.predict(track)
            actual_profile = track.noise_profile
            abs_err = np.mean(np.abs(predicted_profile - actual_profile))
            perc_err = np.mean(np.abs((predicted_profile - actual_profile) / (actual_profile + 1e-12)))
            print(f'MEAN ABS ERR {abs_err:+.3e} MEAN PERC_ERR {perc_err:+06.3f} for {track.get_params()}')

    def cross_validate(self, full_dataset):
        tracks = full_dataset.tracks
        for i in range(len(full_dataset.tracks)):
            print(f'Track {i}')
            train_tracks = [t for j, t in enumerate(tracks) if j != i]
            validation_tracks = [tracks[i]]
            self.fit(train_tracks)
            self.validate(validation_tracks)

    def save(self, model_output_dir):
        model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
        os.makedirs(model_v_output_dir, exist_ok=True)
        np.savez(
            f'{model_v_output_dir}/noise_model.npz',
            coeffs=self.coeffs,
            silent_lead_in_seconds=self.silent_lead_in_seconds,
        )

    def load(self, model_path):
        data = np.load(model_path)
        self.coeffs = data['coeffs']
        self.silent_lead_in_seconds = data['silent_lead_in_seconds']

##########################################################################################################################################
##########################################################################################################################################


if __name__ == "__main__":

    nm = LSNoiseModel(
        param_configs={'d':{'min':1, 'max':7, 'dtype':torch.float32},'f':{'min':1, 'max':7, 'dtype':torch.float32},'v':{'min':1, 'max':7, 'dtype':torch.float32}}
    )

    train_dataset = DataSet(
        '/home/ubuntu/dsp-modeler/data/outputs/manifest_dv3_plus.jsonl', 
        '/home/ubuntu/dsp-modeler/data/input/input.wav', 
        '/home/ubuntu/dsp-modeler/data/outputs', 
        0.03, 
        param_names=['d', 'f', 'v'], 
        param_configs={'d':{'min':1, 'max':7, 'dtype':torch.float32},'f':{'min':1, 'max':7, 'dtype':torch.float32},'v':{'min':1, 'max':7, 'dtype':torch.float32}}, 
        silent_lead_in_seconds=8.0
    )
    train_dataset.compute_noise_profile()
    nm.cross_validate(train_dataset)

    nm.save('/home/ubuntu/dsp-modeler/black_box/model/models/noise_model')

    # gm = LSGainModel(
    #     param_configs={'d':{'min':1, 'max':7, 'dtype':torch.float32},'f':{'min':1, 'max':7, 'dtype':torch.float32},'v':{'min':1, 'max':7, 'dtype':torch.float32}}
    # )

    # train_dataset = DataSet(
    #     '/home/ubuntu/dsp-modeler/data/outputs/manifest_dv3_plus.jsonl', 
    #     '/home/ubuntu/dsp-modeler/data/input/input.wav', 
    #     '/home/ubuntu/dsp-modeler/data/outputs', 
    #     0.03, 
    #     param_names=['d', 'f', 'v'], 
    #     param_configs={'d':{'min':1, 'max':7, 'dtype':torch.float32},'f':{'min':1, 'max':7, 'dtype':torch.float32},'v':{'min':1, 'max':7, 'dtype':torch.float32}}, 
    #     silent_lead_in_seconds=8.0
    # )
    # train_dataset.calculate_noise_profiles()
    # train_dataset.remove_noise()
    # train_dataset.compute_rms_gain()
    # gm.cross_validate(train_dataset)

    # gm.save('/home/ubuntu/dsp-modeler/black_box/model/models/ls_gain_model_dn')

    # gm = LSGainModel() # trained on v>1, d>1
    # # Only one clear outlier: {'d':7,'f':5,'v':3} at 0.0704, +37% over — everything else in the set stays under ±26%.
    # gm.load('/home/ubuntu/dsp-modeler/black_box/model/models/ls_gain_model/2026-08-22_00-24/gain_model.npz')

    nm = LSNoiseModel() # trained on v>1, d>1
    nm.load('/home/ubuntu/dsp-modeler/black_box/model/models/noise_model/2026-08-25_01-10/noise_model.npz')

    # train_dataset = DataSet(
    #     '/home/ubuntu/dsp-modeler/data/outputs/manifest_dv3_plus.jsonl', 
    #     '/home/ubuntu/dsp-modeler/data/input/input.wav', 
    #     '/home/ubuntu/dsp-modeler/data/outputs', 
    #     0.03, 
    #     param_names=['d', 'f', 'v'], 
    #     param_configs={'d':{'min':1, 'max':7, 'dtype':torch.float32},'f':{'min':1, 'max':7, 'dtype':torch.float32},'v':{'min':1, 'max':7, 'dtype':torch.float32}}, 
    #     silent_lead_in_seconds=8.0
    # )

    # for track in train_dataset:
    #     g = gm.predict(track)
    #     rms_w = np.sqrt(np.mean(track.get_wet() ** 2))
    #     print(f"{track.get_params()} => {rms_w/g}")