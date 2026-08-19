import torch, datetime, copy, os
import torch.optim as optim
import numpy as np
from scipy.stats import skew
from model_objects import ConditionedLSTM, combined_loss, TrackDataModel, TrackDataModel2, GainModel
from data_objects import DataSet

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

def train_manifest(
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
    hidden_size=20,
    verbose_time=False,
    verbose_performance = False
):
    # Make out path
    start = datetime.datetime.now()
    model_v_output_dir = f'{model_output_dir}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}'
    os.makedirs(model_v_output_dir, exist_ok=True)

    # Model info
    model = ConditionedLSTM(input_size=len(param_names) + 1, hidden_size=hidden_size).to(device)
    # model.lstm.flatten_parameters()                                 < =======================================
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    prepped = datetime.datetime.now()
    if verbose_time: print(f"Prep took {prepped - start}")

    # Modeling
    best_loss, best_state, bad_epochs_count = float('inf'), None, 0
    print("      |                 LOSS                   |           PREDICTIONS         |         TARGET             ")
    print("EPOCH |    total     esr        dc     pos neg |     mean      std      skew   |    mean       std      skew")
    for epoch in range(epochs):
        e_time = datetime.datetime.now()

        # Training
        model.train()
        for batch in train_dataset.batches_of_random(): # batched segments of no specified track or ts
            # get 
            apply_gain=True
            features_tensors, target_tensors, gains = batch.get_tensors(device, param_names, param_configs, apply_gain=apply_gain) # adds gain param to segments
            print(f"set {gains.shape}")
            pred, _ = model(features_tensors, None)
            if apply_gain: pred = pred / gains[:, None, None]   # (batch,) -> (batch, 1, 1), broadcasts against pred's (batch, seq_len, 1)

            pred_for_loss = pred[:, warmup_samples:, :]
            target_for_loss = target_tensors[:, warmup_samples:, :]
            loss, _, _, _ = combined_loss(pred_for_loss, target_for_loss, batch_size, dc_weight=0.5, pos_neg_weight=0.2, segment_size=train_dataset.segment_size)

            # Backprop
            optimizer.zero_grad() # clear old gradients
            loss.backward() # compute fresh gradients for this accumulated window only
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # cap their magnitude
            optimizer.step() # apply them to the weights

        t_time =  datetime.datetime.now()
        if verbose_time: print(f"Train time: {t_time - e_time}")

        model.eval()
        num_tracks = len(validation_dataset.tracks)
        with torch.no_grad():
        # Predictions
            eval_states = None
            eval_preds = [[] for _ in range(num_tracks)]
            eval_targets = [[] for _ in range(num_tracks)]

            # Per Track group
            batch_groups = validation_dataset.make_window_batches(batch_size=batch_size)
            for batches in batch_groups:
                for batch in batches:
                    input_batch, target_batch, gains = batch.get_tensors(device,param_names,param_configs, apply_gain=apply_gain)
                    pred_batch, eval_states = model(input_batch, eval_states)
                    if apply_gain: pred_batch = pred_batch / gains[:, None, None]
                    # Save pred, tgt in memory structure
                    for i, track in enumerate(batch):
                        eval_preds[i].append(pred_batch[i:i+1])
                        eval_targets[i].append(target_batch[i:i+1])

            p_time =  datetime.datetime.now()
            if verbose_time: print(f"Prediction time: {p_time - t_time}")


        # Validation
            eval_losss = eval_esrs = eval_dcs = eval_pns = pSkewnesss = pPn_rmss = pP999Overp001s = tSkewnesss = tPn_rmss = tP999Overp001s = pMeans = pStds = tMeans = tStds = 0.0
            for p in range(num_tracks):
                eval_pred_p = torch.cat(eval_preds[p], dim=1)
                eval_target_p = torch.cat(eval_targets[p], dim=1)
                eval_loss, eval_esr, eval_dc, eval_pn = combined_loss(eval_pred_p, eval_target_p, batch_size, dc_weight=0.5, pos_neg_weight=0.2, segment_size=validation_dataset.segment_size)
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
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs_count += 1
            if bad_epochs_count > lr_patience:
                # Restore best model weights
                model.load_state_dict(best_state)
                # change optimizer lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_factor
                bad_epochs_count = 0
                print(f"Plateaued: Restored best weights (L = {best_loss:+9.3f}) with LR {optimizer.param_groups[0]['lr']:.2e}")


    # Get best model and save
    model.load_state_dict(best_state)
    torch.save(best_state, f'{model_v_output_dir}/model_best.pt')
    return model

##########################################################################################################################################
##########################################################################################################################################

if __name__ == '__main__':

    param_names=["d", "f", "v"],
    param_configs={
        'd':{'min':1, 'max':7, 'dtype':torch.float32},
        'f':{'min':1, 'max':7, 'dtype':torch.float32},
        'v':{'min':1, 'max':7, 'dtype':torch.float32},
    }
    chunk_seconds=0.03
    trim_noise=True
    silent_lead_in_seconds=8

    # fit gain finder model: ["d", "f", "v"] -> G by segment
    # gain_model = GainModel(param_configs=param_configs)
    # full_dataset = DataSet(
    #     '/home/ubuntu/dsp-modeler/data/outputs/odds-50.jsonl', 
    #     '/home/ubuntu/dsp-modeler/data/input/input.wav', 
    #     '/home/ubuntu/dsp-modeler/data/outputs', 
    #     0.03, 
    #     ["d", "f", "v"], 
    #     {'d':{'min':1, 'max':7, 'dtype':torch.float32},'f':{'min':1, 'max':7, 'dtype':torch.float32},'v':{'min':1, 'max':7, 'dtype':torch.float32}}, 
    #     silent_lead_in_seconds=8, 
    #     trim_noise = True
    # )
    # print("Fit with X-Validation")
    # gain_model.cross_validate(full_dataset)
    # gain_model.save(f'/home/ubuntu/dsp-modeler/black_box/model/models/gain_model')
    gain_model = GainModel(param_configs)
    gain_model.load('/home/ubuntu/dsp-modeler/black_box/model/models/gain_model/2026-08-18_20-17/gain_model.npz')

    example_dataset = DataSet(
        '/home/ubuntu/dsp-modeler/black_box/data/train_single/manifest.jsonl', 
        '/home/ubuntu/dsp-modeler/data/input/input.wav', 
        '/home/ubuntu/dsp-modeler/data/outputs', 
        chunk_seconds, 
        param_names, 
        param_configs, 
        silent_lead_in_seconds=silent_lead_in_seconds, 
        trim_noise = trim_noise
    )
    example_dataset.apply_gain_model(gain_model)

    train_manifest(
        example_dataset,
        example_dataset,
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
        model_output_dir='/home/ubuntu/dsp-modeler/black_box/model/models/transform_model',
        lr_patience = 6,
        lr_factor = 0.5,
        batch_size=30,
        hidden_size=20,
        verbose_time=False,
        verbose_performance = False
    )
    # find noise profile model: ["d", "f", "v"] -> n_p by segment


    # Load data (with removed Noise - noise profile loves on segments)
    # train_dataset = DataSet(
    #     '/home/ubuntu/dsp-modeler/black_box/data/validation/manifest.jsonl', 
    #     '/home/ubuntu/dsp-modeler/data/input/input.wav', 
    #     '/home/ubuntu/dsp-modeler/data/outputs', 
    #     chunk_seconds, 
    #     param_names, 
    #     param_configs, 
    #     silent_lead_in_seconds=silent_lead_in_seconds, 
    #     trim_noise = trim_noise
    # )
    # validation_dataset = DataSet(
    #     '/home/ubuntu/dsp-modeler/black_box/data/train/manifest-5.jsonl', 
    #     '/home/ubuntu/dsp-modeler/data/input/input.wav', 
    #     '/home/ubuntu/dsp-modeler/data/outputs', 
    #     chunk_seconds, 
    #     param_names, 
    #     param_configs, 
    #     silent_lead_in_seconds=silent_lead_in_seconds, 
    #     trim_noise = trim_noise
    # )
    # train_manifest(
    #     train_dataset,
    #     validation_dataset,
    #     learning_rate=5e-4,
    #     epochs=20,
    #     warmup_samples=1000, # only applied to the first chunk -- state is cold there; every later chunk inherits an already-"settled" hidden state
    #     param_names=["d", "f", "v"],
    #     param_configs={
    #         'd':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'f':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'v':{'min':1, 'max':7, 'dtype':torch.float32},
    #     },
    #     device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    #     model_output_dir=f'/home/ubuntu/dsp-modeler/black_box/model/models',
    #     lr_patience = 6,
    #     lr_factor = 0.5,
    #     batch_size=30,
    #     hidden_size=20,
    #     verbose_time=False,
    #     verbose_performance = False
    # )

    # train_manifest(
    #     wet_dir='/home/ubuntu/dsp-modeler/data/outputs',
    #     learning_rate=5e-4,
    #     epochs=20,
    #     warmup_samples=1000, # only applied to the first chunk -- state is cold there; every later chunk inherits an already-"settled" hidden state
    #     silent_lead_in_seconds=8,
    #     chunk_seconds=0.03,
    #     param_names=["d", "f", "v"],
    #     param_configs={
    #         'd':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'f':{'min':1, 'max':7, 'dtype':torch.float32},
    #         'v':{'min':1, 'max':7, 'dtype':torch.float32},
    #     },
    #     train_manifest='/home/ubuntu/dsp-modeler/black_box/data/train/manifest.jsonl',
    #     validation_manifest='/home/ubuntu/dsp-modeler/black_box/data/validation/manifest-15.jsonl',
    #     dry_file='/home/ubuntu/dsp-modeler/data/input/input.wav',
    #     device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'),
    #     model_output_dir = f'/home/ubuntu/dsp-modeler/black_box/model/models',
    #     trim_noise=True,
    #     lr_patience = 6,
    #     lr_factor = 0.5,
    #     verbose_time=False,
    #     verbose_performance = False,
    #     hidden_size=40
    # )

    # track_data_model = TrackDataModel2(k=5, bandwidth=0.5)
    # track_data_model.train(train_dataset)
    # track_data_model.save(f'/home/ubuntu/dsp-modeler/black_box/model/track_models')
    # track_data_model.validate(validation_dataset)