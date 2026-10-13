"""Preprocessing for the speech study.

1. Repo-exact features (swp_features: HG 70-170 Hz, 50 ms win / 10 ms shift, stacking order 4 step 5, 23-bin logMel)
   -> data/features/{pt}_feat.npy, _spec.npy, _procWords.npy, _feat_names.npy  (used by B0; identical to the repo)
2. Per-trial tensors for the neural arms -> data/features/{pt}_trials.npz
   hg     (n_trials, 300, C) unstacked HG envelope, 10 ms frames, window [onset-0.5 s, onset+2.5 s]
   audio16 (n_trials, 32000)  repo-decimated 16 kHz audio, [onset, onset+2 s]
   audio24 (n_trials, 48000)  48 kHz -> 24 kHz (polyphase) for Mimi, same window
   words, onsets_s, frame_idx (index of trial onset in the repo-trimmed frame arrays)
"""
import os, sys, json
import numpy as np, pandas as pd, scipy.signal
import h5py  # NWB files read directly (pynwb/hdmf version mismatch); same datasets as the repo's NWBHDF5IO path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swp_features as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIDS = os.path.join(ROOT, 'data/raw/SingleWordProductionDutch-iBIDS')
OUT = os.path.join(ROOT, 'data/features')
winL, shift, order, step = 0.05, 0.01, 4, 5
EEG_SR, AUD_SR = 1024, 48000
PRE, POST, TRIAL = 0.5, 0.5, 2.0


def run(pt):
    with h5py.File(os.path.join(BIDS, pt, 'ieeg', f'{pt}_task-wordProduction_ieeg.nwb'), 'r') as f:
        eeg = f['acquisition/iEEG/data'][:]
        audio = f['acquisition/Audio/data'][:]
        words = np.array([w.decode() if isinstance(w, bytes) else w for w in f['acquisition/Stimulus/data'][:]], dtype=str)
    chans = np.array(pd.read_csv(os.path.join(BIDS, pt, 'ieeg', f'{pt}_task-wordProduction_channels.tsv'), sep='\t')['name'])
    ev = pd.read_csv(os.path.join(BIDS, pt, 'ieeg', f'{pt}_task-wordProduction_events.tsv'), sep='\t')
    ev = ev[(ev.trial_type == 'word') & (ev.duration > 0)].reset_index(drop=True)   # sub-10: recording ends early, trailing events have duration 0

    # ---- repo-exact pipeline ----
    hg = sf.extractHG(eeg, EEG_SR, windowLength=winL, frameshift=shift)          # (frames, C), frame w starts at w*10 ms
    feat = sf.stackFeatures(hg, modelOrder=order, stepSize=step)
    a16 = scipy.signal.decimate(audio, int(AUD_SR / 16000))
    scaled = np.int16(a16 / np.max(np.abs(a16)) * 32767)
    mel = sf.extractMelSpecs(scaled, 16000, windowLength=winL, frameshift=shift)
    lab = sf.downsampleLabels(words, EEG_SR, windowLength=winL, frameshift=shift)
    lab = lab[order * step:lab.shape[0] - order * step]
    mel = mel[order * step:mel.shape[0] - order * step, :]
    n = min(mel.shape[0], feat.shape[0]); mel, feat, lab = mel[:n], feat[:n], lab[:n]
    names = sf.nameVector(chans[:, None], modelOrder=order)
    np.save(f'{OUT}/{pt}_feat.npy', feat); np.save(f'{OUT}/{pt}_spec.npy', mel)
    np.save(f'{OUT}/{pt}_procWords.npy', lab); np.save(f'{OUT}/{pt}_feat_names.npy', names)

    # ---- per-trial tensors ----
    a24 = scipy.signal.resample_poly(audio.astype(np.float64), 1, 2)
    a16f = scaled.astype(np.float32) / 32767.
    a24 = (a24 / np.max(np.abs(a24))).astype(np.float32)
    nf = int(round((PRE + TRIAL + POST) / shift))
    npad = int(round(PRE / shift)) + 1
    hgp = np.pad(hg, ((npad, npad), (0, 0)), mode='edge')   # edge-pad so trial 0 / last trial keep their context
    H, A16, A24, keep = [], [], [], []
    for i, r in ev.iterrows():
        t0 = r.onset
        f0 = int(round((t0 - PRE) / shift))
        s16, s24 = int(round(t0 * 16000)), int(round(t0 * 24000))
        if s16 + 32000 > len(a16f) or s24 + 48000 > len(a24):
            print(pt, 'skip trial', i, r.value); continue
        H.append(hgp[f0 + npad:f0 + npad + nf]); A16.append(a16f[s16:s16 + 32000]); A24.append(a24[s24:s24 + 48000]); keep.append(i)
    ev = ev.iloc[keep]
    frame_idx = np.round(ev.onset.values / shift).astype(int) - order * step   # onset frame in trimmed repo arrays
    np.savez_compressed(f'{OUT}/{pt}_trials.npz', hg=np.stack(H).astype(np.float32), audio16=np.stack(A16),
                        audio24=np.stack(A24), words=ev.value.values.astype(str), onsets_s=ev.onset.values,
                        frame_idx=frame_idx, channels=chans.astype(str))
    info = dict(pt=pt, n_channels=int(eeg.shape[1]), eeg_s=eeg.shape[0] / EEG_SR, audio_s=len(audio) / AUD_SR,
                n_trials=len(keep), n_unique_words=int(len(set(ev.value))), n_frames=int(n))
    print(json.dumps(info), flush=True)
    return info


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    pts = sys.argv[1:] or [f'sub-{i:02d}' for i in range(1, 11)]
    infos = [run(p) for p in pts]
    if len(pts) == 10:
        pd.DataFrame(infos).to_csv(os.path.join(ROOT, 'results/dataset_summary.csv'), index=False)
