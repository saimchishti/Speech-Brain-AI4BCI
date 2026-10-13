"""Acoustic contamination check (Roussel et al. 2020, J Neural Eng 17:056028 method).

Per participant and channel: magnitude spectrograms of raw sEEG (1024 Hz) and audio (48 kHz) with the same window length
(200 ms, hop 50 ms) so frequency bins coincide (5 Hz grid). audio anti-aliased to 1024 Hz; bins 10-500 Hz, excluding +-3 Hz around 50 Hz harmonics.
Restricted to word-trial windows (where speech occurs). Contamination matrix C[f_audio, f_eeg] = Pearson r over time of
log-magnitudes. Criterion: mean of the diagonal. Null: mean of C[pi(f), f] for random frequency permutations pi
(tests whether same-frequency coupling exceeds cross-frequency coupling, which separates frequency-specific acoustic
leakage from broadband physiological speech-related activity). 10000 permutations.
Flag rule (fixed before looking at results): participant CONTAMINATED if any channel has p < 0.05/n_channels
(Bonferroni), or the channel-averaged matrix has p < 0.01. Also reported for the 70-170 Hz HG band alone.
"""
import os, sys, json
import numpy as np, pandas as pd, h5py, scipy.signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIDS = os.path.join(ROOT, 'data/raw/SingleWordProductionDutch-iBIDS')
NPERM = 10000


def spec(x, sr, win=0.2, hop=0.05):
    f, tt, Z = scipy.signal.stft(x, fs=sr, window='hann', nperseg=int(win * sr), noverlap=int(win * sr) - int(hop * sr),
                                boundary=None, padded=False, axis=0)
    return f, tt, np.log(np.abs(Z) + 1e-8)    # Z: (F, C, T) for axis=0 2-D input


def diag_test(C, rng):
    F = C.shape[-1]
    d = np.mean(np.diagonal(C, axis1=-2, axis2=-1), -1)
    perms = np.stack([rng.permutation(F) for _ in range(NPERM)])            # (NPERM, F)
    null = C[..., perms, np.arange(F)].mean(-1)                              # (..., NPERM)
    p = (1 + (null >= d[..., None]).sum(-1)) / (NPERM + 1)
    return d, p


def sharp_test(C, rng, offs=range(3, 9)):
    """Diagonal minus mean of near-off-diagonals (|offset| 3-8 bins = 15-40 Hz). Leakage is frequency-specific -> sharp
    diagonal; smooth broadband/block physiological coupling -> ~0. Null: sign-flip of per-frequency differences."""
    F = C.shape[-1]
    dg = np.diagonal(C, axis1=-2, axis2=-1)
    off = np.zeros_like(dg); cnt = np.zeros(F)
    for k in offs:
        for s in (k, -k):
            idx = np.arange(F); j = idx + s; ok = (j >= 0) & (j < F)
            off[..., idx[ok]] += C[..., idx[ok], j[ok]]; cnt[idx[ok]] += 1
    diff = dg - off / cnt                                                   # (..., F)
    stat = diff.mean(-1)
    signs = rng.choice([-1., 1.], size=(NPERM, F))
    null = np.einsum('...f,nf->...n', diff, signs) / F
    p = (1 + (null >= stat[..., None]).sum(-1)) / (NPERM + 1)
    return stat, p


def zc(a):  # z-score along last (time) axis
    a = a - a.mean(-1, keepdims=True)
    return a / (a.std(-1, keepdims=True) + 1e-12)


def run(pt, inject=0.0):
    rng = np.random.default_rng(0)
    with h5py.File(f'{BIDS}/{pt}/ieeg/{pt}_task-wordProduction_ieeg.nwb', 'r') as f:
        eeg = f['acquisition/iEEG/data'][:]; audio = f['acquisition/Audio/data'][:]
    if inject > 0:   # POSITIVE CONTROL: add audio (anti-aliased to 1024 Hz) to channel 0 at `inject` x channel RMS
        a = scipy.signal.resample_poly(audio, 16, 750)[:eeg.shape[0]]
        eeg = eeg.copy(); eeg[:len(a), 0] += inject * eeg[:, 0].std() * a / a.std()
    ev = pd.read_csv(f'{BIDS}/{pt}/ieeg/{pt}_task-wordProduction_events.tsv', sep='\t')
    ev = ev[(ev.trial_type == 'word') & (ev.duration > 0)]
    # audio anti-aliased to the sEEG rate (48000*16/750 = 1024 Hz) so both use IDENTICAL STFT frames (no hop drift)
    a1k = scipy.signal.resample_poly(audio, 16, 750)
    n = min(len(a1k), eeg.shape[0])
    fe, tt, Se = spec(eeg[:n], 1024)         # (F, C, T)
    fa, _, Sa = spec(a1k[:n], 1024)          # (F, T)
    Se = np.moveaxis(Se, 0, 1)               # (C, F, T)
    T = min(Se.shape[-1], Sa.shape[-1]); Se, Sa = Se[..., :T], Sa[..., :T]
    tc = tt[:T]                               # window centres (s)
    m = np.zeros(T, bool)
    for o, d in zip(ev.onset, ev.duration): m |= (tc >= o) & (tc < o + d)
    keep = (fe >= 10) & (fe <= 500) & np.all(np.abs(fe[:, None] - np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])[None]) > 3, 1)
    freqs = fe[keep]
    A = zc(Sa[keep][:, m]); E = zc(Se[:, keep][:, :, m])                       # (F, t), (C, F, t)
    C = np.einsum('at,cet->cae', A, E) / A.shape[-1]                         # (C, Fa, Fe)
    d, p = diag_test(C, rng)
    dm, pm = diag_test(C.mean(0), rng)
    hg = (freqs >= 70) & (freqs <= 170)
    dh, ph = diag_test(C[:, hg][:, :, hg], rng)
    sh, psh = sharp_test(C, rng)
    nch = C.shape[0]
    res = dict(pt=pt, n_channels=nch, n_freqs=int(keep.sum()), n_frames=int(m.sum()),
               mean_diag_avgmat=float(dm), p_avgmat=float(pm),
               n_ch_sig_bonf=int((p < 0.05 / nch).sum()), min_p_channel=float(p.min()), max_diag_channel=float(d.max()),
               n_ch_sig_bonf_HG=int((ph < 0.05 / nch).sum()), min_p_channel_HG=float(ph.min()),
               n_ch_p01_uncorr=int((p < 0.01).sum()),
               n_ch_sharp_bonf=int((psh < 0.05 / nch).sum()), n_ch_both_bonf=int(((p < 0.05 / nch) & (psh < 0.05 / nch)).sum()),
               max_sharp=float(sh.max()), inject=inject)
    res['flag_rule1_perm'] = bool(res['n_ch_sig_bonf'] > 0 or pm < 0.01)
    res['flag_rule2_perm_and_sharp'] = bool(res['n_ch_both_bonf'] > 0)
    if inject == 0:
        np.savez_compressed(f'{ROOT}/results/contamination/{pt}.npz', C_avg=C.mean(0), diag=d, p=p, p_hg=ph, p_sharp=psh,
                            sharp=sh, freqs=freqs, C_best=C[np.argmin(p)])
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    os.makedirs(f'{ROOT}/results/contamination', exist_ok=True)
    if sys.argv[1:2] == ['--posctrl']:
        for lvl in [0.01, 0.03, 0.1, 0.3]: run(sys.argv[2], inject=lvl)
        sys.exit()
    pts = sys.argv[1:] or [f'sub-{i:02d}' for i in range(1, 11)]
    rows = [run(p) for p in pts]
    if len(pts) == 10: pd.DataFrame(rows).to_csv(f'{ROOT}/results/contamination/summary.csv', index=False)
