"""B0 reproduction: exactly reconstruction_minimal.py's protocol (frame-level contiguous KFold(10), z-score on train,
PCA(50) fit on train, LinearRegression -> 23-bin logMel; Pearson r per bin per fold). Griffin-Lim synthesis skipped here
(waveforms for the comparison arms are made in the main pipeline). Random control: 1000 circular-shift rounds as in the repo."""
import os, sys, json
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, 'data/features')
pts = [f'sub-{i:02d}' for i in range(1, 11)]
rng = np.random.default_rng(0)
allres, rand = {}, {}
for pt in pts:
    spec = np.load(f'{F}/{pt}_spec.npy'); data = np.load(f'{F}/{pt}_feat.npy')
    rs = np.zeros((10, 23))
    for k, (tr, te) in enumerate(KFold(10, shuffle=False).split(data)):
        mu, sd = data[tr].mean(0), data[tr].std(0)
        Xtr, Xte = (data[tr] - mu) / sd, (data[te] - mu) / sd
        pca = PCA().fit(Xtr)
        Xtr, Xte = Xtr @ pca.components_[:50].T, Xte @ pca.components_[:50].T
        pred = LinearRegression().fit(Xtr, spec[tr]).predict(Xte)
        rs[k] = [pearsonr(spec[te, b], pred[:, b])[0] for b in range(23)]
    rr = []
    for _ in range(1000):
        sp = rng.integers(int(spec.shape[0] * .1), int(spec.shape[0] * .9))
        sh = np.concatenate((spec[sp:], spec[:sp]))
        rr.append(np.mean([np.corrcoef(spec[:, b], sh[:, b])[0, 1] for b in range(23)]))
    allres[pt] = rs; rand[pt] = np.array(rr)
    print(f'{pt} mean r {rs.mean():.3f} (fold sd {rs.mean(1).std():.3f}) | random 95th pct {np.percentile(rr, 95):.3f}', flush=True)
m = np.array([allres[p].mean() for p in pts])
print(f'ALL mean r {m.mean():.3f} +- {m.std():.3f} across participants')
os.makedirs(f'{ROOT}/results', exist_ok=True)
np.savez(f'{ROOT}/results/b0_repro.npz', **{f'{p}_r': allres[p] for p in pts}, **{f'{p}_rand': rand[p] for p in pts})
json.dump({p: float(allres[p].mean()) for p in pts}, open(f'{ROOT}/results/b0_repro.json', 'w'), indent=1)
