"""Train/predict all arms for one participant, 10-fold trial-level CV (contiguous folds, as the repo's non-shuffled KFold).

Fold k: test = fold k, val = fold (k+1)%10 (early stopping / model selection / gate), train = remaining 8 folds.
Arms:  B0 (PCA50 + linear -> 23-bin logMel, repo features; Griffin-Lim), B1 (net -> mel80 -> HiFi-GAN),
       UC (net -> Mimi pre-quant latent, MSE; decoded on Mimi's unquantized path), UD (net -> 8 Mimi codebooks, CE,
       level-1 weight 3), C (net -> level-1 tokens; levels 2-8 from same-speaker TRAINING frames, see cascade_lookup).
Floors: <arm>_shuf (brain<->audio pairing permuted within train+val, same procedure), UD_maj / C_maj (majority token).
Output: results/preds/{pt}/{arm}.npz  wav16 (N,32000) in trial order, plus tokens / native predictions where relevant.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import codecs_util as cu
import MelFilterBank as mel_fb
import reconstructWave as rW

ROOT = cu.ROOT
NF = 10
LVL1_W = 3.0
HP = dict(d=128, drop_in=0.2, drop=0.3, lr=1e-3, wd=1e-2, bs=16, max_ep=300, patience=40)   # fixed a priori, not tuned


def folds(n):
    b = np.array_split(np.arange(n), NF)
    out = []
    for k in range(NF):
        te, va = b[k], b[(k + 1) % NF]
        tr = np.setdiff1d(np.arange(n), np.concatenate([te, va]))
        out.append((tr, va, te))
    return out


# ---------------------------------------------------------------- B0
def griffin_lim_23(spec):
    """repo createAudio for one trial: logMel(23) -> linear -> Griffin-Lim (fftsize 800, overlap 5)"""
    mfb = mel_fb.MelFilterBank(401, 23, 16000)
    s = mfb.fromLogMels(spec)
    np.random.seed(0)                                   # repo's GL uses an unseeded random phase init
    w = rW.reconstructWavFromSpectrogram(s, s.shape[0] * s.shape[1], fftsize=800, overlap=5)
    w = np.asarray(w, np.float32)
    w = w / (np.abs(w).max() + 1e-8) * 0.5
    out = np.zeros(32000, np.float32); out[:min(32000, len(w))] = w[:32000]
    return out


def run_b0(pt, D, shuffle=False, seed=0):
    feat = np.load(f'{ROOT}/data/features/{pt}_feat.npy')
    if D.get('keep_ch') is not None:      # stacked features are flatten((9 offsets, C)) -> column t*C + c
        C = len(D['channels']); feat = feat[:, (np.arange(9)[:, None] * C + D['keep_ch'][None]).ravel()]
    spec = np.load(f'{ROOT}/data/features/{pt}_spec.npy')
    fi = D['frame_idx']; n = len(fi)
    blk = [np.arange(max(f0, 0), min(f0 + 300, len(feat))) for f0 in fi]        # trial block [onset, onset+3 s)
    tf = lambda j: np.arange(max(fi[j], 0), max(fi[j], 0) + 195)                # trial window frames [onset, onset+1.95]
    pred = np.zeros((n, 195, 23), np.float32); val_r = []
    rng = np.random.default_rng(seed)
    for k, (tr, va, te) in enumerate(folds(n)):
        # shuffled floor: brain frames of block j paired with spectrogram frames of block pi(j) (same offset from onset)
        pi = dict(zip(tr, rng.permutation(tr))) if shuffle else {j: j for j in tr}
        m = {j: min(len(blk[j]), len(blk[pi[j]])) for j in tr}
        X = np.concatenate([feat[blk[j][:m[j]]] for j in tr]); Y = np.concatenate([spec[blk[pi[j]][:m[j]]] for j in tr])
        mu, sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - mu) / sd; Z = Z - Z.mean(0)
        ev, V = np.linalg.eigh(Z.T @ Z / (len(Z) - 1)); W = V[:, ::-1][:, :50]     # = PCA(50) components (exact, up to sign)
        lr = LinearRegression().fit(((X - mu) / sd) @ W, Y)
        P = lambda idx: lr.predict(((feat[idx] - mu) / sd) @ W)
        for j in te: pred[j] = P(tf(j))
        vf = np.concatenate([tf(j) for j in va]); pv = P(vf)
        val_r.append(np.mean([np.corrcoef(spec[vf, b], pv[:, b])[0, 1] for b in range(23)]))
    wav = np.stack([griffin_lim_23(pred[j]) for j in range(n)])
    return dict(wav16=wav, spec23_native=pred, val_r=np.array(val_r))


# ---------------------------------------------------------------- neural backbone
class Net(nn.Module):
    def __init__(self, C, out, d=128, drop_in=0.2, drop=0.3, **_):
        super().__init__()
        self.drop_in = nn.Dropout1d(drop_in)
        self.inp = nn.Conv1d(C, d, 1)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2 * dl, dilation=dl) for dl in (1, 2, 4)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, d) for _ in range(3)])
        self.gru = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.out = out                     # head module: (B, T, 2d) -> prediction

    def forward(self, x):                  # x (B, 300, C)
        h = self.inp(self.drop_in(x.transpose(1, 2)))
        for c, nrm in zip(self.convs, self.norms): h = h + self.drop(F.gelu(nrm(c(h))))
        h, _ = self.gru(h.transpose(1, 2))
        return self.out(self.drop(h))


def to_frames(h, T):
    """backbone frames 0..299 cover [-0.5, 2.5) s at 10 ms; take [0, 2.0+) and resample to T target frames"""
    seg = h[:, 50:250].transpose(1, 2)                   # (B, 2d, 200) = [0, 2 s)
    if T == 25: return F.avg_pool1d(seg, 8).transpose(1, 2)                # Mimi 80 ms frames
    return F.interpolate(seg, size=T, mode='linear', align_corners=False).transpose(1, 2)


class RegHead(nn.Module):
    def __init__(self, d2, dim, T):
        super().__init__(); self.l = nn.Linear(d2, dim); self.T = T
    def forward(self, h): return self.l(to_frames(h, self.T)).transpose(1, 2)        # (B, dim, T)


class TokHead(nn.Module):
    """logits = (P_l h) . E_l^T with E_l the frozen Mimi codebook of level l (2048 x 256)"""
    def __init__(self, d2, codebooks):
        super().__init__()
        self.register_buffer('E', torch.stack(codebooks))                               # (L, 2048, 256)
        L, _, e = self.E.shape
        self.P = nn.ModuleList([nn.Linear(d2, e) for _ in range(L)])
        self.scale = nn.Parameter(torch.full((L,), 1.0))
        self.b = nn.Parameter(torch.zeros(L, self.E.shape[1]))
    def forward(self, h):
        h = to_frames(h, 25)                                                             # (B, 25, 2d)
        return torch.stack([self.scale[l] * F.normalize(self.P[l](h), dim=-1) @ F.normalize(self.E[l], dim=-1).T * 10 + self.b[l]
                            for l in range(len(self.P))], 1)                            # (B, L, 25, 2048)


def mimi_codebooks(mi, L):
    q = mi.m.quantizer
    layers = list(q.semantic_residual_vector_quantizer.layers) + list(q.acoustic_residual_vector_quantizer.layers)
    return [layers[l].codebook.embed.detach().float().cpu() for l in range(L)]


def train_net(make, X, Y, tr, va, loss_fn, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=HP['lr'], weight_decay=HP['wd'])
    Xt, Yt = X[tr].cuda(), Y[tr].cuda(); Xv, Yv = X[va].cuda(), Y[va].cuda()
    best, best_state, bad, ep_best = 1e9, None, 0, 0
    for ep in range(HP['max_ep']):
        net.train()
        for b in torch.randperm(len(tr)).split(HP['bs']):
            loss = loss_fn(net(Xt[b]), Yt[b]); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = loss_fn(net(Xv), Yv).item()
        if vl < best: best, bad, ep_best, best_state = vl, 0, ep, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad > HP['patience']: break
    net.load_state_dict(best_state); net.eval()
    return net, best, ep_best


def cascade_lookup(codes_tr, sem_pred, E1):
    """Acoustic levels 2-8 for each predicted level-1 token: the most frequent (lvl2..8) tuple co-occurring with that
    token in the same speaker's TRAINING frames; unseen tokens -> nearest seen token in level-1 codebook space."""
    sem = codes_tr[:, 0].reshape(-1); ac = codes_tr[:, 1:].transpose(0, 2, 1).reshape(-1, codes_tr.shape[1] - 1)
    table = {}
    for s in np.unique(sem):
        tup, cnt = np.unique(ac[sem == s], axis=0, return_counts=True); table[s] = tup[np.argmax(cnt)]
    seen = np.array(sorted(table)); En = F.normalize(E1, dim=-1)
    out = np.zeros((sem_pred.shape[0], codes_tr.shape[1], sem_pred.shape[1]), np.int64); out[:, 0] = sem_pred
    for i in range(sem_pred.shape[0]):
        for t in range(sem_pred.shape[1]):
            s = sem_pred[i, t]
            if s not in table: s = seen[torch.argmax(En[seen] @ En[s]).item()]
            out[i, 1:, t] = table[s]
    return out


def run_neural(arm, pt, D, T, mi, hg, shuffle=False):
    hgx = torch.tensor(D['hg']); n, _, C = hgx.shape
    codes = torch.tensor(T['mimi_codes'].astype(np.int64))
    if arm == 'B1': Y = torch.tensor(T['mel80']); dim = 80
    elif arm == 'UC': Y = torch.tensor(T['mimi_lat']); dim = 512
    elif arm == 'UD': Y = codes; L = cu.NQ
    elif arm == 'C': Y = codes[:, :1]; L = 1
    cbs = mimi_codebooks(mi, cu.NQ) if arm in ('UD', 'C') else None
    preds, tok, info = [None] * n, np.zeros((n, cu.NQ, 25), np.int64), []
    for k, (tr, va, te) in enumerate(folds(n)):
        trva = np.concatenate([tr, va])
        mu = hgx[tr].mean((0, 1)); sd = hgx[tr].std((0, 1)) + 1e-6
        X = (hgx - mu) / sd
        Yk, codes_k = Y.clone(), codes.clone()
        if shuffle:   # brain of trial j paired with the audio targets of trial pi(j), within train+val only
            p = np.random.default_rng(100 + k).permutation(trva)
            Yk[trva] = Y[p]; codes_k[trva] = codes[p]
        if arm in ('B1', 'UC'):
            ym, ys = Yk[tr].mean((0, 2), keepdim=True), Yk[tr].std((0, 2), keepdim=True) + 1e-6
            Yn = (Yk - ym) / ys
            make = lambda: Net(C, RegHead(2 * HP['d'], dim, Y.shape[-1]), **HP)
            loss_fn = lambda p, y: F.mse_loss(p, y)
            net, vl, ep = train_net(make, X, Yn, tr, va, loss_fn, seed=k)
            with torch.no_grad(): P = net(X[te].cuda()).cpu() * ys + ym
            if arm == 'B1': W = hg.wav(P).cpu()
            else: W = torchaudio.functional.resample(mi.decode_latents(P).cpu(), 24000, 16000)
        else:
            w = torch.tensor([LVL1_W] + [1.0] * (L - 1)).cuda()
            make = lambda: Net(C, TokHead(2 * HP['d'], cbs[:L]), **HP)
            loss_fn = lambda p, y: (torch.stack([F.cross_entropy(p[:, l].reshape(-1, p.shape[-1]), y[:, l].reshape(-1))
                                                 for l in range(L)]) * w).sum() / w.sum()
            net, vl, ep = train_net(make, X, Yk, tr, va, loss_fn, seed=k)
            with torch.no_grad(): pt_tok = net(X[te].cuda()).argmax(-1).cpu().numpy()       # (nte, L, 25)
            full = pt_tok if arm == 'UD' else cascade_lookup(codes_k[tr].numpy(), pt_tok[:, 0], cbs[0])
            tok[te] = full
            W = torchaudio.functional.resample(mi.decode_codes(torch.tensor(full)).cpu(), 24000, 16000)
        for i, j in enumerate(te): preds[j] = W[i, :32000].numpy()
        info.append(dict(fold=k, val_loss=vl, best_epoch=ep))
    out = dict(wav16=np.stack(preds).astype(np.float32), info=json.dumps(info))
    if arm in ('UD', 'C'): out['tokens'] = tok
    return out


def run_majority(arm, pt, D, T, mi):
    codes = T['mimi_codes'].astype(np.int64); n = len(codes)
    tok = np.zeros((n, cu.NQ, 25), np.int64)
    cbs = mimi_codebooks(mi, cu.NQ)
    for k, (tr, va, te) in enumerate(folds(n)):
        maj = np.array([np.bincount(codes[tr, l].ravel()).argmax() for l in range(cu.NQ)])
        if arm == 'UD_maj': tok[te] = maj[None, :, None]
        else:   # C_maj: majority level-1 token, acoustic levels via the same training-frame lookup as C
            sem = np.full((len(te), 25), maj[0]); tok[te] = cascade_lookup(codes[tr], sem, cbs[0])
    W = torchaudio.functional.resample(mi.decode_codes(torch.tensor(tok)).cpu(), 24000, 16000)[:, :32000]
    return dict(wav16=W.numpy().astype(np.float32), tokens=tok)


def oracles(pt, D, T, mi, hg):
    """Decoder ceilings from GT targets (not brain): HiFi-GAN(GT mel80), Mimi(GT latent), Mimi(GT 8 codes),
    cascade(GT level-1 + training-frame acoustic lookup), Griffin-Lim(GT 23-bin)."""
    codes = T['mimi_codes'].astype(np.int64); n = len(codes); cbs = mimi_codebooks(mi, 1)
    r = lambda w: torchaudio.functional.resample(w.cpu(), 24000, 16000)[:, :32000].numpy()
    casc = np.zeros_like(codes)
    for k, (tr, va, te) in enumerate(folds(n)): casc[te] = cascade_lookup(codes[tr], codes[te, 0], cbs[0])
    import swp_features as sf
    gl = np.stack([griffin_lim_23(sf.extractMelSpecs(D['audio16'][j] * 32767, 16000)) for j in range(n)])
    return {'ORACLE_B1': hg.wav(torch.tensor(T['mel80'])).cpu().numpy()[:, :32000],
            'ORACLE_UC': r(mi.decode_latents(torch.tensor(T['mimi_lat']))),
            'ORACLE_UD': r(mi.decode_codes(torch.tensor(codes))),
            'ORACLE_C': r(mi.decode_codes(torch.tensor(casc))),
            'ORACLE_B0': gl}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pt'); ap.add_argument('--arms', default='B0,B1,UC,UD,C,B0_shuf,B1_shuf,UC_shuf,UD_shuf,C_shuf,UD_maj,C_maj,ORACLE')
    ap.add_argument('--drop-hg', action='store_true', help='robustness: remove channels with HG-band (70-170 Hz) contamination '
                    'p < 0.05/n_ch; outputs saved as <arm>_noHG')
    a = ap.parse_args()
    od = f'{ROOT}/results/preds/{a.pt}'; os.makedirs(od, exist_ok=True)
    D = dict(np.load(f'{ROOT}/data/features/{a.pt}_trials.npz')); T = dict(np.load(f'{ROOT}/data/features/{a.pt}_targets.npz'))
    D['keep_ch'], suffix = None, ''
    if a.drop_hg:
        ph = np.load(f'{ROOT}/results/contamination/{a.pt}.npz')['p_hg']; nch = len(ph)
        D['keep_ch'] = np.where(ph >= 0.05 / nch)[0]; suffix = '_noHG'
        D['hg'] = D['hg'][:, :, D['keep_ch']]
        print(a.pt, 'drop-hg: removed channels', np.where(ph < 0.05 / nch)[0].tolist(), flush=True)
    mi, hg = cu.Mimi(), cu.HiFiGAN()
    for arm in [x + suffix for x in a.arms.split(',')]:
        if os.path.exists(f'{od}/{arm}.npz') or (arm == 'ORACLE' and os.path.exists(f'{od}/ORACLE_B1.npz')):
            print(a.pt, arm, 'exists, skip', flush=True); continue
        t0 = time.time()
        base, shuf = arm.replace('_noHG', '').replace('_shuf', ''), '_shuf' in arm
        if arm == 'ORACLE':
            for k, v in oracles(a.pt, D, T, mi, hg).items(): np.savez_compressed(f'{od}/{k}.npz', wav16=v.astype(np.float32))
            print(a.pt, arm, f'{time.time() - t0:.0f}s', flush=True); continue
        if base == 'B0': out = run_b0(a.pt, D, shuffle=shuf)
        elif '_maj' in arm: out = run_majority(arm, a.pt, D, T, mi)
        else:
            out = run_neural(base, a.pt, D, T, mi, hg, shuffle=shuf)
        np.savez_compressed(f'{od}/{arm}.npz', **out)
        print(a.pt, arm, f'{time.time() - t0:.0f}s', out.get('info', '')[:300] if isinstance(out.get('info', ''), str) else '',
              ('val_r %.3f' % out['val_r'].mean()) if 'val_r' in out else '', flush=True)
