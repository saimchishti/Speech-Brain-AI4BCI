"""Aggregate scores -> results/per_participant.csv, results/main_table.csv (+ paired tests), gate/drop decisions.
Main claims: CLEAN participants (contamination rule 1 not flagged). All-10 numbers also written."""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONT = pd.read_csv(f'{ROOT}/results/contamination/summary.csv').set_index('pt')['flag_rule1_perm']
METRICS = ['spec_r', 'spec_r_speech', 'stoi', 'estoi', 'mcd', 'wid_top1', 'wid_top5', 'cer', 'spk_cos', 'tok_acc_l1', 'tok_acc_all']
ARMS = ['B0', 'B1', 'UC', 'UD', 'C']
ORDER = ARMS + [a + '_shuf' for a in ARMS] + ['UD_maj', 'C_maj', 'ORACLE_B0', 'ORACLE_B1', 'ORACLE_UC', 'ORACLE_UD', 'ORACLE_C', 'GT']
ROLE = lambda c: 'arm' if c in ARMS else 'floor' if c.endswith(('_shuf', '_maj')) else \
    'ceiling (decoder oracle: GT target -> decoder)' if c.startswith('ORACLE') else 'ceiling (GT audio)' if c == 'GT' else 'robustness'


_VALID = {}


def stoi_valid(pt):
    """pystoi returns ~0 when a clip has < 30 non-silent frames (silence defined on the GT reference) -> same trials for
    every condition; excluded from STOI/ESTOI means. Valid = STOI(GT, GT) >= .99."""
    if pt not in _VALID:
        g = np.load(f'{ROOT}/results/scores/{pt}/GT.npz'); _VALID[pt] = dict(zip(g['idx'], g['stoi'] >= 0.99))
    return _VALID[pt]


def load(pt, cond):
    s = dict(np.load(f'{ROOT}/results/scores/{pt}/{cond}.npz'))
    ok = np.array([stoi_valid(pt).get(j, False) for j in s['idx']])
    s['stoi'] = np.where(ok, s['stoi'], np.nan); s['estoi'] = np.where(ok, s['estoi'], np.nan)
    r = dict(pt=pt, cond=cond, spec_r=s['spec_r_fold'].mean(), spec_r_fold_sd=s['spec_r_fold'].std(),
             stoi=np.nanmean(s['stoi']), estoi=np.nanmean(s['estoi']), stoi_valid_frac=ok.mean(), mcd=s['mcd'].mean(),
             wid_top1=(s['wid_rank'] == 0).mean(), wid_top5=(s['wid_rank'] < 5).mean(), cer=s['cer'].mean(),
             spk_cos=s['spk_cos'].mean() if 'spk_cos' in s else np.nan, n_scored=len(s['idx']))
    if 'spec_r_native_fold' in s: r['spec_r_native'] = s['spec_r_native_fold'].mean()
    if 'spec_r_speech_fold' in s: r['spec_r_speech'] = np.nanmean(s['spec_r_speech_fold'])
    if 'spec_r_speech_native_fold' in s: r['spec_r_speech_native'] = np.nanmean(s['spec_r_speech_native_fold'])
    if 'tok_acc_lvl' in s: r['tok_acc_l1'] = s['tok_acc_lvl'][0]; r['tok_acc_all'] = s['tok_acc_lvl'].mean()
    return r, s


def fold_vals(s, metric, n):
    """per-fold values for fold-level tests (gate)"""
    from arms import folds
    if metric == 'spec_r': return s['spec_r_fold']
    v = (s['wid_rank'] == 0).astype(float) if metric == 'wid_top1' else s[metric]
    pos = {j: i for i, j in enumerate(s['idx'])}
    return np.array([np.mean([v[pos[j]] for j in te if j in pos]) if any(j in pos for j in te) else np.nan for _, _, te in folds(n)])


def gate_test(pt, arm):
    """drop rule: arm 'at floor' if neither Whisper top-1 nor spec r beats <arm>_shuf (one-sided Wilcoxon over 10 folds, p<.05)"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    n = len(np.load(f'{ROOT}/data/features/{pt}_trials.npz')['words'])
    _, a = load(pt, arm); _, f = load(pt, arm + '_shuf')
    out = {}
    for m in ['wid_top1', 'spec_r']:
        x, y = fold_vals(a, m, n), fold_vals(f, m, n)
        d = x - y
        p = wilcoxon(d, alternative='greater').pvalue if np.any(d != 0) else 1.0
        out[m] = dict(arm=float(np.nanmean(x)), floor=float(np.nanmean(y)), p=float(p))
    out['at_floor'] = all(out[m]['p'] >= 0.05 for m in ['wid_top1', 'spec_r'])
    return out


def main(pts=None):
    rows = []
    for d in sorted(glob.glob(f'{ROOT}/results/scores/sub-*')):
        pt = os.path.basename(d)
        if pts and pt not in pts: continue
        for f in sorted(glob.glob(f'{d}/*.npz')):
            rows.append(load(pt, os.path.basename(f)[:-4])[0])
    df = pd.DataFrame(rows); df['contaminated'] = df.pt.map(CONT); df.insert(2, 'role', df.cond.map(ROLE))
    df.to_csv(f'{ROOT}/results/per_participant.csv', index=False)
    tabs = []
    for subset, dd in [('clean', df[~df.contaminated]), ('all', df)]:
        for c in [c for c in ORDER if c in set(dd.cond)]:
            x = dd[dd.cond == c]; r = dict(subset=subset, cond=c, role=ROLE(c), n_participants=len(x))
            for m in METRICS + ['spec_r_native', 'spec_r_speech_native']:
                if m in x and x[m].notna().any(): r[m] = f'{x[m].mean():.3f} ± {x[m].std(ddof=0):.3f}'
            tabs.append(r)
    pd.DataFrame(tabs).to_csv(f'{ROOT}/results/main_table.csv', index=False)
    # paired tests across CLEAN participants
    tests = []
    pairs = [(a, a + '_shuf') for a in ARMS] + [('UD', 'UD_maj'), ('C', 'C_maj'), ('UC', 'B1'), ('UC', 'B0'), ('B1', 'B0'),
             ('UD', 'UC'), ('C', 'UC'), ('UD', 'C'), ('UD', 'B1'), ('C', 'B1')]
    cl = df[~df.contaminated]
    for a, b in pairs:
        for m in ['spec_r', 'spec_r_speech', 'stoi', 'estoi', 'mcd', 'wid_top1', 'cer', 'spk_cos']:
            A = cl[cl.cond == a].set_index('pt')[m]; B = cl[cl.cond == b].set_index('pt')[m]
            common = A.index.intersection(B.index)
            if len(common) < 3 or A[common].isna().any() or B[common].isna().any(): continue
            d = A[common] - B[common]
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            tests.append(dict(a=a, b=b, metric=m, n=len(common), mean_a=A[common].mean(), mean_b=B[common].mean(),
                              mean_diff=d.mean(), wins_a=int((d > 0).sum()), p_wilcoxon_2s=p))
    pd.DataFrame(tests).to_csv(f'{ROOT}/results/paired_tests.csv', index=False)
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--gate', default=None); a = ap.parse_args()
    df = main()
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(df.drop(columns=['spec_r_fold_sd']).round(3).to_string(index=False))
    if a.gate:
        for arm in ['UD', 'C', 'B1', 'UC', 'B0']:
            try: print('GATE', arm, json.dumps(gate_test(a.gate, arm)))
            except FileNotFoundError: pass
