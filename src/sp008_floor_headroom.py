"""SP-008: floor-corrected arm comparisons and headroom, from saved scores only (results/per_participant.csv).
Clean participants only. No training, no GPU."""
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = ['sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-10']
METRICS = {'r_speech': 'spec_r_speech', 'STOI': 'stoi', 'ESTOI': 'estoi', 'WID': 'wid_top1', 'SV': 'spk_cos'}
ARMS = ['B0', 'B1', 'UC', 'UD', 'C']
MIN_DENOM = 0.05

df = pd.read_csv(f'{ROOT}/results/per_participant.csv')
df = df[df.pt.isin(CLEAN)]
V = lambda cond, col: df[df.cond == cond].set_index('pt')[col].reindex(CLEAN)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def wil(d):
    d = np.asarray(d, float)
    return 1.0 if np.all(d == 0) else wilcoxon(d).pvalue        # two-sided; zero differences dropped (scipy default)


# ---------------------------------------------------------------- 1. floor-corrected comparisons
rows = []
for a, b in [('UC', 'B1'), ('UC', 'UD')]:
    block = []
    for name, col in METRICS.items():
        da = V(a, col) - V(a + '_shuf', col); db = V(b, col) - V(b + '_shuf', col)
        diff = da - db
        block.append(dict(comparison=f'{a} vs {b}', metric=name, n=int(diff.notna().sum()),
                          delta_a_mean=da.mean(), delta_a_sd=da.std(ddof=0), delta_b_mean=db.mean(), delta_b_sd=db.std(ddof=0),
                          mean_diff=diff.mean(), wins_a=int((diff > 0).sum()), ties=int((diff == 0).sum()), p_raw=wil(diff)))
    for r, ph in zip(block, holm([r['p_raw'] for r in block])): r['p_holm'] = ph
    rows += block
fc = pd.DataFrame(rows)
fc.to_csv(f'{ROOT}/results/floor_corrected.csv', index=False)

# ---------------------------------------------------------------- 2. headroom
hr_rows, per = [], []
for a in ARMS:
    for name, col in METRICS.items():
        arm, shuf, orc = V(a, col), V(a + '_shuf', col), V('ORACLE_' + a, col)
        den = orc - shuf; h = (arm - shuf) / den
        na = den < MIN_DENOM; h[na] = np.nan
        for pt in CLEAN: per.append(dict(arm=a, metric=name, pt=pt, arm_val=arm[pt], shuf=shuf[pt], oracle=orc[pt],
                                         denom=den[pt], headroom=h[pt]))
        hr_rows.append(dict(arm=a, metric=name, n_used=int(h.notna().sum()), n_NA=int(na.sum()),
                            oracle_available=int(orc.notna().sum()), mean=h.mean(), sd=h.std(ddof=0), median=h.median()))
hr = pd.DataFrame(hr_rows)
hr.to_csv(f'{ROOT}/results/headroom.csv', index=False)
pd.DataFrame(per).to_csv(f'{ROOT}/results/headroom_per_participant.csv', index=False)

# ---------------------------------------------------------------- print
f3 = lambda x: 'NA' if pd.isna(x) else f'{x:.3f}'
print('### Floor-corrected comparisons (delta = arm - arm_shuf; clean n=7; paired two-sided Wilcoxon; Holm over 5 metrics)\n')
print('| comparison | metric | delta A (mean ± SD) | delta B (mean ± SD) | mean diff | A > B | p raw | p Holm |')
print('|---|---|---|---|---|---|---|---|')
for _, r in fc.iterrows():
    print(f"| {r.comparison} | {r.metric} | {r.delta_a_mean:.3f} ± {r.delta_a_sd:.3f} | {r.delta_b_mean:.3f} ± {r.delta_b_sd:.3f} | "
          f"{r.mean_diff:+.3f} | {r.wins_a}/7{' (' + str(r.ties) + ' tie)' if r.ties else ''} | {r.p_raw:.3f} | {r.p_holm:.3f} |")
print(f'\n### Headroom = (arm - shuf) / (ORACLE - shuf); NA if ORACLE - shuf < {MIN_DENOM} (clean n=7)\n')
print('| arm | ' + ' | '.join(METRICS) + ' |'); print('|---|' + '---|' * len(METRICS))
for a in ARMS:
    cells = []
    for m in METRICS:
        r = hr[(hr.arm == a) & (hr.metric == m)].iloc[0]
        cells.append(f"{f3(r['mean'])} ± {f3(r['sd'])} (med {f3(r['median'])}; n={r.n_used}{', NA=' + str(r.n_NA) if r.n_NA else ''})")
    print(f'| {a} | ' + ' | '.join(cells) + ' |')
print('\nORACLE availability per arm (participants):', dict(zip(hr.drop_duplicates('arm').arm, hr.drop_duplicates('arm').oracle_available)))
