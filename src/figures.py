"""Figures: main bar chart with floors/ceilings, discrete-vs-continuous, spectrogram examples. Clean participants only."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, librosa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f'{ROOT}/figures'; os.makedirs(FIG, exist_ok=True)
ARMS = ['B0', 'B1', 'UC', 'UD', 'C']
COL = dict(B0='#2a78d6', B1='#eb6834', UC='#1baf7a', UD='#eda100', C='#e87ba4')     # reference palette slots 1-5, fixed order
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
LABEL = dict(B0='B0 linear→mel\n(Griffin-Lim)', B1='B1 net→mel\n(HiFi-GAN)', UC='UC Mimi\ncontinuous', UD='UD Mimi\ndiscrete', C='C cascade\n(L1 tokens)')
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})


def style(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def main_bars(df):
    mets = [('spec_r', 'Spectral r (all frames)'), ('spec_r_speech', 'Spectral r (speech frames)'), ('estoi', 'ESTOI'),
            ('wid_top1', 'Whisper word ID top-1\n(chance = 0.01)'), ('spk_cos', 'Speaker similarity\n(ECAPA cosine)')]
    fig, axs = plt.subplots(1, len(mets), figsize=(15, 3.6))
    pts = sorted(df.pt.unique())
    for ax, (m, title) in zip(axs, mets):
        for i, a in enumerate(ARMS):
            v = df[df.cond == a].set_index('pt')[m].reindex(pts)
            ax.bar(i, v.mean(), 0.7, color=COL[a], edgecolor='white', linewidth=2, zorder=2)
            ax.scatter(np.full(len(v), i) + np.linspace(-.18, .18, len(v)), v, s=9, color=INK, alpha=.55, zorder=3, lw=0)
            f = df[df.cond == a + '_shuf'][m].mean()
            ax.plot([i - .38, i + .38], [f, f], color=INK, lw=2, zorder=4, solid_capstyle='butt')
            c = df[df.cond == 'ORACLE_' + a][m].mean()
            ax.scatter([i], [c], marker='D', s=26, facecolor='white', edgecolor=INK, lw=1.2, zorder=5)
        if m == 'wid_top1': ax.axhline(0.01, color=INK2, ls=':', lw=1)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS); ax.set_title(title, fontsize=9, color=INK); style(ax)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(color=COL[a], label=LABEL[a].replace('\n', ' ')) for a in ARMS] + \
        [Line2D([], [], color=INK, lw=2, label='shuffled-brain floor (same arm)'),
         Line2D([], [], marker='D', ls='', mfc='white', mec=INK, label='decoder ceiling (GT target → decoder)'),
         Line2D([], [], marker='o', ls='', color=INK, alpha=.55, ms=3, label='participants')]
    fig.legend(handles=h, loc='lower center', ncol=8, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    n = df.pt.nunique()
    fig.suptitle(f'Every arm beats its shuffled floor on spectral r; none yields identifiable words '
                 f'(clean participants, n={n}; bar = mean)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    for ext in ('png', 'pdf'): fig.savefig(f'{FIG}/main_bars.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def disc_vs_cont(df):
    """paired per-participant: gain over own shuffled floor, UC (continuous) vs UD (discrete) vs C (cascade)"""
    mets = [('spec_r_speech', 'Spectral r, speech frames'), ('estoi', 'ESTOI'), ('stoi', 'STOI'), ('spk_cos', 'Speaker similarity')]
    fig, axs = plt.subplots(1, len(mets), figsize=(12, 3.3))
    pts = sorted(df.pt.unique()); arms = ['UC', 'UD', 'C']
    for ax, (m, title) in zip(axs, mets):
        G = np.stack([(df[df.cond == a].set_index('pt')[m] - df[df.cond == a + '_shuf'].set_index('pt')[m]).reindex(pts).values
                      for a in arms])
        for j in range(len(pts)): ax.plot(range(3), G[:, j], color=INK2, lw=0.8, alpha=.6, zorder=2)
        for i, a in enumerate(arms):
            ax.scatter(np.full(len(pts), i), G[i], s=22, color=COL[a], edgecolor='white', lw=1, zorder=3)
            ax.plot([i - .25, i + .25], [np.nanmean(G[i])] * 2, color=INK, lw=2, zorder=4)
        ax.axhline(0, color=INK2, lw=1, ls=':')
        ax.set_xticks(range(3)); ax.set_xticklabels(['UC\ncontinuous', 'UD\ndiscrete', 'C\ncascade']); ax.set_xlim(-.5, 2.5)
        ax.set_title(f'{title}\n(arm − its shuffled floor)', fontsize=9, color=INK); style(ax)
    fig.suptitle(f'Continuous vs discrete Mimi targets: gain over the shuffled-brain floor per participant (n={len(pts)}; bar = mean)',
                 fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ('png', 'pdf'): fig.savefig(f'{FIG}/discrete_vs_continuous.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def spectrograms(clean_pts, seed=0):
    rng = np.random.default_rng(seed)
    picks = []
    for _ in range(3):
        pt = clean_pts[rng.integers(len(clean_pts))]
        n = len(np.load(f'{ROOT}/data/features/{pt}_trials.npz')['words']); picks.append((pt, int(rng.integers(n))))
    rows = [('GT', 'Ground truth'), ('B1', 'B1 mel + HiFi-GAN'), ('UC', 'UC Mimi continuous'), ('UD', 'UD Mimi discrete')]
    fig, axs = plt.subplots(len(rows), 3, figsize=(10, 7), sharex=True, sharey=True)
    for c, (pt, j) in enumerate(picks):
        D = np.load(f'{ROOT}/data/features/{pt}_trials.npz'); word = D['words'][j]
        for r, (cond, lab) in enumerate(rows):
            w = D['audio16'][j] if cond == 'GT' else np.load(f'{ROOT}/results/preds/{pt}/{cond}.npz')['wav16'][j]
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=w.astype(np.float32), sr=16000, n_fft=1024, hop_length=160, n_mels=80), ref=1.0)
            ax = axs[r, c]
            ax.imshow(S, origin='lower', aspect='auto', cmap='magma', extent=(0, len(w) / 16000, 0, 80), vmin=S.max() - 80, vmax=S.max())
            if r == 0: ax.set_title(f'{pt}, trial {j}: "{word}"', fontsize=9, color=INK)
            if c == 0: ax.set_ylabel(f'{lab}\nmel bin', fontsize=8)
            if r == len(rows) - 1: ax.set_xlabel('time from cue (s)')
    fig.suptitle('Spectrogram examples, 3 random test words (seed 0; clean participants; held-out folds)', fontsize=10, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('png', 'pdf'): fig.savefig(f'{FIG}/spectrogram_examples.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return picks


if __name__ == '__main__':
    df = pd.read_csv(f'{ROOT}/results/per_participant.csv'); df = df[~df.contaminated]
    main_bars(df); disc_vs_cont(df); print(spectrograms(sorted(df.pt.unique())))
