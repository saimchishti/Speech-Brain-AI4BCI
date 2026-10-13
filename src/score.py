"""Identical metrics for every condition (arm, floor, oracle, GT) of one participant.

Per trial (2 s clip, 16 kHz):  STOI, ESTOI (pystoi);  MCD (MFCC c1-c13 from 40-band mel, 10 ms hop, DTW-aligned);
  Whisper large-v3 (Dutch): closed-set word ID = argmax log p(word | audio) over the participant's vocabulary
  (forms ' word' / ' Word', + EOT; PMI-calibrated: minus log p(word | silent clip),
  chosen on GT audio only: GT top-1 .77 vs .70-.72 raw; rank of true word), free greedy transcription -> CER vs the true word;
  speaker similarity = ECAPA (spkrec-ecapa-voxceleb) cosine to the GT clip.
Per fold: spectrogram Pearson r = repo 23-bin logMel (50 ms / 10 ms) of output vs GT, test trials concatenated, r per bin,
  mean over bins (literature-comparable; computed from the WAVEFORM for every condition). B0 also: native predicted spectrogram.
Token accuracy per level for conditions with tokens.
--frac f: score a fixed random subset (seed by participant, same for every condition).
Output: results/scores/{pt}/{cond}.npz
"""
import os, sys, glob, argparse, re, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F, librosa, jiwer
from pystoi import stoi
import codecs_util as cu
import swp_features as sf
from arms import folds

ROOT = cu.ROOT


def spec23(w): return sf.extractMelSpecs(np.asarray(w, np.float64) * 32767, 16000)          # (195, 23)


# librosa MFCCs are DCT of dB power (10*log10 P); standard MCD uses natural-log AMPLITUDE cepstra -> scale by ln(10)/20
MCEP_SCALE = np.log(10) / 20


def mfcc(w): return MCEP_SCALE * librosa.feature.mfcc(y=np.asarray(w, np.float32), sr=16000, n_mfcc=14, n_fft=400, hop_length=160, n_mels=40)[1:]


def mcd(a, b):
    A, B = mfcc(a), mfcc(b)
    D, wp = librosa.sequence.dtw(X=A, Y=B, metric='euclidean')
    d = np.linalg.norm(A[:, wp[:, 0]] - B[:, wp[:, 1]], axis=0)
    return float(10 / np.log(10) * np.sqrt(2) * d.mean())


def norm_txt(s):
    s = unicodedata.normalize('NFKC', s).lower().strip()
    return re.sub(r"[^\w\s]", '', s).strip()


class Whisper:
    def __init__(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.p = WhisperProcessor.from_pretrained('openai/whisper-large-v3')
        self.m = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3', torch_dtype=torch.float16).cuda().eval()
        tok = self.p.tokenizer
        self.prefix = tok.convert_tokens_to_ids(['<|startoftranscript|>', '<|nl|>', '<|transcribe|>', '<|notimestamps|>'])
        self.eot = tok.eos_token_id

    def feats(self, wavs):
        f = self.p.feature_extractor(list(wavs), sampling_rate=16000, return_tensors='pt').input_features
        return f.cuda().half()

    @torch.no_grad()
    def vocab_ids(self, vocab):
        seqs, owner = [], []
        for i, w in enumerate(vocab):
            for form in (' ' + w, ' ' + w[:1].upper() + w[1:]):
                seqs.append(self.p.tokenizer(form, add_special_tokens=False).input_ids + [self.eot]); owner.append(i)
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), self.eot); mask = torch.zeros(len(seqs), L)
        for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
        return ids.cuda(), mask.cuda(), np.array(owner)

    @torch.no_grad()
    def cand_ll(self, enc, ids, mask, chunk=int(os.environ.get("WCHUNK", 16))):
        """log p(candidate tokens + EOT | audio, prefix). Prefix run once (cross-attn K/V computed once), cache repeated."""
        from transformers.cache_utils import EncoderDecoderCache
        pre = torch.tensor([self.prefix]).cuda()
        o = self.m(encoder_outputs=(enc[None],), decoder_input_ids=pre, use_cache=True)
        lp0 = torch.log_softmax(o.logits[0, -1].float(), -1)                                  # predicts first cand token
        pkv = o.past_key_values
        legacy = pkv if isinstance(pkv, tuple) else pkv.to_legacy_cache()
        out = []
        for c in range(0, len(ids), chunk):
            I, M = ids[c:c + chunk], mask[c:c + chunk]; n = len(I)
            cache = EncoderDecoderCache.from_legacy_cache(tuple(tuple(t.expand(n, -1, -1, -1).contiguous() for t in layer) for layer in legacy))
            lg = self.m(encoder_outputs=(enc[None].expand(n, -1, -1),), decoder_input_ids=I[:, :-1], past_key_values=cache).logits
            lp = torch.log_softmax(lg.float(), -1).gather(-1, I[:, 1:, None])[..., 0]
            out.append(lp0[I[:, 0]] + (lp * M[:, 1:]).sum(1))
        return torch.cat(out).cpu().numpy()

    @torch.no_grad()
    def score(self, wavs, vocab, bs=8):
        ids, mask, owner = self.vocab_ids(vocab)
        ll_all, hyp = [], []
        for i in range(0, len(wavs), bs):
            fe = self.feats(wavs[i:i + bs])
            enc = self.m.model.encoder(fe).last_hidden_state                                  # (b, 1500, 1280)
            for e in enc:
                s = self.cand_ll(e, ids, mask)
                ll_all.append(np.array([s[owner == k].max() for k in range(len(vocab))]))
            g = self.m.generate(fe, language='nl', task='transcribe', max_new_tokens=12)
            hyp += [norm_txt(t) for t in self.p.batch_decode(g, skip_special_tokens=True)]
        return np.stack(ll_all), hyp


class ECAPA:
    def __init__(self):
        from speechbrain.inference.speaker import EncoderClassifier
        self.m = EncoderClassifier.from_hparams('speechbrain/spkrec-ecapa-voxceleb', savedir=f'{ROOT}/checkpoints/cache/sb_ecapa',
                                                run_opts={'device': 'cuda'})

    @torch.no_grad()
    def emb(self, wavs):
        return torch.cat([self.m.encode_batch(torch.tensor(np.asarray(wavs[i:i + 32]))).squeeze(1) for i in range(0, len(wavs), 32)])


def score_cond(name, P, gt, D, T, W, E, idx, gt_emb):
    words = D['words']; vocab = list(words); n = len(words)
    wav = P['wav16']
    wav = np.nan_to_num(wav)
    out = dict(idx=idx)
    # spectral r per fold (all test trials of the fold, not subsetted: cheap)
    S_gt = np.stack([spec23(gt[j]) for j in range(n)]); S_p = np.stack([spec23(wav[j]) for j in range(n)])
    # speech frames: GT frame mean logMel above the midpoint of the participant's 10th/90th percentiles (fixed rule)
    e = S_gt.mean(-1); thr = 0.5 * (np.percentile(e, 10) + np.percentile(e, 90)); sp = e > thr
    rr = lambda a, b: np.nanmean([np.corrcoef(a[:, q], b[:, q])[0, 1] if b[:, q].std() > 0 else 0 for q in range(23)])
    r_f, r_nat, r_sp, r_sp_nat = [], [], [], []
    for k, (tr, va, te) in enumerate(folds(n)):
        a, b = S_gt[te].reshape(-1, 23), S_p[te].reshape(-1, 23); m = sp[te].reshape(-1)
        r_f.append(rr(a, b)); r_sp.append(rr(a[m], b[m]))
        if 'spec23_native' in P: r_sp_nat.append(rr(a[m], P['spec23_native'][te].reshape(-1, 23)[m]))
        if 'spec23_native' in P:
            c = P['spec23_native'][te].reshape(-1, 23); r_nat.append(np.mean([np.corrcoef(a[:, q], c[:, q])[0, 1] for q in range(23)]))
    out['spec_r_fold'] = np.array(r_f); out['spec_r_speech_fold'] = np.array(r_sp); out['speech_frac'] = sp.mean()
    if r_sp_nat: out['spec_r_speech_native_fold'] = np.array(r_sp_nat)
    if r_nat: out['spec_r_native_fold'] = np.array(r_nat)
    # per-trial metrics on the subset
    out['stoi'] = np.array([stoi(gt[j], wav[j], 16000) for j in idx])
    out['estoi'] = np.array([stoi(gt[j], wav[j], 16000, extended=True) for j in idx])
    out['mcd'] = np.array([mcd(gt[j], wav[j]) for j in idx])
    ll, hyp = W.score(wav[idx], vocab)
    pmi = ll - W.ll0[None]                     # PMI calibration: minus log p(word | silent 2 s clip) (removes short/frequent-word prior)
    out['wid_rank'] = np.array([(pmi[i] > pmi[i, j]).sum() for i, j in enumerate(idx)])   # 0 = correct top-1 (PRIMARY)
    out['wid_rank_raw'] = np.array([(ll[i] > ll[i, j]).sum() for i, j in enumerate(idx)])
    out['wid_ll'] = ll.astype(np.float32)
    out['hyp'] = np.array(hyp)
    out['cer'] = np.array([min(1.0, jiwer.cer(norm_txt(words[j]), h)) if h else 1.0 for j, h in zip(idx, hyp)])
    if name != 'GT':
        e = E.emb(wav[idx]); out['spk_cos'] = F.cosine_similarity(e, gt_emb, dim=-1).cpu().numpy()
    if 'tokens' in P:
        tk, gtc = P['tokens'], T['mimi_codes'].astype(np.int64)
        out['tok_acc_lvl'] = (tk == gtc).mean((0, 2))                       # per level, all test trials
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('pt'); ap.add_argument('--frac', type=float, default=1.0)
    ap.add_argument('--conds', default=None)
    a = ap.parse_args()
    D = dict(np.load(f'{ROOT}/data/features/{a.pt}_trials.npz')); T = dict(np.load(f'{ROOT}/data/features/{a.pt}_targets.npz'))
    gt = D['audio16']; n = len(gt)
    idx = np.arange(n) if a.frac >= 1 else np.sort(np.random.default_rng(int(a.pt[-2:])).choice(n, int(round(n * a.frac)), replace=False))
    od = f'{ROOT}/results/scores/{a.pt}'; os.makedirs(od, exist_ok=True)
    W, E = Whisper(), ECAPA()
    W.ll0 = W.score(np.zeros((1, 32000), np.float32), list(D['words']))[0][0]
    gt_emb = E.emb(gt[idx])
    conds = a.conds.split(',') if a.conds else ['GT'] + sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{ROOT}/results/preds/{a.pt}/*.npz'))
    for c in conds:
        f = f'{od}/{c}.npz'
        if os.path.exists(f): continue
        P = {'wav16': gt} if c == 'GT' else dict(np.load(f'{ROOT}/results/preds/{a.pt}/{c}.npz'))
        o = score_cond(c, P, gt, D, T, W, E, idx, gt_emb)
        np.savez_compressed(f, **o)
        print(a.pt, c, 'r %.3f' % o['spec_r_fold'].mean(), 'stoi %.3f' % o['stoi'].mean(), 'mcd %.2f' % o['mcd'].mean(),
              'wid@1 %.3f' % (o['wid_rank'] == 0).mean(), 'cer %.3f' % o['cer'].mean(),
              ('spk %.3f' % o['spk_cos'].mean()) if 'spk_cos' in o else '', flush=True)
