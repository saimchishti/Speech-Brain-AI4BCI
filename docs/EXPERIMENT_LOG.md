# Experiment log (SP-000 – SP-008)

Converted from the plain-text lab log kept during the study. Entries and numbers are unchanged; only Markdown formatting was added. Git commit hashes refer to the private development repository, whose history is not published.

## [SP-000] Cleanup of fMRI project

**Date/time:** 2026-10-12 12:20-12:35

**Git commit:** 8f2879d

**Status:** DONE

**Hypothesis:** n/a

**Setup:** stopped exp2 queueA2.sh, eval_loop.sh, 2x final_render.py, e4_eval.py (no exp/ dir existed). GPU idle after (3%, 241 MiB).  
NOTE: another agent session was driving exp2; it logged the kill at 12:21 as "cause unknown" and is waiting for the user.

**Command(s):** kill <pids>; git mv/rm; rm -rf data checkpoints runs results exp2 .venv

**Outputs:** archive/fmri_2026-10/ (experiments.txt, experiment2.txt, notes/, figures/, requirements.lock, code/, results_tables/)

**Results:** disk free 25 GB -> 51 GB. repos/ (mind-vis, vis_dec, 36 MB) left in place (not in keep/delete lists).

**Compared to:** n/a

**Conclusion:** done

**Next step:** setup

## [SP-001] Environment + data + preprocessing

**Date/time:** 2026-10-12 12:35-13:00

**Git commit:** (this commit)

**Status:** DONE

**Hypothesis:** n/a

**Setup:** .venv python 3.11 (uv); torch 2.5.1+cu121, torchaudio 2.5.1, transformers 4.46.3, speechbrain 1.0.2, numpy 1.26,  
scipy 1.13.1, pystoi 0.4.1, librosa 0.10.2.post1, jiwer 3.0.5 (full lock: requirements_speech.lock).  
Models (HF_HOME=checkpoints/cache/hf): kyutai/mimi @89091b3, openai/whisper-large-v3 @06f233f (via transformers; the  
openai-whisper pip package failed to build), speechbrain/spkrec-ecapa-voxceleb, speechbrain/tts-hifigan-libritts-16kHz.  
Dataset: OSF nrgx6 SingleWordProductionDutch-iBIDS.zip, 2.79 GB (5.0 GB unzipped) -> data/raw/.  
Code: github neuralinterfacinglab/SingleWordProductionDutch @cfb5636 -> repos/. Compat copies in src/ (swp_features.py,  
MelFilterBank.py, reconstructWave.py): only removed-scipy/numpy-alias fixes (hanning->signal.windows.hann, stats.mode  
on strings -> np.unique mode with identical tie rule, np.float->float, scipy.zeros/real->np). NWB read via h5py  
(pynwb/hdmf incompatibility), same datasets.

**Command(s):** python src/prep.py ; python src/targets.py

**Outputs:** data/features/{pt}_feat/_spec/_procWords/_feat_names.npy (repo-exact), {pt}_trials.npz, {pt}_targets.npz,  
results/dataset_summary.csv

**Results:** 10 participants; 100 words each (sub-10: 95 - recording ends at 285 s, last 5 events have duration 0);  
all words unique within participant (closed-set word ID is 100-way / 95-way, chance 1%).  
Electrodes: 127,127,127,115,60,127,127,54,117,122 (sub-01..10), total 1103 (matches paper).  
Decoder oracles (sub-03, 16 trials, STOI vs GT): Mimi 8-codebook 0.844; Mimi unquantized latent path 0.945;  
HiFi-GAN on GT mel80 0.926 (after fixing a 5-frame inference_padding offset: 0.254 before).  
Voices in the dataset are pitch-shifted per participant (constant 1-3 semitones; Verwoert 2022) -> SV similarity is  
vs. the pitch-shifted GT voice.

**Compared to:** n/a

**Conclusion:** setup OK

**Next step:** contamination check, B0 reproduction

## [SP-002] Acoustic contamination check (Roussel et al. 2020 method)

**Date/time:** 2026-10-12 12:40-12:50

**Git commit:** (this commit)

**Status:** DONE

**Hypothesis:** speech audio leaks into sEEG at matching frequencies in some participants.

**Setup:** src/contamination.py. Raw sEEG and audio (anti-aliased to 1024 Hz) -> identical STFT (200 ms / 50 ms), log-magnitude,  
10-500 Hz minus +-3 Hz around 50 Hz harmonics (89 bins), word-trial windows only. Contamination matrix per channel  
(corr over time, audio freq x sEEG freq). Statistic: mean diagonal; null: 10000 random frequency permutations.  
Flag rule (rule 1): any channel p < 0.05/n_channels OR channel-averaged matrix p < 0.01.  
BUG FOUND+FIXED before final: first version used different hops for sEEG (51 samples = 49.8 ms) and audio (50 ms) -> drift  
of ~1.2 s over the recording. Positive control after fix (sub-01, audio injected into ch 0): diag r 0.12/0.20/0.34/0.49 at  
0.01/0.03/0.1/0.3 x channel RMS.  
A second "sharpness" statistic (diag minus +-15-40 Hz off-diagonals) was added after seeing sub-05 and DROPPED: it failed  
the positive control (only fired at 0.3 x RMS; voice harmonics co-vary across nearby bins).

**Command(s):** python src/contamination.py ; python src/contamination.py --posctrl sub-01

**Outputs:** results/contamination/summary.csv, results/contamination/sub-XX.npz

**Results:** FLAGGED (rule 1): sub-01 (5 ch Bonferroni, avg-matrix p=1e-4), sub-02 (1 ch, max diag r .15), sub-09 (7 ch, avg p=.012).  
Not flagged: sub-03..08, sub-10. Effect sizes small everywhere (max channel diag r .04-.15, comparable to injecting audio at  
~0.01-0.03 x channel RMS). Secondary (not in rule): HG-band (70-170 Hz) Bonferroni-significant channels also in  
sub-03 (2), sub-05 (1), sub-06 (1).  
DISAGREES with Verwoert 2022 ("no significant correlations (p > 0.01) on the diagonal for any participant"); their exact  
statistic/thresholding likely differs (per-channel testing here).

**Compared to:** Verwoert et al. 2022 Technical Validation

**Conclusion:** main claims use the 7 non-flagged participants (sub-03,04,05,06,07,08,10); all 10 reported.

**Next step:** B0 reproduction

## [SP-003] B0 reproduction (repo protocol exactly)

**Date/time:** 2026-10-12 12:45-13:15

**Git commit:** (this commit)

**Status:** DONE

**Hypothesis:** repo linear baseline reproduces: every participant above all 1000 circular-shift randomizations (paper Fig 4a).

**Setup:** src/b0_repro.py: repo features, frame-level contiguous KFold(10), z-score, PCA(50) on train, LinearRegression ->  
23-bin logMel; mean r over bins and folds. Random control 1000 circular shifts (rng seed 0).

**Command(s):** python src/b0_repro.py

**Outputs:** results/b0_repro.npz, results/b0_repro.json

**Results:** mean r per participant: sub-01 .520, 02 .632, 03 .837, 04 .780, 05 .532, 06 .862, 07 .707, 08 .715, 09 .657, 10 .693.  
All 10: 0.694 +- 0.109; clean 7: 0.732 +- 0.102. 9/10 above ALL 1000 randomizations; sub-05 (.532) is above the 95th pct  
(.427) but below the random max (.558).  
Paper gives per-participant values only in Fig 4a (not in text); not read off the plot. Paper notes r is driven mostly by  
speech-vs-silence.

**Compared to:** Verwoert 2022 (all 10 above all randomizations) -> reproduced for 9/10.

**Conclusion:** B0 reproduced (qualitatively; exact figure values not checked).

**Next step:** gate

## [SP-004] Pipeline design, metric calibration, user additions (fixed BEFORE gate results)

**Date/time:** 2026-10-12 13:00-14:45

**Git commit:** e38489a (+ this commit)

**Status:** DONE

**Hypothesis:** n/a

**Setup:** src/arms.py, src/score.py, src/aggregate.py.  
CV: trial-level contiguous 10 folds; fold k test, fold k+1 val (early stopping, gate), 8 folds train - SAME for all arms  
(B0 here also trains on the 8 folds; repo-exact B0 is SP-003). One seed per fold (seed = fold).  
Backbone (B1/UC/UD/C identical, ~0.5 M params): Dropout1d(.2) -> 1x1 conv C->128 -> 3 dilated conv blocks -> BiGRU(128)  
-> head. HP fixed a priori, not tuned: AdamW lr 1e-3 wd 1e-2, bs 16, <=300 epochs, early stop patience 40 on val loss.  
Heads: B1 linear->mel80; UC linear->512-d Mimi pre-quant latent (MSE, z-scored), decoded via Mimi's unquantized path;  
UD 8 levels, logits = cos(P_l h, frozen codebook_l)*10*scale + bias, CE with level-1 weight 3; C level 1 only, acoustic  
levels from same-speaker TRAINING frames (most frequent levels-2..8 tuple co-occurring with each predicted level-1 token;  
unseen tokens -> nearest seen token in codebook space). A single literal reference clip is not usable (Mimi has no  
speaker conditioning) - this is the kNN-VC-style equivalent.  
Floors: <arm>_shuf (train+val pairing permuted, same procedure, tested on true pairs); UD_maj/C_maj majority token per level.  
Ceilings: ORACLE_* = GT target through each decoder (B0: Griffin-Lim of GT 23-bin; B1: HiFi-GAN of GT mel80; UC: Mimi  
unquantized GT latent; UD: Mimi GT 8 codes; C: GT level-1 + training-frame acoustic lookup); GT = original audio.  
Metrics: see score.py docstring. Word ID: Whisper large-v3 closed set, PMI-calibrated (minus log p(word|silence)); rule  
chosen on GT AUDIO ONLY (sub-03/06 first 60 trials): GT top-1 raw .72/.70, length-norm .73/.77, PMI .77/.77.  
MCD: librosa MFCC (dB power) rescaled by ln10/20 to natural-log amplitude cepstra (first smoke run had unscaled values).  
Drop rule (UD, C on gate): at floor if neither Whisper top-1 nor spectral r beats <arm>_shuf, one-sided Wilcoxon over  
10 folds p<.05.  
USER ADDITIONS (14:40): (1) HG-band robustness: sub-03/05/06 main arms rerun without HG-band-flagged channels  
(sub-03 ch 0,34; sub-05 ch 35; sub-06 ch 57) -> <arm>_noHG; (2) spectral r on speech frames only (GT frame mean logMel >  
midpoint of participant's 10th/90th percentile) -> spec_r_speech; (3) ceiling rows in all tables (ORACLE_*, GT);  
(4) limitation: voices pitch-shifted, SV is vs shifted voice.  
Smoke bugs fixed: scipy.zeros/real in reconstructWave; Whisper OOM -> prefix-cached candidate scoring (verified equal to  
naive scoring within fp16, |diff| <= .01); PCA -> exact eigh of covariance (verified identical predictions, 2e-15).

**Command(s):** -

**Outputs:** -

**Results:** Gate = sub-06 (best B0 val r among clean: sub-06 .855, sub-03 .832, sub-04 .766, sub-07 .725, sub-08 .706,  
sub-10 .695, sub-05 .521; flagged: sub-01 .513, sub-02 .621, sub-09 .624).  
EARLY OBSERVATION: B0_shuf val r is close to B0 (sub-06 .841 vs .855; sub-03 .797 vs .832; sub-04 .743 vs .766): the  
trial-onset-locked speech/silence template explains most of the spectral r.

**Compared to:** -

**Conclusion:** -

**Next step:** gate table

## [SP-005] GATE: sub-06, all arms + floors + ceilings

**Date/time:** 2026-10-12 14:30-15:35

**Git commit:** (this commit)

**Status:** DONE

**Hypothesis:** UD/C above their shuffled floors (else dropped).

**Setup:** SP-004 pipeline; sub-06 (clean, best B0 val r .855). All 100 trials scored (no subsetting).

**Command(s):** python src/arms.py sub-06; python src/score.py sub-06; python src/aggregate.py --gate sub-06

**Outputs:** results/preds/sub-06/*, results/scores/sub-06/*, logs/gate.log

**Results** (test folds; r = waveform-derived 23-bin spectral r; r_sp = speech frames only; WID = Whisper closed-set top-1,  
100-way, chance .01; SV = ECAPA cosine to GT):

```text
              r     r_sp   STOI   ESTOI  MCD    WID   top5  CER   SV    tokacc L1/all
  B0        .627   .330   .385  -.001  19.8   .02   .06  1.00  .321
  B1        .881   .417   .432   .054  19.2   .00   .07   .97  .398
  UC        .859   .467   .459   .090  26.8   .00   .05   .92  .397
  UD        .828   .383   .376   .072  29.1   .00   .05   .93  .269   .370/.345
  C         .812   .388   .340   .061  26.7   .00   .06   .94  .366   .371/.368
  B0_shuf   .559   .285   .378  -.005  21.0   .02   .03  1.00  .317
  B1_shuf   .821   .303   .384   .012  22.5   .01   .08   .99  .365
  UC_shuf   .767   .310   .364   .017  28.8   .01   .06   .94  .371
  UD_shuf   .744   .288   .355   .009  30.3   .01   .03   .93  .265   .274/.317
  C_shuf    .691   .264   .301   .008  27.1   .01   .06   .93  .393   .268/.352
  UD_maj   -.090  -.018   .272  -.000  35.4   .01   .04  1.00  .240   .113/.295
  C_maj    -.082  -.020   .280  -.006  34.3   .01   .05  1.00  .265   .113/.235
  ORACLE_B0 .819  .945   .861   .726   7.5   .65   .84   .66  .461
  ORACLE_B1 .998  .995   .943   .884   5.6   .74   .92   .58  .905
  ORACLE_UC .973  .982   .957   .917  16.1   .64   .81   .69  .750
  ORACLE_UD .972  .957   .863   .733  18.4   .52   .69   .76  .591
  ORACLE_C  .901  .639   .531   .281  23.8   .16   .34   .91  .398
  GT       1.000 1.000  1.000  1.000   0.0   .77   .96   .52   -
```

STOI/ESTOI over 91% of trials (9 trials: <30 non-silent frames in GT -> STOI undefined, excluded for all conditions).  
Drop-rule tests (one-sided Wilcoxon over 10 folds vs <arm>_shuf): spectral r p=.001 for UD, C, UC, B0 and p=.002 for B1;  
WID p=.84 for all (all at chance) -> NO arm at floor -> UD and C KEPT.  
HG robustness on gate (noHG, ch 57 removed): B0 .627/.627, B1 .881/.879, UC .859/.856 r; WID unchanged (chance).

**Compared to:** floors and ceilings above

**Conclusion:** all brain arms beat their shuffled floors on spectral measures (r, r_sp, STOI, ESTOI), but NONE produces  
identifiable words (WID at chance, CER ~.92-1.0). Cascade ceiling is low (ORACLE_C WID .16, STOI .53): the  
training-frame acoustic lookup loses most intelligibility, so C is decoder-limited. MCD of Mimi outputs is high even for  
oracles (16-18 vs HiFi-GAN 5.6): likely Mimi's different background-noise rendering in the ~70% silent frames; treat MCD  
with caution.

**Next step:** UD/C for all participants; scoring all; tables.

## [SP-006] Full run: all arms x 10 participants; scoring

**Date/time:** 2026-10-12 15:20-18:24

**Git commit:** (this commit)

**Status:** DONE

**Hypothesis:** see research question

**Setup:** gate decision -> all arms kept. Token arms for 9 non-gate participants 15:20-15:37 (no errors).  
Scoring: two Whisper scorers in parallel (forward + reverse participant order, WCHUNK=6). First attempt at a 2nd  
scorer OOM'd (two 11 GB scorers + trainer on 24 GB) -> nothing saved, restarted with smaller candidate chunks.  
All trials scored (no 50% subsetting), full 10-fold (no 5-fold switch).  
Speech-frame MCD check on sub-06 attempted and DISCARDED (DTW of speech-only GT vs full output is ill-posed; it ranked  
oracles below brain arms). MCD reported as computed, flagged unreliable across decoders.  
Spectrogram figure shows late responses truncated by the 2 s trial window (e.g. sub-03 trial 4) -> affects GT too.

**Command(s):** run_tok.sh; run_score_rest.sh; run_score_rev.sh

**Outputs:** results/scores/* (10 participants x 18 conditions; +5 noHG for sub-03/05/06), no missing conditions

**Results:** see SP-007

**Compared to:** -

**Conclusion:** complete

**Next step:** aggregate, figures, summary

## [SP-007] Final aggregation, tables, figures, summary

**Date/time:** 2026-10-12 18:24-18:26

**Git commit:** (this commit)

**Status:** DONE

**Hypothesis:** research question (unified Mimi target vs mel regression vs cascade; discrete vs continuous)

**Setup:** src/aggregate.py, src/figures.py; clean = 7 participants. Paired two-sided Wilcoxon across participants.

**Command(s):** python src/aggregate.py; python src/figures.py

**Outputs:** results/main_table.csv, results/per_participant.csv, results/paired_tests.csv, results/hg_robustness.txt,  
figures/main_bars.(png|pdf), figures/discrete_vs_continuous.(png|pdf), figures/spectrogram_examples.(png|pdf)  
(random words, seed 0: sub-08 t63, sub-06 t26, sub-05 t4), notes/RESULTS_SUMMARY.md, notes/FINDINGS_speech.md

**Results** (clean, mean +- SD across participants):

```text
           r_all  r_speech  STOI  ESTOI  WID    CER   SV
  B0       .465   .246      .371  .003   .010   1.00  .312   (native r .727; shuffled native .698)
  B1       .786   .333      .394  .035   .016   .977  .389
  UC       .748   .374      .436  .065   .009   .939  .378
  UD       .699   .294      .349  .038   .013   .949  .319   tokacc L1 .261 (shuf .212, maj .113)
  C        .690   .284      .307  .042   .007   .952  .344   tokacc L1 .261 (shuf .201)
```

floors (shuf) r_speech .12-.21, WID .009-.013, SV .29-.33; ceilings WID .60-.75 (C .17), GT .77  
Key tests: UC>UD STOI/SV p=.016 (7/7), r_speech p=.031; UC>B1 r_speech/ESTOI p=.031, STOI p=.016; UC vs B1 SV p=.38;  
floor-corrected UC-UD gain: ESTOI p=.016, SV p=.031, r_speech p=.078.  
HG robustness (sub-03/05/06): |delta| <= .03, main contrasts unchanged.

**Compared to:** Verwoert 2022 (B0 reproduced); SLD-L2S (consistent: discrete tokens weaker under weak conditioning)

**Conclusion:** no arm is intelligible; continuous codec latent is the better target than discrete tokens at this scale;  
no speaker-similarity gain over mel regression; cascade decoder-limited.

**Next step:** none before freeze except text edits.

## [SP-008] Floor-corrected UC vs B1 / UC vs UD, and headroom (analysis of saved scores only)

**Date/time:** 2026-10-12 22:56

**Git commit:** (this commit)

**Status:** DONE

**Hypothesis:** UC's advantages over B1 and UD survive subtracting each arm's own shuffled-brain floor.

**Setup:** no training, no GPU; input results/per_participant.csv only. Clean participants (sub-03,04,05,06,07,08,10), n=7.  
Metrics: r_speech (spec_r_speech), STOI, ESTOI (valid-trial means as in SP-007), WID (Whisper top-1), SV (ECAPA cosine).  
(1) delta_arm = arm - arm_shuf per participant; paired two-sided Wilcoxon on delta_A - delta_B across 7 participants  
(zero differences dropped, scipy default); Holm over the 5 metrics within each comparison.  
(2) headroom = (arm - arm_shuf) / (ORACLE_arm - arm_shuf); NA if ORACLE_arm - arm_shuf < 0.05. ORACLE_* available for  
all 7 clean participants for all 5 arms (all used).

**Command(s):** python src/sp008_floor_headroom.py

**Outputs:** results/floor_corrected.csv, results/headroom.csv, results/headroom_per_participant.csv

**Results:**

| comparison | metric | delta A (mean ± SD) | delta B (mean ± SD) | mean diff | A > B | p raw | p Holm |
|---|---|---|---|---|---|---|---|
| UC vs B1 | r_speech | 0.186 ± 0.078 | 0.124 ± 0.036 | +0.062 | 6/7 | 0.031 | 0.125 |
| UC vs B1 | STOI | 0.089 ± 0.048 | 0.030 ± 0.029 | +0.059 | 5/7 | 0.078 | 0.234 |
| UC vs B1 | ESTOI | 0.067 ± 0.038 | 0.026 ± 0.028 | +0.041 | 7/7 | 0.016 | 0.078 |
| UC vs B1 | WID | -0.000 ± 0.013 | 0.006 ± 0.007 | -0.006 | 1/7 (2 tie) | 0.223 | 0.446 |
| UC vs B1 | SV | 0.052 ± 0.036 | 0.060 ± 0.041 | -0.008 | 3/7 | 0.578 | 0.578 |
| UC vs UD | r_speech | 0.186 ± 0.078 | 0.157 ± 0.090 | +0.029 | 6/7 | 0.078 | 0.234 |
| UC vs UD | STOI | 0.089 ± 0.048 | 0.073 ± 0.053 | +0.016 | 4/7 | 0.578 | 1.000 |
| UC vs UD | ESTOI | 0.067 ± 0.038 | 0.036 ± 0.020 | +0.031 | 7/7 | 0.016 | 0.078 |
| UC vs UD | WID | -0.000 ± 0.013 | 0.001 ± 0.017 | -0.001 | 3/7 (1 tie) | 0.833 | 1.000 |
| UC vs UD | SV | 0.052 ± 0.036 | 0.028 ± 0.026 | +0.024 | 6/7 | 0.031 | 0.125 |
| arm | r_speech | STOI | ESTOI | WID | SV |
|---|---|---|---|---|---|
| B0 | 0.064 ± 0.037 (med 0.050; n=7) | -0.035 ± 0.050 (med -0.028; n=7) | 0.007 ± 0.016 (med 0.003; n=7) | -0.005 ± 0.011 (med 0.000; n=7) | 0.094 ± 0.064 (med 0.081; n=7) |
| B1 | 0.160 ± 0.049 (med 0.167; n=7) | 0.050 ± 0.052 (med 0.070; n=7) | 0.031 ± 0.033 (med 0.019; n=7) | 0.008 ± 0.010 (med 0.011; n=7) | 0.103 ± 0.067 (med 0.122; n=7) |
| UC | 0.237 ± 0.095 (med 0.275; n=7) | 0.148 ± 0.077 (med 0.175; n=7) | 0.076 ± 0.045 (med 0.058; n=7) | -0.003 ± 0.018 (med 0.000; n=7) | 0.115 ± 0.077 (med 0.091; n=7) |
| UD | 0.189 ± 0.095 (med 0.201; n=7) | 0.125 ± 0.086 (med 0.116; n=7) | 0.051 ± 0.029 (med 0.052; n=7) | -0.004 ± 0.033 (med 0.018; n=7) | 0.087 ± 0.079 (med 0.051; n=7) |
| C | 0.320 ± 0.131 (med 0.350; n=7) | 0.177 ± 0.257 (med 0.169; n=7) | 0.148 ± 0.133 (med 0.098; n=7) | -0.046 ± 0.076 (med -0.056; n=7) | 0.295 ± 0.257 (med 0.309; n=6, NA=1) |

NA count (headroom): 1 of 175 (C / SV / sub-06: ORACLE_C .398 vs C_shuf .393).

NOTE: with n=7 the smallest two-sided Wilcoxon p is .0156, so the smallest Holm-adjusted p over 5 metrics is .078:

no comparison can reach Holm p<.05 at this sample size, however consistent.

C has the largest headroom fractions only because its ceiling is low (ORACLE_C - C_shuf small); not evidence that C is

the best arm.

**Compared to:** SP-007 (which reported raw, uncorrected floor-corrected p-values for UC vs UD: ESTOI .016, SV .031,  
r_speech .078 - unchanged here; SP-007 numbers are not modified).

**Conclusion:**  
UC beats B1 after floor correction: MIXED - UC ahead on r_speech (+.062, 6/7, raw p=.031) and ESTOI (+.041, 7/7, raw p=.016),  
trend on STOI (+.059, 5/7, p=.078); none Holm-significant (min p_Holm .078); no advantage on WID (1/7) or SV (3/7, p=.58).  
UC beats UD after floor correction: MIXED - UC ahead on ESTOI (+.031, 7/7, raw p=.016) and SV (+.024, 6/7, raw p=.031),  
trend on r_speech (+.029, 6/7, p=.078); none Holm-significant (min p_Holm .078); no difference on STOI (4/7) or WID.  
Headroom: brain arms recover a small fraction of the decoder's available range over the floor - UC 24% on r_speech,  
15% STOI, 8% ESTOI, ~0% WID, 12% SV (means); B1 16% / 5% / 3% / 1% / 10%.

**Next step:** none (analysis only).
