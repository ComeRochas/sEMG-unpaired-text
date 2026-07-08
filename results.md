# Experiment log — sEMG silent-speech (EMG→text)

Here are the notable results from my experiments across all the internship. 
What stands out is that UML pretraining has NEVER yielded any SIGNIFICANT improvement, despite trying :
- multiple unpaired modality or datasets,
- different pretraining objectives (both in supervised and unsupervised settings), 
- sharing different modules of the architecture (transformer only, transformer + CTC head, ...)

**Metric notes.** WER/CER via CTC + KenLM **beam** decode unless stated. Phase-1 and Phase-2 §B
report the **test** split (99 utts); Phase-2 §C/§C.2 report the **dev** split (200 utts, the
metric the finetune logs select on). All beam/LM numbers are inflated by LM leakage (test = *War
of the Worlds* ∈ LibriSpeech KenLM) — trust relative ordering and CER, not absolutes. In tbenst experiments, 
they removed the excerpts from war of the worlds from the KenLM training dataset.

---

## PHASE 1 — UML with unpaired audio from 2 datasets : LibriSpeech, and the audio from Gaddy's utterances


date : 04/27/2026–05/11/2026
Experiment run : Supervised CTC baseline (`GaddyRawEMGEncoder`, 8× resolution, char), 200 and
400 epochs.
Results (test) : 200 ep = **WER 0.325 / CER 0.144**; 400 ep = **0.307 / 0.130**.

date : 05/08/2026 – 05/12/2026
Experiment run : UML dual-branch pipeline, batches alternate between EMG and audio — EMG + LibriSpeech/Gaddy audio encodings (encoded by wav2vec2) sharing one
Transformer, per-branch CTC head; then EMG-only finetune (`finetune_from_uml.py`). Sweep over
audio source × λ_uml.
Results (test, UML EMG-branch → after finetune) :
| Audio | λ | no finetune | + finetune |
|---|---|---|---|
| Gaddy | 1.0 | 0.342 / 0.157 | 0.329 / 0.146 |
| Gaddy | 0.5 | 0.319 / 0.147 | 0.306 / 0.134 |
| **Gaddy** | **0.3** | 0.321 / 0.144 | **0.287 / 0.127 ← BEST** |
| Libri | 1.0 | 0.312 / 0.144 | 0.306 / 0.137 |
| Libri | 0.5 | 0.309 / 0.143 | 0.311 / 0.138 |
| Libri | 0.3 | 0.311 / 0.141 | 0.294 / 0.133 |
Verdict : **UML wins** — Gaddy λ=0.3 + finetune = 0.287 (−3.8 pp vs 200-ep baseline). λ=0.3 best
for both sources; finetune essential (+1–3 pp). Lessons: `clip_grad_norm=1.0` breaks the Gaddy
audio path (all-blank); `share_ctc_head` −0.5 pp; `epoch_mode=both` unbalanced collapses EMG.

---

## PHASE 2 - SSL using unpaired modality (audio and text)

date : 05/14/2026-05/25/2026
Experiment run : JEPA (BYOL-style) EMG pretrain → CTC finetune. Tried BYOL, VICReg (inspired by Cell-JEPA paper)
Results : dev WER ~0.403 (linear-probe proj. ~0.33); **never test-evaluated**. No transfer at
full labels.

date : 05/29/2026 – 06/01/2026
Experiment run : Unsupervised SSL on the shared transformer — three objectives: data2vec, w2v2
(Gumbel-VQ + InfoNCE), NTP (next-frame).
Results : data2vec finetune **test 0.328 / 0.144** (= baseline, no gain); w2v2 **codebook
collapse** (cancelled); NTP **CTC all-blank** (dev 1.0 all epochs, fails as init). Two prior
VICReg attempts also collapsed. **Unsupervised direction archived** — SSL extracts nothing the
supervised CTC doesn't already get at ~100 h audio.

---

## PHASE 3 — using text as auxiliary modality, and figuring out which target unit is best (spoiler : characters)

date : 06/02/2026 – 06/03/2026
Experiment run : Phase-2 scaffolding — pluggable target unit (char/subword/phoneme), configurable
token resolution (`--downsample-factor`), per-unit duration analysis, sweep infra.
Results : infrastructure; char @ 8× reference baseline = **test 0.315 / 0.145** (reproduces and
beats phase-1 0.325).

date : 06/03/2026 – 06/15/2026
Experiment run : Supervised baseline sweep over (unit × resolution) — char/subword-{60,100,250,
500,1000}/phoneme × factor {8,10,16,20,25,32}.
Results (test WER, beam+KenLM; phoneme = greedy PER) :
| unit | best test WER (factor) | curve |
|---|---|---|
| **char** | **0.292 (20×) / 0.296 (16×)** | 8× .315 → 10× .300 → 16× .296 → 20× .292 |
| subword-60 | 0.276 (20×) | 10× .316 / 16× .280 / 20× .276 |
| subword-100 | 0.287 (16×) | 10× .322 / 16× .287 / 20× .290 |
| subword-250 | 0.283 (25×) | 16× .290 / 20× .294 / 25× .283 |
| subword-500 | 0.298 (16×) | 16× .298 / 20× .319 / 25× .357 |
| subword-1000 | 0.333 (16×) | 16× .333 / 20× .335 |
| phoneme | dropped | PER 0.161 (16×); 8× collapses to all-blank |
Verdict : coarser-than-phase-1 wins; optimum ~16–20× (signal info-rate, not unit length).
Apples-to-apples flashlight lexicon decoder: char-16× **0.248** < phoneme-16× **0.275**.
Caveat on the subword numbers: on the **test** split the small subwords actually posted the
lowest WER — subword-60 @20× = 0.276 and @16× = 0.280, nominally below char (0.292). I did NOT
treat this as a real win. On the **dev** split (200 utts, the split checkpoints are selected on)
char is best and subword-60 is worse (char-16× dev 0.301 vs subword-60-16× dev 0.311, subword-60-
20× dev 0.357), and a ~0.01–0.016 gap on the 99-utterance test set sits inside the run-to-run
variance (see the ~.01 WER note below). So I kept characters — best on dev, simplest to decode,
and the robust choice.
One thing to take into account is that, in order to get a WER from phonemes decoding, I had to 
convert the predicted chain of phonemes into words. In order to do this, I therefore had to use a lexicon
which biased the results because it helped the words decoding. When using the same lexicon for character decoding, 
I achieved even better results with character as the target unit, so I chose not to pursue phonemes decoding.
**DECISION → predict CHARACTERS @ 16×.**

So here, I beat my previous baseline by just changing (mulitplying by 2) the temporal resolution of my EMG representations 
after the convolutions (and therefore after the transformer, ie before the CTC head). 
And from then on, the UML didn't improve the training anymore, so even though I initially thought it helped improve it, 
the results do not hold. 
Important note : there is a ~.01 WER variance for two runs trained with the exact same scripts.

date : 06/16/2026 – 06/17/2026
Experiment run : §C Unpaired TEXT branch (denoising-CTC: corrupt char stream → shared
transformer → CTC), sources libri + gaddy, frontends embed and frozen (CANINE / ByT5). char-16×,
+ EMG finetune.
Results (dev WER) : text-gaddy 0.296, text-libri 0.410, text-gaddy-frozen 0.282,
text-libri-frozen 0.284, text-libri λ=0.1 0.284, λ=0 control 0.304. No text arm beats the control.

date : 06/17/2026 – 06/18/2026
Experiment run : §C Single-aux AUDIO at char-16× (Gaddy-internal + LibriSpeech), full wav2vec2
frontend, + EMG finetune. (λ=0 control at 8×.)
Results (dev WER) : **Gaddy-audio-16× + finetune = 0.276** (best Phase-2 dev number to date);
Gaddy-audio-8× 0.290; λ=0 control (8×) 0.295. ≈ ties control band.

date : 07/01/2026 – 07/06/2026
Experiment run : §C Multi-auxiliary UML — EMG + audio + text sharing the transformer (Fisher-info
compounding), seed-matched (0,1) λ=0 controls, CANINE and ByT5 text encoders. char-16×, +finetune.
Results (dev WER, seeds 0 / 1; raw UML EMG-branch → after EMG finetune) :
| arm | raw UML dev | finetuned dev |
|---|---|---|
| **λ=0 control** | 0.306 / 0.319 | **0.280 / 0.313** (best 0.280) |
| multi-aux audio+text, CANINE | 0.334 / 0.343 | 0.308 / 0.324 |
| multi-aux audio+text, ByT5 | 0.336 | 0.308 |
Verdict : **no unpaired auxiliary (audio, text, or both; CANINE or ByT5) beats the pure-EMG λ=0
control.** Seed spread (0.033) > the effect (0.019). ByT5 == CANINE. UML is a wash at full labels.

---

## PHASE 4 — decoding to HuBERT units 

I tried decoding directly to HuBERT units and see if UML could improve training in that setting.
To get a WER at eval (HuBERT units are not text), I re-synthesized speech from the predicted units
and transcribed it: units → Tacotron2 (units → mel-spectrogram) → WaveGlow (mel → audio) →
Whisper ASR (audio → text), then WER against the transcript. The same chain run on the
ground-truth units gives the ceiling (topline) imposed by the vocoder + ASR.


date : 06/25/2026 – 06/26/2026
Experiment run : First EMG → HuBERT-unit baseline (λ=0, 8× resolution), units precomputed from each
utterance's OWN `_audio_clean.flac`.
Results : dev **UER 0.730** (unit error rate; greedy, since units have no word LM). **BUG found:**
dev/test are 100% SILENT utterances, and a silent recording's own audio is near-silence, so its
HuBERT units are "units of silence" (noise), not the spoken content — silent-utt unit-entropy 3.74
bits vs 6.44 for voiced; a silent utt's units vs its parallel voiced utt of the SAME sentence differ
by UER 0.926 (≈ unrelated). The target itself was degenerate.

date : 06/29/2026
Experiment run : FIX — for silent utterances, target the parallel VOICED recording's units (matched
by book_location, 100% coverage) via `--silent-target voiced`; re-precompute + retrain the λ=0
baseline. Added a held-out VOICED eval split ("vdev": voiced EMG → its own units) to separate
silent-speech difficulty from target degeneracy.
Results : dev UER 0.730 → **0.506**; vdev (voiced) **0.373**. The 0.506-vs-0.373 gap is the genuine
silent-vs-voiced difficulty; the rest of the old 0.730 was target degeneracy.

date : 06/30/2026 – 07/01/2026
Experiment run : Real UML with unit targets — audio branch predicts the SAME km100 units as EMG
(units precomputed from the audio-branch waveforms), λ=0.3, sources LibriSpeech and Gaddy-internal;
then resynth word-WER eval on the 3 checkpoints.
Results (dev UER / vdev) : λ=0 control 0.506 / 0.373; libri-UML 0.510 / 0.373; gaddy-UML 0.509 /
0.373 → **no transfer** (identical to the control). Resynth word-WER (test): pred 0.487 / 0.535 /
0.498; but the TOPLINE (ground-truth units → resynth → ASR, so model-independent) came out
0.387 / 0.455 / 0.214 on IDENTICAL gold inputs — WaveGlow samples fresh, unseeded noise, so the
vocoder is stochastic and resynth-WER is too noisy to rank models at n=99. UER (deterministic) is
the reliable metric. **Verdict : with meaningful (voiced) targets EMG→HuBERT-units trains fine
(~0.51 UER) but UML gives no gain; the resynth eval is dominated by vocoder stochasticity.**


---

## PHASE 5 — Back to unpaired with EMG + audio : adapting the EMG encoder (or the audio encoder) so both unpaired modalities go through the same transformations before the shared modules

date : 07/06/2026 – 07/07/2026
Experiment run : Frontend symmetry (Option 2) — conv-only audio frontend (`--audio-frontend
conv`, wav2vec2 feature-extractor only, no attention), vs the known full-wav2vec2 frontend.
char-16×, seed 0, Gaddy + Libri, λ 0.3 / 1.0, + finetune.
Results (dev WER / CER) :
| frontend | source | λ | dev WER / CER |
|---|---|---|---|
| FULL (12 attn) | Gaddy | 0.3 | **0.276 / 0.132** |
| conv-only | Libri | 1.0 | 0.288 / 0.139 |
| conv-only | Libri | 0.3 | 0.292 / 0.139 |
| conv-only | Gaddy | 1.0 | 0.336 / 0.167 |
| conv-only | Gaddy | 0.3 | OOM @ep126 (l40s-44GB) |
| λ=0 control | — | 0 | 0.280 / 0.139 |
Verdict : **conv-only FAILS** — Gaddy full 0.276 → conv 0.336 (+0.060); Libri conv (CER 0.139)
== λ=0 control (inert). wav2vec2's transformer IS the transfer source; asymmetry is a feature.

date : 07/07/2026
Experiment run : EMG-private pre-transformer (Option 1) — `--num-private-layers 6` gives the EMG
branch its own 6-layer attention stack before the shared transformer (encoder 58M→102M params),
audio kept FULL wav2vec2. char-16×, seed 0, Gaddy + Libri × λ 0.3 / 1.0, + λ=0 same-arch control,
+ chained EMG finetune (jobs 12861113–12861122, a100).
Results : **RUNNING — pending.** Comparator = full-frontend Gaddy 0.276 and λ=0 control 0.280.
