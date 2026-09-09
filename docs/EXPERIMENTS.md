# Experiment Inventory

Every model and experiment run in this project, across all three
modalities, with verified numbers pulled directly from each run's
`metrics.json` (or `segmentation_history.json` where no `metrics.json`
exists). This is the "what's actually been tried" reference — see
[`TRAINING.md`](TRAINING.md) for the shared architecture/fine-tuning recipe
every classifier below uses, and [`ENSEMBLE_MODELS.md`](ENSEMBLE_MODELS.md)
for how the brain MRI ensembles work mechanically.

## Chest X-ray

| Run | Backbone | Training set | Result | Status |
|---|---|---|---|---|
| `ml/artifacts/rsna_real` | EfficientNet-B0 | 5,000-image subset | 64.0% acc / 0.8253 AUC | ✅ live |
| `ml/artifacts/rsna_full_seed42` | EfficientNet-B0 | **full 26,684** | 65.87% acc / **0.8507** AUC | evaluated, not deployed |

3-class RSNA pneumonia task (Normal / Lung Opacity / No Lung Opacity-Not
Normal). The live model was trained on a 5,000-image class-balanced subset
(not the full ~26k set) on CPU — a deliberate reproducibility baseline, not
the strongest possible number; see `frontend/lib/evaluation-data.ts`'s
`chest_xray.literature` section for the honest gap-to-literature writeup.

### Full-dataset retrain: significant on AUC, not on accuracy

The subset was a self-imposed CPU-era constraint, so the obvious test was
retraining on all 26,684 images with the identical recipe. Both models were
then scored on the **same 750 test images** — verified first to have *zero*
leakage in either direction (neither model trained on any image in the
other's test split), with the 750 being a strict subset of the full test
split the new model never saw.

| Metric (same 750 images) | 5k subset | Full 26,684 | Δ | p |
|---|---|---|---|---|
| Accuracy | 64.00% | 65.87% | +1.87 pts | 0.449 — **not significant** |
| **ROC-AUC** | 0.8253 | **0.8507** | **+0.0254** | **<0.0001 — significant** |

The AUC difference has a paired-bootstrap 95% CI of **[+0.0135, +0.0379]**
(2,000 resamples, `ml/scripts/auc_bootstrap.py`), which excludes zero.

**Why the two metrics disagree, and why AUC is the one to believe here:**
accuracy at n=750 is a thresholded, high-variance statistic — the p=0.449
reflects insufficient power, not absence of effect. ROC-AUC is
threshold-independent and lower-variance, so the same 750 images resolve
decisively what accuracy could not. On the metric that actually matters for
a screening tool — ranking ability independent of operating point — more
data genuinely helped.

On its own larger test split (n=4,003) the full-data model scores 67.00% acc
/ 0.8540 AUC, but that is *not* comparable to the 64.0% figure (different
sample); the 750-image paired comparison above is the honest one.

**Caveat, stated plainly:** both arms are **single seeds**. The bootstrap
proves these two specific models differ; it does not fully separate "more
data helps" from "this seed was lucky." A 5-seed campaign was started and
abandoned after ~12 GPU-hours (each run is ~6-7 h at ~25 min/epoch, and
repeated session teardowns plus host memory exhaustion made it
impractical). Partial evidence: seed 0 reached 65.6% val vs seed 42's 64.8%
at the same epoch before it was interrupted, i.e. tracking closely rather
than diverging. For scale, EfficientNet-B0's 5-seed AUC spread on brain MRI
was ~0.013, roughly half the +0.0254 gap observed here.

**Not deployed.** The accuracy gain is inside seed noise, and a production
swap on a single-seed AUC result did not meet this project's own bar.

**Not a result**: the loose files at `ml/artifacts/metrics.json` /
`history.json` / `confusion_matrix.png` / `roc_curve.png` (repo root of
`ml/artifacts/`) are a 5-image smoke-test fixture (acc 0.8, AUC 1.0, classes
`normal`/`pneumonia`) used to generate `gradcam_demo.png`/`pipeline_demo.png`
for documentation — don't mistake it for a trained model's metrics.

## Brain MRI

4-class tumor classification (glioma / meningioma / pituitary / no tumor),
3,264 images, same manifest/split across every run below — the cleanest
apples-to-apples comparison in the project.

| Run | Config | Accuracy | ROC-AUC | Status |
|---|---|---|---|---|
| `brain_mri` | EfficientNet-B0 | 82.04% | 0.9564 | ✅ live |
| `brain_mri_resnet50_solo` | ResNet50, seed 42 | 85.51% | 0.9701 | not deployed |
| `brain_mri_resnet50_seed0` | ResNet50, seed 0 | 82.45% | 0.9631 | not deployed |
| `brain_mri_resnet50_seed1` | ResNet50, seed 1 | 84.08% | 0.9664 | not deployed |
| `brain_mri_resnet50_seed2` | ResNet50, seed 2 | 85.71% | 0.9684 | not deployed |
| `brain_mri_resnet50_seed3` | ResNet50, seed 3 | 85.71% | 0.9682 | not deployed |
| ResNet50, mean of 5 seeds (42,0,1,2,3) | — | **84.69%** | **0.9673** | 95% CI: 82.9–86.5% acc, 96.4–97.1% AUC |
| `brain_mri_densenet121_solo` | DenseNet121 | 83.06% | 0.9576 | not deployed |
| `brain_mri_vgg16_solo` | VGG16 | 85.31% | 0.9684 | not deployed |
| `ens2_seed{42,0,1,2,3}` | Ensemble: EfficientNet-B0 + DenseNet121, **mean of 5 seeds** | **87.67%** | **0.9741** | 95% CI: 85.6–89.7% acc, 96.8–98.0% AUC — not deployed |
| `ens3_seed{42,0,1,2,3}` | Ensemble: EfficientNet-B0 + ResNet50 + DenseNet121 (3-way), **mean of 5 seeds** | **88.12%** | **0.9771** | 95% CI: 87.4–88.9% acc, 97.4–98.0% AUC — not deployed (memory ceiling — see `ENSEMBLE_MODELS.md`) |
| `brain_mri_2way_effnet_dense` | 2-way ensemble, seed 42 only (superseded) | 84.90% | 0.9660 | single run — kept for provenance |
| `brain_mri_ensemble` | 3-way ensemble, seed 42 only (superseded) | 87.14% | 0.9727 | single run — kept for provenance |

Two things worth being precise about, since the project has a history of
over/under-stating exactly this kind of thing:

- **The single-run ResNet50 number (85.5%) is not the honest comparison
  point.** The 5-seed mean (84.69%) is what's statistically defensible — the
  published ResNet101 baseline it's compared against (84.5% acc) falls
  *inside* that 5-seed 95% CI, so the accuracy win over that baseline is
  **not statistically significant** (one-sample t-test p=0.80). The ROC-AUC
  win over that same baseline (90.1%) *is* significant (our CI floor is
  96.4%, p<0.000001). See `frontend/lib/evaluation-data.ts`'s `brain_mri`
  `literature` block for the full writeup — this is already correctly
  reflected on the live `/evaluation` page.
- **DenseNet121 solo (83.06%) has never been mentioned in any narrative
  copy** (README/ROADMAP/evaluation-data.ts all cite ResNet50 and VGG16 as
  "both beat EfficientNet-B0" without mentioning DenseNet121's own solo
  result, which also beats it, just less than the others). Listed here so
  the full backbone comparison is actually complete.
- **Both ensembles are now 5-seed results, and seed 42 was the worst seed
  for both.** The previously-published single-run numbers (3-way 87.14%,
  2-way 84.90%) were therefore pessimistic, not optimistic — re-running
  seeds 0–3 raised the 3-way mean to 88.12% and the 2-way mean to 87.67%.
  The seed-42 runs reproduce the old artifacts exactly, which is what
  validates the new evaluation setup.

### The deployed brain MRI model is the worst of five seeds

Evaluating the four extra EfficientNet-B0 seed checkpoints standalone (they
were trained as ensemble members, but never scored on their own) produced
the most actionable finding in this whole line of work:

| Plain EfficientNet-B0 | Accuracy | ROC-AUC |
|---|---|---|
| seed 42 — *previously deployed* | 82.04% | 0.9564 |
| seed 0 | 84.49% | 0.9670 |
| seed 1 | 84.90% | 0.9681 |
| seed 2 | 84.90% | 0.9693 |
| seed 3 — ✅ **now deployed** | **85.71%** | 0.9690 |
| **5-seed mean** | **84.41%** (95% CI 82.67–86.14) | **0.9660** (95% CI 0.9593–0.9726) |

The old live model sat at the very bottom of its own seed distribution.
Swapping to seed 3 was **+3.67 accuracy points** at byte-identical
checkpoint size (16.35MB), zero memory change, and zero deployment risk — a
bigger, cheaper, and far more certain win than either the ensemble (blocked
on memory) or distillation (below). The originally-published 82.04% was
never wrong, it was just unlucky; nothing about it was known to be a low
draw until the other seeds existed.

**This has been actioned:** `backend/model/model_efficientnet_b0_brain_mri.pt`
is now the seed-3 checkpoint (verified: identical class ordering, loads
through the unmodified `Predictor`). Headline brain MRI numbers across the
README, ROADMAP, and `/evaluation` now read 85.71% / 0.969.

One caveat stated plainly: this gain is measured on the same held-out test
split used for every other number in this project. It is the best available
estimate and is consistent with how everything else here is reported, but it
is not independent validation on a second dataset — see the cross-dataset
item in the roadmap.

### Knowledge distillation — tested, and it does not work here

Full implementation in `ml/src/medchron/models/distill.py` (Hinton KD,
T=4.0, α=0.7) with teacher soft targets cached by
`ml/scripts/cache_teacher_logits.py` from the 3-way ensemble. The premise:
the student is the same architecture and size as the deployed model, so any
accuracy recovered from the 88.12% teacher would ship at zero memory cost.

It made things **significantly worse**, on the same 5 seeds, paired t-test:

| | Plain (5 seeds) | Distilled (5 seeds) | Δ | p |
|---|---|---|---|---|
| Accuracy | 84.41% | **81.88%** (95% CI 80.15–83.61) | **−2.53 pts** | 0.027 |
| ROC-AUC | 0.9660 | **0.9580** (95% CI 0.9563–0.9598) | **−0.79 pts** | 0.014 |

**Diagnosed mechanism, not a mystery.** The teacher scores only **87.35% on
the training split itself** (val 85.10%, test 87.14%) — so roughly 1 in 8
soft targets points at the wrong class, and α=0.7 weights those errors at
more than double the true hard labels. Standard KD assumes a teacher that
has effectively memorised its training data (typically >95% train
accuracy); this ensemble never did, because its members were early-stopped
around 8 epochs with augmentation. With only ~3.7 points of teacher
headroom over a plain student and 12.65% of the guidance actively wrong,
the soft targets are net noise rather than "dark knowledge".

Worth testing if revisited (not attempted here): a much lower α so hard
labels dominate, or masking the soft term on samples where the teacher
disagrees with ground truth. Neither is likely to beat simply reseeding,
which costs nothing and delivers more.

The distillation code stays in the repo — it is correct, tested, and the
negative result is only meaningful because the implementation is sound.

### 3-way ensemble vs. the published NAS paper (5 seeds, one-sample t-test)

| Comparison | Ours (n=5) | Reference | Result |
|---|---|---|---|
| Accuracy vs ResNet101 baseline | 88.12% | 84.52% | **significantly above** (p=0.00018) |
| Accuracy vs LeaSE+DARTS SOTA | 88.12% | 90.61% | **significantly below** (p=0.00076) |
| ROC-AUC vs ResNet101 baseline | 97.71% | 90.06% | **significantly above** (p<0.000001) |
| ROC-AUC vs LeaSE+DARTS SOTA | 97.71% | 95.60% | **significantly above** (p=0.000057) |

The honest headline: on this dataset the 3-way ensemble **beats the
published architecture-search SOTA on ROC-AUC** while remaining
**significantly below it on accuracy**. Both directions are real and
statistically supported; neither should be quoted without the other. The
2-way ensemble also significantly beats the ResNet101 baseline on accuracy
(87.67%, p=0.0126) but has a much wider CI than the 3-way (±2.03 vs ±0.75
points), i.e. it is less stable across seeds.

## Mammography

The platform's one open, honestly-documented gap — full seven-attempt
history. Two baselines apply, and they are **not interchangeable**: MIAS
(3-class, attempt 1 only) has a 63.3% majority baseline; every CBIS-DDSM
full-mammogram run (attempts 2, 4, 5, 6, 7) has its own, separate 55.0%
majority baseline (test split: 241 Benign / 197 Malignant = 55.02%). Mixing
these up produced a real error in an earlier draft of this project's own
docs — corrected as of this writing.

| # | Run(s) | Dataset | Result | Verdict |
|---|---|---|---|---|
| 1 | `mammography` | MIAS, full mammograms, 3-class | 59.18% acc / 0.618 AUC | below 63.3% baseline — fail |
| 2 | `mammography_cbis` | CBIS-DDSM, full mammograms | 59.13% acc / 0.642 AUC | above 55.0% baseline, thinly — thin win |
| 3 | `mammography_cbis_cropped` | CBIS-DDSM, official lesion crops | **71.12% acc / 0.785 AUC** | real win — best standalone result, but needs a pre-cropped image the upload flow doesn't provide |
| 4 | `mammography_localized` | bbox regressor → attempt 3's classifier | 49.77% acc / 0.486 AUC | below baseline — fail |
| 5 | `mammography_localized_segmentation` | U-Net segmenter → attempt 3's classifier | 48.86% acc / 0.541 AUC | below baseline — fail, despite a much better localizer |
| 6a | `mammography_gtcrop` (standalone) | classifier retrained on ground-truth-box crops | 62.53% acc / 0.660 AUC | — |
| 6b | `mammography_localized_gtcrop` (pipeline) | segmenter → 6a's classifier | 52.97% acc / 0.534 AUC | below baseline — partial recovery, still fails |
| 7a | `mammography_autocrop` (standalone) | classifier retrained on segmenter's own predicted crops | 56.39% acc / 0.566 AUC | — |
| 7b | `pipe_autocrop_seed{42,0,1,2,3}` (pipeline, **mean of 5 seeds**) | segmenter → 7a's classifier | **56.58% acc / 0.584 AUC** | 95% CI 55.17–57.98% acc — thin win, best full-pipeline result yet |

**Attempt 7 statistics (5 seeds, added after the fact — the original
write-up was a single run):** accuracy mean 56.58%, 95% CI [55.17%, 57.98%];
ROC-AUC mean 58.35%, 95% CI [56.05%, 60.66%]. Against the 55.02% majority
baseline the accuracy win is *statistically* significant (one-sample t-test,
p=0.0377) but the CI's lower bound clears the baseline by only 0.15
points — significant, yet practically negligible. The more robust finding is
that ROC-AUC sits well above chance (58.35% vs 0.500, p=0.00055), i.e. the
pipeline has genuine discriminative signal rather than just tracking the
majority class. Per-seed accuracies ranged 55.02–58.22%, so any single-run
number from this configuration is worth ±1.4 points of seed noise. **Still
not deployed** — a 1.6-point margin over majority guessing is not clinically
useful regardless of its p-value.

Attempt 4's localizer (the bounding-box regressor, `SpatialBBoxNet` in
`ml/src/medchron/models/detect.py`) has **no saved metrics file** — its
IoU 0.043 → 0.068 progression across three architecture iterations (naive
pooled head → spatial soft-argmax → attention-weighted size pooling) is
documented only in the module's docstring, not as a `metrics.json`. Attempt
5's segmenter (`ml/artifacts/mammography_segmentation`, no `metrics.json`
either) has a `segmentation_history.json` whose best validation Dice is
0.2785 (epoch 10) — the "test Dice 0.254" figure cited in `ROADMAP.md` is a
separately-run held-out test evaluation, not the same number as the training
curve's best val Dice; both are real, they're just measuring different
splits.

Full method write-ups for each attempt live in `ROADMAP.md`'s Version 2
section and the live `/case-study` page — this table exists to give the
verified numbers a single place to be checked against source files.

## Cross-dataset generalization

Every other number in this document comes from a split of the *same* dataset
the model trained on. This section is the only evidence here that anything
survives contact with data from a different source — the question a clinical
reviewer asks first.

### The leakage trap (and why deduplication *is* the experiment)

The natural second brain-MRI dataset,
`masoudnickparvar/brain-tumor-mri-dataset` (7,200 images), shares this
project's exact 4 classes — a rare label-compatible match. It is also a
*compilation* that re-packages the SARTAJ set this project trains on.
Measured with `ml/scripts/find_duplicate_images.py` (MD5 for exact copies,
64-bit dHash with Hamming ≤ 5 for re-encoded/resized copies):

| | Count | Share of external set |
|---|---|---|
| Exact MD5 duplicates of our training images | 2,633 | 36.6% |
| Near-duplicates (dHash ≤ 5) | 1,939 | 26.9% |
| **Total overlapping** | **4,572** | **63.5%** |
| Genuinely unseen | 2,628 | 36.5% |

**63.5% of that "independent" dataset is our own training data.** Evaluating
on it naively — which is the common practice — would report a generalization
number that is mostly memorisation.

Threshold choice, stated because it is a judgement call: same-class agreement
among flagged pairs degrades with Hamming distance (100% for exact/0, 94% at
1, 85% at 2, 76% at 3, 71% at 4, 65% at 5, against ~25–30% chance). Brain MRI
slices genuinely collide at 8×8, so roughly a third of the far near-dup flags
are coincidental. Keeping ≤ 5 is deliberately **conservative**: it discards
~670 genuinely-unseen images, costing test-set size, but it cannot leave
leakage in — and leakage is the error that would *inflate* the claim. The
strict (exact-only) variant bounds the other side. The detector was
sanity-checked against the training set *itself* first, where it must and does
report 100% overlap.

### Brain MRI — it generalizes, and better than in-domain

Deployed checkpoint (EfficientNet-B0, seed 3), zero-shot, no retraining:

| Evaluation set | n | Accuracy | 95% CI | ROC-AUC |
|---|---|---|---|---|
| In-domain test (own split) | 490 | 85.71% | — | 0.9690 |
| **External, clean (unseen only)** | **2,628** | **89.31%** | **[88.06, 90.46]** | **0.9786** |
| External, strict (exact removed) | 4,567 | 90.23% | [89.34, 91.08] | 0.9814 |
| External, all (contaminated) | 7,200 | 89.19% | [88.45, 89.90] | 0.9793 |
| *3-way ensemble*, external clean | 2,628 | 90.64% | [89.46, 91.73] | 0.9822 |

The **3-way ensemble also generalizes, and stays ahead of the single model
out-of-domain** (90.64% vs 89.31% on the same 2,628 unseen images). Its CI
overlaps the single model's, so this is a consistent edge rather than a
significant one — but it holds in the same direction as in-domain, which is
reassuring for an ensemble that is still blocked from deployment on memory
grounds rather than accuracy.

Two things worth reading carefully:

**The contaminated set scores no higher than the clean set** (89.19% vs
89.31%). If leakage were inflating results, the set that is 63.5% training
images should have scored *higher*. It did not — meaning this model never
memorised its training data. That independently corroborates two earlier
measurements: the 3-way ensemble reached only 87.35% on the training split,
and that same lack of memorisation is precisely why distillation failed
(above). So the leakage here was **massive in extent but not inflationary**,
which is what makes the 89.31% trustworthy rather than lucky.

**External accuracy (89.31%) exceeds in-domain (85.71%)** — raw gap +3.59
points, which is backwards from the usual domain-shift expectation. An
earlier draft of this document attributed that to label noise in the
`sartajbhuvaji` source. **That explanation was tested and is wrong**, and
what replaces it is measured rather than inferred.

*Hypothesis 1 — label noise. Refuted.* If SARTAJ mislabels images and the
external compilation corrected them, the same physical image would carry
different labels in the two datasets. Across all **2,633 byte-identical
(MD5) images present in both**, labels agree **2,633 / 2,633 = 100.00%**.
There is zero label disagreement, so label noise cannot explain the gap.

*Hypothesis 2 — class composition. Confirmed, and it explains about a
third.* Accuracy is a support-weighted average of per-class recall, and
deduplication changed the class mix: the external clean set holds fewer of
this model's hardest class (meningioma, 21.0% vs 28.8% in-domain) and more
of its easiest (no-tumor, 25.1% vs 15.3%). Reweighting the external
per-class recalls to the *in-domain* class proportions:

| | Accuracy |
|---|---|
| External, raw | 89.31% |
| External, reweighted to in-domain class mix | **88.11%** |
| In-domain | 85.71% |

Composition alone accounts for **33%** of the raw gap.

*What survives.* The residual +2.39 points is **not statistically
significant** (z=1.40, p=0.163, 95% CI **[−0.97, +5.75]** points) — the
interval includes zero. The raw gap was significant (p=0.021); after
adjusting for class composition, it is not.

**The defensible claim is therefore "generalizes at least as well as
in-domain," not "generalizes better than in-domain."** The stronger
statement does not survive scrutiny, and no speculative explanation is
needed once composition is accounted for. What remains solid is the
headline result itself: 89.31% on 2,628 genuinely-unseen images from an
independent compilation, with a tight CI.

### Mammography — it does not generalize

MIAS is a genuinely independent source (different institutions, different
era, different digitisation) from the CBIS-DDSM these models trained on, so
no deduplication is needed. Its 115 Benign/Malignant images, evaluated
zero-shot through the full production pipeline (segmenter → crop →
classifier):

| Classifier (via localized pipeline) | Accuracy | 95% CI | ROC-AUC | vs 55.65% baseline |
|---|---|---|---|---|
| CBIS official-crop trained | 49.57% | [40.11, 59.04] | 0.604 | below (p=0.19) |
| Auto-crop trained | 52.17% | [42.66, 61.57] | 0.595 | below (p=0.46) |

**Both land below the 55.65% majority-class baseline** — on a different
dataset, this pipeline is worse than always guessing "Benign". Neither gap is
statistically significant at n=115, so the honest statement is "no evidence
it beats majority guessing," not "proven worse."

One nuance that matters: **ROC-AUC stays near 0.60, above chance**, while
accuracy sits below baseline. That combination means the model retains some
genuine ranking ability across domains, but its *decision threshold* does not
transfer — it is miscalibrated for MIAS's class balance. A recalibrated
threshold might recover accuracy; the underlying signal is weak but not
absent.

Stated upfront rather than buried: n=115 gives roughly a ±9-point CI. This
can answer "does it collapse to chance?" — it can't support a precise
accuracy claim.

### What this section does and doesn't establish

- Brain MRI generalizes to an independent compilation at 89.31% (single
  model) / 90.64% (3-way ensemble) — a real, well-powered (n=2,628) result.
- Mammography does not transfer to MIAS, consistent with it remaining
  undeployed.
- Chest X-ray was **not** tested cross-dataset; no label-compatible second
  source was on hand. That gap is still open.
- All of this is still one *modality-level* comparison per model, not a
  multi-site clinical validation.

## Full `ml/artifacts/` reference table

Every subfolder as of this writing, so nothing in the repo is a mystery:

| Folder | Has `metrics.json`? | Headline |
|---|---|---|
| `brain_mri` | yes | 82.04% acc / 0.9564 AUC |
| `brain_mri_2way_effnet_dense` | yes | 84.90% acc / 0.9660 AUC (seed 42, superseded by `ens2_seed*`) |
| `brain_mri_densenet121_seed0..3` | no (checkpoints only) | ensemble members, evaluated via `ens2_*`/`ens3_*` |
| `brain_mri_densenet121_solo` | yes | 83.06% acc / 0.9576 AUC |
| `brain_mri_distilled` | yes | 81.84% acc / 0.9560 AUC (distilled, seed 42) |
| `brain_mri_distilled_seed0..3` | no (checkpoints only) | distilled students, evaluated via `distilled_eval_*` |
| `brain_mri_efficientnet_b0_seed0..3` | no (checkpoints only) | ensemble members; standalone scores via `plain_effnet_seed*` |
| `brain_mri_teacher_soft.npz` | (loose file) | cached 3-way ensemble soft targets, 3,264 rows |
| `distilled_eval_seed0..3` | yes (each) | distilled student per seed; mean 81.88% / 0.9580 |
| `plain_effnet_seed0..3` | yes (each) | plain EfficientNet-B0 per seed; mean (with seed 42) 84.41% / 0.9660 |
| `brain_mri_overlap.csv` | (loose file) | 4,572 leakage pairs between our training set and the external dataset |
| `xdata_single_{clean,strict,all}` | yes (each) | cross-dataset brain MRI, deployed model; clean 89.31% / 0.9786 |
| `xdata_ens3_clean` | yes | cross-dataset brain MRI, 3-way ensemble; 90.64% / 0.9822 |
| `xdata_mias_cbiscrop` | yes | cross-dataset mammography (MIAS), CBIS-crop classifier; 49.57% / 0.604 |
| `xdata_mias_autocrop` | yes | cross-dataset mammography (MIAS), auto-crop classifier; 52.17% / 0.595 |
| `brain_mri_ensemble` | yes | 87.14% acc / 0.9727 AUC (seed 42, superseded by `ens3_seed*`) |
| `brain_mri_resnet50_seed0..3` | yes (each) | see table above |
| `ens2_seed{42,0,1,2,3}` | yes (each) | 2-way ensemble per seed; mean 87.67% / 0.9741 |
| `ens3_seed{42,0,1,2,3}` | yes (each) | 3-way ensemble per seed; mean 88.12% / 0.9771 |
| `pipe_autocrop_seed{42,0,1,2,3}` | yes (each) | mammography pipeline per seed; mean 56.58% / 0.5835 |
| `brain_mri_resnet50_solo` | yes | 85.51% acc / 0.9701 AUC |
| `brain_mri_resnet50_multiseed.json` | (loose file, not a folder) | raw per-seed array backing the 5-seed mean above |
| `brain_mri_vgg16_solo` | yes | 85.31% acc / 0.9684 AUC |
| `mammography` | yes | 59.18% acc / 0.618 AUC (MIAS) |
| `mammography_autocrop` | yes | 56.39% acc / 0.566 AUC (standalone classifier, seed 42) |
| `mammography_autocrop_seed0..3` | no (checkpoints only) | pipeline members, evaluated via `pipe_autocrop_*` |
| `mammography_cbis` | yes | 59.13% acc / 0.642 AUC |
| `mammography_cbis_cropped` | yes | 71.12% acc / 0.785 AUC |
| `mammography_gtcrop` | yes | 62.53% acc / 0.660 AUC |
| `mammography_localized` | yes | 49.77% acc / 0.486 AUC |
| `mammography_localized_autocrop` | yes | 56.62% acc / 0.574 AUC |
| `mammography_localized_gtcrop` | yes | 52.97% acc / 0.534 AUC |
| `mammography_localized_segmentation` | yes | 48.86% acc / 0.541 AUC |
| `mammography_segmentation` | no (`segmentation_history.json` only) | best val Dice 0.2785 |
| `rsna_real` | yes | 64.0% acc / 0.825 AUC |
| root loose files | yes (toy fixture) | 80% acc / 1.0 AUC — 5-image smoke test, not a real model |

## How results reach the live site

Two separate, non-overlapping paths — don't confuse them:

- **`ml/scripts/log_experiment.py`** publishes a run's `metrics.json` to the
  backend's `ExperimentRun` table (`backend/app/models_db.py`, `kind` one of
  `classification`/`bbox_regression`/`segmentation`/`ensemble`), which
  powers the authenticated, role-gated **Research Workspace**
  (`/dashboard/(app)/research`) — a live, DB-driven table, not static
  content. This is where every run above is expected to be logged as it's
  produced.
- **`frontend/lib/evaluation-data.ts`** is a separate, hand-curated dataset
  backing the public **`/evaluation`** page — headline metrics, training
  curves, and literature comparisons for the three *deployed-or-headline*
  models (chest X-ray, brain MRI, mammography's best standalone result). It
  does not auto-sync from `ml/artifacts/` or the Research Workspace; updates
  to it are a manual editing step, same as this document.
