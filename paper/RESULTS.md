# Generated paper assets

Every number below is computed from `ml/artifacts/*/metrics.json` by `ml/scripts/make_paper_assets.py`.
Regenerate with that script; do not edit by hand.

```
=== TABLES ===
  EfficientNet-B0 (single)   acc 84.41 [82.67,86.14]  auc 0.9660
  ResNet50 (single)          acc 84.69 [82.92,86.47]  auc 0.9673
  Distilled student          acc 81.88 [80.15,83.61]  auc 0.9580
  Ensemble, 2-way            acc 87.67 [85.64,89.71]  auc 0.9741
  Ensemble, 3-way            acc 88.12 [87.38,88.87]  auc 0.9771
  wrote paper\tables\tab_brain_mri.tex
  wrote paper\tables\tab_chest_xray.tex
  accuracy 64.00 -> 65.87 (p=0.449); AUC 0.8253 -> 0.8507 (p<0.0001)
  All (contaminated)             n= 7200  acc=89.19  auc=0.9793
  Exact duplicates removed       n= 4567  acc=90.23  auc=0.9814
  Exact + near removed (clean)   n= 2628  acc=89.31  auc=0.9786
  Clean, 3-way ensemble          n= 2628  acc=90.64  auc=0.9822
  wrote paper\tables\tab_crossdataset.tex
  1. MIAS, full images               acc=59.18  auc=0.6176
  2. CBIS-DDSM, full images          acc=59.13  auc=0.6416
  3. CBIS-DDSM, official crops       acc=71.12  auc=0.7849
  4. Pipeline: bbox regressor        acc=49.77  auc=0.4861
  5. Pipeline: U-Net segmenter       acc=48.86  auc=0.5409
  6. Pipeline: GT-crop classifier    acc=52.97  auc=0.5341
  wrote paper\tables\tab_mammography.tex
  7. auto-crop pipeline (5 seeds)   acc=56.58 [55.17,57.98]

=== FIGURES ===
  wrote paper\figures\fig_seed_lottery.pdf/.png
  seed lottery: min 82.04 max 85.71 spread 3.67 pts
  wrote paper\figures\fig_leakage.pdf/.png
  contaminated 89.19% vs clean 89.31% -> no inflation
  wrote paper\figures\fig_acc_vs_auc.pdf/.png
  wrote paper\figures\fig_crossdataset.pdf/.png
  in-domain 85.71  external 89.31  adjusted 88.11
```
