# Model Architecture & Fine-Tuning

How every MedChron image classifier — regardless of backbone or modality —
is built and trained. This is the shared recipe behind every checkpoint in
`ml/artifacts/` (see [`ENSEMBLE_MODELS.md`](ENSEMBLE_MODELS.md) for how
several of these checkpoints get combined into an ensemble).

## Supported backbones

Defined in [`ml/src/medchron/models/backbone.py`](../ml/src/medchron/models/backbone.py):
`vgg16`, `resnet50`, `densenet121`, `efficientnet_b0`. All four are
torchvision models loaded with **ImageNet-pretrained weights**
(`_BACKBONES`, [backbone.py:9](../ml/src/medchron/models/backbone.py#L9)); only
the head is task-specific.

## Architecture: head replacement

`create_model()` ([backbone.py:27](../ml/src/medchron/models/backbone.py#L27))
swaps each backbone's final classification layer for one sized to the
manifest's class count:

| Backbone | Original head | Replacement |
|---|---|---|
| `vgg16` | `classifier[6]` (last FC of the 3-layer classifier block) | `nn.Linear(in_f, num_classes)` — the two FC+ReLU+Dropout layers before it stay pretrained |
| `resnet50` | `fc` | `nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_f, num_classes))` — ResNet's stock `fc` has no dropout, so one is added (`dropout=0.5` default) |
| `densenet121` | `classifier` | `nn.Linear(in_f, num_classes)` |
| `efficientnet_b0` | `classifier[1]` (the Linear inside its built-in `Dropout→Linear` sequential) | `nn.Linear(in_f, num_classes)` — keeps EfficientNet's own dropout at index 0 |

Everything upstream (conv stacks, batch norms, attention blocks, etc.) is
the unmodified pretrained backbone at this point — nothing is frozen yet;
freezing happens in the training loop, not in `create_model`.

## Data pipeline

`build_dataloaders()` ([`ml/src/medchron/models/dataset.py:121`](../ml/src/medchron/models/dataset.py#L121))
turns a list of `Sample`s (path + label + split, read from a manifest CSV)
into train/val/test `DataLoader`s:

- **Preprocessing**: images go through `model_image()` (modality-specific
  preprocessing preset — CLAHE, resize, etc.), then standard ImageNet
  normalization (`IMAGENET_MEAN`/`IMAGENET_STD`).
- **Train-time augmentation only**: `RandomAffine(degrees=7, translate=5%,
  scale=95–105%)` + `ColorJitter(brightness=0.10, contrast=0.10)` — deliberately
  mild, since aggressive geometric distortion can erase small lesions.
  Val/test get no augmentation.
- **Class imbalance**: `class_weights()` computes inverse-frequency weights
  per class for multiclass tasks (`CrossEntropyLoss(weight=...)`);
  `multilabel_pos_weights()` computes per-label positive-class weights for
  multilabel tasks (`BCEWithLogitsLoss(pos_weight=...)`). Both are derived
  from the **train split only**, never val/test.
- **Multiclass vs multilabel**: one label per image (e.g. brain MRI tumor
  type, RSNA pneumonia) vs. several `|`-delimited findings per image (e.g.
  NIH ChestX-ray14-style datasets) — selected via `--task`.

## Fine-tuning: two-phase transfer learning

`train()` ([`ml/src/medchron/models/train.py:182`](../ml/src/medchron/models/train.py#L182))
runs the identical two-phase recipe for every backbone/modality:

### Phase 1 — "head" ([train.py:241](../ml/src/medchron/models/train.py#L241))
- `freeze_backbone()` ([backbone.py:49](../ml/src/medchron/models/backbone.py#L49))
  freezes every parameter except the head — matched by name prefix via
  `HEAD_PREFIX` (`"classifier"` for vgg16/densenet121/efficientnet_b0,
  `"fc"` for resnet50).
- Only the new (randomly initialized) head trains, for `epochs_head`
  (default 5), AdamW, `lr_head=1e-3`, `weight_decay=1e-4`.
- Purpose: let the random head converge toward the pretrained feature space
  first, so early large gradients don't smear pretrained conv filters once
  they're unfrozen in phase 2.

### Phase 2 — "finetune" ([train.py:258](../ml/src/medchron/models/train.py#L258))
- `unfreeze_top_fraction(model, finetune_fraction=0.25)`
  ([backbone.py:55](../ml/src/medchron/models/backbone.py#L55)) unfreezes the
  **last 25% of the model's parameter tensors**, ordered as
  `model.parameters()` yields them — i.e. the deepest/output-facing layers
  (head + roughly the last block of the backbone). Earlier conv layers stay
  frozen throughout.
- Trains for `epochs_finetune` (default 10), AdamW, a much smaller
  `lr_finetune=1e-5` so pretrained weights are nudged rather than
  overwritten, plus `ReduceLROnPlateau(factor=0.5, patience=2)` on val loss.

### Shared mechanics (both phases)
- **Early stopping** (`_EarlyStopper`, [train.py:107](../ml/src/medchron/models/train.py#L107)):
  tracks val loss, stops after `patience` (default 5) non-improving epochs,
  keeps a deep-copied best `state_dict` and reloads it at the end of the
  phase — a phase never ends on an overfit epoch.
- **AMP mixed precision** on CUDA (`tcfg.amp`, default on); no-op on CPU.
- **Resumability**: every epoch, full state (`model`, `optimizer`,
  `scheduler`, `best_val`, `best_state`, `history`, plus `class_to_idx` /
  `model_config` / `train_config` / `preprocess`) is written to
  `training_state_{backbone}.pt`. `--resume` picks this up mid-phase, so an
  interrupted run (e.g. a killed Colab session) doesn't lose progress. This
  state file is deleted once training finishes normally.
- **Final checkpoint**: only written once both phases complete —
  `model_{backbone}.pt` bundles `state_dict`, `class_to_idx`, `model_config`,
  `train_config`, `task`, `preprocess`, and the full per-epoch `history`.
  `history.json` is written alongside it.

## Grad-CAM target layer

Each backbone also has a fixed Grad-CAM hook point
(`default_gradcam_layer()`, [backbone.py:70](../ml/src/medchron/models/backbone.py#L70))
— the last spatial feature layer before the head, i.e. the finest-grained
layer where "which pixels drove this prediction" is still meaningful:

| Backbone | Grad-CAM layer |
|---|---|
| `vgg16` | `features[28]` (last conv layer) |
| `resnet50` | `layer4[-1]` (last bottleneck block) |
| `densenet121` | `features.norm5` (final batch norm before the classifier) |
| `efficientnet_b0` | `features[-1]` (last conv block) |

## Running it (CLI)

[`ml/scripts/train.py`](../ml/scripts/train.py) wraps `train()`:

```bash
# from a prepared manifest (has a train/val/test split column)
python ml/scripts/train.py --manifest ml/data/brain_mri/manifest.csv --backbone resnet50

# or straight from an ImageFolder-style directory (manifest + stratified split built on the fly)
python ml/scripts/train.py --root ml/data/rsna_pneumonia --backbone efficientnet_b0 --epochs-finetune 15

# resume an interrupted run
python ml/scripts/train.py --manifest ml/data/brain_mri/manifest.csv --backbone vgg16 --resume
```

Every hyperparameter in `TrainConfig` (epochs, LRs, batch size, backbone,
task, out-dir, device, …) is exposed as a flag; see the CLI's `--help` or
`TrainConfig` in [train.py:28](../ml/src/medchron/models/train.py#L28) for
the full field list. Checkpoints land in `--out-dir` (default
`ml/artifacts/`) — see `docs/ENSEMBLE_MODELS.md` for how multiple such
checkpoints (different backbones or seeds) get soft-voted into an ensemble.
