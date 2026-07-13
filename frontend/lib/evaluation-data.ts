// Real held-out test-set evaluation of the deployed model.
// Source of truth: ml/artifacts/rsna_real/{metrics.json, history.json}, produced
// by `python ml/scripts/evaluate.py` (metrics computed with scikit-learn) on the
// leak-safe test split. These numbers are the actual output, embedded verbatim.

export const MODEL_INFO = {
  name: "EfficientNet-B0",
  task: "3-class chest X-ray classification",
  dataset: "RSNA Pneumonia Detection Challenge",
  trainedOn: "5,000-image class-balanced subset of 26,684 labeled RSNA images",
  split: { train: 3500, val: 750, test: 750 },
  classes: ["Lung Opacity", "No Lung Opacity / Not Normal", "Normal"],
  device: "CPU",
  schedule: "3 head + 5 fine-tune epochs (two-phase transfer learning)",
};

export const HEADLINE = {
  roc_auc: 0.825297198554294, // macro one-vs-rest
  accuracy: 0.64,
  macro_f1: 0.6346396965865991,
  test_images: 750,
};

export interface PerClass {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export const PER_CLASS: PerClass[] = [
  { label: "Normal", precision: 0.680379746835443, recall: 0.8634538152610441, f1: 0.7610619469026548, support: 249 },
  { label: "Lung Opacity", precision: 0.5324074074074074, recall: 0.6804733727810651, f1: 0.5974025974025974, support: 169 },
  { label: "No Lung Opacity / Not Normal", precision: 0.6880733944954128, recall: 0.45180722891566266, f1: 0.5454545454545454, support: 332 },
];

// rows = true class, cols = predicted class; order matches CM_LABELS
export const CM_LABELS = ["Lung Opacity", "No Lung Opacity / Not Normal", "Normal"];
export const CONFUSION_MATRIX: number[][] = [
  [115, 43, 11],
  [92, 150, 90],
  [9, 25, 215],
];

export interface EpochRecord {
  step: number;      // global epoch index 1..8
  phase: "head" | "finetune";
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
}

export const HISTORY: EpochRecord[] = [
  { step: 1, phase: "head", train_loss: 0.9232, val_loss: 0.9225, train_acc: 0.5257, val_acc: 0.5213 },
  { step: 2, phase: "head", train_loss: 0.8475, val_loss: 0.8327, train_acc: 0.5791, val_acc: 0.5853 },
  { step: 3, phase: "head", train_loss: 0.8344, val_loss: 0.9189, train_acc: 0.5903, val_acc: 0.5173 },
  { step: 4, phase: "finetune", train_loss: 0.8329, val_loss: 0.8489, train_acc: 0.5883, val_acc: 0.5787 },
  { step: 5, phase: "finetune", train_loss: 0.7866, val_loss: 0.8294, train_acc: 0.5991, val_acc: 0.5933 },
  { step: 6, phase: "finetune", train_loss: 0.7871, val_loss: 0.8049, train_acc: 0.6103, val_acc: 0.6213 },
  { step: 7, phase: "finetune", train_loss: 0.7824, val_loss: 0.8163, train_acc: 0.6049, val_acc: 0.6093 },
  { step: 8, phase: "finetune", train_loss: 0.7683, val_loss: 0.7880, train_acc: 0.6271, val_acc: 0.6267 },
];

// Series colors — CVD-safe blue/orange pair, theme-aware via CSS vars so each
// mode uses a step that passes ≥3:1 contrast on its surface (computed:
// dark #3b82f6 5.47:1, #d97706 6.32:1 · light #2563eb 4.72:1, #b45309 4.58:1;
// deuteranopia-sim separation 223/441).
export const SERIES = {
  train: "var(--series-train)",
  val: "var(--series-val)",
};

export const PROCEDURES = [
  {
    title: "Leak-safe patient split",
    body: "Train/val/test split at the patient level, not the image level — the same patient never appears in two splits, so test accuracy isn't inflated by memorization.",
  },
  {
    title: "Held-out test set",
    body: "The 750-image test split was never seen during training or model selection. All headline metrics are computed on it alone.",
  },
  {
    title: "Standard metrics via scikit-learn",
    body: "Accuracy, per-class precision/recall/F1 and macro one-vs-rest ROC-AUC are computed with scikit-learn, not hand-rolled.",
  },
  {
    title: "Identical preprocessing, train and serve",
    body: "The same DIP path (denoise + CLAHE, ImageNet normalization) runs at training time and at inference time, so there is no train/serve skew.",
  },
  {
    title: "Automated test suite",
    body: "22 unit/integration tests cover the imaging pipeline, leak-safe splitting, model wiring, Grad-CAM, and the API — run on every change.",
  },
];
