
export interface PerClass {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface EpochRecord {
  step: number;
  phase: "head" | "finetune";
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
}

export interface ModalityEvaluation {
  key: string;
  label: string;
  deployed: boolean;
  notDeployedReason?: string;
  modelInfo: {
    name: string;
    task: string;
    dataset: string;
    trainedOn: string;
    split: { train: number; val: number; test: number };
    schedule: string;
  };
  headline: {
    roc_auc: number;
    accuracy: number;
    macro_f1: number;
    test_images: number;
  };
  perClass: PerClass[];
  cmLabels: string[];
  confusionMatrix: number[][];
  shortLabels?: Record<string, string>;
  history: EpochRecord[];
  randomBaseline: number;
}

export const EVALUATIONS: Record<string, ModalityEvaluation> = {
  chest_xray: {
    key: "chest_xray",
    label: "Chest X-ray",
    deployed: true,
    modelInfo: {
      name: "EfficientNet-B0",
      task: "3-class chest X-ray classification",
      dataset: "RSNA Pneumonia Detection Challenge",
      trainedOn: "5,000-image class-balanced subset of 26,684 labeled RSNA images",
      split: { train: 3500, val: 750, test: 750 },
      schedule: "3 head + 5 fine-tune epochs (two-phase transfer learning), CPU",
    },
    headline: {
      roc_auc: 0.825297198554294,
      accuracy: 0.64,
      macro_f1: 0.6346396965865991,
      test_images: 750,
    },
    perClass: [
      { label: "Normal", precision: 0.680379746835443, recall: 0.8634538152610441, f1: 0.7610619469026548, support: 249 },
      { label: "Lung Opacity", precision: 0.5324074074074074, recall: 0.6804733727810651, f1: 0.5974025974025974, support: 169 },
      { label: "No Lung Opacity / Not Normal", precision: 0.6880733944954128, recall: 0.45180722891566266, f1: 0.5454545454545454, support: 332 },
    ],
    cmLabels: ["Lung Opacity", "No Lung Opacity / Not Normal", "Normal"],
    shortLabels: {
      "Lung Opacity": "Opacity",
      "No Lung Opacity / Not Normal": "No opacity",
      Normal: "Normal",
    },
    confusionMatrix: [
      [115, 43, 11],
      [92, 150, 90],
      [9, 25, 215],
    ],
    history: [
      { step: 1, phase: "head", train_loss: 0.9232, val_loss: 0.9225, train_acc: 0.5257, val_acc: 0.5213 },
      { step: 2, phase: "head", train_loss: 0.8475, val_loss: 0.8327, train_acc: 0.5791, val_acc: 0.5853 },
      { step: 3, phase: "head", train_loss: 0.8344, val_loss: 0.9189, train_acc: 0.5903, val_acc: 0.5173 },
      { step: 4, phase: "finetune", train_loss: 0.8329, val_loss: 0.8489, train_acc: 0.5883, val_acc: 0.5787 },
      { step: 5, phase: "finetune", train_loss: 0.7866, val_loss: 0.8294, train_acc: 0.5991, val_acc: 0.5933 },
      { step: 6, phase: "finetune", train_loss: 0.7871, val_loss: 0.8049, train_acc: 0.6103, val_acc: 0.6213 },
      { step: 7, phase: "finetune", train_loss: 0.7824, val_loss: 0.8163, train_acc: 0.6049, val_acc: 0.6267 },
      { step: 8, phase: "finetune", train_loss: 0.7683, val_loss: 0.7880, train_acc: 0.6271, val_acc: 0.6267 },
    ],
    randomBaseline: 1 / 3,
  },

  brain_mri: {
    key: "brain_mri",
    label: "Brain MRI",
    deployed: true,
    modelInfo: {
      name: "EfficientNet-B0",
      task: "4-class brain tumor classification",
      dataset: "Brain Tumor Classification (MRI) — sartajbhuvaji/Kaggle",
      trainedOn: "3,264 images (glioma / meningioma / pituitary / no tumor), image-level split (no native patient IDs)",
      split: { train: 2284, val: 490, test: 490 },
      schedule: "3 head + 5 fine-tune epochs (two-phase transfer learning), CPU",
    },
    headline: {
      roc_auc: 0.9564474963280549,
      accuracy: 0.8204081632653061,
      macro_f1: 0.8187924683185762,
      test_images: 490,
    },
    perClass: [
      { label: "glioma_tumor", precision: 0.865546218487395, recall: 0.7410071942446043, f1: 0.7984496124031008, support: 139 },
      { label: "meningioma_tumor", precision: 0.7983870967741935, recall: 0.7021276595744681, f1: 0.7471698113207547, support: 141 },
      { label: "no_tumor", precision: 0.6915887850467289, recall: 0.9866666666666667, f1: 0.8131868131868132, support: 75 },
      { label: "pituitary_tumor", precision: 0.9, recall: 0.9333333333333333, f1: 0.9163636363636364, support: 135 },
    ],
    cmLabels: ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
    shortLabels: {
      glioma_tumor: "Glioma",
      meningioma_tumor: "Meningioma",
      no_tumor: "No tumor",
      pituitary_tumor: "Pituitary",
    },
    confusionMatrix: [
      [103, 20, 11, 5],
      [14, 99, 19, 9],
      [1, 0, 74, 0],
      [1, 5, 3, 126],
    ],
    history: [
      { step: 1, phase: "head", train_loss: 1.0168, val_loss: 0.8113, train_acc: 0.5898, val_acc: 0.7122 },
      { step: 2, phase: "head", train_loss: 0.7818, val_loss: 0.6845, train_acc: 0.704, val_acc: 0.7531 },
      { step: 3, phase: "head", train_loss: 0.6995, val_loss: 0.6383, train_acc: 0.7268, val_acc: 0.7673 },
      { step: 4, phase: "finetune", train_loss: 0.6721, val_loss: 0.6084, train_acc: 0.7364, val_acc: 0.7837 },
      { step: 5, phase: "finetune", train_loss: 0.612, val_loss: 0.5687, train_acc: 0.7706, val_acc: 0.802 },
      { step: 6, phase: "finetune", train_loss: 0.6062, val_loss: 0.5381, train_acc: 0.7649, val_acc: 0.802 },
      { step: 7, phase: "finetune", train_loss: 0.5752, val_loss: 0.5314, train_acc: 0.7758, val_acc: 0.8041 },
      { step: 8, phase: "finetune", train_loss: 0.5518, val_loss: 0.5028, train_acc: 0.7863, val_acc: 0.8082 },
    ],
    randomBaseline: 1 / 4,
  },

  mammography: {
    key: "mammography",
    label: "Mammography",
    deployed: false,
    notDeployedReason:
      "This model is NOT live — its test accuracy (59.2%) is below the majority-class " +
      "baseline (63.3%, always predicting Normal), with 0% recall on the Benign class. " +
      "The MIAS dataset has only 322 images total (35-44 per minority class), too few to " +
      "train a reliable classifier. The DIP pipeline and quality gate work on mammograms " +
      "today; classification needs a larger dataset (CBIS-DDSM, ~10k+ images) before it " +
      "can ship. Shown here for transparency, not as a working feature.",
    modelInfo: {
      name: "EfficientNet-B0",
      task: "3-class mammography classification",
      dataset: "MIAS Mammography Database",
      trainedOn: "322 images (Normal / Benign / Malignant, derived from official Info.txt annotations)",
      split: { train: 224, val: 49, test: 49 },
      schedule: "3 head + 5 fine-tune epochs (two-phase transfer learning), CPU",
    },
    headline: {
      roc_auc: 0.6175547665319499,
      accuracy: 0.5918367346938775,
      macro_f1: 0.37121212121212127,
      test_images: 49,
    },
    perClass: [
      { label: "Benign", precision: 0.0, recall: 0.0, f1: 0.0, support: 10 },
      { label: "Malignant", precision: 0.6666666666666666, recall: 0.25, f1: 0.36363636363636365, support: 8 },
      { label: "Normal", precision: 0.6585365853658537, recall: 0.8709677419354839, f1: 0.75, support: 31 },
    ],
    cmLabels: ["Benign", "Malignant", "Normal"],
    confusionMatrix: [
      [0, 1, 9],
      [1, 2, 5],
      [4, 0, 27],
    ],
    history: [],
    randomBaseline: 1 / 3,
  },
};

export const SERIES = {
  train: "var(--series-train)",
  val: "var(--series-val)",
};

export const PROCEDURES = [
  {
    title: "Leak-safe patient split",
    body: "Splits are made at the patient level wherever the dataset provides patient IDs (chest X-ray), so the same patient never appears in two splits. Datasets without native patient IDs (brain MRI, mammography) use an image-level stratified split instead — documented per modality above.",
  },
  {
    title: "Held-out test set",
    body: "The test split was never seen during training or model selection for any modality. All headline metrics are computed on it alone.",
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
    title: "Models that don't clear the bar don't ship",
    body: "The mammography model is shown here but not deployed live — its accuracy is below the majority-class baseline. A model that loses to a constant guess doesn't serve predictions, per the same principle behind the quality gate.",
  },
];
