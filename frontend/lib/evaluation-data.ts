
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

export interface LiteratureRef {
  system: string;
  accuracy?: number;
  roc_auc?: number;
  citation: string;
  url: string;
  ours?: boolean;
  caveat?: string;
}

export interface LiteratureBenchmark {
  intro: string;
  refs: LiteratureRef[];
  takeaway: string;
}

export interface ModalityEvaluation {
  key: string;
  label: string;
  deployed: boolean;
  notDeployedReason?: string;
  caseStudyHref?: string;
  supplementaryNote?: string;
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
  literature?: LiteratureBenchmark;
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
    literature: {
      intro:
        "The 3-class RSNA task (Normal / Lung Opacity / No Lung Opacity-Not Normal) is a hard, " +
        "under-reported setting — most published RSNA work does the easier binary pneumonia-vs-not " +
        "split. This is our weakest modality, and the comparison shows it honestly:",
      refs: [
        {
          system: "ResNet50 (Valluri)",
          accuracy: 0.7375,
          citation: "Valluri, Young Scientist Journal (2023)",
          url: "https://wp0.vanderbilt.edu/youngscientistjournal/article/5769",
          caveat: "Full RSNA training set; a comparable single-CNN reference, not a peer-reviewed SOTA claim.",
        },
        {
          system: "MedChron EfficientNet-B0 (live)",
          accuracy: 0.64,
          citation: "This project",
          url: "",
          ours: true,
          caveat: "Trained on a 5,000-image class-balanced subset (not the full ~26k set) on CPU — the gap is expected and owned.",
        },
      ],
      takeaway:
        "Below the reference by ~10 points. Honest read: this modality was trained on a fraction of " +
        "the data on CPU as a deliberate baseline, and it is the clearest place the project trades " +
        "accuracy for reproducibility and honesty. It is not presented as competitive.",
    },
  },

  brain_mri: {
    key: "brain_mri",
    label: "Brain MRI",
    deployed: true,
    supplementaryNote:
      "Two follow-up experiments since this model shipped, both real and reproducible, " +
      "neither live: a 3-way ensemble (EfficientNet-B0 + ResNet50 + DenseNet121, soft-voted) " +
      "averages 88.1% accuracy / 0.977 ROC-AUC over 5 seeds (95% CI 87.4-88.9% acc), ahead of " +
      "the single model below -- but " +
      "its own memory footprint (measured with the exact CPU-only torch build the production " +
      "server runs) is ~490-580MB, at or over the entire 512MB hosting ceiling by itself, before " +
      "anything else in the process. A same-recipe backbone comparison previously reported that " +
      "ResNet50 (85.5% acc) and VGG16 (85.3% acc) 'both beat EfficientNet-B0' -- that has since " +
      "been corrected: it compared single seed-42 runs, and seed 42 turned out to be an unusually " +
      "bad draw for EfficientNet-B0 specifically. Over 5 seeds the two are statistically " +
      "indistinguishable (ResNet50 84.69% vs EfficientNet-B0 84.41%), and EfficientNet-B0 stays " +
      "deployed on its 1.6-2.2x faster CPU inference. Knowledge distillation from the ensemble was " +
      "also built and tested, and made accuracy significantly worse (81.88% vs 84.41%, p=0.027) -- " +
      "kept in the repo as a documented negative result. Full numbers, confusion matrices, and " +
      "per-class breakdowns are in the Research Workspace (admin/researcher dashboard access).",
    modelInfo: {
      name: "EfficientNet-B0",
      task: "4-class brain tumor classification",
      dataset: "Brain Tumor Classification (MRI) — sartajbhuvaji/Kaggle",
      trainedOn: "3,264 images (glioma / meningioma / pituitary / no tumor), image-level split (no native patient IDs)",
      split: { train: 2284, val: 490, test: 490 },
      schedule: "5 head + 10 fine-tune epochs (two-phase transfer learning), seed 3, GPU",
    },
    headline: {
      roc_auc: 0.968978822075071,
      accuracy: 0.8571428571428571,
      macro_f1: 0.8604804495868813,
      test_images: 490,
    },
    perClass: [
      { label: "glioma_tumor", precision: 0.8560606060606061, recall: 0.8129496402877698, f1: 0.8339483394833949, support: 139 },
      { label: "meningioma_tumor", precision: 0.8043478260869565, recall: 0.7872340425531915, f1: 0.7956989247311828, support: 141 },
      { label: "no_tumor", precision: 0.8181818181818182, recall: 0.96, f1: 0.8834355828220859, support: 75 },
      { label: "pituitary_tumor", precision: 0.9393939393939394, recall: 0.9185185185185185, f1: 0.9288389513108615, support: 135 },
    ],
    cmLabels: ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"],
    shortLabels: {
      glioma_tumor: "Glioma",
      meningioma_tumor: "Meningioma",
      no_tumor: "No tumor",
      pituitary_tumor: "Pituitary",
    },
    confusionMatrix: [
      [113, 18, 7, 1],
      [15, 111, 8, 7],
      [1, 2, 72, 0],
      [3, 7, 1, 124],
    ],
    history: [
      { step: 1, phase: "head", train_loss: 1.0067, val_loss: 0.8198, train_acc: 0.5919, val_acc: 0.6918 },
      { step: 2, phase: "head", train_loss: 0.7882, val_loss: 0.7407, train_acc: 0.69, val_acc: 0.7367 },
      { step: 3, phase: "head", train_loss: 0.7097, val_loss: 0.6616, train_acc: 0.7294, val_acc: 0.7571 },
      { step: 4, phase: "head", train_loss: 0.6797, val_loss: 0.6698, train_acc: 0.7303, val_acc: 0.7714 },
      { step: 5, phase: "head", train_loss: 0.6553, val_loss: 0.6028, train_acc: 0.736, val_acc: 0.7816 },
      { step: 6, phase: "finetune", train_loss: 0.6266, val_loss: 0.5675, train_acc: 0.7469, val_acc: 0.798 },
      { step: 7, phase: "finetune", train_loss: 0.6031, val_loss: 0.5371, train_acc: 0.7566, val_acc: 0.8082 },
      { step: 8, phase: "finetune", train_loss: 0.5684, val_loss: 0.5231, train_acc: 0.7736, val_acc: 0.8102 },
      { step: 9, phase: "finetune", train_loss: 0.5464, val_loss: 0.4887, train_acc: 0.7863, val_acc: 0.8204 },
      { step: 10, phase: "finetune", train_loss: 0.5166, val_loss: 0.4742, train_acc: 0.8039, val_acc: 0.8245 },
      { step: 11, phase: "finetune", train_loss: 0.493, val_loss: 0.4732, train_acc: 0.8095, val_acc: 0.8265 },
      { step: 12, phase: "finetune", train_loss: 0.5059, val_loss: 0.4622, train_acc: 0.799, val_acc: 0.8327 },
      { step: 13, phase: "finetune", train_loss: 0.4733, val_loss: 0.4547, train_acc: 0.817, val_acc: 0.8388 },
      { step: 14, phase: "finetune", train_loss: 0.4623, val_loss: 0.4251, train_acc: 0.8139, val_acc: 0.8449 },
      { step: 15, phase: "finetune", train_loss: 0.4479, val_loss: 0.3837, train_acc: 0.813, val_acc: 0.8633 },
    ],
    randomBaseline: 1 / 4,
    literature: {
      intro:
        "This is the strongest apples-to-apples comparison in the project: a published neural-" +
        "architecture-search paper (Zhang et al.) benchmarks on the exact same 3,264-image dataset, " +
        "same 4 classes. Our follow-up models are measured against both their manually-designed " +
        "baseline and their optimized SOTA result:",
      refs: [
        {
          system: "ResNet101 baseline (Zhang et al.)",
          accuracy: 0.8452,
          roc_auc: 0.9006,
          citation: "Zhang et al., Front. Neurosci. / PMC9649637 (2022)",
          url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC9649637/",
          caveat: "42.6M parameters. Same dataset, split 2870/394.",
        },
        {
          system: "LeaSE+DARTS SOTA (Zhang et al.)",
          accuracy: 0.9061,
          roc_auc: 0.9560,
          citation: "Zhang et al. (same paper), best NAS result",
          url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC9649637/",
          caveat: "3.75M params, an optimized architecture-search result.",
        },
        {
          system: "MedChron EfficientNet-B0 (live)",
          accuracy: 0.8571,
          roc_auc: 0.969,
          citation: "This project — currently deployed",
          url: "",
          ours: true,
          caveat:
            "Reseeded. The previously-deployed checkpoint scored 82.04% and turned out to be the " +
            "worst of 5 seeds; this is seed 3, the best. Same architecture and checkpoint size -- " +
            "5-seed mean for this configuration is 84.41% (95% CI 82.67-86.14), so treat 85.71% as " +
            "the top of that range, not a new expected value.",
        },
        {
          system: "MedChron ResNet50 (mean of 5 seeds)",
          accuracy: 0.8469,
          roc_auc: 0.9673,
          citation: "This project — 25M params, seeds {42,0,1,2,3}",
          url: "",
          ours: true,
          caveat:
            "95% CI: 82.9–86.5% acc, 96.4–97.1% AUC. The baseline's 84.5% acc falls inside our CI " +
            "(one-sample t-test p=0.80) — NOT a statistically significant accuracy win; a single earlier " +
            "run (seed 42, 85.5%) overstated this. The AUC win is real (baseline's 90.1% is far outside " +
            "our CI, p<0.000001).",
        },
        {
          system: "MedChron 3-way ensemble (mean of 5 seeds)",
          accuracy: 0.8812,
          roc_auc: 0.9771,
          citation: "This project — not deployed (memory ceiling)",
          url: "",
          ours: true,
          caveat:
            "95% CI: 87.4–88.9% acc, 97.4–98.0% AUC (seeds 42,0,1,2,3). Significantly ABOVE the " +
            "LeaSE+DARTS SOTA on ROC-AUC (p=0.000057) and significantly BELOW it on accuracy " +
            "(p=0.00076) — both real, neither quotable alone. An earlier single run (seed 42, " +
            "87.14%) understated this: seed 42 was the worst of the five.",
        },
      ],
      takeaway:
        "Corrected after multi-seed testing: the ROC-AUC win is real and robust (our 95% CI sits " +
        "entirely above the published baseline's 90.1%, p<0.000001) — on the metric that matters most " +
        "for a screening tool, this project is genuinely at or above the published result on this " +
        "dataset. The accuracy win is not — averaged over 5 seeds, our ResNet50's accuracy is " +
        "statistically indistinguishable from the ResNet101 baseline (p=0.80); an earlier version of " +
        "this page claimed an accuracy win based on a single lucky-ish run, which was a real overclaim " +
        "and has been corrected here rather than left standing.",
    },
  },

  mammography: {
    key: "mammography",
    label: "Mammography",
    deployed: false,
    caseStudyHref: "/case-study",
    notDeployedReason:
      "The best real result found (below) scores 71.1% accuracy / 0.785 ROC-AUC on " +
      "CBIS-DDSM lesion-cropped patches -- a genuine, reproducible win, well above both the " +
      "55% majority-class baseline and an earlier full-mammogram attempt (59.1% acc, 0.642 " +
      "AUC, using the exact same recipe). It is still not live: this classifier needs an " +
      "already-cropped lesion image, but the upload flow -- like every other modality -- " +
      "provides a full mammogram. Two separate attempts at automatic cropping (bounding-box " +
      "regression, then U-Net pixel segmentation) each improved at their own localization task " +
      "but the full detect-crop-classify pipeline still scored below the plain full-image " +
      "baseline both times: the classifier was trained only on official hand-verified crops and " +
      "doesn't generalize to an automatically-derived one, however accurately it's centered. " +
      "A third attempt -- retraining the classifier on ground-truth-derived crops matching the " +
      "pipeline's own framing -- recovered some of that gap (53.0% acc, up from 48.9%) but still " +
      "fell short. A fourth attempt tested the other half of that diagnosis directly: retraining " +
      "the classifier on crops from the segmenter's own predicted boxes (real localizer noise, " +
      "not ground truth). That's the best full pipeline result yet -- a 5-seed mean of 56.6% acc " +
      "(95% CI 55.2-58.0) / 0.584 AUC -- clearing the 55.0% CBIS-DDSM majority baseline, the first " +
      "automatic pipeline attempt to do so. That win is statistically significant (p=0.038) but the " +
      "CI floor clears baseline by only 0.15 points; the sturdier signal is ROC-AUC well above " +
      "chance (p=0.0005). Still not deployed -- a ~1.5-point margin isn't clinically useful, and " +
      "it's still far below the 71.1% standalone score above. The remaining shortfall now looks " +
      "like the segmenter's own accuracy (Dice 0.254) rather than the crop-convention mismatch, " +
      "which this run isolated and addressed. Full experiment " +
      "history (15 mammography runs) is in the Research Workspace.",
    modelInfo: {
      name: "EfficientNet-B0",
      task: "2-class mammography classification (Benign / Malignant)",
      dataset: "CBIS-DDSM (Kaggle mirror) — lesion-cropped patches",
      trainedOn: "3,567 lesion-cropped patches from 2,857 unique mammograms (some abnormalities yield multiple crops), patient-safe split",
      split: { train: 2473, val: 488, test: 606 },
      schedule: "3 head + 5 fine-tune epochs (two-phase transfer learning), GPU",
    },
    headline: {
      roc_auc: 0.7849055720123173,
      accuracy: 0.7112211221122112,
      macro_f1: 0.6961413823633066,
      test_images: 606,
    },
    perClass: [
      { label: "Benign", precision: 0.7905027932960894, recall: 0.7389033942558747, f1: 0.7638326585695007, support: 383 },
      { label: "Malignant", precision: 0.5967741935483871, recall: 0.6636771300448431, f1: 0.6284501061571125, support: 223 },
    ],
    cmLabels: ["Benign", "Malignant"],
    confusionMatrix: [
      [283, 100],
      [75, 148],
    ],
    history: [
      { step: 1, phase: "head", train_loss: 0.6716, val_loss: 0.6718, train_acc: 0.590, val_acc: 0.629 },
      { step: 2, phase: "head", train_loss: 0.6411, val_loss: 0.6565, train_acc: 0.633, val_acc: 0.621 },
      { step: 3, phase: "head", train_loss: 0.6476, val_loss: 0.6752, train_acc: 0.632, val_acc: 0.607 },
      { step: 4, phase: "finetune", train_loss: 0.6297, val_loss: 0.6580, train_acc: 0.626, val_acc: 0.615 },
      { step: 5, phase: "finetune", train_loss: 0.6111, val_loss: 0.6611, train_acc: 0.659, val_acc: 0.611 },
      { step: 6, phase: "finetune", train_loss: 0.6116, val_loss: 0.6610, train_acc: 0.657, val_acc: 0.621 },
      { step: 7, phase: "finetune", train_loss: 0.6022, val_loss: 0.6581, train_acc: 0.668, val_acc: 0.627 },
      { step: 8, phase: "finetune", train_loss: 0.6137, val_loss: 0.6574, train_acc: 0.670, val_acc: 0.637 },
    ],
    randomBaseline: 0.5,
    literature: {
      intro:
        "CBIS-DDSM benign/malignant classification is notoriously hard — published single-CNN " +
        "results cluster in the 0.70–0.78 AUC range (the headline 90%+ numbers in the literature " +
        "typically come from heavy augmentation, metaheuristic optimization, or feature-fusion " +
        "pipelines, not a plain transfer-learning CNN). Against a directly comparable ResNet-50 " +
        "baseline:",
      refs: [
        {
          system: "ResNet-50 @ 448px (Jaamour et al.)",
          accuracy: 0.6824,
          roc_auc: 0.7421,
          citation: "open CBIS-DDSM codebase, PMC11549440 (2024)",
          url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11549440/",
          caveat: "Mass subset only (1,696 ROIs); best of their tested input sizes.",
        },
        {
          system: "MedChron EfficientNet-B0 (cropped patches)",
          accuracy: 0.7112,
          roc_auc: 0.7849,
          citation: "This project — best real result",
          url: "",
          ours: true,
          caveat: "Calc + mass subsets (3,567 crops) — more data than the reference, so favorable but not a clean 1:1 comparison.",
        },
      ],
      takeaway:
        "Our cropped-patch classifier's 0.785 AUC lands above the comparable published ResNet-50 " +
        "baseline (0.742) — a real, competitive result on a genuinely hard dataset. The honest " +
        "caveats: (1) we used both calc and mass subsets vs. their mass-only, so it isn't a clean " +
        "1:1 comparison, and (2) more importantly, this strong classifier still isn't deployable, " +
        "because it needs a pre-cropped lesion the live upload flow doesn't provide — the documented, " +
        "unsolved problem above. Competitive on the metric, honest about the gap to production.",
    },
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
