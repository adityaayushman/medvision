export interface QualityReport {
  focus: number;
  brightness: number;
  contrast: number;
  passed: boolean;
  reasons: string[];
}

export interface ROI {
  bbox: number[]; // [x, y, w, h]
  area: number;
}

export interface Prediction {
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  backbone?: string;
  explained_class?: string;
}

export interface AnalyzeResponse {
  study_id?: number;
  modality: string;
  model_loaded: boolean;
  quality: QualityReport;
  num_rois: number;
  rois: ROI[];
  prediction?: Prediction | null;
  image_url: string;
  annotated_url: string;
  heatmap_url?: string | null;
  stages?: { name: string; url: string }[];
}

export interface Patient {
  id: number;
  name: string;
  sex?: string | null;
  birth_year?: number | null;
  created_at: string;
  study_count: number;
  last_study_at?: string | null;
  last_label?: string | null;
}

export interface StudyRead {
  id: number;
  patient_id?: number | null;
  patient_name?: string | null;
  modality: string;
  uploaded_at: string;
  quality_passed: boolean;
  num_rois: number;
  image_url: string;
  annotated_url?: string | null;
  prediction?: Prediction | null;
}

export interface Health {
  status: string;
  model_loaded: boolean;
  modality: string;
  disclaimer: string;
}

export interface DatasetSpec {
  key: string;
  name: string;
  modality: string;
  task: string;
  access: "open" | "kaggle" | "credentialed";
  roi_support: boolean;
  approx_images: string;
  url: string;
  notes: string;
  recommended_for: string[];
}
