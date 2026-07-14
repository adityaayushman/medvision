export interface QualityCheck {
  name: string;
  status: "ok" | "warn" | "fail";
  detail: string;
}

export interface QualityReport {
  focus: number;
  brightness: number;
  contrast: number;
  passed: boolean;
  reasons: string[];
  overall_score: number;
  focus_score: number;
  brightness_score: number;
  contrast_score: number;
  motion_blur_ratio: number;
  motion_blur_detected: boolean;
  checks: QualityCheck[];
  recommendation: string;
}

export interface PipelineStep {
  name: string;
  status: "done" | "skipped" | "stopped";
  detail: string;
}

export interface RoiConfidence {
  overall_score: number;
  overall_label: "High" | "Medium" | "Low";
  per_roi: number[];
  note: string;
}

export interface ProcessingMetadata {
  preprocessing_ops: string[];
  segmentation_success: boolean;
  foreground_ratio: number;
  roi_confidence: RoiConfidence | null;
  processing_time_ms: number;
  model_version: string | null;
  inference_time_ms: number | null;
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
  model_version?: string;
  inference_time_ms?: number;
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
  analysis_stopped: boolean;
  pipeline_steps: PipelineStep[];
  processing_metadata: ProcessingMetadata;
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
  quality_score?: number | null;
  analysis_stopped: boolean;
  model_version?: string | null;
  num_rois: number;
  image_url: string;
  annotated_url?: string | null;
  prediction?: Prediction | null;
}

export interface Health {
  status: string;
  model_loaded: boolean;
  modality: string;
  modalities: Record<string, boolean>;
  disclaimer: string;
}

export const MODALITY_LABELS: Record<string, string> = {
  chest_xray: "Chest X-ray",
  brain_mri: "Brain MRI",
  mammography: "Mammography",
};

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
