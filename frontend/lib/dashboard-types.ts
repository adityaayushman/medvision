export type DashboardRole = "admin" | "radiologist" | "researcher";

export interface DashboardUser {
  id: number;
  org_id: number;
  email: string;
  role: DashboardRole;
  name?: string | null;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: DashboardUser;
}

export interface AuditLogRead {
  id: number;
  actor_user_id: number;
  actor_email?: string | null;
  action: string;
  target_type: string;
  target_id: number;
  meta?: Record<string, unknown> | null;
  created_at: string;
}

export type ExperimentKind = "classification" | "bbox_regression" | "segmentation" | "ensemble";

export interface ExperimentRunRead {
  id: number;
  kind: ExperimentKind;
  modality: string;
  backbone: string;
  label: string;
  metrics: Record<string, unknown>;
  notes?: string | null;
  created_by_user_id: number;
  created_by_email?: string | null;
  created_at: string;
}
