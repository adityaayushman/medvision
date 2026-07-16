export interface DashboardUser {
  id: number;
  org_id: number;
  email: string;
  role: "admin" | "radiologist";
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
