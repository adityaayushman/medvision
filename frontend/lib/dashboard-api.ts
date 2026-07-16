import type { Patient, ReviewStatus, StudyRead } from "./types";
import type { AuditLogRead, DashboardUser, TokenResponse } from "./dashboard-types";

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

function authHeaders(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

export async function signup(orgName: string, email: string, password: string, name?: string): Promise<TokenResponse> {
  const res = await fetch("/api/auth/signup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ org_name: orgName, email, password, name }),
  });
  return jsonOrThrow<TokenResponse>(res);
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  const res = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  return jsonOrThrow<TokenResponse>(res);
}

export async function getMe(token: string): Promise<DashboardUser> {
  return jsonOrThrow<DashboardUser>(
    await fetch("/api/auth/me", { headers: authHeaders(token), cache: "no-store" }),
  );
}

export async function listTeammates(token: string): Promise<DashboardUser[]> {
  return jsonOrThrow<DashboardUser[]>(
    await fetch("/api/auth/users", { headers: authHeaders(token), cache: "no-store" }),
  );
}

export async function createTeammate(
  token: string,
  data: { email: string; password: string; role: "admin" | "radiologist"; name?: string },
): Promise<DashboardUser> {
  const res = await fetch("/api/auth/users", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(data),
  });
  return jsonOrThrow<DashboardUser>(res);
}

export async function listDashboardPatients(token: string): Promise<Patient[]> {
  return jsonOrThrow<Patient[]>(
    await fetch("/api/dashboard/patients", { headers: authHeaders(token), cache: "no-store" }),
  );
}

export async function createDashboardPatient(
  token: string,
  data: { name: string; sex?: string; birth_year?: number },
): Promise<Patient> {
  const res = await fetch("/api/dashboard/patients", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify(data),
  });
  return jsonOrThrow<Patient>(res);
}

export async function getDashboardPatient(token: string, patientId: number): Promise<Patient> {
  return jsonOrThrow<Patient>(
    await fetch(`/api/dashboard/patients/${patientId}`, { headers: authHeaders(token), cache: "no-store" }),
  );
}

export async function getDashboardTimeline(token: string, patientId: number): Promise<StudyRead[]> {
  return jsonOrThrow<StudyRead[]>(
    await fetch(`/api/dashboard/patients/${patientId}/timeline`, {
      headers: authHeaders(token),
      cache: "no-store",
    }),
  );
}

export async function analyzeDashboardStudy(
  token: string,
  file: File,
  patientId?: number,
  modality?: string,
): Promise<StudyRead & { study_id?: number }> {
  const form = new FormData();
  form.append("file", file);
  if (patientId) form.append("patient_id", String(patientId));
  if (modality) form.append("modality", modality);
  const res = await fetch("/api/dashboard/studies/analyze", {
    method: "POST",
    headers: authHeaders(token),
    body: form,
  });
  return jsonOrThrow(res);
}

export async function listDashboardStudies(token: string, reviewStatus?: ReviewStatus): Promise<StudyRead[]> {
  const qs = reviewStatus ? `?review_status=${reviewStatus}` : "";
  return jsonOrThrow<StudyRead[]>(
    await fetch(`/api/dashboard/studies${qs}`, { headers: authHeaders(token), cache: "no-store" }),
  );
}

export async function getDashboardStudy(token: string, studyId: number): Promise<StudyRead> {
  return jsonOrThrow<StudyRead>(
    await fetch(`/api/dashboard/studies/${studyId}`, { headers: authHeaders(token), cache: "no-store" }),
  );
}

export async function updateReviewStatus(
  token: string,
  studyId: number,
  reviewStatus: ReviewStatus,
  note?: string,
): Promise<StudyRead> {
  const res = await fetch(`/api/dashboard/studies/${studyId}/review`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ review_status: reviewStatus, note }),
  });
  return jsonOrThrow<StudyRead>(res);
}

export async function assignDashboardPatient(token: string, studyId: number, patientId: number): Promise<StudyRead> {
  const res = await fetch(`/api/dashboard/studies/${studyId}/patient`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders(token) },
    body: JSON.stringify({ patient_id: patientId }),
  });
  return jsonOrThrow<StudyRead>(res);
}

export async function listAuditLog(token: string): Promise<AuditLogRead[]> {
  return jsonOrThrow<AuditLogRead[]>(
    await fetch("/api/dashboard/audit", { headers: authHeaders(token), cache: "no-store" }),
  );
}
