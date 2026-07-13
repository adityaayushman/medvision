import type { AnalyzeResponse, Health, Patient, StudyRead } from "./types";

// All requests are relative — next.config.mjs rewrites /api and /static to the
// FastAPI backend, so there is a single origin in dev and no CORS friction.

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function analyze(file: File, patientId?: number): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  if (patientId) form.append("patient_id", String(patientId));
  const res = await fetch("/api/analyze", { method: "POST", body: form });
  return jsonOrThrow<AnalyzeResponse>(res);
}

export async function getHealth(): Promise<Health> {
  return jsonOrThrow<Health>(await fetch("/health", { cache: "no-store" }));
}

export async function listPatients(): Promise<Patient[]> {
  return jsonOrThrow<Patient[]>(await fetch("/api/patients", { cache: "no-store" }));
}

export async function createPatient(data: {
  name: string;
  sex?: string;
  birth_year?: number;
}): Promise<Patient> {
  const res = await fetch("/api/patients", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return jsonOrThrow<Patient>(res);
}

export async function getTimeline(patientId: number): Promise<StudyRead[]> {
  return jsonOrThrow<StudyRead[]>(
    await fetch(`/api/patients/${patientId}/timeline`, { cache: "no-store" }),
  );
}
