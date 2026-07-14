import type { AnalyzeResponse, DatasetSpec, Health, Patient, StudyRead } from "./types";

// All requests are relative — next.config.mjs rewrites /api and /static to the
// FastAPI backend, so there is a single origin in dev and no CORS friction.

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function analyze(
  file: File,
  patientId?: number,
  modality?: string,
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  if (patientId) form.append("patient_id", String(patientId));
  if (modality) form.append("modality", modality);
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

export async function getPatient(patientId: number): Promise<Patient> {
  return jsonOrThrow<Patient>(await fetch(`/api/patients/${patientId}`, { cache: "no-store" }));
}

export async function getTimeline(patientId: number): Promise<StudyRead[]> {
  return jsonOrThrow<StudyRead[]>(
    await fetch(`/api/patients/${patientId}/timeline`, { cache: "no-store" }),
  );
}

export async function listDatasets(): Promise<DatasetSpec[]> {
  return jsonOrThrow<DatasetSpec[]>(await fetch("/api/datasets", { cache: "no-store" }));
}

export async function listStudies(): Promise<StudyRead[]> {
  return jsonOrThrow<StudyRead[]>(await fetch("/api/studies", { cache: "no-store" }));
}

export async function assignPatient(studyId: number, patientId: number): Promise<StudyRead> {
  const res = await fetch(`/api/studies/${studyId}/patient`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_id: patientId }),
  });
  return jsonOrThrow<StudyRead>(res);
}
