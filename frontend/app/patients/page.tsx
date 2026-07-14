"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Loader2, UserPlus, ChevronRight, FileStack, Clock } from "lucide-react";
import { createPatient, listPatients } from "@/lib/api";
import type { Patient } from "@/lib/types";

export default function PatientsPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [name, setName] = useState("");
  const [sex, setSex] = useState("");
  const [birthYear, setBirthYear] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      setPatients(await listPatients());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load patients");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function add(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await createPatient({
        name: name.trim(),
        sex: sex || undefined,
        birth_year: birthYear ? Number(birthYear) : undefined,
      });
      setName("");
      setSex("");
      setBirthYear("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create patient");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Patients</h1>

      <form onSubmit={add} className="card flex flex-wrap items-end gap-3 p-4">
        <Field label="Name">
          <input value={name} onChange={(e) => setName(e.target.value)} required
            className="input" placeholder="Jane Doe" />
        </Field>
        <Field label="Sex">
          <select value={sex} onChange={(e) => setSex(e.target.value)} className="input">
            <option value="">—</option>
            <option value="F">F</option>
            <option value="M">M</option>
            <option value="O">O</option>
          </select>
        </Field>
        <Field label="Birth year">
          <input value={birthYear} onChange={(e) => setBirthYear(e.target.value)} type="number"
            className="input w-28" placeholder="1990" />
        </Field>
        <button className="btn-primary" disabled={saving}>
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <UserPlus className="h-4 w-4" />}
          Add patient
        </button>
      </form>
      {error && <p className="text-sm text-bad">{error}</p>}

      <div className="card divide-y divide-line">
        {loading ? (
          <div className="grid place-items-center p-10 text-ink-4">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : patients.length === 0 ? (
          <p className="p-6 text-sm text-ink-4">No patients yet. Add one above.</p>
        ) : (
          patients.map((p) => (
            <Link key={p.id} href={`/patients/${p.id}`}
              className="flex items-center justify-between gap-3 p-4 transition hover:bg-surface">
              <div className="min-w-0">
                <p className="font-medium">{p.name}</p>
                <p className="text-xs text-ink-4">
                  {[p.sex, p.birth_year].filter(Boolean).join(" · ") || "—"}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-3">
                {p.study_count > 0 ? (
                  <>
                    <span className="hidden items-center gap-1 text-xs text-ink-3 sm:inline-flex">
                      <FileStack className="h-3.5 w-3.5" />
                      {p.study_count} scan{p.study_count === 1 ? "" : "s"}
                    </span>
                    {p.last_study_at && (
                      <span className="hidden items-center gap-1 text-xs text-ink-4 md:inline-flex">
                        <Clock className="h-3.5 w-3.5" />
                        {new Date(p.last_study_at).toLocaleDateString()}
                      </span>
                    )}
                    {p.last_label && (
                      <span className="rounded-full bg-surface-2 px-2 py-0.5 text-[11px] font-medium capitalize text-ink-2">
                        {p.last_label}
                      </span>
                    )}
                  </>
                ) : (
                  <span className="rounded-full bg-surface-2 px-2 py-0.5 text-[11px] text-ink-4">
                    No scans yet
                  </span>
                )}
                <ChevronRight className="h-4 w-4 text-ink-4" />
              </div>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
      {label}
      {children}
    </label>
  );
}
