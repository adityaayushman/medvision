"use client";

import { useEffect, useState } from "react";
import { Loader2, ShieldAlert, UserCog, UserPlus } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";
import { createTeammate, listTeammates } from "@/lib/dashboard-api";
import type { DashboardUser } from "@/lib/dashboard-types";

export default function TeamPage() {
  const { token, user } = useAuth();
  const [teammates, setTeammates] = useState<DashboardUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<"admin" | "radiologist">("radiologist");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    if (!token) return;
    try {
      setTeammates(await listTeammates(token));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load team");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function add(e: React.FormEvent) {
    e.preventDefault();
    if (!token) return;
    setSaving(true);
    setError(null);
    try {
      await createTeammate(token, { email, password, role });
      setEmail("");
      setPassword("");
      setRole("radiologist");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add teammate");
    } finally {
      setSaving(false);
    }
  }

  if (user && user.role !== "admin") {
    return (
      <p className="card flex items-center gap-2 p-6 text-sm text-bad">
        <ShieldAlert className="h-4 w-4" /> Admin access required.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <UserCog className="h-5 w-5 text-brand-600 dark:text-brand-400" />
        <h1 className="text-2xl font-bold">Team</h1>
      </div>

      <form onSubmit={add} className="card flex flex-wrap items-end gap-3 p-4">
        <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
          Email
          <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} className="input" placeholder="rad@hospital.org" />
        </label>
        <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
          Initial password
          <input type="password" required minLength={6} value={password} onChange={(e) => setPassword(e.target.value)} className="input" />
        </label>
        <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
          Role
          <select value={role} onChange={(e) => setRole(e.target.value as "admin" | "radiologist")} className="input">
            <option value="radiologist">Radiologist</option>
            <option value="admin">Admin</option>
          </select>
        </label>
        <button className="btn-primary" disabled={saving}>
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <UserPlus className="h-4 w-4" />}
          Add teammate
        </button>
      </form>
      {error && <p className="text-sm text-bad">{error}</p>}

      <div className="card divide-y divide-line">
        {loading ? (
          <div className="grid place-items-center p-10 text-ink-4">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : teammates.length === 0 ? (
          <p className="p-6 text-sm text-ink-4">No teammates yet.</p>
        ) : (
          teammates.map((u) => (
            <div key={u.id} className="flex items-center justify-between gap-3 p-4">
              <div className="min-w-0">
                <p className="font-medium">{u.name || u.email}</p>
                <p className="text-xs text-ink-4">{u.email}</p>
              </div>
              <span className="rounded-full bg-surface-2 px-2 py-0.5 text-[11px] font-medium capitalize text-ink-3">{u.role}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
