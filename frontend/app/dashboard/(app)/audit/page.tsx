"use client";

import { useEffect, useState } from "react";
import { Loader2, ScrollText, ShieldAlert } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";
import { listAuditLog } from "@/lib/dashboard-api";
import type { AuditLogRead } from "@/lib/dashboard-types";

export default function AuditPage() {
  const { token, user } = useAuth();
  const [entries, setEntries] = useState<AuditLogRead[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;
    listAuditLog(token)
      .then(setEntries)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load audit log"))
      .finally(() => setLoading(false));
  }, [token]);

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
        <ScrollText className="h-5 w-5 text-brand-600 dark:text-brand-400" />
        <h1 className="text-2xl font-bold">Audit trail</h1>
      </div>
      <p className="max-w-2xl text-sm text-ink-3">Who viewed and attached what, across this organization.</p>

      {loading ? (
        <div className="grid place-items-center p-10 text-ink-4">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      ) : error ? (
        <p className="text-sm text-bad">{error}</p>
      ) : entries.length === 0 ? (
        <p className="card p-6 text-sm text-ink-4">No activity recorded yet.</p>
      ) : (
        <div className="card divide-y divide-line">
          {entries.map((e) => (
            <div key={e.id} className="flex flex-wrap items-center justify-between gap-2 p-4 text-sm">
              <div>
                <span className="font-medium">{e.actor_email ?? `user #${e.actor_user_id}`}</span>{" "}
                <span className="text-ink-4">
                  {e.action.replace(/_/g, " ")} {e.target_type} #{e.target_id}
                </span>
              </div>
              <span className="text-xs text-ink-4">{new Date(e.created_at).toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
