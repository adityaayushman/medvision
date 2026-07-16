"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Building2, Loader2 } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";

export default function DashboardSignupPage() {
  const { signup } = useAuth();
  const router = useRouter();
  const [orgName, setOrgName] = useState("");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await signup(orgName, email, password, name || undefined);
      router.push("/dashboard/queue");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Signup failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="card space-y-5 p-6">
      <div>
        <h1 className="text-xl font-bold">Create your organization</h1>
        <p className="mt-1 text-sm text-ink-4">
          You become the admin. Add radiologist teammates from the Team page afterward.
        </p>
      </div>
      <form onSubmit={onSubmit} className="space-y-3">
        <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
          Organization name
          <input
            required
            value={orgName}
            onChange={(e) => setOrgName(e.target.value)}
            className="input"
            placeholder="Acme Radiology"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
          Your name (optional)
          <input value={name} onChange={(e) => setName(e.target.value)} className="input" placeholder="Ada Admin" />
        </label>
        <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
          Email
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="input"
            placeholder="you@hospital.org"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs font-medium text-ink-4">
          Password
          <input
            type="password"
            required
            minLength={6}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="input"
          />
        </label>
        {error && <p className="text-sm text-bad">{error}</p>}
        <button className="btn-primary w-full" disabled={submitting}>
          {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Building2 className="h-4 w-4" />}
          Create organization
        </button>
      </form>
      <p className="text-center text-sm text-ink-4">
        Already have an account?{" "}
        <Link href="/dashboard/login" className="text-brand-600 hover:underline dark:text-brand-400">
          Sign in
        </Link>
      </p>
    </div>
  );
}
