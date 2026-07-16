"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Loader2, LogIn } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";

export default function DashboardLoginPage() {
  const { login } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(email, password);
      router.push("/dashboard/queue");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="card space-y-5 p-6">
      <div>
        <h1 className="text-xl font-bold">Clinical dashboard sign in</h1>
        <p className="mt-1 text-sm text-ink-4">Research/educational software — not for clinical use.</p>
      </div>
      <form onSubmit={onSubmit} className="space-y-3">
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
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="input"
          />
        </label>
        {error && <p className="text-sm text-bad">{error}</p>}
        <button className="btn-primary w-full" disabled={submitting}>
          {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogIn className="h-4 w-4" />}
          Sign in
        </button>
      </form>
      <p className="text-center text-sm text-ink-4">
        No organization yet?{" "}
        <Link href="/dashboard/signup" className="text-brand-600 hover:underline dark:text-brand-400">
          Create one
        </Link>
      </p>
    </div>
  );
}
