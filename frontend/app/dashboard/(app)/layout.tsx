"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";
import { DashboardNav } from "@/components/dashboard/DashboardNav";

export default function DashboardAppLayout({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) router.replace("/dashboard/login");
  }, [loading, user, router]);

  if (loading || !user) {
    return (
      <div className="grid min-h-[50vh] place-items-center text-ink-4">
        <Loader2 className="h-6 w-6 animate-spin" />
      </div>
    );
  }

  return (
    <>
      <DashboardNav />
      <div className="pt-6">{children}</div>
    </>
  );
}
