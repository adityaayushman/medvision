"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { useAuth } from "@/components/dashboard/AuthContext";

export default function DashboardIndex() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading) return;
    router.replace(user ? "/dashboard/queue" : "/dashboard/login");
  }, [loading, user, router]);

  return (
    <div className="grid min-h-[50vh] place-items-center text-ink-4">
      <Loader2 className="h-6 w-6 animate-spin" />
    </div>
  );
}
