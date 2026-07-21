"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ClipboardList, FlaskConical, LogOut, ScrollText, Users, UserCog } from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "@/components/theme";
import type { DashboardRole } from "@/lib/dashboard-types";
import { useAuth } from "./AuthContext";

const LINKS: { href: string; label: string; icon: typeof ClipboardList; allowedRoles?: DashboardRole[] }[] = [
  { href: "/dashboard/queue", label: "Queue", icon: ClipboardList },
  { href: "/dashboard/patients", label: "Patients", icon: Users },
  { href: "/dashboard/research", label: "Research", icon: FlaskConical, allowedRoles: ["admin", "researcher"] },
  { href: "/dashboard/team", label: "Team", icon: UserCog, allowedRoles: ["admin"] },
  { href: "/dashboard/audit", label: "Audit", icon: ScrollText, allowedRoles: ["admin"] },
];

export function DashboardNav() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useAuth();

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-page/70 backdrop-blur-xl">
      <nav className="mx-auto flex max-w-6xl items-center justify-between gap-2 px-4 py-3">
        <Link href="/dashboard/queue" className="group flex items-center gap-2.5 font-bold tracking-tight">
          <span className="relative grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-brand-400 to-brand-700 text-white shadow-[0_8px_20px_-8px_rgba(47,143,255,0.9)]">
            <span className="absolute inset-0 rounded-xl ring-1 ring-inset ring-white/25" />
            M
          </span>
          <span className="text-[15px]">
            MedChron<span className="text-brand-500 dark:text-brand-400"> Dashboard</span>
          </span>
        </Link>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-full border border-line bg-surface p-1 backdrop-blur">
            {LINKS.filter((l) => !l.allowedRoles || (user && l.allowedRoles.includes(user.role))).map(({ href, label, icon: Icon }) => {
              const active = pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-medium transition",
                    active
                      ? "bg-brand-500/20 text-brand-700 shadow-[inset_0_0_0_1px_rgba(120,170,255,0.35)] dark:text-brand-200"
                      : "text-ink-3 hover:bg-surface-2 hover:text-ink",
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{label}</span>
                </Link>
              );
            })}
          </div>
          {user && (
            <span className="hidden text-xs text-ink-4 md:inline" title={user.email}>
              {user.email} · {user.role}
            </span>
          )}
          <button
            type="button"
            onClick={() => {
              logout();
              router.replace("/dashboard/login");
            }}
            aria-label="Log out"
            title="Log out"
            className="grid h-9 w-9 place-items-center rounded-full border border-line bg-surface text-ink-3 transition hover:bg-surface-2 hover:text-ink"
          >
            <LogOut className="h-4 w-4" />
          </button>
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
