"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, BarChart3, Database, FileStack, ScanLine, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "./theme";

const LINKS = [
  { href: "/", label: "Overview", icon: Activity },
  { href: "/analyze", label: "Analyze", icon: ScanLine },
  { href: "/records", label: "Records", icon: FileStack },
  { href: "/evaluation", label: "Evaluation", icon: BarChart3 },
  { href: "/patients", label: "Patients", icon: Users },
  { href: "/datasets", label: "Datasets", icon: Database },
];

export function Navbar() {
  const pathname = usePathname();
  if (pathname.startsWith("/dashboard")) return null;
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-page/70 backdrop-blur-xl">
      <nav className="mx-auto flex max-w-6xl items-center justify-between gap-2 px-4 py-3">
        <Link href="/" className="group flex items-center gap-2.5 font-bold tracking-tight">
          <span className="relative grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-brand-400 to-brand-700 text-white shadow-[0_8px_20px_-8px_rgba(47,143,255,0.9)]">
            <span className="absolute inset-0 rounded-xl ring-1 ring-inset ring-white/25" />
            M
          </span>
          <span className="text-[15px]">
            MedChron<span className="text-brand-500 dark:text-brand-400"> AI</span>
          </span>
        </Link>

        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-full border border-line bg-surface p-1 backdrop-blur">
            {LINKS.map(({ href, label, icon: Icon }) => {
              const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
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
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
