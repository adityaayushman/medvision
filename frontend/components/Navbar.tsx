"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, BarChart3, Database, FileStack, ScanLine, Users } from "lucide-react";
import { cn } from "@/lib/utils";

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
  return (
    <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-[#05070e]/70 backdrop-blur-xl">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link href="/" className="group flex items-center gap-2.5 font-bold tracking-tight">
          <span className="relative grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-brand-400 to-brand-700 text-white shadow-[0_8px_20px_-8px_rgba(47,143,255,0.9)]">
            <span className="absolute inset-0 rounded-xl ring-1 ring-inset ring-white/25" />
            M
          </span>
          <span className="text-[15px]">
            MedChron<span className="text-brand-400"> AI</span>
          </span>
        </Link>

        <div className="flex items-center gap-1 rounded-full border border-white/[0.07] bg-white/[0.03] p-1 backdrop-blur">
          {LINKS.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm font-medium transition",
                  active
                    ? "bg-brand-500/20 text-brand-200 shadow-[inset_0_0_0_1px_rgba(120,170,255,0.25)]"
                    : "text-slate-400 hover:bg-white/[0.06] hover:text-slate-100",
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
