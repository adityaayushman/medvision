"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ScanLine,
  Sparkles,
  Layers,
  Eye,
  History,
  ShieldCheck,
  ArrowRight,
  Activity,
  CircleDot,
} from "lucide-react";
import HeroCanvas from "./HeroCanvas";
import { getHealth } from "@/lib/api";
import type { Health } from "@/lib/types";

const PIPELINE = [
  "Quality check",
  "Denoise + CLAHE",
  "Segmentation",
  "ROI extraction",
  "Classify",
  "Grad-CAM",
  "Patient record",
];

const FEATURES = [
  {
    icon: Layers,
    title: "Full DIP pipeline",
    body: "Every scan is quality-gated, denoised, CLAHE-enhanced, segmented and reduced to regions of interest before the model ever sees it.",
  },
  {
    icon: Eye,
    title: "Explainable, not opaque",
    body: "Grad-CAM overlays show where the model looked, alongside calibrated class probabilities — the why, not just the what.",
  },
  {
    icon: History,
    title: "Longitudinal record",
    body: "Each patient accumulates a timeline of studies and predictions, turning one-shot diagnosis into disease monitoring.",
  },
  {
    icon: Sparkles,
    title: "Modality-agnostic",
    body: "Chest X-ray ships first; brain MRI and mammography are configuration presets, not rewrites.",
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0 },
};

export default function Landing() {
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  return (
    <div className="space-y-16">
      {/* ---------- HERO ---------- */}
      <section className="card relative overflow-hidden">
        {/* 3D scene, right-anchored, fading into the card */}
        <div className="absolute inset-y-0 right-0 w-full sm:w-[62%]">
          <HeroCanvas />
          <div className="absolute inset-0 bg-gradient-to-r from-hero via-hero/70 to-transparent" />
        </div>

        <div className="relative px-6 py-14 sm:px-10 sm:py-20">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="mb-5 inline-flex items-center gap-2 rounded-full border border-line bg-surface-2 px-3 py-1 text-xs font-medium text-ink-2 backdrop-blur"
          >
            <ShieldCheck className="h-3.5 w-3.5 text-brand-600 dark:text-brand-400" />
            Research &amp; educational · not for clinical use
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.05 }}
            className="max-w-2xl text-4xl font-bold leading-[1.08] tracking-tight sm:text-6xl"
          >
            <span className="text-gradient">Medical imaging intelligence,</span>
            <br />
            from raw scan to explainable insight.
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.12 }}
            className="mt-5 max-w-xl text-base text-ink-2/90 sm:text-lg"
          >
            MedChron AI processes a medical image, extracts the clinically relevant
            region, classifies it with an explainable model, and files the result
            into a longitudinal patient record.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mt-8 flex flex-wrap items-center gap-3"
          >
            <Link href="/analyze" className="btn-primary">
              <ScanLine className="h-4 w-4" /> Analyze a scan
            </Link>
            <Link href="/patients" className="btn-ghost">
              View patients <ArrowRight className="h-4 w-4" />
            </Link>
          </motion.div>

          {/* live model status */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mt-8 inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-xs backdrop-blur"
          >
            <CircleDot
              className={`h-3.5 w-3.5 ${health?.model_loaded ? "text-ok" : "text-warn"}`}
            />
            <span className="text-ink-2">
              {health === null
                ? "Connecting to inference API…"
                : health.model_loaded
                  ? "Live model: RSNA chest X-ray classifier (EfficientNet-B0)"
                  : "API online · preprocess-only mode (no model loaded)"}
            </span>
          </motion.div>
        </div>
      </section>

      {/* ---------- PIPELINE ---------- */}
      <section>
        <SectionLabel icon={Activity}>The pipeline</SectionLabel>
        <motion.div
          variants={{ show: { transition: { staggerChildren: 0.05 } } }}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          className="flex flex-wrap items-center gap-2"
        >
          {PIPELINE.map((step, i) => (
            <motion.div key={step} variants={fadeUp} className="flex items-center gap-2">
              <span className="rounded-xl border border-line bg-surface px-3.5 py-2 text-sm font-medium text-ink backdrop-blur">
                {step}
              </span>
              {i < PIPELINE.length - 1 && <ArrowRight className="h-4 w-4 text-ink-5" />}
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ---------- FEATURES ---------- */}
      <section>
        <SectionLabel icon={Sparkles}>What makes it different</SectionLabel>
        <motion.div
          variants={{ show: { transition: { staggerChildren: 0.08 } } }}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-80px" }}
          className="grid gap-4 sm:grid-cols-2"
        >
          {FEATURES.map(({ icon: Icon, title, body }) => (
            <motion.div key={title} variants={fadeUp} className="card card-hover p-6">
              <div className="mb-4 grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-brand-500/25 to-brand-700/10 text-brand-600 dark:text-brand-300 ring-1 ring-inset ring-line">
                <Icon className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold">{title}</h3>
              <p className="mt-1.5 text-sm text-ink-3">{body}</p>
            </motion.div>
          ))}
        </motion.div>
      </section>

      {/* ---------- CTA ---------- */}
      <motion.section
        variants={fadeUp}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true }}
        className="card relative overflow-hidden px-6 py-10 text-center sm:px-10 sm:py-14"
      >
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_120%_at_50%_0%,rgba(47,143,255,0.18),transparent_60%)]" />
        <div className="relative">
          <h2 className="text-2xl font-bold sm:text-3xl">See the whole pipeline on one scan.</h2>
          <p className="mx-auto mt-2 max-w-lg text-sm text-ink-3">
            Upload a chest X-ray and watch quality assessment, ROI extraction,
            classification and Grad-CAM run end to end.
          </p>
          <Link href="/analyze" className="btn-primary mx-auto mt-6">
            <ScanLine className="h-4 w-4" /> Try the analyzer
          </Link>
        </div>
      </motion.section>
    </div>
  );
}

function SectionLabel({ icon: Icon, children }: { icon: any; children: React.ReactNode }) {
  return (
    <div className="mb-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-ink-4">
      <Icon className="h-3.5 w-3.5" />
      {children}
    </div>
  );
}
