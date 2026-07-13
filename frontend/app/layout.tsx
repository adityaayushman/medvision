import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Navbar } from "@/components/Navbar";

const inter = Inter({ subsets: ["latin"], display: "swap" });

export const metadata: Metadata = {
  title: "MedChron AI — Medical Imaging Intelligence",
  description:
    "Explainable medical image analysis: preprocessing, ROI, classification, Grad-CAM, and a longitudinal patient record. Research/educational use only.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`ambient min-h-screen ${inter.className}`}>
        <Navbar />
        <main className="mx-auto max-w-6xl px-4 pb-16 pt-6">{children}</main>
        <footer className="mx-auto max-w-6xl px-4 py-10 text-xs text-muted">
          MedChron AI · Research &amp; educational software —{" "}
          <strong className="text-slate-300">not a medical device and not for clinical use.</strong>
        </footer>
      </body>
    </html>
  );
}
