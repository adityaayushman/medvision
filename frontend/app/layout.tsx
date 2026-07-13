import "./globals.css";
import type { Metadata } from "next";
import { Navbar } from "@/components/Navbar";

export const metadata: Metadata = {
  title: "MedChron AI — Medical Imaging Intelligence",
  description:
    "Explainable medical image analysis: preprocessing, ROI, VGG16 classification, Grad-CAM, and a longitudinal patient record. Research/educational use only.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Navbar />
        <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
        <footer className="mx-auto max-w-6xl px-4 py-10 text-xs text-slate-500">
          MedChron AI · Research &amp; educational software — <strong>not a medical
          device and not for clinical use.</strong>
        </footer>
      </body>
    </html>
  );
}
