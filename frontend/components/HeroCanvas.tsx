"use client";

import React from "react";
import dynamic from "next/dynamic";

// WebGL must be client-only (no SSR).
const Hero3D = dynamic(() => import("./Hero3D"), { ssr: false, loading: () => null });

/** If WebGL is unavailable or the scene throws, render nothing and let the
 *  gradient behind the canvas carry the hero — the page never breaks. */
class SafeBoundary extends React.Component<{ children: React.ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export default function HeroCanvas() {
  return (
    <div className="pointer-events-none absolute inset-0">
      <SafeBoundary>
        <Hero3D />
      </SafeBoundary>
    </div>
  );
}
