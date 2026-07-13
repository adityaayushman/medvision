"use client";

import React from "react";
import dynamic from "next/dynamic";

// WebGL must be client-only (no SSR).
const Backdrop3D = dynamic(() => import("./Backdrop3D"), { ssr: false, loading: () => null });

/** If WebGL is unavailable or the scene throws, render nothing — the CSS
 *  aurora/grid behind it still carries the page. Never breaks the site. */
class SafeBoundary extends React.Component<{ children: React.ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export default function BackdropCanvas() {
  return (
    // z-[-1]: above the CSS aurora (-2/-3), below all page content
    <div aria-hidden className="pointer-events-none fixed inset-0 z-[-1]">
      <SafeBoundary>
        <Backdrop3D />
      </SafeBoundary>
    </div>
  );
}
