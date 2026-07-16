"use client";

import React from "react";
import dynamic from "next/dynamic";

const Backdrop3D = dynamic(() => import("./Backdrop3D"), { ssr: false, loading: () => null });

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
    <div aria-hidden className="pointer-events-none fixed inset-0 z-[-1]">
      <SafeBoundary>
        <Backdrop3D />
      </SafeBoundary>
    </div>
  );
}
