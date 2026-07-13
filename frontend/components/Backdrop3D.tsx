"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useTheme } from "./theme";

/* Global ambient 3D scene rendered behind every page:
   a slowly drifting particle field + floating wireframe polyhedra,
   with gentle mouse parallax. Theme-aware colours; static under
   prefers-reduced-motion. Kept deliberately sparse so text stays legible. */

type Palette = {
  particle: string;
  particleOpacity: number;
  wire: string;
  wireOpacity: number;
  blending: THREE.Blending;
};

const DARK: Palette = {
  particle: "#7cc0ff",
  particleOpacity: 0.5,
  wire: "#2f8fff",
  wireOpacity: 0.1,
  blending: THREE.AdditiveBlending,
};

const LIGHT: Palette = {
  particle: "#1670f5",
  particleOpacity: 0.32,
  wire: "#1670f5",
  wireOpacity: 0.13,
  blending: THREE.NormalBlending,
};

function Particles({ palette, animate }: { palette: Palette; animate: boolean }) {
  const group = useRef<THREE.Group>(null);
  const mouse = useRef({ x: 0, y: 0 });

  const { positions, count } = useMemo(() => {
    const N = 850;
    const arr = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 26;      // x: wide
      arr[i * 3 + 1] = (Math.random() - 0.5) * 15;  // y
      arr[i * 3 + 2] = (Math.random() - 0.5) * 10;  // z: depth
    }
    return { positions: arr, count: N };
  }, []);

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      mouse.current.x = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.current.y = (e.clientY / window.innerHeight) * 2 - 1;
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);

  useFrame((state, delta) => {
    if (!group.current || !animate) return;
    // slow drift + parallax easing toward the pointer
    group.current.rotation.y += delta * 0.012;
    group.current.rotation.x = THREE.MathUtils.lerp(
      group.current.rotation.x, mouse.current.y * 0.06, 0.02);
    group.current.rotation.z = THREE.MathUtils.lerp(
      group.current.rotation.z, mouse.current.x * 0.04, 0.02);
    group.current.position.y = Math.sin(state.clock.elapsedTime * 0.14) * 0.35;
  });

  return (
    <group ref={group}>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} count={count} />
        </bufferGeometry>
        <pointsMaterial
          color={palette.particle}
          size={0.035}
          sizeAttenuation
          transparent
          opacity={palette.particleOpacity}
          blending={palette.blending}
          depthWrite={false}
        />
      </points>
    </group>
  );
}

function FloatingShape({
  palette,
  animate,
  position,
  radius,
  detail = 1,
  speed = 1,
  kind = "ico",
}: {
  palette: Palette;
  animate: boolean;
  position: [number, number, number];
  radius: number;
  detail?: number;
  speed?: number;
  kind?: "ico" | "octa";
}) {
  const mesh = useRef<THREE.Mesh>(null);
  const base = useRef(position[1]);

  useFrame((state, delta) => {
    if (!mesh.current || !animate) return;
    mesh.current.rotation.x += delta * 0.06 * speed;
    mesh.current.rotation.y += delta * 0.09 * speed;
    mesh.current.position.y =
      base.current + Math.sin(state.clock.elapsedTime * 0.3 * speed + position[0]) * 0.4;
  });

  return (
    <mesh ref={mesh} position={position}>
      {kind === "ico" ? (
        <icosahedronGeometry args={[radius, detail]} />
      ) : (
        <octahedronGeometry args={[radius, detail]} />
      )}
      <meshBasicMaterial color={palette.wire} wireframe transparent opacity={palette.wireOpacity} />
    </mesh>
  );
}

export default function Backdrop3D() {
  const { theme } = useTheme();
  const palette = theme === "light" ? LIGHT : DARK;
  const [reduced, setReduced] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(mq.matches);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return (
    <Canvas
      camera={{ position: [0, 0, 9], fov: 50 }}
      dpr={[1, 1.5]}
      frameloop={reduced ? "demand" : "always"}
      gl={{ antialias: false, alpha: true, powerPreference: "low-power" }}
    >
      <Particles palette={palette} animate={!reduced} />
      <FloatingShape palette={palette} animate={!reduced} position={[-7.5, 2.6, -3]} radius={2.4} speed={0.8} />
      <FloatingShape palette={palette} animate={!reduced} position={[8, -2.8, -4]} radius={3.1} speed={0.6} />
      <FloatingShape palette={palette} animate={!reduced} position={[2.5, 4.2, -6]} radius={1.6} speed={1.2} kind="octa" />
    </Canvas>
  );
}
