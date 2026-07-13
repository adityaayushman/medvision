"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useTheme } from "./theme";

/* Theme palettes: additive light-blue glow on dark; solid deeper blues on light
   (additive blending washes out on a light background). */
const PALETTES = {
  dark: {
    particle: "#7cc0ff", particleOpacity: 0.85, blending: THREE.AdditiveBlending,
    shell: "#2f8fff", shellOpacity: 0.14,
    core: "#1670f5", emissive: "#2f8fff",
  },
  light: {
    particle: "#1670f5", particleOpacity: 0.55, blending: THREE.NormalBlending,
    shell: "#0f59e1", shellOpacity: 0.22,
    core: "#2f8fff", emissive: "#59b0ff",
  },
} as const;

type Palette = (typeof PALETTES)[keyof typeof PALETTES];

/** A rotating point-cloud sphere — evokes a volumetric scan / feature space. */
function ScanVolume({ palette }: { palette: Palette }) {
  const points = useRef<THREE.Points>(null);
  const shell = useRef<THREE.Mesh>(null);
  const core = useRef<THREE.Mesh>(null);

  const positions = useMemo(() => {
    const N = 2600;
    const arr = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      // fill a ball (cbrt for uniform radial density)
      const r = 2.15 * Math.cbrt(Math.random());
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      arr[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      arr[i * 3 + 2] = r * Math.cos(phi);
    }
    return arr;
  }, []);

  useFrame((state, delta) => {
    if (points.current) {
      points.current.rotation.y += delta * 0.09;
      points.current.rotation.x += delta * 0.025;
    }
    if (shell.current) {
      shell.current.rotation.y -= delta * 0.05;
      shell.current.rotation.z += delta * 0.02;
    }
    if (core.current) {
      const s = 1 + Math.sin(state.clock.elapsedTime * 1.2) * 0.04;
      core.current.scale.setScalar(s);
    }
  });

  return (
    <group>
      <points ref={points}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial
          color={palette.particle}
          size={0.022}
          sizeAttenuation
          transparent
          opacity={palette.particleOpacity}
          blending={palette.blending}
          depthWrite={false}
        />
      </points>

      <mesh ref={shell}>
        <icosahedronGeometry args={[2.35, 1]} />
        <meshBasicMaterial color={palette.shell} wireframe transparent opacity={palette.shellOpacity} />
      </mesh>

      <mesh ref={core}>
        <icosahedronGeometry args={[0.7, 2]} />
        <meshStandardMaterial
          color={palette.core}
          emissive={palette.emissive}
          emissiveIntensity={0.6}
          roughness={0.35}
          metalness={0.4}
          transparent
          opacity={0.9}
        />
      </mesh>
    </group>
  );
}

export default function Hero3D() {
  const { theme } = useTheme();
  const palette = PALETTES[theme];
  return (
    <Canvas
      camera={{ position: [0, 0, 6], fov: 45 }}
      dpr={[1, 1.8]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
    >
      <ambientLight intensity={0.6} />
      <pointLight position={[4, 3, 5]} intensity={2.2} color="#8ec5ff" />
      <pointLight position={[-5, -2, -3]} intensity={1.4} color="#6366f1" />
      <ScanVolume palette={palette} />
    </Canvas>
  );
}
