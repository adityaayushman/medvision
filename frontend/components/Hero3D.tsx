"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

/** A rotating point-cloud sphere — evokes a volumetric scan / feature space. */
function ScanVolume() {
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
          color="#7cc0ff"
          size={0.022}
          sizeAttenuation
          transparent
          opacity={0.85}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
        />
      </points>

      <mesh ref={shell}>
        <icosahedronGeometry args={[2.35, 1]} />
        <meshBasicMaterial color="#2f8fff" wireframe transparent opacity={0.14} />
      </mesh>

      <mesh ref={core}>
        <icosahedronGeometry args={[0.7, 2]} />
        <meshStandardMaterial
          color="#1670f5"
          emissive="#2f8fff"
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
  return (
    <Canvas
      camera={{ position: [0, 0, 6], fov: 45 }}
      dpr={[1, 1.8]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
    >
      <ambientLight intensity={0.6} />
      <pointLight position={[4, 3, 5]} intensity={2.2} color="#8ec5ff" />
      <pointLight position={[-5, -2, -3]} intensity={1.4} color="#6366f1" />
      <ScanVolume />
    </Canvas>
  );
}
