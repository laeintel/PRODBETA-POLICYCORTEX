'use client';

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Box, Text } from '@react-three/drei';
import * as THREE from 'three';

interface ComplianceCell {
  x: number;
  z: number;
  compliance: number;
  policy: string;
  status: 'compliant' | 'warning' | 'violation';
}

const ComplianceBox: React.FC<{ cell: ComplianceCell }> = ({ cell }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      const targetHeight = cell.compliance / 100 * 2;
      meshRef.current.scale.y = THREE.MathUtils.lerp(
        meshRef.current.scale.y,
        hovered ? targetHeight * 1.2 : targetHeight,
        0.1
      );
      
      // Pulse effect for violations
      if (cell.status === 'violation') {
        const pulse = Math.sin(state.clock.elapsedTime * 3) * 0.1 + 1;
        meshRef.current.scale.x = pulse;
        meshRef.current.scale.z = pulse;
      }
    }
  });

  const colors = {
    compliant: '#00FF88',
    warning: '#FFB800',
    violation: '#FF0040',
  };

  const color = colors[cell.status];
  const height = cell.compliance / 100 * 2;

  return (
    <group position={[cell.x * 1.2, height / 2, cell.z * 1.2]}>
      <Box
        ref={meshRef}
        args={[1, height, 1]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshPhongMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.2}
          transparent
          opacity={0.8}
        />
      </Box>
      {hovered && (
        <Text
          position={[0, height / 2 + 0.5, 0]}
          fontSize={0.15}
          color="#F0F9FF"
          anchorX="center"
        >
          {cell.policy}
          {'\n'}
          {cell.compliance}%
        </Text>
      )}
    </group>
  );
};

import { useState } from 'react';

const ComplianceGrid: React.FC = () => {
  const cells = useMemo(() => {
    const policies = [
      'Data Encryption', 'Access Control', 'Audit Logging',
      'Network Security', 'Backup Policy', 'Disaster Recovery',
      'Identity Management', 'Compliance Monitoring', 'Vulnerability Scanning'
    ];
    
    const grid: ComplianceCell[] = [];
    for (let x = -2; x <= 2; x++) {
      for (let z = -2; z <= 2; z++) {
        const compliance = Math.floor(Math.random() * 40) + 60;
        let status: 'compliant' | 'warning' | 'violation';
        if (compliance >= 90) status = 'compliant';
        else if (compliance >= 70) status = 'warning';
        else status = 'violation';
        
        grid.push({
          x,
          z,
          compliance,
          policy: policies[Math.floor(Math.random() * policies.length)],
          status,
        });
      }
    }
    return grid;
  }, []);

  return (
    <Canvas camera={{ position: [5, 5, 5], fov: 50 }}>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.7} />
      <pointLight position={[-10, 10, -10]} intensity={0.5} color="#8B5CF6" />
      
      {/* Grid base */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <planeGeometry args={[7, 7]} />
        <meshBasicMaterial color="#0A0E27" transparent opacity={0.8} />
      </mesh>
      
      {/* Grid lines */}
      <gridHelper args={[7, 7, '#00D4FF', '#1A1F3A']} />
      
      {/* Compliance cells */}
      {cells.map((cell, idx) => (
        <ComplianceBox key={idx} cell={cell} />
      ))}
      
      <fog attach="fog" args={['#0A0E27', 5, 15]} />
    </Canvas>
  );
};

export default ComplianceGrid;