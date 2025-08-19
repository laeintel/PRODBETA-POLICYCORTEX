'use client';

import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Box, Sphere, Line, MeshDistortMaterial } from '@react-three/drei';
import * as THREE from 'three';

interface ResourceNode {
  id: string;
  name: string;
  type: string;
  position: [number, number, number];
  connections: string[];
  status: 'healthy' | 'warning' | 'critical';
  size: number;
}

const ResourceSphere: React.FC<{ node: ResourceNode }> = ({ node }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.01;
      if (hovered) {
        meshRef.current.scale.lerp(new THREE.Vector3(1.2, 1.2, 1.2), 0.1);
      } else {
        meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1);
      }
    }
  });

  const color = {
    healthy: '#00FF88',
    warning: '#FFB800',
    critical: '#FF0040',
  }[node.status];

  return (
    <group position={node.position}>
      <Sphere
        ref={meshRef}
        args={[node.size, 32, 32]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <MeshDistortMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.5}
          roughness={0.1}
          metalness={0.8}
          distort={0.2}
          speed={2}
        />
      </Sphere>
      {hovered && (
        <Text
          position={[0, node.size + 0.5, 0]}
          fontSize={0.3}
          color="#00D4FF"
          anchorX="center"
          anchorY="middle"
        >
          {node.name}
        </Text>
      )}
    </group>
  );
};

const ConnectionLine: React.FC<{ start: [number, number, number], end: [number, number, number] }> = ({ start, end }) => {
  const lineRef = useRef<any>(null);

  useFrame((state) => {
    if (lineRef.current && lineRef.current.material) {
      const material = lineRef.current.material as THREE.LineBasicMaterial;
      material.opacity = 0.3 + Math.sin(state.clock.elapsedTime * 2) * 0.2;
    }
  });

  const points = useMemo(() => {
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(...start),
      new THREE.Vector3(
        (start[0] + end[0]) / 2,
        (start[1] + end[1]) / 2 + 1,
        (start[2] + end[2]) / 2
      ),
      new THREE.Vector3(...end),
    ]);
    return curve.getPoints(50);
  }, [start, end]);

  return (
    <Line
      ref={lineRef}
      points={points}
      color="#00D4FF"
      lineWidth={1}
      transparent
      opacity={0.3}
    />
  );
};

const ParticleField: React.FC = () => {
  const particlesRef = useRef<THREE.Points>(null);
  
  const particles = useMemo(() => {
    const count = 1000;
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 20;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 20;
      
      colors[i * 3] = 0;
      colors[i * 3 + 1] = 0.83;
      colors[i * 3 + 2] = 1;
    }
    
    return { positions, colors };
  }, []);

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y = state.clock.elapsedTime * 0.05;
      particlesRef.current.rotation.x = state.clock.elapsedTime * 0.03;
    }
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particles.positions.length / 3}
          array={particles.positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={particles.colors.length / 3}
          array={particles.colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial size={0.05} vertexColors transparent opacity={0.6} />
    </points>
  );
};

import { useState } from 'react';

const AzureResourceMap: React.FC = () => {
  // Sample resource data
  const resources: ResourceNode[] = [
    { id: '1', name: 'VM-PROD-01', type: 'VM', position: [0, 0, 0], connections: ['2', '3'], status: 'healthy', size: 0.5 },
    { id: '2', name: 'SQL-DB-01', type: 'Database', position: [3, 1, -2], connections: ['1', '4'], status: 'healthy', size: 0.6 },
    { id: '3', name: 'STORAGE-01', type: 'Storage', position: [-3, -1, 2], connections: ['1', '5'], status: 'warning', size: 0.4 },
    { id: '4', name: 'APP-SVC-01', type: 'AppService', position: [2, 2, 3], connections: ['2', '5', '6'], status: 'healthy', size: 0.7 },
    { id: '5', name: 'KEY-VAULT', type: 'KeyVault', position: [-2, 0, -3], connections: ['3', '4'], status: 'healthy', size: 0.3 },
    { id: '6', name: 'CONTAINER-01', type: 'Container', position: [0, -2, 1], connections: ['4', '7'], status: 'critical', size: 0.5 },
    { id: '7', name: 'COSMOS-DB', type: 'CosmosDB', position: [4, -1, 0], connections: ['6'], status: 'healthy', size: 0.6 },
  ];

  const resourceMap = useMemo(() => {
    const map = new Map<string, ResourceNode>();
    resources.forEach(r => map.set(r.id, r));
    return map;
  }, []);

  return (
    <Canvas camera={{ position: [5, 5, 5], fov: 60 }}>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} color="#8B5CF6" />
      
      {/* Background particles */}
      <ParticleField />
      
      {/* Resource nodes */}
      {resources.map((node) => (
        <ResourceSphere key={node.id} node={node} />
      ))}
      
      {/* Connections */}
      {resources.map((node) =>
        node.connections.map((targetId) => {
          const target = resourceMap.get(targetId);
          if (target && node.id < targetId) {
            return (
              <ConnectionLine
                key={`${node.id}-${targetId}`}
                start={node.position}
                end={target.position}
              />
            );
          }
          return null;
        })
      )}
      
      {/* Central glow */}
      <Sphere args={[0.2, 32, 32]} position={[0, 0, 0]}>
        <meshBasicMaterial color="#00D4FF" transparent opacity={0.8} />
      </Sphere>
      
      <OrbitControls
        enablePan={false}
        enableZoom={true}
        maxDistance={15}
        minDistance={3}
        autoRotate
        autoRotateSpeed={0.5}
      />
      
      <fog attach="fog" args={['#0A0E27', 10, 30]} />
    </Canvas>
  );
};

export default AzureResourceMap;