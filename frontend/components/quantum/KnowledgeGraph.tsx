'use client';

import React, { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Text, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

interface KnowledgeNode {
  id: string;
  label: string;
  category: string;
  position: THREE.Vector3;
  connections: string[];
  importance: number;
}

const KnowledgeNodeMesh: React.FC<{ 
  node: KnowledgeNode; 
  isActive: boolean;
  onHover: (id: string | null) => void;
}> = ({ node, isActive, onHover }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const textRef = useRef<any>(null);
  const [hovered, setHovered] = React.useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      // Gentle rotation
      meshRef.current.rotation.y += 0.005;
      
      // Pulse effect for active nodes
      if (isActive) {
        const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.1;
        meshRef.current.scale.setScalar(scale);
      }
      
      // Hover scale
      const targetScale = hovered ? 1.3 : 1;
      meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });

  const categoryColors: { [key: string]: string } = {
    policy: '#00D4FF',
    resource: '#10F4B1',
    compliance: '#8B5CF6',
    security: '#FF0040',
    cost: '#FFB800',
  };

  const color = categoryColors[node.category] || '#00D4FF';
  const size = 0.1 + node.importance * 0.2;

  return (
    <group position={node.position}>
      <Sphere
        ref={meshRef}
        args={[size, 16, 16]}
        onPointerOver={() => {
          setHovered(true);
          onHover(node.id);
        }}
        onPointerOut={() => {
          setHovered(false);
          onHover(null);
        }}
      >
        <meshPhongMaterial
          color={color}
          emissive={color}
          emissiveIntensity={isActive ? 0.5 : 0.2}
          transparent
          opacity={0.8}
        />
      </Sphere>
      
      {/* Orbital rings */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <ringGeometry args={[size * 1.5, size * 1.7, 32]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={hovered ? 0.4 : 0.1}
        />
      </mesh>
      
      {hovered && (
        <Text
          ref={textRef}
          position={[0, size + 0.3, 0]}
          fontSize={0.12}
          color="#F0F9FF"
          anchorX="center"
          anchorY="middle"
        >
          {node.label}
        </Text>
      )}
    </group>
  );
};

const ConnectionLine: React.FC<{
  start: THREE.Vector3;
  end: THREE.Vector3;
  strength: number;
  active: boolean;
}> = ({ start, end, strength, active }) => {
  const lineRef = useRef<any>(null);

  useFrame((state) => {
    if (lineRef.current && lineRef.current.material) {
      const opacity = active 
        ? 0.4 + Math.sin(state.clock.elapsedTime * 3) * 0.2
        : 0.1 + Math.sin(state.clock.elapsedTime) * 0.05;
      lineRef.current.material.opacity = opacity;
    }
  });

  const midPoint = new THREE.Vector3(
    (start.x + end.x) / 2,
    (start.y + end.y) / 2,
    (start.z + end.z) / 2
  );

  const curve = new THREE.QuadraticBezierCurve3(start, midPoint, end);
  const points = curve.getPoints(30);

  return (
    <Line
      ref={lineRef}
      points={points}
      color={active ? '#00D4FF' : '#8B5CF6'}
      lineWidth={strength * 2}
      transparent
      opacity={0.2}
    />
  );
};

const ParticleSystem: React.FC = () => {
  const particlesRef = useRef<THREE.Points>(null);

  const particles = useMemo(() => {
    const count = 500;
    const positions = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const r = 3 + Math.random() * 2;
      
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }
    
    return positions;
  }, []);

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y = state.clock.elapsedTime * 0.02;
      particlesRef.current.rotation.x = state.clock.elapsedTime * 0.01;
    }
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particles.length / 3}
          array={particles}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.02}
        color="#00D4FF"
        transparent
        opacity={0.3}
        sizeAttenuation
      />
    </points>
  );
};

interface KnowledgeGraphProps {
  context?: {
    query?: string;
    topics?: string[];
    connections?: number;
  };
}

const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({ context }) => {
  const [hoveredNode, setHoveredNode] = React.useState<string | null>(null);
  const [activeNodes, setActiveNodes] = React.useState<Set<string>>(new Set());

  // Generate knowledge nodes based on context
  const nodes: KnowledgeNode[] = useMemo(() => {
    const categories = context?.topics || ['policy', 'resource', 'compliance', 'security', 'cost'];
    const nodeCount = context?.connections || 15;
    
    return Array.from({ length: nodeCount }, (_, i) => {
      const theta = (i / nodeCount) * Math.PI * 2;
      const phi = Math.acos((i / nodeCount) * 2 - 1);
      const r = 1.5 + Math.random();
      
      return {
        id: `node-${i}`,
        label: `${categories[i % categories.length].toUpperCase()}-${i}`,
        category: categories[i % categories.length],
        position: new THREE.Vector3(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi)
        ),
        connections: [],
        importance: Math.random(),
      };
    });
  }, [context]);

  // Create connections
  const nodesWithConnections = useMemo(() => {
    const connected = [...nodes];
    const nodeMap = new Map<string, KnowledgeNode>();
    
    connected.forEach(node => {
      nodeMap.set(node.id, node);
      
      // Create connections to nearby nodes
      const numConnections = Math.floor(Math.random() * 3) + 1;
      for (let i = 0; i < numConnections; i++) {
        const targetIdx = Math.floor(Math.random() * nodes.length);
        if (nodes[targetIdx].id !== node.id) {
          node.connections.push(nodes[targetIdx].id);
        }
      }
    });
    
    return { nodes: connected, nodeMap };
  }, [nodes]);

  // Animate active nodes based on context
  useEffect(() => {
    if (context?.topics) {
      const active = new Set<string>();
      nodesWithConnections.nodes.forEach(node => {
        if (context.topics?.includes(node.category)) {
          active.add(node.id);
        }
      });
      setActiveNodes(active);
    }
  }, [context, nodesWithConnections.nodes]);

  return (
    <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.5} color="#00D4FF" />
      <pointLight position={[-10, -10, -10]} intensity={0.3} color="#8B5CF6" />
      
      {/* Background particles */}
      <ParticleSystem />
      
      {/* Central core */}
      <mesh>
        <dodecahedronGeometry args={[0.2, 0]} />
        <meshPhongMaterial
          color="#00D4FF"
          emissive="#00D4FF"
          emissiveIntensity={0.5}
          wireframe
        />
      </mesh>
      
      {/* Knowledge nodes */}
      {nodesWithConnections.nodes.map(node => (
        <KnowledgeNodeMesh
          key={node.id}
          node={node}
          isActive={activeNodes.has(node.id)}
          onHover={setHoveredNode}
        />
      ))}
      
      {/* Connections */}
      {nodesWithConnections.nodes.map(node =>
        node.connections.map((targetId, idx) => {
          const target = nodesWithConnections.nodeMap.get(targetId);
          if (target && node.id < targetId) {
            const isActive = hoveredNode === node.id || hoveredNode === targetId;
            return (
              <ConnectionLine
                key={`${node.id}-${targetId}`}
                start={node.position}
                end={target.position}
                strength={0.5}
                active={isActive}
              />
            );
          }
          return null;
        })
      )}
      
      <fog attach="fog" args={['#0A0E27', 3, 10]} />
    </Canvas>
  );
};

export default KnowledgeGraph;