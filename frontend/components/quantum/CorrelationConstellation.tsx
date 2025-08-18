'use client';

import React, { useRef, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Sphere, Line } from '@react-three/drei';
import * as THREE from 'three';

interface CorrelationNode {
  id: string;
  label: string;
  value: number;
  position: THREE.Vector3;
  connections: { target: string; strength: number }[];
  category: 'security' | 'cost' | 'compliance' | 'performance';
}

const NodeSphere: React.FC<{ node: CorrelationNode; allNodes: Map<string, CorrelationNode> }> = ({ node, allNodes }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const [pulseIntensity, setPulseIntensity] = useState(0);

  useFrame((state) => {
    if (meshRef.current) {
      // Gentle floating animation
      meshRef.current.position.y = node.position.y + Math.sin(state.clock.elapsedTime + node.value) * 0.1;
      
      // Pulse on data flow
      if (pulseIntensity > 0) {
        setPulseIntensity(prev => Math.max(0, prev - 0.02));
      }
      
      // Hover effect
      const targetScale = hovered ? 1.5 : 1;
      meshRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.1);
    }
  });

  const colors = {
    security: '#FF0040',
    cost: '#FFB800',
    compliance: '#00FF88',
    performance: '#00D4FF',
  };

  const nodeColor = colors[node.category];
  const nodeSize = 0.1 + (node.value / 100) * 0.3;

  return (
    <group position={node.position}>
      <Sphere
        ref={meshRef}
        args={[nodeSize, 16, 16]}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        onClick={() => setPulseIntensity(1)}
      >
        <meshPhongMaterial
          color={nodeColor}
          emissive={nodeColor}
          emissiveIntensity={0.2 + pulseIntensity}
          transparent
          opacity={0.8}
        />
      </Sphere>
      
      {/* Outer glow ring */}
      <mesh>
        <ringGeometry args={[nodeSize * 1.5, nodeSize * 2, 32]} />
        <meshBasicMaterial
          color={nodeColor}
          transparent
          opacity={hovered ? 0.3 : 0.1}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {hovered && (
        <Text
          position={[0, nodeSize + 0.3, 0]}
          fontSize={0.15}
          color="#F0F9FF"
          anchorX="center"
          anchorY="middle"
        >
          {node.label}
          {'\n'}
          <Text fontSize={0.1} color={nodeColor}>
            {node.category} • {node.value}%
          </Text>
        </Text>
      )}
    </group>
  );
};

const CorrelationLine: React.FC<{ 
  start: THREE.Vector3; 
  end: THREE.Vector3; 
  strength: number;
}> = ({ start, end, strength }) => {
  const lineRef = useRef<any>(null);
  const [flowOffset, setFlowOffset] = useState(0);

  useFrame((state) => {
    setFlowOffset(state.clock.elapsedTime * 0.5);
    if (lineRef.current && lineRef.current.material) {
      lineRef.current.material.opacity = 0.2 + strength * 0.3 + Math.sin(state.clock.elapsedTime * 2) * 0.1;
    }
  });

  const midPoint = new THREE.Vector3(
    (start.x + end.x) / 2,
    (start.y + end.y) / 2 + strength,
    (start.z + end.z) / 2
  );

  const curve = new THREE.QuadraticBezierCurve3(start, midPoint, end);
  const points = curve.getPoints(50);

  return (
    <group>
      <Line
        ref={lineRef}
        points={points}
        color="#8B5CF6"
        lineWidth={strength * 3}
        transparent
        opacity={0.3}
        dashed
        dashScale={5}
        dashSize={0.5}
        gapSize={0.5}
        dashOffset={flowOffset}
      />
      {/* Energy particles along the line */}
      {[0.2, 0.5, 0.8].map((t, i) => {
        const pos = curve.getPoint((t + flowOffset * 0.1) % 1);
        return (
          <mesh key={i} position={pos}>
            <sphereGeometry args={[0.02, 8, 8]} />
            <meshBasicMaterial color="#00D4FF" transparent opacity={0.8} />
          </mesh>
        );
      })}
    </group>
  );
};

const BackgroundGrid: React.FC = () => {
  const gridRef = useRef<THREE.GridHelper>(null);
  
  useFrame((state) => {
    if (gridRef.current) {
      gridRef.current.rotation.y = state.clock.elapsedTime * 0.02;
    }
  });

  return (
    <>
      <gridHelper
        ref={gridRef}
        args={[20, 20, '#00D4FF', '#0A0E27']}
        position={[0, -3, 0]}
      />
      <gridHelper
        args={[20, 20, '#8B5CF6', '#0A0E27']}
        position={[0, 3, 0]}
        rotation={[Math.PI, 0, 0]}
      />
    </>
  );
};

const CorrelationConstellation: React.FC = () => {
  // Generate correlation nodes
  const nodes: CorrelationNode[] = useMemo(() => {
    const categories: Array<'security' | 'cost' | 'compliance' | 'performance'> = 
      ['security', 'cost', 'compliance', 'performance'];
    
    return Array.from({ length: 20 }, (_, i) => {
      const angle = (i / 20) * Math.PI * 2;
      const radius = 2 + Math.random() * 2;
      const height = (Math.random() - 0.5) * 3;
      
      return {
        id: `node-${i}`,
        label: `Pattern ${i + 1}`,
        value: Math.floor(Math.random() * 100),
        position: new THREE.Vector3(
          Math.cos(angle) * radius,
          height,
          Math.sin(angle) * radius
        ),
        connections: [],
        category: categories[Math.floor(Math.random() * categories.length)],
      };
    });
  }, []);

  // Create connections between nodes
  const nodesWithConnections = useMemo(() => {
    const connectedNodes = [...nodes];
    const nodeMap = new Map<string, CorrelationNode>();
    
    connectedNodes.forEach(node => {
      nodeMap.set(node.id, node);
      
      // Create 1-3 random connections
      const numConnections = Math.floor(Math.random() * 3) + 1;
      for (let i = 0; i < numConnections; i++) {
        const targetIndex = Math.floor(Math.random() * nodes.length);
        const targetNode = nodes[targetIndex];
        if (targetNode.id !== node.id) {
          node.connections.push({
            target: targetNode.id,
            strength: Math.random(),
          });
        }
      }
    });
    
    return { nodes: connectedNodes, nodeMap };
  }, [nodes]);

  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 60 }}>
      <ambientLight intensity={0.2} />
      <pointLight position={[10, 10, 10]} intensity={0.5} color="#00D4FF" />
      <pointLight position={[-10, -10, -10]} intensity={0.3} color="#8B5CF6" />
      
      {/* Background grid */}
      <BackgroundGrid />
      
      {/* Central core */}
      <mesh position={[0, 0, 0]}>
        <icosahedronGeometry args={[0.3, 1]} />
        <meshPhongMaterial
          color="#00D4FF"
          emissive="#00D4FF"
          emissiveIntensity={0.5}
          wireframe
        />
      </mesh>
      
      {/* Correlation nodes */}
      {nodesWithConnections.nodes.map((node) => (
        <NodeSphere
          key={node.id}
          node={node}
          allNodes={nodesWithConnections.nodeMap}
        />
      ))}
      
      {/* Correlation lines */}
      {nodesWithConnections.nodes.map((node) =>
        node.connections.map((connection, idx) => {
          const targetNode = nodesWithConnections.nodeMap.get(connection.target);
          if (targetNode) {
            return (
              <CorrelationLine
                key={`${node.id}-${connection.target}-${idx}`}
                start={node.position}
                end={targetNode.position}
                strength={connection.strength}
              />
            );
          }
          return null;
        })
      )}
      
      <OrbitControls
        enablePan={false}
        enableZoom={true}
        maxDistance={12}
        minDistance={3}
        autoRotate
        autoRotateSpeed={0.3}
      />
      
      <fog attach="fog" args={['#0A0E27', 5, 20]} />
    </Canvas>
  );
};

export default CorrelationConstellation;