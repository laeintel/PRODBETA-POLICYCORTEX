'use client';

import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Line, Text, Sphere } from '@react-three/drei';
import * as THREE from 'three';

interface ThreatBlip {
  id: string;
  angle: number;
  distance: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  label: string;
  detected: number;
}

const RadarGrid: React.FC = () => {
  const scanLineRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (scanLineRef.current) {
      scanLineRef.current.rotation.z = -state.clock.elapsedTime;
    }
  });

  return (
    <group>
      {/* Concentric circles */}
      {[1, 2, 3, 4].map((radius) => (
        <Line
          key={radius}
          points={Array.from({ length: 65 }, (_, i) => {
            const angle = (i / 64) * Math.PI * 2;
            return new THREE.Vector3(
              Math.cos(angle) * radius,
              Math.sin(angle) * radius,
              0
            );
          })}
          color="#00D4FF"
          lineWidth={1}
          transparent
          opacity={0.3}
        />
      ))}
      
      {/* Cross lines */}
      <Line
        points={[new THREE.Vector3(-4, 0, 0), new THREE.Vector3(4, 0, 0)]}
        color="#00D4FF"
        lineWidth={1}
        transparent
        opacity={0.3}
      />
      <Line
        points={[new THREE.Vector3(0, -4, 0), new THREE.Vector3(0, 4, 0)]}
        color="#00D4FF"
        lineWidth={1}
        transparent
        opacity={0.3}
      />
      
      {/* Scanning line */}
      <mesh ref={scanLineRef}>
        <planeGeometry args={[8, 0.1]} />
        <meshBasicMaterial
          color="#00FF88"
          transparent
          opacity={0.5}
          side={THREE.DoubleSide}
        />
      </mesh>
      
      {/* Scan trail */}
      <mesh ref={scanLineRef}>
        <ringGeometry args={[0, 4, 32, 1, 0, Math.PI / 4]} />
        <meshBasicMaterial
          color="#00FF88"
          transparent
          opacity={0.1}
          side={THREE.DoubleSide}
        />
      </mesh>
    </group>
  );
};

const ThreatBlipMesh: React.FC<{ threat: ThreatBlip }> = ({ threat }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [pulse, setPulse] = useState(0);

  useFrame((state) => {
    if (meshRef.current) {
      const timeSinceDetected = state.clock.elapsedTime - threat.detected;
      const fadeOut = Math.max(0, 1 - timeSinceDetected / 5);
      
      meshRef.current.scale.setScalar(
        (1 + Math.sin(state.clock.elapsedTime * 5) * 0.2) * fadeOut
      );
      
      if (threat.severity === 'critical') {
        const material = meshRef.current.material as THREE.MeshBasicMaterial;
        material.opacity = 0.8 + Math.sin(state.clock.elapsedTime * 10) * 0.2;
      }
    }
  });

  const colors = {
    low: '#10F4B1',
    medium: '#FFB800',
    high: '#FF6B00',
    critical: '#FF0040',
  };

  const x = Math.cos(threat.angle) * threat.distance;
  const y = Math.sin(threat.angle) * threat.distance;

  return (
    <group position={[x, y, 0]}>
      <Sphere args={[0.1, 16, 16]}>
        <meshBasicMaterial
          color={colors[threat.severity]}
          transparent
          opacity={0.8}
        />
      </Sphere>
      {/* Ripple effect for critical threats */}
      {threat.severity === 'critical' && (
        <>
          <mesh>
            <ringGeometry args={[0.2, 0.3, 32]} />
            <meshBasicMaterial
              color={colors[threat.severity]}
              transparent
              opacity={0.3}
              side={THREE.DoubleSide}
            />
          </mesh>
          <mesh>
            <ringGeometry args={[0.3, 0.4, 32]} />
            <meshBasicMaterial
              color={colors[threat.severity]}
              transparent
              opacity={0.2}
              side={THREE.DoubleSide}
            />
          </mesh>
        </>
      )}
    </group>
  );
};

const ThreatRadar: React.FC = () => {
  const [threats, setThreats] = useState<ThreatBlip[]>([]);
  
  useEffect(() => {
    // Simulate threat detection
    const interval = setInterval(() => {
      const newThreat: ThreatBlip = {
        id: `threat-${Date.now()}`,
        angle: Math.random() * Math.PI * 2,
        distance: Math.random() * 3.5 + 0.5,
        severity: ['low', 'medium', 'high', 'critical'][Math.floor(Math.random() * 4)] as any,
        label: ['SQL Injection', 'DDoS Attack', 'Malware', 'Data Breach', 'Unauthorized Access'][
          Math.floor(Math.random() * 5)
        ],
        detected: Date.now() / 1000,
      };
      
      setThreats(prev => [...prev.slice(-10), newThreat]);
    }, 3000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <Canvas camera={{ position: [0, 0, 8], fov: 50 }}>
      <ambientLight intensity={0.2} />
      
      {/* Radar background */}
      <mesh position={[0, 0, -0.1]}>
        <circleGeometry args={[4.5, 64]} />
        <meshBasicMaterial color="#0A0E27" transparent opacity={0.9} />
      </mesh>
      
      {/* Radar grid */}
      <RadarGrid />
      
      {/* Center dot */}
      <mesh position={[0, 0, 0]}>
        <circleGeometry args={[0.1, 32]} />
        <meshBasicMaterial color="#00FF88" />
      </mesh>
      
      {/* Threat blips */}
      {threats.map((threat) => (
        <ThreatBlipMesh key={threat.id} threat={threat} />
      ))}
      
      {/* Labels */}
      <Text
        position={[0, 4.5, 0]}
        fontSize={0.2}
        color="#00D4FF"
        anchorX="center"
      >
        N
      </Text>
      <Text
        position={[4.5, 0, 0]}
        fontSize={0.2}
        color="#00D4FF"
        anchorX="center"
      >
        E
      </Text>
      <Text
        position={[0, -4.5, 0]}
        fontSize={0.2}
        color="#00D4FF"
        anchorX="center"
      >
        S
      </Text>
      <Text
        position={[-4.5, 0, 0]}
        fontSize={0.2}
        color="#00D4FF"
        anchorX="center"
      >
        W
      </Text>
    </Canvas>
  );
};

export default ThreatRadar;