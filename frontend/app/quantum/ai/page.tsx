'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import dynamic from 'next/dynamic';
import QuantumLayout from '@/components/quantum/QuantumLayout';
import { Canvas } from '@react-three/fiber';
import { Sphere, MeshDistortMaterial, OrbitControls } from '@react-three/drei';

// Dynamic import for knowledge graph
const KnowledgeGraph = dynamic(
  () => import('@/components/quantum/KnowledgeGraph'),
  { ssr: false }
);

interface Message {
  id: string;
  role: 'user' | 'ai';
  content: string;
  timestamp: Date;
  confidence?: number;
  sources?: string[];
  visualData?: any;
}

const AIOrb: React.FC<{ isThinking: boolean; mood: string }> = ({ isThinking, mood }) => {
  const colors = {
    idle: '#00D4FF',
    thinking: '#8B5CF6',
    responding: '#10F4B1',
    alert: '#FF0040',
  };

  return (
    <Canvas camera={{ position: [0, 0, 3] }}>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} />
      
      <Sphere args={[1, 64, 64]}>
        <MeshDistortMaterial
          color={colors[mood as keyof typeof colors] || colors.idle}
          emissive={colors[mood as keyof typeof colors] || colors.idle}
          emissiveIntensity={isThinking ? 0.8 : 0.4}
          roughness={0.1}
          metalness={0.8}
          distort={isThinking ? 0.4 : 0.2}
          speed={isThinking ? 5 : 2}
        />
      </Sphere>
      
      {/* Outer glow */}
      <Sphere args={[1.2, 32, 32]}>
        <meshBasicMaterial
          color={colors[mood as keyof typeof colors] || colors.idle}
          transparent
          opacity={0.1}
        />
      </Sphere>
      
      <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={1} />
    </Canvas>
  );
};

export default function AIConversationNexus() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [aiMood, setAiMood] = useState('idle');
  const [showKnowledgeGraph, setShowKnowledgeGraph] = useState(false);
  const [currentContext, setCurrentContext] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsThinking(true);
    setAiMood('thinking');

    // Simulate AI processing
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: 'ai',
        content: generateAIResponse(input),
        timestamp: new Date(),
        confidence: Math.floor(Math.random() * 20) + 80,
        sources: ['Azure Policy Engine', 'Compliance Database', 'ML Model v2.4'],
        visualData: { type: 'correlation', data: generateVisualizationData() },
      };

      setMessages(prev => [...prev, aiResponse]);
      setIsThinking(false);
      setAiMood('responding');
      
      // Update context for knowledge graph
      setCurrentContext({
        query: input,
        topics: extractTopics(input),
        connections: Math.floor(Math.random() * 10) + 5,
      });

      setTimeout(() => setAiMood('idle'), 2000);
    }, 2000);
  };

  const generateAIResponse = (query: string): string => {
    const responses = [
      "I've analyzed your Azure infrastructure and identified 3 critical compliance gaps in your storage accounts. The cross-domain correlation engine shows these are linked to recent policy changes.",
      "Based on predictive analysis, there's a 87% probability of cost overrun in the next billing cycle. I recommend implementing the suggested auto-scaling policies immediately.",
      "The quantum compliance grid shows all resources are within acceptable thresholds. However, I've detected unusual patterns in the East US region that require attention.",
      "Your security posture score has improved by 12% since last week. The neural network has identified 5 additional optimizations that could increase it by another 8%.",
    ];
    return responses[Math.floor(Math.random() * responses.length)];
  };

  const extractTopics = (text: string): string[] => {
    const topics = ['compliance', 'security', 'cost', 'performance', 'governance'];
    return topics.filter(() => Math.random() > 0.5);
  };

  const generateVisualizationData = () => {
    return Array.from({ length: 10 }, (_, i) => ({
      x: i,
      y: Math.random() * 100,
      category: ['security', 'cost', 'compliance'][Math.floor(Math.random() * 3)],
    }));
  };

  const suggestedQueries = [
    "Show me all non-compliant resources",
    "Predict next month's Azure costs",
    "Analyze security vulnerabilities",
    "Optimize resource allocation",
  ];

  return (
    <QuantumLayout>
      <div className="grid grid-cols-3 gap-6 h-[calc(100vh-8rem)]">
        {/* Left Panel - AI Orb and Status */}
        <motion.div
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass rounded-2xl p-6 border border-quantum-blue/20 flex flex-col"
        >
          <h2 className="text-xl font-display font-bold text-quantum-blue mb-4">
            AI CONSCIOUSNESS
          </h2>
          
          <div className="h-64 relative rounded-xl overflow-hidden bg-dark-matter/50 mb-4">
            <AIOrb isThinking={isThinking} mood={aiMood} />
          </div>
          
          <div className="space-y-3 flex-1">
            <div className="glass-light rounded-lg p-3">
              <p className="text-xs font-mono text-photon-white/60">STATUS</p>
              <p className="text-sm font-bold text-quantum-blue uppercase">
                {isThinking ? 'PROCESSING' : 'READY'}
              </p>
            </div>
            
            <div className="glass-light rounded-lg p-3">
              <p className="text-xs font-mono text-photon-white/60">NEURAL ACTIVITY</p>
              <div className="mt-2 space-y-1">
                {['Pattern Recognition', 'Correlation Analysis', 'Prediction Engine'].map((activity, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <span className="text-xs text-photon-white/80">{activity}</span>
                    <div className="w-16 h-1 bg-dark-matter rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.random() * 100}%` }}
                        transition={{ duration: 1, repeat: Infinity, repeatType: 'reverse' }}
                        className="h-full bg-gradient-to-r from-quantum-blue to-neural-purple"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="glass-light rounded-lg p-3">
              <p className="text-xs font-mono text-photon-white/60">KNOWLEDGE BASE</p>
              <p className="text-2xl font-bold text-neural-purple">2.4M</p>
              <p className="text-xs text-photon-white/60">Data Points</p>
            </div>
          </div>
        </motion.div>

        {/* Center - Conversation */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-2xl p-6 border border-quantum-blue/20 flex flex-col"
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-display font-bold text-quantum-blue">
              NEURAL DIALOGUE
            </h2>
            <button
              onClick={() => setShowKnowledgeGraph(!showKnowledgeGraph)}
              className="px-3 py-1 glass-light rounded-lg text-xs font-mono text-quantum-blue hover:glass transition-all"
            >
              {showKnowledgeGraph ? 'HIDE' : 'SHOW'} GRAPH
            </button>
          </div>
          
          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-4 mb-4 pr-2 scrollbar-thin scrollbar-thumb-quantum-blue/20">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-[80%] ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
                    <div
                      className={`
                        rounded-xl p-4
                        ${message.role === 'user' 
                          ? 'glass-light border border-quantum-blue/30' 
                          : 'glass border border-neural-purple/30'}
                      `}
                    >
                      <p className="text-sm text-photon-white">{message.content}</p>
                      
                      {message.confidence && (
                        <div className="mt-2 flex items-center space-x-2">
                          <span className="text-xs font-mono text-photon-white/60">
                            CONFIDENCE
                          </span>
                          <div className="flex-1 h-1 bg-dark-matter rounded-full overflow-hidden">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{ width: `${message.confidence}%` }}
                              className="h-full bg-gradient-to-r from-plasma-green to-quantum-blue"
                            />
                          </div>
                          <span className="text-xs font-mono text-plasma-green">
                            {message.confidence}%
                          </span>
                        </div>
                      )}
                      
                      {message.sources && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {message.sources.map((source, i) => (
                            <span
                              key={i}
                              className="text-xs px-2 py-1 glass-light rounded-full text-quantum-cyan"
                            >
                              {source}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    
                    <p className="text-xs text-photon-white/40 mt-1 px-2">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            
            {isThinking && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="glass rounded-xl p-4 flex items-center space-x-2">
                  <div className="flex space-x-1">
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="w-2 h-2 bg-quantum-blue rounded-full"
                        animate={{ y: [0, -10, 0] }}
                        transition={{ duration: 0.6, delay: i * 0.2, repeat: Infinity }}
                      />
                    ))}
                  </div>
                  <span className="text-xs font-mono text-photon-white/60">
                    NEURAL PROCESSING
                  </span>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
          
          {/* Input */}
          <form onSubmit={handleSubmit} className="space-y-3">
            <div className="relative">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me about your Azure governance..."
                className="w-full px-4 py-3 glass rounded-xl border border-quantum-blue/30 text-photon-white placeholder-photon-white/40 focus:outline-none focus:border-quantum-blue focus:shadow-lg focus:shadow-quantum-blue/20 transition-all"
                disabled={isThinking}
              />
              <button
                type="submit"
                disabled={isThinking || !input.trim()}
                className="absolute right-2 top-1/2 -translate-y-1/2 px-4 py-1.5 bg-gradient-to-r from-quantum-blue to-neural-purple rounded-lg text-white font-medium disabled:opacity-50 hover:shadow-lg hover:shadow-quantum-blue/30 transition-all"
              >
                TRANSMIT
              </button>
            </div>
            
            {/* Suggested Queries */}
            <div className="flex flex-wrap gap-2">
              {suggestedQueries.map((query, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setInput(query)}
                  className="text-xs px-3 py-1 glass-light rounded-full text-quantum-cyan hover:glass transition-all"
                >
                  {query}
                </button>
              ))}
            </div>
          </form>
        </motion.div>

        {/* Right Panel - Knowledge Graph */}
        <motion.div
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass rounded-2xl p-6 border border-quantum-blue/20"
        >
          <h2 className="text-xl font-display font-bold text-neural-purple mb-4">
            KNOWLEDGE GRAPH
          </h2>
          
          <div className="h-96 relative rounded-xl overflow-hidden bg-dark-matter/50 mb-4">
            {showKnowledgeGraph && <KnowledgeGraph context={currentContext} />}
          </div>
          
          {currentContext && (
            <div className="space-y-3">
              <div className="glass-light rounded-lg p-3">
                <p className="text-xs font-mono text-photon-white/60">ACTIVE TOPICS</p>
                <div className="mt-2 flex flex-wrap gap-1">
                  {currentContext.topics?.map((topic: string, i: number) => (
                    <span
                      key={i}
                      className="text-xs px-2 py-1 bg-gradient-to-r from-quantum-blue/20 to-neural-purple/20 rounded-full text-photon-white"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              </div>
              
              <div className="glass-light rounded-lg p-3">
                <p className="text-xs font-mono text-photon-white/60">NEURAL CONNECTIONS</p>
                <p className="text-2xl font-bold text-quantum-blue">
                  {currentContext.connections || 0}
                </p>
              </div>
            </div>
          )}
        </motion.div>
      </div>

      <style jsx>{`
        .glass {
          background: rgba(10, 14, 39, 0.6);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
        }

        .glass-light {
          background: rgba(240, 249, 255, 0.05);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
        }

        .scrollbar-thin::-webkit-scrollbar {
          width: 4px;
        }

        .scrollbar-thumb-quantum-blue\\/20::-webkit-scrollbar-thumb {
          background: rgba(0, 212, 255, 0.2);
          border-radius: 2px;
        }

        .text-quantum-blue { color: #00D4FF; }
        .text-neural-purple { color: #8B5CF6; }
        .text-plasma-green { color: #10F4B1; }
        .text-quantum-cyan { color: #00F5FF; }
        .text-photon-white { color: #F0F9FF; }
        .bg-dark-matter { background: #0A0E27; }
        .border-quantum-blue { border-color: #00D4FF; }
        .border-neural-purple { border-color: #8B5CF6; }
      `}</style>
    </QuantumLayout>
  );
}