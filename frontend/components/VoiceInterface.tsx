/**
 * PATENT NOTICE: This code implements methods covered by:
 * - US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
 * - US Patent Application 17/123,457 - Conversational Governance Intelligence System
 * - US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
 * - US Patent Application 17/123,459 - Predictive Policy Compliance Engine
 * Unauthorized use, reproduction, or distribution may constitute patent infringement.
 * © 2024 PolicyCortex. All rights reserved.
 */

'use client'

import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  Sparkles,
  Brain,
  Zap,
  Activity,
  AlertCircle,
  CheckCircle,
  DollarSign,
  Shield,
  Users,
  Server
} from 'lucide-react'
import { useConversation } from '../lib/api'

interface VoiceInterfaceProps {
  onActionTrigger?: (action: string, data?: any) => void;
}

export default function VoiceInterface({ onActionTrigger }: VoiceInterfaceProps) {
  const [isListening, setIsListening] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [lastCommand, setLastCommand] = useState('')
  const [voiceEnabled, setVoiceEnabled] = useState(true)
  const [isProcessing, setIsProcessing] = useState(false)
  const [cortexResponse, setCortexResponse] = useState('')
  const [pulse, setPulse] = useState(false)
  const [isClient, setIsClient] = useState(false)
  
  const { sendMessage } = useConversation()
  const recognitionRef = useRef<any | null>(null)
  const synthRef = useRef<SpeechSynthesis | null>(null)

  // Voice commands mapping
  const voiceCommands = {
    // Assessment commands
    'assess soc compliance': () => handleAssessment('soc'),
    'assess security posture': () => handleAssessment('security'),
    'assess cost optimization': () => handleAssessment('cost'),
    'assess rbac compliance': () => handleAssessment('rbac'),
    'assess network security': () => handleAssessment('network'),
    'assess resource utilization': () => handleAssessment('resources'),
    
    // Quick actions
    'show dashboard': () => onActionTrigger?.('navigate', '/dashboard'),
    'show policies': () => onActionTrigger?.('navigate', '/dashboard?module=policies'),
    'show costs': () => onActionTrigger?.('navigate', '/dashboard?module=costs'),
    'show chat': () => onActionTrigger?.('navigate', '/chat'),
    'show recommendations': () => onActionTrigger?.('action', 'show_recommendations'),
    
    // Analysis commands
    'analyze spending': () => handleAnalysis('cost'),
    'analyze compliance': () => handleAnalysis('compliance'),
    'analyze threats': () => handleAnalysis('security'),
    'analyze permissions': () => handleAnalysis('rbac'),
    
    // Emergency commands
    'security alert': () => handleEmergency('security'),
    'cost alert': () => handleEmergency('cost'),
    'compliance issue': () => handleEmergency('compliance'),
  }

  useEffect(() => {
    setIsClient(true)
  }, [])

  useEffect(() => {
    // Only run on client side after component is mounted
    if (!isClient) return
    
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      synthRef.current = window.speechSynthesis
    }
  }, [isClient])

  useEffect(() => {
    // Only run on client side after component is mounted
    if (!isClient) return
    
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition
      
      if (!SpeechRecognition) {
        console.log('Speech recognition not supported in this browser')
        return
      }
      const recognition = new SpeechRecognition()
      
      recognition.continuous = false
      recognition.interimResults = true
      recognition.lang = 'en-US'
      
      recognition.onstart = () => {
        setIsListening(true)
        setPulse(true)
      }
      
      recognition.onresult = (event: any) => {
        const current = event.resultIndex
        const transcript = event.results[current][0].transcript
        setTranscript(transcript)
        
        if (event.results[current].isFinal) {
          handleVoiceCommand(transcript.toLowerCase())
        }
      }
      
      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error)
        setIsListening(false)
        setPulse(false)
      }
      
      recognition.onend = () => {
        setIsListening(false)
        setPulse(false)
      }
      
      recognitionRef.current = recognition
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    }
  }, [isClient])

  const handleVoiceCommand = async (command: string) => {
    setLastCommand(command)
    setIsProcessing(true)

    // Check for exact matches first
    const exactMatch = voiceCommands[command as keyof typeof voiceCommands]
    if (exactMatch) {
      exactMatch()
      speakResponse(`Executing ${command}`)
      setIsProcessing(false)
      return
    }

    // Check for partial matches
    const partialMatch = Object.keys(voiceCommands).find(key => 
      command.includes(key.split(' ')[0]) && command.includes(key.split(' ')[1])
    )
    
    if (partialMatch) {
      voiceCommands[partialMatch as keyof typeof voiceCommands]()
      speakResponse(`Executing ${partialMatch}`)
      setIsProcessing(false)
      return
    }

    // Send to AI for processing
    try {
      const response = await sendMessage(command)
      setCortexResponse(response.response)
      speakResponse(response.response)
      
      // Extract actions from AI response (guarded)
      const actions = (response as any)?.suggested_actions as string[] | undefined
      if (Array.isArray(actions) && actions.length > 0) {
        onActionTrigger?.('ai_suggestions', actions)
      }
    } catch (error) {
      speakResponse("I'm having trouble processing that request. Please try again.")
    }
    
    setIsProcessing(false)
  }

  const handleAssessment = async (type: string) => {
    const assessments = {
      soc: "Initiating SOC 2 compliance assessment across all cloud resources...",
      security: "Analyzing security posture including network, identity, and data protection...", 
      cost: "Evaluating cost optimization opportunities and budget utilization...",
      rbac: "Reviewing role-based access controls and permission assignments...",
      network: "Scanning network security groups, firewalls, and traffic patterns...",
      resources: "Analyzing resource utilization, idle resources, and optimization opportunities..."
    }
    
    const message = assessments[type as keyof typeof assessments] || "Starting comprehensive assessment..."
    speakResponse(message)
    onActionTrigger?.('assessment', { type, message })
  }

  const handleAnalysis = (type: string) => {
    speakResponse(`Starting deep ${type} analysis with predictive insights...`)
    onActionTrigger?.('analysis', type)
  }

  const handleEmergency = (type: string) => {
    speakResponse(`Alert acknowledged. Escalating ${type} issue to priority queue.`)
    onActionTrigger?.('emergency', type)
  }

  const speakResponse = (text: string) => {
    if (!isClient || typeof window === 'undefined') return
    
    if (voiceEnabled && synthRef.current && window.SpeechSynthesisUtterance) {
      const utterance = new window.SpeechSynthesisUtterance(text)
      utterance.rate = 0.9
      utterance.pitch = 1.1
      utterance.volume = 0.8
      synthRef.current.speak(utterance)
    }
  }

  const startListening = () => {
    if (recognitionRef.current && !isListening) {
      setTranscript('')
      try {
        recognitionRef.current.start()
      } catch (error) {
        console.error('Failed to start speech recognition:', error)
        // If already started, try to stop and restart
        try {
          recognitionRef.current.stop()
          setTimeout(() => {
            recognitionRef.current.start()
          }, 100)
        } catch (restartError) {
          console.error('Failed to restart speech recognition:', restartError)
        }
      }
    }
  }

  // Experimental: Start Azure OpenAI Realtime via WebRTC using SDP exchange
  const startRealtime = async () => {
    try {
      const pc = new RTCPeerConnection()
      pc.ontrack = (event) => {
        const audioEl = document.getElementById('cortex-audio') as HTMLAudioElement | null
        if (audioEl) {
          audioEl.srcObject = event.streams[0]
          audioEl.play().catch(()=>{})
        }
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      stream.getTracks().forEach(t => pc.addTrack(t, stream))

      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      const resp = await fetch('/api/v1/voice/realtime/sdp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/sdp', 'Accept': 'application/sdp' },
        body: offer.sdp || ''
      })
      if (!resp.ok) throw new Error(`SDP exchange failed: ${resp.status}`)
      const answerSdp = await resp.text()
      await pc.setRemoteDescription({ type: 'answer', sdp: answerSdp })
      setIsListening(true)
      setPulse(true)
    } catch (e) {
      console.warn('Realtime setup failed; falling back to local STT', e)
      startListening()
    }
  }

  const stopListening = () => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop()
    }
  }

  const toggleVoice = () => {
    setVoiceEnabled(!voiceEnabled)
    if (synthRef.current) {
      if (voiceEnabled) {
        synthRef.current.cancel()
      } else {
        speakResponse("Voice feedback enabled")
      }
    }
  }

  if (!isClient) {
    return null
  }

  return (
    <>
      <audio id="cortex-audio" className="hidden" />
      {/* Floating Voice Button */}
      <motion.div
        className="fixed bottom-6 right-6 z-50"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 260, damping: 20 }}
      >
        <motion.button
          onClick={() => setIsExpanded(!isExpanded)}
          className={`w-16 h-16 rounded-full flex items-center justify-center text-white shadow-2xl transition-all ${
            isListening || pulse 
              ? 'bg-gradient-to-r from-red-500 to-pink-500 shadow-red-500/50' 
              : isProcessing
              ? 'bg-gradient-to-r from-yellow-500 to-orange-500 shadow-yellow-500/50'
              : 'bg-gradient-to-r from-purple-600 to-indigo-600 shadow-purple-500/50'
          }`}
          animate={pulse ? { scale: [1, 1.1, 1] } : {}}
          transition={{ duration: 0.6, repeat: pulse ? Infinity : 0 }}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          {isProcessing ? (
            <Brain className="w-8 h-8 animate-pulse" />
          ) : isListening ? (
            <motion.div
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 0.5, repeat: Infinity }}
            >
              <Mic className="w-8 h-8" />
            </motion.div>
          ) : (
            <Sparkles className="w-8 h-8" />
          )}
        </motion.button>

        {/* CorTex Label */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="absolute right-20 top-1/2 transform -translate-y-1/2 bg-black/80 text-white px-3 py-1 rounded-lg text-sm font-semibold"
        >
          CorTex AI
        </motion.div>
      </motion.div>

      {/* Expanded Voice Interface */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 20 }}
            className="fixed bottom-28 right-6 w-80 bg-black/90 backdrop-blur-md rounded-2xl border border-white/20 p-6 z-40 shadow-2xl"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white font-bold text-lg flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                CorTex Voice AI
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={toggleVoice}
                  className={`p-2 rounded-full transition-colors ${
                    voiceEnabled ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
                  }`}
                >
                  {voiceEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                </button>
                <button
                  onClick={() => setIsExpanded(false)}
                  className="p-2 rounded-full bg-gray-600 text-white hover:bg-gray-500 transition-colors"
                >
                  ×
                </button>
              </div>
            </div>

            {/* Voice Status */}
            <div className="mb-4">
              {isListening ? (
                <div className="flex items-center gap-2 text-red-400">
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.5, repeat: Infinity }}
                  >
                    <Activity className="w-4 h-4" />
                  </motion.div>
                  <span className="text-sm">Listening...</span>
                </div>
              ) : isProcessing ? (
                <div className="flex items-center gap-2 text-yellow-400">
                  <Brain className="w-4 h-4 animate-pulse" />
                  <span className="text-sm">Processing...</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-green-400">
                  <CheckCircle className="w-4 h-4" />
                  <span className="text-sm">Ready for voice commands</span>
                </div>
              )}
            </div>

            {/* Transcript */}
            {transcript && (
              <div className="mb-4 p-3 bg-purple-900/30 rounded-lg">
                <p className="text-white text-sm">{transcript}</p>
              </div>
            )}

            {/* Last Command */}
            {lastCommand && (
              <div className="mb-4 p-3 bg-blue-900/30 rounded-lg">
                <p className="text-blue-300 text-xs mb-1">Last Command:</p>
                <p className="text-white text-sm">{lastCommand}</p>
              </div>
            )}

            {/* AI Response */}
            {cortexResponse && (
              <div className="mb-4 p-3 bg-green-900/30 rounded-lg">
                <p className="text-green-300 text-xs mb-1">CorTex Response:</p>
                <p className="text-white text-sm">{cortexResponse}</p>
              </div>
            )}

            {/* Voice Controls */}
            <div className="flex gap-3">
              <button
                onClick={isListening ? stopListening : startRealtime}
                disabled={isProcessing}
                className={`flex-1 py-3 px-4 rounded-lg font-semibold transition-all ${
                  isListening
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : isProcessing
                    ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                    : 'bg-purple-600 hover:bg-purple-700 text-white'
                }`}
              >
                {isListening ? (
                  <>
                    <MicOff className="w-4 h-4 inline mr-2" />
                    Stop Listening
                  </>
                ) : isProcessing ? (
                  <>
                    <Brain className="w-4 h-4 inline mr-2 animate-pulse" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Mic className="w-4 h-4 inline mr-2" />
                    Start Listening
                  </>
                )}
              </button>
            </div>

            {/* Quick Commands */}
            <div className="mt-4">
              <p className="text-gray-400 text-xs mb-2">Quick Commands:</p>
              <div className="grid grid-cols-2 gap-1 text-xs">
                <span className="text-purple-300">"Assess SOC compliance"</span>
                <span className="text-blue-300">"Show dashboard"</span>
                <span className="text-green-300">"Analyze spending"</span>
                <span className="text-yellow-300">"Security alert"</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

// Type declarations for Speech API
declare global {
  interface Window {
    SpeechRecognition: any;
    webkitSpeechRecognition: any;
  }
}