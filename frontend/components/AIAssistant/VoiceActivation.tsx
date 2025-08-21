/**
 * PATENT NOTICE: This component implements Patent #2 - Conversational Governance Intelligence System
 * Voice-enabled natural language processing for hands-free governance operations
 */

'use client'

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Mic, MicOff, Volume2, Loader2, CheckCircle, XCircle, AlertCircle } from 'lucide-react'

interface VoiceActivationProps {
  onCommand: (command: string) => void
  onTranscript?: (text: string) => void
  isEnabled: boolean
  hotword?: string
}

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList
  resultIndex: number
}

interface SpeechRecognitionResultList {
  length: number
  item(index: number): SpeechRecognitionResult
  [index: number]: SpeechRecognitionResult
}

interface SpeechRecognitionResult {
  isFinal: boolean
  length: number
  item(index: number): SpeechRecognitionAlternative
  [index: number]: SpeechRecognitionAlternative
}

interface SpeechRecognitionAlternative {
  transcript: string
  confidence: number
}

declare global {
  interface Window {
    SpeechRecognition: any
    webkitSpeechRecognition: any
  }
}

type RecognitionState = 'idle' | 'listening' | 'processing' | 'speaking' | 'error'

export default function VoiceActivation({
  onCommand,
  onTranscript,
  isEnabled,
  hotword = 'hey policycortex'
}: VoiceActivationProps) {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [interimTranscript, setInterimTranscript] = useState('')
  const [state, setState] = useState<RecognitionState>('idle')
  const [error, setError] = useState<string | null>(null)
  const [confidence, setConfidence] = useState(0)
  const [isHotwordActive, setIsHotwordActive] = useState(false)
  const recognitionRef = useRef<any>(null)
  const timeoutRef = useRef<NodeJS.Timeout>()

  // Initialize speech recognition
  useEffect(() => {
    if (!isEnabled) return

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    
    if (!SpeechRecognition) {
      setError('Speech recognition not supported in this browser')
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = 'en-US'
    recognition.maxAlternatives = 3

    recognition.onstart = () => {
      setState('listening')
      setError(null)
      setTranscript('')
      setInterimTranscript('')
    }

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let finalTranscript = ''
      let interimText = ''

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        const text = result[0].transcript
        const conf = result[0].confidence

        if (result.isFinal) {
          finalTranscript += text
          setConfidence(conf * 100)
        } else {
          interimText += text
        }
      }

      if (finalTranscript) {
        setTranscript(prev => prev + ' ' + finalTranscript)
        setInterimTranscript('')
        
        // Check for hotword
        const lowerTranscript = finalTranscript.toLowerCase()
        if (lowerTranscript.includes(hotword.toLowerCase())) {
          setIsHotwordActive(true)
          speak('Yes, how can I help you?')
          
          // Extract command after hotword
          const hotwordIndex = lowerTranscript.indexOf(hotword.toLowerCase())
          const command = finalTranscript.substring(hotwordIndex + hotword.length).trim()
          
          if (command) {
            processCommand(command)
          }
        } else if (isHotwordActive) {
          processCommand(finalTranscript)
        }
        
        if (onTranscript) {
          onTranscript(finalTranscript)
        }
      } else {
        setInterimTranscript(interimText)
      }
    }

    recognition.onerror = (event: any) => {
      console.error('Speech recognition error:', event.error)
      setState('error')
      
      switch (event.error) {
        case 'no-speech':
          setError('No speech detected. Please try again.')
          break
        case 'audio-capture':
          setError('Microphone not found. Please check your settings.')
          break
        case 'not-allowed':
          setError('Microphone access denied. Please enable permissions.')
          break
        case 'network':
          setError('Network error. Please check your connection.')
          break
        default:
          setError(`Speech recognition error: ${event.error}`)
      }
      
      setIsListening(false)
      stopListening()
    }

    recognition.onend = () => {
      setState('idle')
      setIsListening(false)
      
      // Auto-restart if still enabled and was listening
      if (isEnabled && isListening) {
        setTimeout(() => startListening(), 1000)
      }
    }

    recognitionRef.current = recognition

    // Start listening automatically
    startListening()

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    }
  }, [isEnabled, hotword])

  const startListening = useCallback(() => {
    if (recognitionRef.current && !isListening) {
      try {
        recognitionRef.current.start()
        setIsListening(true)
        setState('listening')
        
        // Auto-stop after 30 seconds of inactivity
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current)
        }
        timeoutRef.current = setTimeout(() => {
          stopListening()
        }, 30000)
      } catch (error) {
        console.error('Failed to start recognition:', error)
        setError('Failed to start voice recognition')
      }
    }
  }, [isListening])

  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop()
      setIsListening(false)
      setState('idle')
      setIsHotwordActive(false)
      
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [isListening])

  const processCommand = useCallback((command: string) => {
    if (!command.trim()) return
    
    setState('processing')
    setIsHotwordActive(false)
    
    // Process the command
    onCommand(command)
    
    // Provide voice feedback
    const response = generateVoiceResponse(command)
    speak(response)
    
    // Reset after processing
    setTimeout(() => {
      setState('listening')
      setTranscript('')
    }, 2000)
  }, [onCommand])

  const speak = (text: string) => {
    if (!window.speechSynthesis) return
    
    setState('speaking')
    
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = 1.0
    utterance.pitch = 1.0
    utterance.volume = 0.8
    utterance.lang = 'en-US'
    
    // Select a good voice if available
    const voices = window.speechSynthesis.getVoices()
    const preferredVoice = voices.find(v => 
      v.name.includes('Google') || v.name.includes('Microsoft')
    )
    if (preferredVoice) {
      utterance.voice = preferredVoice
    }
    
    utterance.onend = () => {
      setState('listening')
    }
    
    window.speechSynthesis.speak(utterance)
  }

  const generateVoiceResponse = (command: string): string => {
    const lowerCommand = command.toLowerCase()
    
    if (lowerCommand.includes('compliance')) {
      return 'Checking compliance status now'
    } else if (lowerCommand.includes('cost') || lowerCommand.includes('savings')) {
      return 'Analyzing cost optimization opportunities'
    } else if (lowerCommand.includes('risk')) {
      return 'Reviewing active risks and threats'
    } else if (lowerCommand.includes('resource')) {
      return 'Opening resource management'
    } else if (lowerCommand.includes('prediction')) {
      return 'Showing AI predictions'
    } else if (lowerCommand.includes('help')) {
      return 'I can help you check compliance, find cost savings, review risks, manage resources, and view AI predictions'
    } else {
      return 'Processing your request'
    }
  }

  // Visual indicator states
  const getStateColor = () => {
    switch (state) {
      case 'listening': return 'bg-green-500'
      case 'processing': return 'bg-blue-500'
      case 'speaking': return 'bg-purple-500'
      case 'error': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getStateIcon = () => {
    switch (state) {
      case 'listening': return <Mic className="w-4 h-4" />
      case 'processing': return <Loader2 className="w-4 h-4 animate-spin" />
      case 'speaking': return <Volume2 className="w-4 h-4" />
      case 'error': return <AlertCircle className="w-4 h-4" />
      default: return <MicOff className="w-4 h-4" />
    }
  }

  if (!isEnabled) return null

  return (
    <>
      {/* Floating Voice Indicator */}
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.8 }}
        className="fixed bottom-4 left-4 z-50"
      >
        <div className="relative">
          {/* Pulse animation when listening */}
          {state === 'listening' && (
            <div className="absolute inset-0 rounded-full bg-green-500 animate-ping opacity-20" />
          )}
          
          {/* Main indicator */}
          <button
            onClick={isListening ? stopListening : startListening}
            className={`
              relative flex items-center gap-2 px-4 py-2 rounded-full shadow-lg
              transition-all duration-300 ${getStateColor()} text-white
              hover:scale-105 active:scale-95
            `}
          >
            {getStateIcon()}
            <span className="text-sm font-medium">
              {state === 'listening' && isHotwordActive ? 'Listening...' :
               state === 'listening' ? `Say "${hotword}"` :
               state === 'processing' ? 'Processing...' :
               state === 'speaking' ? 'Speaking...' :
               state === 'error' ? 'Error' :
               'Voice Inactive'}
            </span>
            
            {/* Confidence indicator */}
            {confidence > 0 && state === 'processing' && (
              <span className="text-xs opacity-75">
                {confidence.toFixed(0)}%
              </span>
            )}
          </button>
        </div>
      </motion.div>

      {/* Transcript Display */}
      <AnimatePresence>
        {(transcript || interimTranscript) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-20 left-4 max-w-md z-40"
          >
            <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl">
              <div className="flex items-start gap-2">
                <Mic className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  {transcript && (
                    <p className="text-sm text-white">{transcript}</p>
                  )}
                  {interimTranscript && (
                    <p className="text-sm text-gray-400 italic">{interimTranscript}</p>
                  )}
                </div>
                <button
                  onClick={() => {
                    setTranscript('')
                    setInterimTranscript('')
                  }}
                  className="text-gray-400 hover:text-white"
                >
                  <XCircle className="w-4 h-4" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error Message */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="fixed top-20 left-4 max-w-sm z-50"
          >
            <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-3">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-red-400" />
                <p className="text-sm text-red-400">{error}</p>
                <button
                  onClick={() => setError(null)}
                  className="ml-auto text-red-400 hover:text-red-300"
                >
                  <XCircle className="w-4 h-4" />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Voice Commands Help */}
      {state === 'listening' && !isHotwordActive && !transcript && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed bottom-20 right-4 max-w-xs z-40"
        >
          <div className="bg-gray-900/90 backdrop-blur border border-gray-700 rounded-lg p-3">
            <p className="text-xs text-gray-400 mb-2">Voice commands:</p>
            <div className="space-y-1">
              <p className="text-xs text-gray-300">• "{hotword}, check compliance"</p>
              <p className="text-xs text-gray-300">• "{hotword}, show cost savings"</p>
              <p className="text-xs text-gray-300">• "{hotword}, what are the risks?"</p>
              <p className="text-xs text-gray-300">• "{hotword}, open AI chat"</p>
            </div>
          </div>
        </motion.div>
      )}
    </>
  )
}