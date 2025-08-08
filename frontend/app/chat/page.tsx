'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Send,
  Bot,
  User,
  Sparkles,
  Code,
  CheckCircle,
  DollarSign,
  Shield,
  Users,
  Server,
  Brain
} from 'lucide-react'
import { useConversation } from '../../lib/api'
import AppLayout from '../../components/AppLayout'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  intent?: string
  confidence?: number
  suggestedActions?: string[]
  generatedPolicy?: string
}

export default function ChatPage() {
  const { sendMessage, loading } = useConversation()
  const [messages, setMessages] = useState<Message[]>([])
  const [isInitialized, setIsInitialized] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    // Initialize with welcome message only on client side
    if (!isInitialized) {
      setMessages([
        {
          id: '1',
          type: 'assistant',
          content: "Hello! I'm your AI-powered cloud governance expert. I've been trained specifically on your Azure environment and can help with policies, RBAC, cost optimization, network security, and resource management. What would you like to know?",
          timestamp: new Date(),
          intent: 'greeting',
          confidence: 100,
        }
      ])
      setIsInitialized(true)
    }
  }, [isInitialized])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!inputValue.trim() || loading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')

    try {
      const response = await sendMessage(inputValue)
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.response,
        timestamp: new Date(),
        intent: response.intent,
        confidence: response.confidence,
        suggestedActions: response.suggested_actions,
        generatedPolicy: response.generated_policy,
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "I'm sorry, I'm having trouble connecting to the AI service right now. Please try again later.",
        timestamp: new Date(),
        intent: 'error',
        confidence: 0,
      }
      setMessages(prev => [...prev, errorMessage])
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const quickActions = [
    {
      icon: DollarSign,
      text: "Show me cost optimization opportunities",
      color: "yellow"
    },
    {
      icon: Shield,
      text: "What are my biggest security risks?",
      color: "red"
    },
    {
      icon: Users,
      text: "Review RBAC permissions for admin roles",
      color: "blue"
    },
    {
      icon: Server,
      text: "Find idle resources I can shut down",
      color: "green"
    }
  ]

  return (
    <AppLayout>
      <div className="h-screen flex flex-col">
        {/* Header */}
        <div className="border-b border-white/10 bg-black/20 backdrop-blur-md">
          <div className="px-6 py-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center">
                <Brain className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">AI Governance Assistant</h1>
                <p className="text-sm text-gray-400">Patent #3: Conversational Intelligence</p>
              </div>
            </div>
          </div>
        </div>

      {/* Chat Messages */}
      <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-6 overflow-auto">
        <div className="space-y-6">
          <AnimatePresence>
            {messages.map((message, index) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                className={`flex gap-4 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.type === 'assistant' && (
                  <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                )}
                
                <div className={`max-w-2xl ${message.type === 'user' ? 'order-1' : ''}`}>
                  <div className={`p-4 rounded-xl ${
                    message.type === 'user' 
                      ? 'bg-purple-600 text-white' 
                      : 'bg-white/10 backdrop-blur-md border border-white/20 text-white'
                  }`}>
                    <p className="whitespace-pre-wrap">{message.content}</p>
                    
                    {message.intent && message.type === 'assistant' && (
                      <div className="mt-3 flex items-center gap-2 text-xs text-gray-300">
                        <Sparkles className="w-3 h-3" />
                        <span>Intent: {message.intent}</span>
                        <span>•</span>
                        <span>Confidence: {message.confidence?.toFixed(1)}%</span>
                      </div>
                    )}

                    {message.suggestedActions && message.suggestedActions.length > 0 && (
                      <div className="mt-3">
                        <p className="text-sm text-gray-300 mb-2">Suggested actions:</p>
                        <div className="space-y-1">
                          {message.suggestedActions.map((action, idx) => (
                            <div key={idx} className="flex items-center gap-2 text-sm">
                              <CheckCircle className="w-3 h-3 text-green-400" />
                              <span className="text-gray-200">{action}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {message.generatedPolicy && (
                      <div className="mt-3">
                        <p className="text-sm text-gray-300 mb-2 flex items-center gap-2">
                          <Code className="w-3 h-3" />
                          Generated Azure Policy:
                        </p>
                        <pre className="bg-black/30 p-3 rounded text-xs text-gray-200 overflow-x-auto">
                          <code>{JSON.stringify(JSON.parse(message.generatedPolicy), null, 2)}</code>
                        </pre>
                      </div>
                    )}
                  </div>
                  
                  <p className="text-xs text-gray-500 mt-2">
                    {isInitialized ? message.timestamp.toLocaleTimeString() : ''}
                  </p>
                </div>

                {message.type === 'user' && (
                  <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>
          
          {loading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex gap-4 justify-start"
            >
              <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-4">
                <div className="flex items-center gap-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  <span className="text-gray-300 text-sm">AI is analyzing your request...</span>
                </div>
              </div>
            </motion.div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick Actions */}
      {messages.length <= 1 && (
        <div className="max-w-4xl mx-auto w-full px-4 py-4">
          <p className="text-gray-300 text-sm mb-3">Quick actions to get started:</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {quickActions.map((action, index) => (
              <motion.button
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                onClick={() => setInputValue(action.text)}
                className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg transition-all text-left"
              >
                <action.icon className={`w-5 h-5 text-${action.color}-400`} />
                <span className="text-white text-sm">{action.text}</span>
              </motion.button>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-white/10 bg-black/20 backdrop-blur-md">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me about policies, costs, security, RBAC, or resources..."
                disabled={loading}
                className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-gray-400 resize-none focus:outline-none focus:border-purple-400 disabled:opacity-50"
                rows={1}
                style={{ minHeight: '44px', maxHeight: '120px' }}
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || loading}
              className="bg-purple-600 hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-lg px-4 py-3 transition-colors flex items-center justify-center"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
          
          <p className="text-xs text-gray-500 mt-2">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
      </div>
    </AppLayout>
  )
}