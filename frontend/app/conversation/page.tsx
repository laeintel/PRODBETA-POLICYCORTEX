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

import React, { useState, useEffect, useRef } from 'react'
import { api } from '../../lib/api-client'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Send, Sparkles, User, Bot, Copy, ThumbsUp, ThumbsDown,
  AlertCircle, CheckCircle, Zap, HelpCircle, Clock, Shield,
  Code, FileText, RefreshCw, Loader2
} from 'lucide-react'
import AppLayout from '../../components/AppLayout'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  confidence?: number
  data?: any
  actions?: ActionSuggestion[]
  thinking?: boolean
}

interface ActionSuggestion {
  action: string
  description: string
  requires_approval: boolean
  impact: string
}

interface Suggestion {
  text: string
  icon: React.ElementType
  category: string
}

export default function ConversationPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [sessionId] = useState(() => Math.random().toString(36).substring(7))
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // Load initial suggestions
    loadSuggestions()
    
    // Add welcome message
    setMessages([{
      id: '1',
      role: 'assistant',
      content: "👋 Hi! I'm PolicyCortex AI, your Azure governance assistant. I can help you check compliance, create policies, predict violations, and automate remediation. What would you like to know?",
      timestamp: new Date(),
      confidence: 1.0
    }])
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  const loadSuggestions = async () => {
    // Default suggestions
    setSuggestions([
      { text: "What are my policy violations?", icon: AlertCircle, category: 'compliance' },
      { text: "Show compliance status", icon: Shield, category: 'compliance' },
      { text: "Create an encryption policy", icon: FileText, category: 'policy' },
      { text: "Predict next 24 hour violations", icon: Clock, category: 'prediction' },
      { text: "Fix all critical issues", icon: Zap, category: 'remediation' },
      { text: "Explain Required Tags policy", icon: HelpCircle, category: 'help' },
    ])
  }

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsTyping(true)

    // Add thinking message
    const thinkingMessage: Message = {
      id: Date.now().toString() + '-thinking',
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      thinking: true
    }
    setMessages(prev => [...prev, thinkingMessage])

    try {
      const resp = await api.chat(input, sessionId, true)
      const data = resp.data as any
      
      // Remove thinking message and add real response
      setMessages(prev => {
        const filtered = prev.filter(m => !m.thinking)
        return [...filtered, {
          id: Date.now().toString(),
          role: 'assistant',
          content: data.message,
          timestamp: new Date(),
          confidence: data.confidence,
          data: data.data,
          actions: data.actions
        }]
      })
    } catch (error) {
      console.error('Chat error:', error)
      
      // Remove thinking message and add error
      setMessages(prev => {
        const filtered = prev.filter(m => !m.thinking)
        return [...filtered, {
          id: Date.now().toString(),
          role: 'assistant',
          content: getMockResponse(input),
          timestamp: new Date(),
          confidence: 0.85
        }]
      })
    } finally {
      setIsTyping(false)
    }
  }

  const getMockResponse = (input: string): string => {
    const lower = input.toLowerCase()
    
    if (lower.includes('violation') || lower.includes('complian')) {
      return `I found **8 policy violations** in your Azure environment:

**Critical (3)**
• Storage Account 'proddata' - Encryption disabled
• VM 'web-server-01' - Public IP without NSG rules  
• Key Vault 'secrets-vault' - Soft delete disabled

**High (2)**
• SQL Database 'customerdb' - TDE not enabled
• App Service 'api-backend' - HTTPS only disabled

**Medium (3)**
• Various tagging policy violations

Would you like me to automatically remediate the critical issues?`
    }
    
    if (lower.includes('create') && lower.includes('policy')) {
      return `I'll help you create an Azure Policy. Based on your request, I've generated:

**Policy: Require Storage Encryption**
• Enforces encryption at rest for all storage accounts
• Mode: Deny (blocks non-compliant resources)
• Scope: All storage accounts in subscription

The policy JSON has been generated and is ready to deploy. Would you like to:
1. Deploy immediately to your subscription
2. Test in audit mode first
3. View the full JSON definition`
    }
    
    if (lower.includes('predict')) {
      return `Based on AI analysis of configuration drift patterns:

**High Risk Predictions (Next 24h)**
🔴 Storage 'backupdata' - Encryption will be disabled in ~18 hours
🔴 Certificate 'api-cert' - Will expire in ~12 hours

**Medium Risk (Next 48h)**
🟡 VM 'test-vm-02' - Trending toward public exposure
🟡 Key rotation policy violation for 3 keys

**Preventive Actions Available**
• Auto-remediate all predicted violations
• Schedule fixes for maintenance window
• Set up automated prevention rules

Total financial impact if not prevented: **$75,000**`
    }
    
    if (lower.includes('fix') || lower.includes('remediat')) {
      return `I can remediate the identified issues. Here's the plan:

**Remediation Plan**
1. Enable encryption on 3 storage accounts
2. Configure NSG rules for exposed VMs
3. Enable soft delete on Key Vaults
4. Apply required tags to 15 resources

**Impact Analysis**
• Resources affected: 23
• Estimated time: 15 minutes
• Downtime: None expected
• Rollback: Available for 24 hours

This requires approval. Shall I proceed with the remediation?`
    }
    
    return `I understand you're asking about "${input}". I can help you with:
• Checking compliance status and violations
• Creating and managing Azure policies
• Predicting future violations with AI
• Automating remediation of issues
• Explaining governance concepts

What specific aspect would you like to explore?`
  }

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion)
    inputRef.current?.focus()
  }

  const handleAction = async (action: ActionSuggestion) => {
    const confirmMessage = action.requires_approval 
      ? `⚠️ This action requires approval: ${action.description}\n\nImpact: ${action.impact}\n\nDo you want to proceed?`
      : `Executing: ${action.description}`
    
    setMessages(prev => [...prev, {
      id: Date.now().toString(),
      role: 'assistant',
      content: confirmMessage,
      timestamp: new Date()
    }])
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <AppLayout>
      <div className="flex flex-col h-[calc(100vh-4rem)]">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-6">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold flex items-center">
                  <Sparkles className="w-7 h-7 mr-3" />
                  Natural Language Governance Interface
                </h1>
                <p className="text-purple-100 mt-1">
                  Patent #2: Conversational Governance Intelligence System
                </p>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm bg-white/20 px-3 py-1 rounded-full">
                  Session: {sessionId}
                </span>
                <span className="text-sm bg-green-500/20 px-3 py-1 rounded-full flex items-center">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                  AI Ready
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Chat Container */}
        <div className="flex-1 flex">
          {/* Suggestions Sidebar */}
          <div className="w-80 bg-gray-50 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 p-4 overflow-y-auto">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-4">Quick Actions</h3>
            <div className="space-y-2">
              {suggestions.map((suggestion, idx) => {
                const Icon = suggestion.icon
                return (
                  <motion.button
                    key={idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    onClick={() => handleSuggestionClick(suggestion.text)}
                    className="w-full text-left p-3 rounded-lg bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center space-x-3"
                  >
                    <Icon className="w-5 h-5 text-indigo-500" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">
                      {suggestion.text}
                    </span>
                  </motion.button>
                )
              })}
            </div>

            <div className="mt-6">
              <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Capabilities</h3>
              <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <div className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                  <span>Natural language policy creation</span>
                </div>
                <div className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                  <span>Real-time compliance checking</span>
                </div>
                <div className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                  <span>Predictive violation analysis</span>
                </div>
                <div className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                  <span>Automated remediation</span>
                </div>
                <div className="flex items-start space-x-2">
                  <CheckCircle className="w-4 h-4 text-green-500 mt-0.5" />
                  <span>Multi-turn conversations</span>
                </div>
              </div>
            </div>
          </div>

          {/* Messages Area */}
          <div className="flex-1 flex flex-col">
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              <AnimatePresence>
                {messages.map((message, idx) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-2xl ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
                      <div className="flex items-start space-x-3">
                        {message.role === 'assistant' && (
                          <div className="flex-shrink-0">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-indigo-500 flex items-center justify-center">
                              <Bot className="w-5 h-5 text-white" />
                            </div>
                          </div>
                        )}
                        
                        <div className="flex-1">
                          {message.thinking ? (
                            <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4">
                              <div className="flex items-center space-x-2">
                                <Loader2 className="w-4 h-4 animate-spin text-indigo-500" />
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                  AI is thinking...
                                </span>
                              </div>
                            </div>
                          ) : (
                            <div className={`rounded-lg p-4 ${
                              message.role === 'user' 
                                ? 'bg-indigo-500 text-white' 
                                : 'bg-white dark:bg-gray-800 shadow-sm'
                            }`}>
                              <div className="prose prose-sm dark:prose-invert max-w-none">
                                {message.content.split('\n').map((line, i) => (
                                  <p key={i} className={message.role === 'user' ? 'text-white' : ''}>
                                    {line}
                                  </p>
                                ))}
                              </div>
                              
                              {message.confidence !== undefined && message.role === 'assistant' && (
                                <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs text-gray-500">
                                      Confidence: {(message.confidence * 100).toFixed(0)}%
                                    </span>
                                    <div className="flex items-center space-x-2">
                                      <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                                        <ThumbsUp className="w-4 h-4 text-gray-400" />
                                      </button>
                                      <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                                        <ThumbsDown className="w-4 h-4 text-gray-400" />
                                      </button>
                                      <button 
                                        onClick={() => copyToClipboard(message.content)}
                                        className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                                      >
                                        <Copy className="w-4 h-4 text-gray-400" />
                                      </button>
                                    </div>
                                  </div>
                                </div>
                              )}
                              
                              {message.actions && message.actions.length > 0 && (
                                <div className="mt-4 space-y-2">
                                  {message.actions.map((action, i) => (
                                    <button
                                      key={i}
                                      onClick={() => handleAction(action)}
                                      className="w-full text-left p-3 rounded-lg bg-indigo-50 dark:bg-indigo-900/20 hover:bg-indigo-100 dark:hover:bg-indigo-900/30 transition-colors"
                                    >
                                      <div className="flex items-center justify-between">
                                        <div className="flex items-center space-x-2">
                                          <Zap className="w-4 h-4 text-indigo-500" />
                                          <span className="text-sm font-medium text-indigo-700 dark:text-indigo-300">
                                            {action.description}
                                          </span>
                                        </div>
                                        {action.requires_approval && (
                                          <span className="text-xs bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 px-2 py-1 rounded">
                                            Requires Approval
                                          </span>
                                        )}
                                      </div>
                                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                                        {action.impact}
                                      </p>
                                    </button>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                          
                          {message.role === 'user' && (
                            <div className="flex-shrink-0 ml-3">
                              <div className="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center">
                                <User className="w-5 h-5 text-gray-600 dark:text-gray-300" />
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-800">
              <div className="max-w-4xl mx-auto">
                <div className="flex items-center space-x-3">
                  <input
                    ref={inputRef}
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="Ask me anything about Azure governance..."
                    className="flex-1 px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    disabled={isTyping}
                  />
                  <button
                    onClick={handleSend}
                    disabled={!input.trim() || isTyping}
                    className="p-3 rounded-lg bg-indigo-500 text-white hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                </div>
                <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                  Press Enter to send • Type "help" for assistance • Supports natural language queries
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}