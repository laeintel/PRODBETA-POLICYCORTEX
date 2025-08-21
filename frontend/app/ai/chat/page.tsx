'use client';

import React, { useState, useEffect, useRef, KeyboardEvent } from 'react';
import { 
  Send, 
  Mic, 
  MicOff, 
  Plus, 
  Search,
  Bot,
  User,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  ChevronDown,
  Sparkles,
  Command,
  AlertCircle,
  CheckCircle,
  Info,
  Settings,
  History,
  BookOpen,
  Zap,
  Shield,
  DollarSign,
  Activity,
  Code,
  FileText,
  Download,
  Share2,
  Pin,
  Archive
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { toast } from '@/hooks/useToast'
import { handleExport } from '@/lib/exportUtils'
import ConfigurationDialog from '@/components/ConfigurationDialog'
import { sanitizeMarkdown, textToSafeHTML } from '@/lib/sanitize'

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  actions?: QuickAction[];
  confidence?: number;
  sources?: string[];
  pinned?: boolean;
  feedback?: 'positive' | 'negative' | null;
}

interface QuickAction {
  label: string;
  icon: React.ElementType;
  action: string;
  type: 'primary' | 'secondary';
}

interface Suggestion {
  text: string;
  category: string;
  icon: React.ElementType;
}

const predefinedSuggestions: Suggestion[] = [
  { text: "What are my compliance violations?", category: "Compliance", icon: Shield },
  { text: "Show me cost optimization opportunities", category: "Cost", icon: DollarSign },
  { text: "Analyze security risks in production", category: "Security", icon: AlertCircle },
  { text: "Generate a compliance report for SOC2", category: "Reports", icon: FileText },
  { text: "What resources are non-compliant?", category: "Resources", icon: Activity },
  { text: "Predict next month's cloud costs", category: "AI Predictions", icon: Sparkles },
  { text: "Show me unused resources", category: "Optimization", icon: Zap },
  { text: "Explain my risk score", category: "Analysis", icon: Info }
];

const commandPalette = [
  { command: '/analyze', description: 'Analyze resource or policy', icon: Activity },
  { command: '/predict', description: 'Get AI predictions', icon: Sparkles },
  { command: '/remediate', description: 'Fix compliance issues', icon: CheckCircle },
  { command: '/report', description: 'Generate reports', icon: FileText },
  { command: '/optimize', description: 'Optimize costs', icon: DollarSign },
  { command: '/secure', description: 'Security analysis', icon: Shield },
  { command: '/code', description: 'Generate code', icon: Code },
  { command: '/help', description: 'Get help', icon: BookOpen }
];

export default function AIChatPage() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [selectedCommand, setSelectedCommand] = useState<number>(-1);
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [configOpen, setConfigOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Load initial welcome message
    setMessages([
      {
        id: '1',
        role: 'assistant',
        content: `Hello! I'm your AI governance assistant powered by 4 patented technologies. I can help you with:

• **Compliance Analysis** - Check violations and generate reports
• **Cost Optimization** - Find savings and predict future costs  
• **Security Assessment** - Identify risks and vulnerabilities
• **Resource Management** - Analyze and optimize your cloud resources
• **Predictive Insights** - Get AI-powered predictions and recommendations

How can I assist you today?`,
        timestamp: new Date(),
        confidence: 100,
        sources: ['Patent #2: Conversational AI']
      }
    ]);

    // Load chat history
    setChatHistory([
      { id: '1', title: 'Compliance analysis for Q4', date: '2 hours ago', messages: 5 },
      { id: '2', title: 'Cost optimization recommendations', date: 'Yesterday', messages: 12 },
      { id: '3', title: 'Security vulnerability scan', date: '3 days ago', messages: 8 },
      { id: '4', title: 'Resource tagging strategy', date: '1 week ago', messages: 15 }
    ]);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    setShowCommandPalette(false);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse = generateAIResponse(input);
      setMessages(prev => [...prev, aiResponse]);
      setIsTyping(false);
    }, 1500);
  };

  const generateAIResponse = (userInput: string): Message => {
    const lowerInput = userInput.toLowerCase();
    let response: Message = {
      id: Date.now().toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      confidence: 95,
      sources: []
    };

    // Pattern matching for different types of queries
    if (lowerInput.includes('compliance') || lowerInput.includes('violation')) {
      response.content = `Based on my analysis of your Azure environment, I've identified **12 compliance violations** across 3 policy categories:

## Critical Violations (3)
• **Unencrypted Storage**: 2 storage accounts lack encryption at rest
• **Public Access**: 1 SQL database has public endpoint enabled

## Medium Priority (9)
• **Missing Tags**: 9 resources missing required compliance tags
• **Backup Policy**: 5 VMs without backup configuration
• **Access Reviews**: 3 overdue access review assignments

### Recommended Actions:
1. Enable encryption on storage accounts immediately
2. Restrict SQL database to private endpoints
3. Apply required tags using the bulk tagging tool
4. Configure backup policies for production VMs

Would you like me to create a remediation plan or generate a detailed compliance report?`;
      response.sources = ['Azure Policy', 'Compliance Engine', 'Patent #1: Cross-Domain Correlation'];
      response.actions = [
        { label: 'View All Violations', icon: AlertCircle, action: '/governance/compliance', type: 'primary' },
        { label: 'Start Remediation', icon: CheckCircle, action: '/operations/remediation', type: 'secondary' }
      ];
    } else if (lowerInput.includes('cost') || lowerInput.includes('optimize') || lowerInput.includes('savings')) {
      response.content = `I've analyzed your cloud spending patterns and identified **$45,000/month** in potential savings:

## Top Optimization Opportunities

### 1. Compute Resources ($28,000/mo)
• **Idle VMs**: 15 VMs with <5% CPU usage
• **Oversized Instances**: 8 VMs can be rightsized
• **Dev/Test Resources**: Running 24/7 unnecessarily

### 2. Storage ($10,000/mo)
• **Orphaned Disks**: 23 unattached managed disks
• **Archive Tier**: 500GB suitable for archive storage
• **Snapshots**: 180 old snapshots can be deleted

### 3. Database ($7,000/mo)
• **Elastic Pools**: Consolidate 5 SQL databases
• **Reserved Capacity**: Save 40% with 3-year commitment

## AI Prediction
Based on current trends, your costs will increase by **23%** next month unless action is taken.

Shall I create an optimization plan with specific recommendations?`;
      response.sources = ['Cost Management API', 'Patent #4: Predictive Analytics', 'Azure Advisor'];
      response.actions = [
        { label: 'View Cost Analysis', icon: DollarSign, action: '/governance/cost', type: 'primary' },
        { label: 'Apply Optimizations', icon: Zap, action: '/operations/optimization', type: 'secondary' }
      ];
    } else if (lowerInput.includes('security') || lowerInput.includes('risk')) {
      response.content = `Security assessment completed. Here's your current security posture:

## Risk Score: 32/100 (Low Risk)

### Active Threats (0)
✅ No active security threats detected

### Vulnerabilities Found (7)
• **Critical (1)**: Unpatched SQL Server instance
• **High (2)**: Missing network segmentation in prod
• **Medium (4)**: Weak password policies on 4 accounts

### Security Recommendations
1. **Enable Azure Defender** on all subscriptions
2. **Implement MFA** for 12 admin accounts
3. **Update Network Security Groups** with stricter rules
4. **Enable Just-In-Time VM Access**
5. **Configure Private Endpoints** for PaaS services

### Compliance Status
• **ISO 27001**: 94% compliant
• **SOC 2**: 89% compliant
• **PCI-DSS**: Not applicable

Would you like me to create a security improvement roadmap?`;
      response.sources = ['Azure Security Center', 'Patent #1: Cross-Domain Correlation', 'Threat Intelligence'];
      response.actions = [
        { label: 'Security Dashboard', icon: Shield, action: '/security', type: 'primary' },
        { label: 'View Vulnerabilities', icon: AlertCircle, action: '/security/vulnerabilities', type: 'secondary' }
      ];
    } else if (lowerInput.includes('predict') || lowerInput.includes('forecast')) {
      response.content = `Based on my predictive models, here are the AI-generated insights for the next 30 days:

## Predictive Analytics Results

### 📊 Compliance Forecast
• **Prediction**: Compliance score will drop to 87% in 7 days
• **Confidence**: 92%
• **Root Cause**: 5 certificates expiring, 3 policies becoming obsolete
• **Action**: Renew certificates and update policies immediately

### 💰 Cost Projection
• **Prediction**: 23% cost increase by month-end
• **Confidence**: 87%
• **Driver**: Auto-scaling triggered by increased traffic
• **Savings Opportunity**: Implement reserved instances to save $12K

### 🔒 Security Prediction
• **Prediction**: 2 potential security incidents likely
• **Confidence**: 78%
• **Risk Factors**: Unpatched systems, expired certificates
• **Prevention**: Apply security updates within 48 hours

### 🚀 Performance Forecast
• **Prediction**: 15% performance degradation expected
• **Confidence**: 85%
• **Bottleneck**: Database reaching capacity limits
• **Solution**: Scale database tier or implement caching

These predictions are generated using Patent #4: Predictive Policy Compliance Engine with 99.2% model accuracy.`;
      response.sources = ['ML Models', 'Patent #4: Predictive Engine', 'Historical Data Analysis'];
      response.actions = [
        { label: 'View Predictions', icon: Sparkles, action: '/ai/predictive', type: 'primary' },
        { label: 'Configure Alerts', icon: AlertCircle, action: '/operations/alerts', type: 'secondary' }
      ];
    } else {
      // Default helpful response
      response.content = `I understand you're asking about "${userInput}". Let me help you with that.

Based on the context, here are some relevant insights:

1. **Current Status**: Your environment is operating normally with 94.5% compliance
2. **Recent Changes**: 12 new resources were added in the last 24 hours
3. **Recommendations**: Consider reviewing your resource tagging strategy

Would you like me to:
• Perform a detailed analysis of this topic?
• Generate a report with specific recommendations?
• Show you relevant dashboards and metrics?
• Connect you with documentation?

Just let me know how I can best assist you!`;
      response.sources = ['General Knowledge Base', 'Patent #2: Conversational AI'];
    }

    return response;
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    if (e.key === '/' && input === '') {
      setShowCommandPalette(true);
    }
    if (showCommandPalette && e.key === 'Escape') {
      setShowCommandPalette(false);
      setSelectedCommand(-1);
    }
    if (showCommandPalette && e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedCommand(prev => (prev + 1) % commandPalette.length);
    }
    if (showCommandPalette && e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedCommand(prev => prev <= 0 ? commandPalette.length - 1 : prev - 1);
    }
    if (showCommandPalette && e.key === 'Enter' && selectedCommand >= 0) {
      e.preventDefault();
      setInput(commandPalette[selectedCommand].command + ' ');
      setShowCommandPalette(false);
      setSelectedCommand(-1);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
    inputRef.current?.focus();
  };

  const handleVoiceInput = () => {
    setIsListening(!isListening);
    // Implement voice recognition
    if (!isListening) {
      console.log('Starting voice recognition...');
    } else {
      console.log('Stopping voice recognition...');
    }
  };

  const handleFeedback = (messageId: string, feedback: 'positive' | 'negative') => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, feedback } : msg
    ));
  };

  const handlePin = (messageId: string) => {
    setMessages(prev => prev.map(msg => 
      msg.id === messageId ? { ...msg, pinned: !msg.pinned } : msg
    ));
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    // Show toast notification
  };

  const MessageComponent = ({ message }: { message: Message }) => {
    const isUser = message.role === 'user';
    
    return (
      <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
        {!isUser && (
          <div className="flex-shrink-0">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <Bot className="h-6 w-6 text-white" />
            </div>
          </div>
        )}
        
        <div className={`max-w-3xl ${isUser ? 'order-1' : 'order-2'}`}>
          <div className={`rounded-lg p-4 ${
            isUser 
              ? 'bg-blue-600 text-white' 
              : 'bg-white border border-gray-200'
          }`}>
            <div className="prose prose-sm max-w-none">
              <div 
                className={isUser ? 'text-white' : 'text-gray-800'}
                dangerouslySetInnerHTML={{ 
                  __html: isUser ? textToSafeHTML(message.content) : sanitizeMarkdown(message.content.replace(/\n/g, '<br />').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'))
                }}
              />
            </div>
            
            {message.actions && (
              <div className="flex gap-2 mt-4">
                {message.actions.map((action, idx) => (
                  <button type="button"
                    key={idx}
                    onClick={() => router.push(action.action)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      action.type === 'primary'
                        ? 'bg-blue-600 text-white hover:bg-blue-700'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    <action.icon className="h-4 w-4" />
                    {action.label}
                  </button>
                ))}
              </div>
            )}
            
            {!isUser && (
              <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-200">
                <div className="flex items-center gap-4">
                  {message.confidence && (
                    <span className="text-xs text-gray-500">
                      {message.confidence}% confidence
                    </span>
                  )}
                  {message.sources && message.sources.length > 0 && (
                    <span className="text-xs text-gray-500">
                      Sources: {message.sources.join(', ')}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button type="button"
                    onClick={() => copyToClipboard(message.content)}
                    className="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
                    title="Copy"
                  >
                    <Copy className="h-4 w-4" />
                  </button>
                  <button type="button"
                    onClick={() => handleFeedback(message.id, 'positive')}
                    className={`p-1.5 rounded-lg ${
                      message.feedback === 'positive' 
                        ? 'text-green-600 bg-green-50' 
                        : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                    }`}
                    title="Helpful"
                  >
                    <ThumbsUp className="h-4 w-4" />
                  </button>
                  <button type="button"
                    onClick={() => handleFeedback(message.id, 'negative')}
                    className={`p-1.5 rounded-lg ${
                      message.feedback === 'negative' 
                        ? 'text-red-600 bg-red-50' 
                        : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                    }`}
                    title="Not helpful"
                  >
                    <ThumbsDown className="h-4 w-4" />
                  </button>
                  <button type="button"
                    onClick={() => handlePin(message.id)}
                    className={`p-1.5 rounded-lg ${
                      message.pinned 
                        ? 'text-blue-600 bg-blue-50' 
                        : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                    }`}
                    title="Pin message"
                  >
                    <Pin className="h-4 w-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
            <span>{message.timestamp.toLocaleTimeString()}</span>
            {message.pinned && (
              <span className="flex items-center gap-1 text-blue-600">
                <Pin className="h-3 w-3" />
                Pinned
              </span>
            )}
          </div>
        </div>
        
        {isUser && (
          <div className="flex-shrink-0 order-2">
            <div className="w-10 h-10 rounded-lg bg-gray-200 flex items-center justify-center">
              <User className="h-6 w-6 text-gray-600" />
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Chat History Sidebar */}
      <div className={`${showHistory ? 'w-80' : 'w-0'} transition-all duration-300 bg-white border-r border-gray-200 overflow-hidden`}>
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Chat History</h2>
            <button type="button"
              onClick={() => setShowHistory(false)}
              className="p-1 text-gray-400 hover:text-gray-600"
            >
              <ChevronDown className="h-5 w-5 rotate-90" />
            </button>
          </div>
          <div className="mt-3">
            <input
              type="text"
              placeholder="Search conversations..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
        <div className="overflow-y-auto">
          {chatHistory.map((chat) => (
            <button type="button"
              key={chat.id}
              className="w-full p-4 text-left hover:bg-gray-50 border-b border-gray-100"
            >
              <div className="font-medium text-sm text-gray-900">{chat.title}</div>
              <div className="text-xs text-gray-500 mt-1">{chat.date} • {chat.messages} messages</div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {!showHistory && (
                <button
                  type="button"
                  onClick={() => setShowHistory(true)}
                  className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg"
                  title="Show history"
                >
                  <History className="h-5 w-5" />
                </button>
              )}
              <div>
                <h1 className="text-xl font-semibold text-gray-900">AI Assistant</h1>
                <p className="text-sm text-gray-500">Powered by Patent #2: Conversational Intelligence</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button type="button" 
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg" 
                onClick={() => handleExport({
                  data: messages,
                  filename: 'chat-conversation',
                  format: 'json',
                  title: 'AI Chat Conversation'
                })}>
                <Download className="h-5 w-5" />
              </button>
              <button type="button" 
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg" 
                onClick={() => {
                  const shareUrl = `${window.location.origin}/ai/chat?session=${Date.now()}`;
                  navigator.clipboard.writeText(shareUrl);
                  toast({ title: 'Link copied', description: 'Share link copied to clipboard' });
                }}>
                <Share2 className="h-5 w-5" />
              </button>
              <button type="button" 
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg" 
                onClick={() => setConfigOpen(true)}>
                <Settings className="h-5 w-5" />
              </button>
              <button type="button"
                onClick={() => setMessages([messages[0]])}
                className="px-3 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg flex items-center gap-2"
              >
                <Plus className="h-4 w-4" />
                New Chat
              </button>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {/* Pinned Messages */}
          {messages.some(m => m.pinned) && (
            <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <h3 className="text-sm font-medium text-blue-900 mb-2 flex items-center gap-2">
                <Pin className="h-4 w-4" />
                Pinned Messages
              </h3>
              <div className="space-y-2">
                {messages.filter(m => m.pinned).map(msg => (
                  <div key={msg.id} className="text-sm text-blue-800">
                    {msg.content.substring(0, 100)}...
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map((message) => (
            <MessageComponent key={message.id} message={message} />
          ))}
          
          {isTyping && (
            <div className="flex gap-4 mb-6">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <Bot className="h-6 w-6 text-white" />
                </div>
              </div>
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Suggestions */}
        {messages.length === 1 && (
          <div className="px-6 pb-4">
            <p className="text-sm text-gray-600 mb-3">Suggested questions:</p>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
              {predefinedSuggestions.map((suggestion, idx) => (
                <button type="button"
                  key={idx}
                  onClick={() => handleSuggestionClick(suggestion.text)}
                  className="p-3 bg-white border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors text-left group"
                >
                  <div className="flex items-start gap-2">
                    <suggestion.icon className="h-4 w-4 text-gray-400 group-hover:text-blue-600 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-gray-700 group-hover:text-blue-700">
                        {suggestion.text}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">{suggestion.category}</p>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Command Palette */}
        {showCommandPalette && (
          <div className="absolute bottom-20 left-6 right-6 mx-auto max-w-2xl bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
            <div className="p-2 border-b border-gray-200">
              <p className="text-xs text-gray-500 px-2">Commands</p>
            </div>
            <div className="max-h-64 overflow-y-auto">
              {commandPalette.map((cmd, idx) => (
                <button type="button"
                  key={idx}
                  onClick={() => {
                    setInput(cmd.command + ' ');
                    setShowCommandPalette(false);
                    inputRef.current?.focus();
                  }}
                  className={`w-full px-4 py-3 flex items-center gap-3 hover:bg-gray-50 ${
                    selectedCommand === idx ? 'bg-blue-50' : ''
                  }`}
                >
                  <cmd.icon className="h-4 w-4 text-gray-400" />
                  <div className="text-left">
                    <p className="text-sm font-medium text-gray-900">{cmd.command}</p>
                    <p className="text-xs text-gray-500">{cmd.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="border-t border-gray-200 bg-white px-6 py-4">
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <div className="relative">
                <textarea
                  ref={inputRef}
                  data-testid="chat-input"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your message or / for commands..."
                  className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={1}
                  style={{ minHeight: '48px', maxHeight: '120px' }}
                />
                <button type="button"
                  onClick={handleVoiceInput}
                  className={`absolute right-3 bottom-3 p-1.5 rounded-lg transition-colors ${
                    isListening 
                      ? 'text-red-600 bg-red-50 animate-pulse' 
                      : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {isListening ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
                </button>
              </div>
              <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <Command className="h-3 w-3" />
                  Press / for commands
                </span>
                <span>Shift+Enter for new line</span>
              </div>
            </div>
            <button type="button"
              onClick={handleSend}
              disabled={!input.trim()}
              className="px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              <Send className="h-5 w-5" />
              Send
            </button>
          </div>
        </div>
      </div>
      
      <ConfigurationDialog
        isOpen={configOpen}
        onClose={() => setConfigOpen(false)}
        title="AI Chat"
        configType="ai"
      />
    </div>
  );
}