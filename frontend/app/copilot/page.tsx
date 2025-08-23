'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import {
  Brain,
  Send,
  Mic,
  Sparkles,
  Shield,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  Code,
  FileText,
  HelpCircle,
  Lightbulb,
  User,
  Bot,
  Settings,
  History,
  Bookmark,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RefreshCw,
  X
} from 'lucide-react';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  suggestions?: string[];
  actions?: { label: string; action: () => void }[];
  confidence?: number;
  sources?: string[];
}

interface Suggestion {
  icon: React.ComponentType<{ className?: string }>;
  text: string;
  category: 'cost' | 'security' | 'compliance' | 'performance';
}

export default function GovernanceCopilot() {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const quickActions = [
    "What are my biggest cost optimization opportunities?",
    "Show me security vulnerabilities that need immediate attention",
    "Am I compliant with HIPAA requirements?",
    "Why did my AWS bill spike last month?",
    "How can I improve my cloud security posture?",
    "What resources are underutilized?",
    "Help me create a budget alert",
    "Explain this policy violation"
  ];

  const proactiveSuggestions: Suggestion[] = [
    {
      icon: DollarSign,
      text: "You could save $45K/month by rightsizing your EC2 instances",
      category: 'cost'
    },
    {
      icon: Shield,
      text: "3 security groups have overly permissive rules - fix now?",
      category: 'security'
    },
    {
      icon: AlertTriangle,
      text: "Compliance drift detected in production - review changes",
      category: 'compliance'
    },
    {
      icon: Lightbulb,
      text: "Enable auto-scaling for better performance and cost efficiency",
      category: 'performance'
    }
  ];

  useEffect(() => {
    // Initial greeting
    setMessages([
      {
        id: '1',
        role: 'assistant',
        content: "👋 Hi! I'm your Governance Copilot, powered by our patented AI. I can help you with cloud costs, security, compliance, and optimization. What would you like to know?",
        timestamp: new Date(),
        suggestions: [
          "Cost optimization opportunities",
          "Security vulnerabilities",
          "Compliance status",
          "Performance improvements"
        ]
      }
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

    // Simulate AI response
    setTimeout(() => {
      const response = generateAIResponse(input);
      setMessages(prev => [...prev, response]);
      setIsTyping(false);
    }, 2000);
  };

  const generateAIResponse = (query: string): Message => {
    const lowerQuery = query.toLowerCase();
    
    if (lowerQuery.includes('cost') || lowerQuery.includes('save') || lowerQuery.includes('optimization')) {
      return {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Based on my analysis of your cloud spending, I've identified several optimization opportunities:

**Immediate Savings Available: $45,230/month**

1. **Rightsize EC2 Instances** - Save $17,000/month
   - Your prod-web-servers are using m5.4xlarge but only need m5.2xlarge
   - CPU utilization averaging 23%
   
2. **Implement Auto-shutdown** - Save $6,400/month
   - Dev environments running 24/7 but only used 8 hours/day
   
3. **Move to Glacier** - Save $7,200/month
   - Backups older than 90 days can use cheaper storage

Would you like me to automatically implement any of these optimizations?`,
        timestamp: new Date(),
        confidence: 94,
        actions: [
          { label: 'Auto-optimize all', action: () => router.push('/finops/optimization') },
          { label: 'Review details', action: () => router.push('/finops') }
        ],
        sources: ['AWS Cost Explorer', 'Azure Cost Management', 'ML Analysis']
      };
    } else if (lowerQuery.includes('security') || lowerQuery.includes('vulnerabilities')) {
      return {
        id: Date.now().toString(),
        role: 'assistant',
        content: `I've detected several security issues that need your attention:

**Critical Security Findings:**

🔴 **3 High-Risk Issues**
- S3 bucket 'customer-data' is publicly accessible
- Security group sg-0abc123 allows SSH from 0.0.0.0/0
- 12 IAM users haven't rotated credentials in 90+ days

🟡 **5 Medium-Risk Issues**
- CloudTrail logging disabled in eu-west-1
- No MFA on 4 admin accounts
- Unencrypted EBS volumes in production

I can help you fix these issues immediately. The estimated time to remediate all issues is 15 minutes with my automation.`,
        timestamp: new Date(),
        confidence: 91,
        actions: [
          { label: 'Fix critical issues now', action: () => router.push('/security') },
          { label: 'View security dashboard', action: () => router.push('/security') }
        ],
        sources: ['AWS Security Hub', 'Azure Security Center', 'Custom Scans']
      };
    } else if (lowerQuery.includes('compliance') || lowerQuery.includes('hipaa') || lowerQuery.includes('gdpr')) {
      return {
        id: Date.now().toString(),
        role: 'assistant',
        content: `Here's your compliance status across major frameworks:

**Compliance Dashboard:**

✅ **HIPAA** - 94% Compliant
- 186 of 198 controls passing
- 12 controls need attention (mostly encryption and access logging)

✅ **GDPR** - 91% Compliant  
- Data residency requirements met
- Need to update data retention policies

⚠️ **SOC 2** - 87% Compliant
- Security controls strong
- Need better change management documentation

**Next Audit:** March 15, 2025 (52 days)

I can generate a compliance report or help you address the gaps. What would you prefer?`,
        timestamp: new Date(),
        confidence: 96,
        actions: [
          { label: 'Generate compliance report', action: () => router.push('/governance/compliance') },
          { label: 'Fix compliance gaps', action: () => router.push('/governance/compliance') }
        ],
        sources: ['Compliance Scanner', 'Policy Engine', 'Audit Logs']
      };
    } else {
      return {
        id: Date.now().toString(),
        role: 'assistant',
        content: `I understand you're asking about "${query}". Let me help you with that.

Based on your infrastructure patterns, here are some relevant insights:

1. **Current Status**: Your systems are operating normally with 99.98% uptime
2. **Recent Changes**: 3 deployments in the last 24 hours, all successful
3. **Recommendations**: Consider implementing the suggested optimizations to improve efficiency

Would you like me to provide more specific information about any particular area?`,
        timestamp: new Date(),
        confidence: 85,
        suggestions: [
          "Tell me more about costs",
          "Show security status",
          "Check compliance",
          "View recent changes"
        ]
      };
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
    handleSend();
  };

  const handleVoiceInput = () => {
    setIsListening(!isListening);
    // In production, this would use Web Speech API
    if (!isListening) {
      setTimeout(() => {
        setInput("Show me my biggest cost optimization opportunities");
        setIsListening(false);
      }, 2000);
    }
  };

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 border-b dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">Governance Copilot</h1>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  AI-powered cloud governance assistant
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
                <History className="h-5 w-5" />
              </button>
              <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
                <Bookmark className="h-5 w-5" />
              </button>
              <button className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg">
                <Settings className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role === 'assistant' && (
                <div className="flex-shrink-0">
                  <div className="p-2 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg">
                    <Bot className="h-5 w-5 text-white" />
                  </div>
                </div>
              )}
              
              <div className={`max-w-2xl ${message.role === 'user' ? 'order-1' : ''}`}>
                <div className={`p-4 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white dark:bg-gray-800 border dark:border-gray-700'
                }`}>
                  <div className="whitespace-pre-wrap">{message.content}</div>
                  
                  {message.confidence && (
                    <div className="mt-2 text-sm opacity-75">
                      Confidence: {message.confidence}%
                    </div>
                  )}
                  
                  {message.actions && (
                    <div className="flex gap-2 mt-3">
                      {message.actions.map((action, idx) => (
                        <button
                          key={idx}
                          onClick={action.action}
                          className="px-3 py-1 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700"
                        >
                          {action.label}
                        </button>
                      ))}
                    </div>
                  )}
                  
                  {message.suggestions && (
                    <div className="mt-3 pt-3 border-t dark:border-gray-600">
                      <p className="text-sm mb-2 opacity-75">Suggested follow-ups:</p>
                      <div className="flex flex-wrap gap-2">
                        {message.suggestions.map((suggestion, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSuggestionClick(suggestion)}
                            className="px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-sm hover:bg-gray-200 dark:hover:bg-gray-600"
                          >
                            {suggestion}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {message.sources && (
                    <div className="mt-2 text-xs opacity-60">
                      Sources: {message.sources.join(', ')}
                    </div>
                  )}
                </div>
                
                {message.role === 'assistant' && (
                  <div className="flex gap-2 mt-2 text-sm">
                    <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                      <Copy className="h-4 w-4" />
                    </button>
                    <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                      <ThumbsUp className="h-4 w-4" />
                    </button>
                    <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                      <ThumbsDown className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </div>
              
              {message.role === 'user' && (
                <div className="flex-shrink-0 order-2">
                  <div className="p-2 bg-gray-200 dark:bg-gray-700 rounded-lg">
                    <User className="h-5 w-5" />
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {isTyping && (
            <div className="flex gap-3">
              <div className="p-2 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg">
                <Bot className="h-5 w-5 text-white" />
              </div>
              <div className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-lg p-4">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white dark:bg-gray-800 border-t dark:border-gray-700 p-4">
          <div className="flex gap-2">
            <button
              onClick={handleVoiceInput}
              className={`p-2 rounded-lg ${
                isListening
                  ? 'bg-red-600 text-white animate-pulse'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              <Mic className="h-5 w-5" />
            </button>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Ask me anything about your cloud governance..."
              className="flex-1 px-4 py-2 border dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim()}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
            >
              <Send className="h-5 w-5" />
              Send
            </button>
          </div>
          
          {/* Quick Actions */}
          <div className="mt-3 flex gap-2 overflow-x-auto pb-2">
            {quickActions.slice(0, 4).map((action, idx) => (
              <button
                key={idx}
                onClick={() => handleSuggestionClick(action)}
                className="px-3 py-1 bg-gray-100 dark:bg-gray-700 rounded-full text-sm whitespace-nowrap hover:bg-gray-200 dark:hover:bg-gray-600"
              >
                {action}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Proactive Suggestions Sidebar */}
      <div className="w-80 bg-gray-50 dark:bg-gray-900 border-l dark:border-gray-700 p-4">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-yellow-500" />
          Proactive Insights
        </h2>
        
        <div className="space-y-3">
          {proactiveSuggestions.map((suggestion, idx) => {
            const Icon = suggestion.icon;
            return (
              <div
                key={idx}
                className="bg-white dark:bg-gray-800 rounded-lg p-3 cursor-pointer hover:shadow-md transition-shadow"
                onClick={() => handleSuggestionClick(suggestion.text)}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-lg ${
                    suggestion.category === 'cost' ? 'bg-green-50 dark:bg-green-900/20' :
                    suggestion.category === 'security' ? 'bg-red-50 dark:bg-red-900/20' :
                    suggestion.category === 'compliance' ? 'bg-yellow-50 dark:bg-yellow-900/20' :
                    'bg-blue-50 dark:bg-blue-900/20'
                  }`}>
                    <Icon className={`h-5 w-5 ${
                      suggestion.category === 'cost' ? 'text-green-600 dark:text-green-400' :
                      suggestion.category === 'security' ? 'text-red-600 dark:text-red-400' :
                      suggestion.category === 'compliance' ? 'text-yellow-600 dark:text-yellow-400' :
                      'text-blue-600 dark:text-blue-400'
                    }`} />
                  </div>
                  <p className="text-sm">{suggestion.text}</p>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-6">
          <h3 className="text-sm font-semibold mb-2 text-gray-600 dark:text-gray-400">
            Learning Resources
          </h3>
          <div className="space-y-2">
            <button className="w-full text-left px-3 py-2 bg-white dark:bg-gray-800 rounded-lg text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
              📚 Cloud Governance Best Practices
            </button>
            <button className="w-full text-left px-3 py-2 bg-white dark:bg-gray-800 rounded-lg text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
              🎓 FinOps Certification Guide
            </button>
            <button className="w-full text-left px-3 py-2 bg-white dark:bg-gray-800 rounded-lg text-sm hover:bg-gray-100 dark:hover:bg-gray-700">
              🔒 Security Compliance Checklist
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}