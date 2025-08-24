'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  MessageSquare, Send, Sparkles, HelpCircle, BookOpen, 
  TrendingUp, Shield, DollarSign, Users, Zap, Brain,
  ChevronRight, CheckCircle, AlertTriangle, Info
} from 'lucide-react';
import { useButtonActions } from '@/lib/button-actions';
import { toast } from 'react-hot-toast';

// Sample conversations
const sampleQueries = [
  "Which resources are not compliant with our encryption policy?",
  "Show me our cloud spend trend for the last 3 months",
  "What permissions does user john.doe have across all clouds?",
  "Create a cost optimization report for our dev environment",
  "Which security groups have port 22 open to the internet?",
  "Help me set up automated tagging for all new resources"
];

// Conversation history
const conversationHistory = [
  {
    type: 'user',
    message: "Show me all unencrypted databases in production",
    timestamp: '2 minutes ago'
  },
  {
    type: 'assistant',
    message: "I found 3 unencrypted databases in your production environment:",
    data: {
      type: 'findings',
      items: [
        { name: 'prod-mysql-01', cloud: 'AWS', region: 'us-east-1', risk: 'high' },
        { name: 'analytics-postgres', cloud: 'Azure', region: 'eastus', risk: 'high' },
        { name: 'reporting-db', cloud: 'GCP', region: 'us-central1', risk: 'medium' }
      ]
    },
    actions: ['Enable Encryption', 'Create Exception', 'View Details'],
    timestamp: '1 minute ago'
  },
  {
    type: 'user',
    message: "Enable encryption for all of them",
    timestamp: '1 minute ago'
  },
  {
    type: 'assistant',
    message: "I'll enable encryption for all 3 databases. This will require a brief maintenance window.",
    status: 'in_progress',
    steps: [
      { step: 'Creating snapshots', status: 'completed' },
      { step: 'Enabling encryption', status: 'in_progress' },
      { step: 'Validating configuration', status: 'pending' },
      { step: 'Updating compliance records', status: 'pending' }
    ],
    timestamp: 'Just now'
  }
];

// Learning resources
const learningResources = [
  {
    title: 'Cloud Cost Optimization',
    level: 'Beginner',
    duration: '15 min',
    completed: 75,
    topics: ['Reserved Instances', 'Auto-scaling', 'Right-sizing']
  },
  {
    title: 'Security Best Practices',
    level: 'Intermediate',
    duration: '30 min',
    completed: 45,
    topics: ['IAM Policies', 'Network Security', 'Encryption']
  },
  {
    title: 'Compliance Automation',
    level: 'Advanced',
    duration: '45 min',
    completed: 20,
    topics: ['Policy as Code', 'Audit Preparation', 'Evidence Collection']
  }
];

// Executive insights
const executiveInsights = [
  {
    metric: 'Cost Savings',
    value: '$47,320',
    change: '+23%',
    insight: 'AI recommendations saved 23% more than last month'
  },
  {
    metric: 'Compliance Score',
    value: '94.2%',
    change: '+5.3%',
    insight: 'Automated remediation improved compliance by 5.3%'
  },
  {
    metric: 'Security Posture',
    value: '87/100',
    change: '+12',
    insight: '156 vulnerabilities auto-remediated this week'
  },
  {
    metric: 'Team Efficiency',
    value: '3.2x',
    change: '+0.8x',
    insight: 'Teams resolving issues 3.2x faster with AI assistance'
  }
];

export default function IntelligentAssistantPage() {
  const router = useRouter();
  const actions = useButtonActions(router);
  const [message, setMessage] = useState('');
  const [selectedQuery, setSelectedQuery] = useState('');

  const handleSendMessage = () => {
    if (message.trim()) {
      actions.sendMessage(message, () => {
        setMessage('');
      });
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">AI-Powered Governance Assistant</h1>
          <p className="text-gray-600 mt-1">Natural language interface for cloud governance - no expertise required</p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant="outline"
            onClick={() => actions.openLearningCenter()}
          >
            <BookOpen className="w-4 h-4 mr-2" />
            Learning Center
          </Button>
          <Button 
            className="bg-purple-600 hover:bg-purple-700"
            onClick={() => actions.startAITraining()}
          >
            <Brain className="w-4 h-4 mr-2" />
            AI Training
          </Button>
        </div>
      </div>

      {/* Key Metrics for Non-Technical Users */}
      <div className="grid grid-cols-4 gap-4">
        {executiveInsights.map((insight, idx) => (
          <Card key={idx}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-gray-600">{insight.metric}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-baseline gap-2">
                <div className="text-2xl font-bold">{insight.value}</div>
                <span className={`text-sm ${
                  insight.change.startsWith('+') ? 'text-green-600' : 'text-red-600'
                }`}>
                  {insight.change}
                </span>
              </div>
              <div className="text-xs text-gray-600 mt-1">{insight.insight}</div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Conversational Interface */}
        <div className="col-span-2">
          <Card className="h-[600px] flex flex-col">
            <CardHeader>
              <CardTitle className="flex items-center">
                <MessageSquare className="w-5 h-5 mr-2" />
                Conversational Governance
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              {/* Chat History */}
              <div className="flex-1 overflow-y-auto space-y-4 mb-4">
                {conversationHistory.map((item, idx) => (
                  <div key={idx} className={`flex ${item.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[80%] ${item.type === 'user' ? 'order-2' : ''}`}>
                      <div className={`p-3 rounded-lg ${
                        item.type === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100'
                      }`}>
                        <div className="text-sm">{item.message}</div>
                        
                        {/* Data Display */}
                        {item.data && item.data.type === 'findings' && (
                          <div className="mt-3 space-y-2">
                            {item.data.items.map((finding, fidx) => (
                              <div key={fidx} className="bg-white bg-opacity-20 p-2 rounded text-xs">
                                <div className="flex justify-between">
                                  <span className="font-medium">{finding.name}</span>
                                  <span className={`px-2 py-1 rounded ${
                                    finding.risk === 'high' ? 'bg-red-100 text-red-700' :
                                    'bg-yellow-100 text-yellow-700'
                                  }`}>
                                    {finding.risk} risk
                                  </span>
                                </div>
                                <div className="text-gray-600 mt-1">
                                  {finding.cloud} • {finding.region}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Progress Steps */}
                        {item.steps && (
                          <div className="mt-3 space-y-1">
                            {item.steps.map((step, sidx) => (
                              <div key={sidx} className="flex items-center gap-2 text-xs">
                                {step.status === 'completed' ? (
                                  <CheckCircle className="w-3 h-3 text-green-500" />
                                ) : step.status === 'in_progress' ? (
                                  <div className="w-3 h-3 border-2 border-blue-500 rounded-full animate-spin" />
                                ) : (
                                  <div className="w-3 h-3 border-2 border-gray-300 rounded-full" />
                                )}
                                <span>{step.step}</span>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Action Buttons */}
                        {item.actions && (
                          <div className="flex gap-2 mt-3">
                            {item.actions.map((action, aidx) => (
                              <Button 
                                key={aidx} 
                                size="sm" 
                                variant="secondary" 
                                className="text-xs"
                                onClick={() => {
                                  if (action === 'Enable Encryption') {
                                    toast.success('Enabling encryption...');
                                    setTimeout(() => {
                                      toast.success('Encryption enabled successfully!');
                                    }, 2000);
                                  } else if (action === 'Create Exception') {
                                    actions.createException();
                                  } else if (action === 'View Details') {
                                    actions.viewDetails('database', 'db-001');
                                  } else {
                                    actions.handleUnimplementedAction(action);
                                  }
                                }}
                              >
                                {action}
                              </Button>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1 px-1">{item.timestamp}</div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Sample Queries */}
              <div className="border-t pt-3 mb-3">
                <div className="text-xs text-gray-500 mb-2">Try asking:</div>
                <div className="flex flex-wrap gap-2">
                  {sampleQueries.map((query, idx) => (
                    <button
                      key={idx}
                      onClick={() => setMessage(query)}
                      className="text-xs px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
                    >
                      {query}
                    </button>
                  ))}
                </div>
              </div>

              {/* Input Area */}
              <div className="flex gap-2">
                <Input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Ask anything about your cloud governance..."
                  className="flex-1"
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                />
                <Button 
                  onClick={handleSendMessage} 
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  <Send className="w-4 h-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Learning & Guidance */}
        <div className="space-y-6">
          {/* AI Capabilities */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-base">
                <Sparkles className="w-5 h-5 mr-2 text-purple-500" />
                AI Capabilities
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start gap-2">
                <DollarSign className="w-4 h-4 text-green-500 mt-0.5" />
                <div>
                  <div className="text-sm font-medium">Cost Optimization</div>
                  <div className="text-xs text-gray-600">Find savings opportunities</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Shield className="w-4 h-4 text-blue-500 mt-0.5" />
                <div>
                  <div className="text-sm font-medium">Security Analysis</div>
                  <div className="text-xs text-gray-600">Identify vulnerabilities</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-purple-500 mt-0.5" />
                <div>
                  <div className="text-sm font-medium">Compliance Check</div>
                  <div className="text-xs text-gray-600">Audit readiness assessment</div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Zap className="w-4 h-4 text-orange-500 mt-0.5" />
                <div>
                  <div className="text-sm font-medium">Auto-Remediation</div>
                  <div className="text-xs text-gray-600">Fix issues automatically</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Guided Learning */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-base">
                <BookOpen className="w-5 h-5 mr-2 text-blue-500" />
                Guided Learning Paths
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {learningResources.map((resource, idx) => (
                <div 
                  key={idx} 
                  className="p-3 border rounded-lg hover:bg-gray-50 cursor-pointer"
                  onClick={() => {
                    toast(`Opening ${resource.title} course...`, {
                      icon: '📚',
                    });
                    router.push(`/ai/learning-center/${resource.title.toLowerCase().replace(/\s+/g, '-')}`);
                  }}
                >
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <div className="font-medium text-sm">{resource.title}</div>
                      <div className="text-xs text-gray-600">
                        {resource.level} • {resource.duration}
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-gray-400" />
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5 mb-2">
                    <div 
                      className="bg-blue-600 h-1.5 rounded-full" 
                      style={{ width: `${resource.completed}%` }}
                    />
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {resource.topics.map((topic, tidx) => (
                      <span key={tidx} className="text-xs px-2 py-0.5 bg-gray-100 rounded">
                        {topic}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Help Tips */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center text-base">
                <HelpCircle className="w-5 h-5 mr-2 text-green-500" />
                Quick Tips
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-start gap-2">
                <Info className="w-3 h-3 text-blue-500 mt-0.5" />
                <p className="text-xs">Use natural language - no technical jargon needed</p>
              </div>
              <div className="flex items-start gap-2">
                <Info className="w-3 h-3 text-blue-500 mt-0.5" />
                <p className="text-xs">Ask for explanations if something is unclear</p>
              </div>
              <div className="flex items-start gap-2">
                <Info className="w-3 h-3 text-blue-500 mt-0.5" />
                <p className="text-xs">The AI learns from your patterns over time</p>
              </div>
              <div className="flex items-start gap-2">
                <Info className="w-3 h-3 text-blue-500 mt-0.5" />
                <p className="text-xs">All actions can be reviewed before execution</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}