// Conversational Chat Demo Fallback
// Provides intelligent demo responses when OpenAI API is unavailable

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
}

interface DemoResponse {
  message: string;
  suggestions?: string[];
  actions?: Array<{
    type: string;
    label: string;
    data: any;
  }>;
}

const demoResponses: Record<string, DemoResponse> = {
  // Governance queries
  'compliance': {
    message: "Based on our analysis, your Azure environment has a 92% compliance score. There are 3 policy violations that need attention: \n1. Storage accounts without encryption at rest\n2. VMs without backup configured\n3. Network security groups with overly permissive rules.\n\nWould you like me to generate a remediation plan?",
    suggestions: ["Show policy violations", "Generate remediation plan", "View compliance trends"]
  },
  
  'cost': {
    message: "Your current monthly Azure spend is $45,231 with a projected 12% increase next month. Top optimization opportunities:\n1. Right-size 15 oversized VMs (save ~$3,200/month)\n2. Delete 8 unattached disks (save ~$450/month)\n3. Convert 5 VMs to Reserved Instances (save ~$2,100/month)",
    suggestions: ["Show cost breakdown", "Apply optimizations", "View cost trends"]
  },
  
  'security': {
    message: "Security posture score: 88/100. Recent findings:\n• 2 critical vulnerabilities detected in container images\n• 3 storage accounts with public access enabled\n• 5 users with excessive permissions\n\nAll issues have automated remediation available.",
    suggestions: ["View security alerts", "Apply remediations", "Review permissions"]
  },
  
  'resources': {
    message: "You have 342 Azure resources across 5 subscriptions:\n• 125 Virtual Machines\n• 87 Storage Accounts\n• 45 Databases\n• 35 Network Resources\n• 50 Other Resources\n\n15 resources are tagged as non-compliant.",
    suggestions: ["View resource inventory", "Show non-compliant resources", "Analyze resource usage"]
  },
  
  // Default response
  'default': {
    message: "I'm PolicyCortex AI, your Azure governance assistant. I can help you with:\n• Compliance monitoring and remediation\n• Cost optimization recommendations\n• Security posture assessment\n• Resource governance and tagging\n• Predictive policy compliance\n\nWhat would you like to know about your Azure environment?",
    suggestions: ["Check compliance", "Analyze costs", "Review security", "Show resources"]
  }
};

export class ChatFallbackService {
  private static instance: ChatFallbackService;
  
  private constructor() {}
  
  static getInstance(): ChatFallbackService {
    if (!ChatFallbackService.instance) {
      ChatFallbackService.instance = new ChatFallbackService();
    }
    return ChatFallbackService.instance;
  }
  
  // Analyze user intent from message
  private analyzeIntent(message: string): string {
    const lowercaseMessage = message.toLowerCase();
    
    if (lowercaseMessage.includes('compliance') || lowercaseMessage.includes('policy')) {
      return 'compliance';
    }
    if (lowercaseMessage.includes('cost') || lowercaseMessage.includes('spend') || lowercaseMessage.includes('optimize')) {
      return 'cost';
    }
    if (lowercaseMessage.includes('security') || lowercaseMessage.includes('threat') || lowercaseMessage.includes('vulnerab')) {
      return 'security';
    }
    if (lowercaseMessage.includes('resource') || lowercaseMessage.includes('inventory')) {
      return 'resources';
    }
    
    return 'default';
  }
  
  // Generate demo response
  async generateResponse(message: string, context?: ChatMessage[]): Promise<ChatMessage> {
    // Simulate processing delay
    await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));
    
    const intent = this.analyzeIntent(message);
    const response = demoResponses[intent] || demoResponses.default;
    
    return {
      role: 'assistant',
      content: response.message,
      timestamp: new Date()
    };
  }
  
  // Get suggested actions based on context
  getSuggestions(context?: ChatMessage[]): string[] {
    if (!context || context.length === 0) {
      return demoResponses.default.suggestions || [];
    }
    
    const lastMessage = context[context.length - 1];
    if (lastMessage.role === 'assistant') {
      const intent = this.analyzeIntent(lastMessage.content);
      return demoResponses[intent]?.suggestions || demoResponses.default.suggestions || [];
    }
    
    return demoResponses.default.suggestions || [];
  }
  
  // Check if we should use fallback
  static shouldUseFallback(): boolean {
    return !process.env.OPENAI_API_KEY && 
           !process.env.AZURE_OPENAI_API_KEY &&
           (process.env.USE_DEMO_CHAT === 'true' || process.env.NODE_ENV === 'development');
  }
}

// Export singleton instance
export const chatFallback = ChatFallbackService.getInstance();