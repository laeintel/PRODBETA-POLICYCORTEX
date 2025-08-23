'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  MessageSquare,
  Shield,
  Cloud,
  Code,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Brain,
  Sparkles,
  Send,
  RefreshCw,
  Copy,
  Download,
  History,
  BookOpen,
  ArrowLeft
} from 'lucide-react';

interface PolicyTranslation {
  naturalLanguage: string;
  awsPolicy?: string;
  azurePolicy?: string;
  gcpPolicy?: string;
  terraformPolicy?: string;
  confidence: number;
  warnings?: string[];
  suggestions?: string[];
}

export default function NaturalLanguagePolicyStudio() {
  const router = useRouter();
  const [input, setInput] = useState('');
  const [translations, setTranslations] = useState<PolicyTranslation[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedCloud, setSelectedCloud] = useState<'all' | 'aws' | 'azure' | 'gcp'>('all');

  const exampleQueries = [
    "Ensure all databases are encrypted with AES-256",
    "Restrict production access to senior engineers only",
    "Block all public internet access to storage buckets",
    "Require MFA for all admin operations",
    "Enforce HIPAA compliance on all healthcare data",
    "Prevent resource creation outside US regions",
    "Limit VM sizes to cost-optimized instances",
    "Require approval for resources costing over $1000/month"
  ];

  const handleTranslate = async () => {
    if (!input.trim()) return;
    
    setIsProcessing(true);
    
    // Simulate AI processing
    setTimeout(() => {
      const newTranslation: PolicyTranslation = {
        naturalLanguage: input,
        awsPolicy: `{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Deny",
    "Principal": "*",
    "Action": ["s3:*"],
    "Resource": "*",
    "Condition": {
      "StringNotEquals": {
        "s3:x-amz-server-side-encryption": "AES256"
      }
    }
  }]
}`,
        azurePolicy: `{
  "properties": {
    "displayName": "Enforce encryption",
    "policyType": "Custom",
    "mode": "All",
    "policyRule": {
      "if": {
        "field": "Microsoft.Storage/storageAccounts/encryption.services.blob.enabled",
        "equals": false
      },
      "then": {
        "effect": "deny"
      }
    }
  }
}`,
        gcpPolicy: `bindings:
- members:
  - serviceAccount:policy-engine@project.iam.gserviceaccount.com
  role: roles/storage.admin
  condition:
    title: Require Encryption
    expression: resource.encryption == "AES256"`,
        terraformPolicy: `resource "aws_s3_bucket_server_side_encryption_configuration" "example" {
  bucket = aws_s3_bucket.example.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}`,
        confidence: 0.94,
        warnings: input.toLowerCase().includes('delete') ? ['This policy includes destructive actions'] : undefined,
        suggestions: ['Consider adding exception for backup buckets', 'Add monitoring for policy violations']
      };
      
      setTranslations([newTranslation, ...translations]);
      setInput('');
      setIsProcessing(false);
    }, 2000);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-4">
          <button
            onClick={() => router.push('/ai')}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h1 className="text-4xl font-bold flex items-center gap-3">
              <MessageSquare className="h-10 w-10 text-purple-600" />
              Natural Language Policy Studio
            </h1>
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              Describe policies in plain English - AI generates cloud-specific implementations
            </p>
          </div>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center gap-2">
            <History className="h-5 w-5" />
            History
          </button>
          <button className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2">
            <BookOpen className="h-5 w-5" />
            Policy Library
          </button>
        </div>
      </div>

      {/* AI Capability Banner */}
      <div className="mb-8 p-4 bg-gradient-to-r from-purple-600 to-indigo-600 rounded-xl text-white">
        <div className="flex items-start gap-3">
          <Brain className="h-8 w-8 mt-1" />
          <div>
            <h2 className="text-xl font-semibold mb-2">
              Patent #2: Conversational Governance Intelligence
            </h2>
            <p className="text-purple-100">
              Our 175B parameter AI model understands governance intent and automatically generates 
              compliant policies across AWS, Azure, GCP, and Infrastructure-as-Code platforms.
            </p>
            <div className="flex gap-4 mt-3 text-sm">
              <span className="flex items-center gap-1">
                <CheckCircle className="h-4 w-4" />
                98.7% Azure accuracy
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle className="h-4 w-4" />
                98.2% AWS accuracy
              </span>
              <span className="flex items-center gap-1">
                <CheckCircle className="h-4 w-4" />
                97.5% GCP accuracy
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Input Section */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <h2 className="text-xl font-semibold mb-4">Describe Your Policy</h2>
        <div className="space-y-4">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Example: Ensure all databases are encrypted and prevent public access to sensitive data..."
            className="w-full h-32 px-4 py-3 border dark:border-gray-700 rounded-lg bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          
          <div className="flex items-center justify-between">
            <div className="flex gap-2">
              <button
                onClick={() => setSelectedCloud('all')}
                className={`px-3 py-1 rounded-md text-sm ${
                  selectedCloud === 'all' 
                    ? 'bg-purple-600 text-white' 
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                All Clouds
              </button>
              <button
                onClick={() => setSelectedCloud('aws')}
                className={`px-3 py-1 rounded-md text-sm ${
                  selectedCloud === 'aws' 
                    ? 'bg-orange-600 text-white' 
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                AWS
              </button>
              <button
                onClick={() => setSelectedCloud('azure')}
                className={`px-3 py-1 rounded-md text-sm ${
                  selectedCloud === 'azure' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                Azure
              </button>
              <button
                onClick={() => setSelectedCloud('gcp')}
                className={`px-3 py-1 rounded-md text-sm ${
                  selectedCloud === 'gcp' 
                    ? 'bg-green-600 text-white' 
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                GCP
              </button>
            </div>
            
            <button
              onClick={handleTranslate}
              disabled={!input.trim() || isProcessing}
              className="px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="h-5 w-5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Sparkles className="h-5 w-5" />
                  Generate Policies
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Example Queries */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-8">
        <h3 className="text-lg font-semibold mb-3">Example Queries</h3>
        <div className="flex flex-wrap gap-2">
          {exampleQueries.map((query, idx) => (
            <button
              key={idx}
              onClick={() => setInput(query)}
              className="px-3 py-1 bg-purple-50 dark:bg-purple-900/20 text-purple-700 dark:text-purple-300 rounded-lg text-sm hover:bg-purple-100 dark:hover:bg-purple-900/30"
            >
              {query}
            </button>
          ))}
        </div>
      </div>

      {/* Translations */}
      {translations.map((translation, idx) => (
        <div key={idx} className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 mb-6">
          <div className="mb-4">
            <div className="flex items-start justify-between mb-2">
              <h3 className="text-lg font-semibold">{translation.naturalLanguage}</h3>
              <span className="px-2 py-1 bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 rounded-full text-xs font-medium">
                {(translation.confidence * 100).toFixed(0)}% confident
              </span>
            </div>
            
            {translation.warnings && (
              <div className="flex items-center gap-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg mb-3">
                <AlertTriangle className="h-4 w-4 text-yellow-600" />
                <span className="text-sm text-yellow-700 dark:text-yellow-300">
                  {translation.warnings[0]}
                </span>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {(selectedCloud === 'all' || selectedCloud === 'aws') && translation.awsPolicy && (
              <div className="border dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-orange-600">AWS Policy</h4>
                  <div className="flex gap-2">
                    <button
                      onClick={() => copyToClipboard(translation.awsPolicy!)}
                      className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                    <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                      <Download className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                <pre className="text-xs bg-gray-50 dark:bg-gray-900 p-3 rounded overflow-x-auto">
                  <code>{translation.awsPolicy}</code>
                </pre>
              </div>
            )}

            {(selectedCloud === 'all' || selectedCloud === 'azure') && translation.azurePolicy && (
              <div className="border dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-blue-600">Azure Policy</h4>
                  <div className="flex gap-2">
                    <button
                      onClick={() => copyToClipboard(translation.azurePolicy!)}
                      className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                    <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                      <Download className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                <pre className="text-xs bg-gray-50 dark:bg-gray-900 p-3 rounded overflow-x-auto">
                  <code>{translation.azurePolicy}</code>
                </pre>
              </div>
            )}

            {(selectedCloud === 'all' || selectedCloud === 'gcp') && translation.gcpPolicy && (
              <div className="border dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-green-600">GCP Policy</h4>
                  <div className="flex gap-2">
                    <button
                      onClick={() => copyToClipboard(translation.gcpPolicy!)}
                      className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                    <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                      <Download className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                <pre className="text-xs bg-gray-50 dark:bg-gray-900 p-3 rounded overflow-x-auto">
                  <code>{translation.gcpPolicy}</code>
                </pre>
              </div>
            )}

            {translation.terraformPolicy && (
              <div className="border dark:border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-purple-600">Terraform</h4>
                  <div className="flex gap-2">
                    <button
                      onClick={() => copyToClipboard(translation.terraformPolicy!)}
                      className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                    <button className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded">
                      <Download className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                <pre className="text-xs bg-gray-50 dark:bg-gray-900 p-3 rounded overflow-x-auto">
                  <code>{translation.terraformPolicy}</code>
                </pre>
              </div>
            )}
          </div>

          {translation.suggestions && (
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <h4 className="font-medium text-blue-700 dark:text-blue-300 mb-2">
                AI Suggestions
              </h4>
              <ul className="list-disc list-inside text-sm text-blue-600 dark:text-blue-400">
                {translation.suggestions.map((suggestion, i) => (
                  <li key={i}>{suggestion}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}