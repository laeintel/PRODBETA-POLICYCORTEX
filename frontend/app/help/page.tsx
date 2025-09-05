'use client';

import { Card } from '@/components/ui/card';
import { HelpCircle, Book, MessageSquare, FileText, Video, Mail, Github, ExternalLink } from 'lucide-react';

export default function HelpPage() {
  const resources = [
    {
      icon: <Book className="h-6 w-6 text-blue-500" />,
      title: 'Documentation',
      description: 'Comprehensive guides and API references',
      link: '/docs'
    },
    {
      icon: <Video className="h-6 w-6 text-purple-500" />,
      title: 'Video Tutorials',
      description: 'Step-by-step video walkthroughs',
      link: '/tutorials'
    },
    {
      icon: <MessageSquare className="h-6 w-6 text-green-500" />,
      title: 'Community Forum',
      description: 'Get help from the community',
      link: '/community'
    },
    {
      icon: <Github className="h-6 w-6 text-gray-700 dark:text-gray-300" />,
      title: 'GitHub',
      description: 'Report issues and contribute',
      link: 'https://github.com/policycortex'
    }
  ];

  const faqs = [
    {
      question: 'How do I connect my Azure subscription?',
      answer: 'Navigate to Settings > Integrations and click on "Connect Azure". You\'ll need your subscription ID and appropriate permissions.'
    },
    {
      question: 'What compliance frameworks are supported?',
      answer: 'PolicyCortex supports major frameworks including SOC2, ISO 27001, HIPAA, PCI-DSS, and custom policies.'
    },
    {
      question: 'How often are resources scanned?',
      answer: 'By default, resources are scanned every 6 hours. You can configure this in Settings > Compliance.'
    },
    {
      question: 'Can I export compliance reports?',
      answer: 'Yes, navigate to the Audit page and click "Export Report" to download reports in PDF or CSV format.'
    }
  ];

  return (
    <div className="container mx-auto p-6 space-y-6 max-w-6xl">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold flex items-center justify-center gap-3 mb-4">
          <HelpCircle className="h-8 w-8" />
          Help Center
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Find answers to your questions and learn how to use PolicyCortex effectively
        </p>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {resources.map((resource, idx) => (
          <Card key={idx} className="p-6 hover:shadow-lg transition-shadow cursor-pointer">
            <div className="flex flex-col items-center text-center space-y-3">
              {resource.icon}
              <h3 className="font-semibold">{resource.title}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {resource.description}
              </p>
              <a href={resource.link} className="text-blue-600 text-sm hover:underline flex items-center gap-1">
                Learn more <ExternalLink className="h-3 w-3" />
              </a>
            </div>
          </Card>
        ))}
      </div>

      {/* FAQs */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-6">Frequently Asked Questions</h2>
        <div className="space-y-6">
          {faqs.map((faq, idx) => (
            <div key={idx} className="border-b border-gray-200 dark:border-gray-700 pb-4 last:border-0">
              <h3 className="font-medium mb-2">{faq.question}</h3>
              <p className="text-gray-600 dark:text-gray-400">{faq.answer}</p>
            </div>
          ))}
        </div>
      </Card>

      {/* Contact Support */}
      <Card className="p-6 bg-blue-50 dark:bg-blue-950">
        <div className="flex items-start gap-4">
          <Mail className="h-6 w-6 text-blue-600 mt-1" />
          <div className="flex-1">
            <h2 className="text-xl font-semibold mb-2">Need More Help?</h2>
            <p className="text-gray-700 dark:text-gray-300 mb-4">
              Our support team is here to help you with any questions or issues.
            </p>
            <div className="flex gap-4">
              <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                Contact Support
              </button>
              <button className="px-4 py-2 border border-blue-600 text-blue-600 rounded-md hover:bg-blue-50">
                Schedule a Demo
              </button>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}