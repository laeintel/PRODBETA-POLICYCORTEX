'use client';

import { useState } from 'react';

export default function PolicyComposer() {
  const [step, setStep] = useState(1);
  const [policy, setPolicy] = useState({
    baseline: '',
    scope: { accounts: [], tags: [], resources: [] },
    parameters: {},
    enforcement: 'observe',
    name: ''
  });

  return (
    <div className="space-y-6">
      <header className="border-b border-gray-200 dark:border-gray-700 pb-4">
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">Policy Composer</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-1">
          Create custom policies with predictive impact analysis
        </p>
      </header>

      {/* Progress Steps */}
      <div className="flex items-center justify-between mb-8">
        {['Baseline', 'Scope', 'Parameters', 'Enforcement', 'Preview'].map((label, idx) => (
          <div key={label} className="flex items-center">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
              idx + 1 <= step ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
            }`}>
              {idx + 1}
            </div>
            <span className={`ml-2 text-sm font-medium ${
              idx + 1 <= step ? 'text-gray-900 dark:text-white' : 'text-gray-500 dark:text-gray-400'
            }`}>{label}</span>
            {idx < 4 && <div className="w-12 h-0.5 mx-2 bg-gray-200 dark:bg-gray-700" />}
          </div>
        ))}
      </div>

      {/* Step Content */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        {step === 1 && (
          <div>
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Step 1: Choose Baseline</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <button className="p-4 border-2 border-gray-200 dark:border-gray-600 rounded-lg hover:border-blue-500 dark:hover:border-blue-400 text-left">
                <h4 className="font-medium text-gray-900 dark:text-white">Start from Policy Pack</h4>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Use a curated baseline and customize</p>
              </button>
              <button className="p-4 border-2 border-gray-200 dark:border-gray-600 rounded-lg hover:border-blue-500 dark:hover:border-blue-400 text-left">
                <h4 className="font-medium text-gray-900 dark:text-white">Start from Blank</h4>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Build your policy from scratch</p>
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div>
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Step 2: Define Scope</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Accounts</label>
                <input type="text" placeholder="Select accounts..." className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Tags</label>
                <input type="text" placeholder="Add tags..." className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Resource Selectors</label>
                <input type="text" placeholder="e.g., resource.type = 'AWS::EC2::Instance'" className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white" />
              </div>
            </div>
          </div>
        )}

        {step === 3 && (
          <div>
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Step 3: Configure Parameters</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Allowed Regions</label>
                <select className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white">
                  <option>us-east-1, us-west-2</option>
                  <option>All regions</option>
                  <option>Custom...</option>
                </select>
              </div>
              <div className="flex items-center">
                <input type="checkbox" id="encryption" className="mr-2" defaultChecked />
                <label htmlFor="encryption" className="text-sm font-medium text-gray-700 dark:text-gray-300">Require encryption at rest</label>
              </div>
              <div className="flex items-center">
                <input type="checkbox" id="mfa" className="mr-2" defaultChecked />
                <label htmlFor="mfa" className="text-sm font-medium text-gray-700 dark:text-gray-300">Require MFA for admin operations</label>
              </div>
            </div>
          </div>
        )}

        {step === 4 && (
          <div>
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Step 4: Enforcement Mode</h3>
            <div className="space-y-4">
              <label className="flex items-center p-4 border-2 border-gray-200 dark:border-gray-600 rounded-lg cursor-pointer hover:border-blue-500">
                <input type="radio" name="enforcement" value="observe" className="mr-3" defaultChecked />
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">Observe</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Monitor and report violations only</p>
                </div>
              </label>
              <label className="flex items-center p-4 border-2 border-gray-200 dark:border-gray-600 rounded-lg cursor-pointer hover:border-blue-500">
                <input type="radio" name="enforcement" value="pr-block" className="mr-3" />
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">PR-Block</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Block deployments that violate policy</p>
                </div>
              </label>
              <label className="flex items-center p-4 border-2 border-gray-200 dark:border-gray-600 rounded-lg cursor-pointer hover:border-blue-500">
                <input type="radio" name="enforcement" value="auto-fix" className="mr-3" />
                <div>
                  <p className="font-medium text-gray-900 dark:text-white">Auto-fix</p>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Automatically remediate violations</p>
                </div>
              </label>
            </div>
          </div>
        )}

        {step === 5 && (
          <div>
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Step 5: Preview & Simulate</h3>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                <p className="text-sm text-blue-600 dark:text-blue-400 font-medium">Predicted Drift Reduction</p>
                <p className="text-2xl font-bold text-blue-900 dark:text-blue-200 mt-1">87%</p>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                <p className="text-sm text-green-600 dark:text-green-400 font-medium">Risk Avoided</p>
                <p className="text-2xl font-bold text-green-900 dark:text-green-200 mt-1">High</p>
              </div>
              <div className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-4">
                <p className="text-sm text-amber-600 dark:text-amber-400 font-medium">Cost Impact</p>
                <p className="text-2xl font-bold text-amber-900 dark:text-amber-200 mt-1">$156K/yr</p>
              </div>
            </div>
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <pre className="text-sm text-gray-800 dark:text-gray-200 overflow-x-auto">
{`package main

deny[msg] {
  resource := input.resource
  resource.type == "AWS::S3::Bucket"
  not resource.properties.encryption
  msg := "S3 buckets must have encryption enabled"
}`}
              </pre>
            </div>
          </div>
        )}

        {/* Navigation Buttons */}
        <div className="flex justify-between mt-6">
          <button 
            onClick={() => setStep(Math.max(1, step - 1))}
            disabled={step === 1}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50"
          >
            Previous
          </button>
          {step < 5 ? (
            <button 
              onClick={() => setStep(step + 1)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Next
            </button>
          ) : (
            <button className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
              Create Policy
            </button>
          )}
        </div>
      </div>
    </div>
  );
}