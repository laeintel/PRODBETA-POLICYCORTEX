'use client';

import React, { useEffect, useState } from 'react';
import { Eye, AlertCircle, CheckCircle, XCircle } from 'lucide-react';

interface AccessibilityIssue {
  type: 'error' | 'warning' | 'success';
  message: string;
  element?: string;
}

export function AccessibilityChecker() {
  const [issues, setIssues] = useState<AccessibilityIssue[]>([]);
  const [isVisible, setIsVisible] = useState(false);
  const [isChecking, setIsChecking] = useState(false);

  const checkAccessibility = () => {
    setIsChecking(true);
    const newIssues: AccessibilityIssue[] = [];

    // Check for images without alt text
    const images = document.querySelectorAll('img');
    let imagesWithoutAlt = 0;
    images.forEach(img => {
      if (!img.alt) {
        imagesWithoutAlt++;
      }
    });

    if (imagesWithoutAlt > 0) {
      newIssues.push({
        type: 'error',
        message: `${imagesWithoutAlt} image(s) missing alt text`,
        element: 'img'
      });
    } else if (images.length > 0) {
      newIssues.push({
        type: 'success',
        message: 'All images have alt text',
        element: 'img'
      });
    }

    // Check for proper heading hierarchy
    const h1Count = document.querySelectorAll('h1').length;
    const h2Count = document.querySelectorAll('h2').length;
    
    if (h1Count === 0) {
      newIssues.push({
        type: 'error',
        message: 'No H1 heading found on page',
        element: 'h1'
      });
    } else if (h1Count > 1) {
      newIssues.push({
        type: 'warning',
        message: `Multiple H1 headings found (${h1Count})`,
        element: 'h1'
      });
    } else {
      newIssues.push({
        type: 'success',
        message: 'Proper H1 heading structure',
        element: 'h1'
      });
    }

    // Check for buttons without accessible text
    const buttons = document.querySelectorAll('button');
    let buttonsWithoutText = 0;
    buttons.forEach(button => {
      const hasText = button.textContent?.trim() || button.getAttribute('aria-label');
      if (!hasText) {
        buttonsWithoutText++;
      }
    });

    if (buttonsWithoutText > 0) {
      newIssues.push({
        type: 'error',
        message: `${buttonsWithoutText} button(s) missing accessible text`,
        element: 'button'
      });
    } else if (buttons.length > 0) {
      newIssues.push({
        type: 'success',
        message: 'All buttons have accessible text',
        element: 'button'
      });
    }

    // Check for form labels
    const inputs = document.querySelectorAll('input, select, textarea');
    let inputsWithoutLabels = 0;
    inputs.forEach(input => {
      const id = input.id;
      const hasLabel = id && document.querySelector(`label[for="${id}"]`);
      const hasAriaLabel = input.getAttribute('aria-label');
      if (!hasLabel && !hasAriaLabel) {
        inputsWithoutLabels++;
      }
    });

    if (inputsWithoutLabels > 0) {
      newIssues.push({
        type: 'warning',
        message: `${inputsWithoutLabels} form field(s) missing labels`,
        element: 'input'
      });
    } else if (inputs.length > 0) {
      newIssues.push({
        type: 'success',
        message: 'All form fields have labels',
        element: 'input'
      });
    }

    // Check color contrast (simplified check)
    const hasThemeToggle = document.querySelector('[aria-label="Toggle theme"]');
    if (hasThemeToggle) {
      newIssues.push({
        type: 'success',
        message: 'Theme toggle available for contrast preferences',
        element: 'theme'
      });
    }

    // Check for keyboard navigation
    const focusableElements = document.querySelectorAll(
      'a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusableElements.length > 0) {
      newIssues.push({
        type: 'success',
        message: `${focusableElements.length} keyboard-navigable elements`,
        element: 'keyboard'
      });
    }

    setIssues(newIssues);
    setIsChecking(false);
  };

  useEffect(() => {
    if (process.env.NODE_ENV === 'development' && isVisible) {
      checkAccessibility();
    }
  }, [isVisible]);

  if (process.env.NODE_ENV !== 'development') return null;

  const getIcon = (type: AccessibilityIssue['type']) => {
    switch (type) {
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'warning':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
    }
  };

  const errorCount = issues.filter(i => i.type === 'error').length;
  const warningCount = issues.filter(i => i.type === 'warning').length;

  return (
    <>
      <button
        onClick={() => setIsVisible(!isVisible)}
        className="fixed bottom-4 left-4 z-50 p-2 bg-gray-900 dark:bg-gray-800 text-white rounded-full shadow-lg hover:scale-110 transition-transform relative"
        aria-label="Toggle accessibility checker"
      >
        <Eye className="w-5 h-5" />
        {errorCount > 0 && (
          <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-xs flex items-center justify-center">
            {errorCount}
          </span>
        )}
      </button>

      {isVisible && (
        <div className="fixed bottom-16 left-4 z-50 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-4 min-w-[300px] max-w-[400px]">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
              <Eye className="w-4 h-4" />
              Accessibility Checker
            </h3>
            <button
              onClick={checkAccessibility}
              disabled={isChecking}
              className="text-xs bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-600 disabled:opacity-50"
            >
              {isChecking ? 'Checking...' : 'Recheck'}
            </button>
          </div>

          <div className="space-y-2 max-h-[300px] overflow-y-auto">
            {issues.length === 0 ? (
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Click "Recheck" to scan for accessibility issues
              </div>
            ) : (
              issues.map((issue, index) => (
                <div
                  key={index}
                  className="flex items-start gap-2 text-sm p-2 bg-gray-50 dark:bg-gray-800 rounded"
                >
                  {getIcon(issue.type)}
                  <div className="flex-1">
                    <div className="text-gray-900 dark:text-white">
                      {issue.message}
                    </div>
                    {issue.element && (
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        Element: {issue.element}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>

          {issues.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700 flex justify-between text-xs">
              <span className="text-red-500">
                Errors: {errorCount}
              </span>
              <span className="text-yellow-500">
                Warnings: {warningCount}
              </span>
              <span className="text-green-500">
                Passed: {issues.filter(i => i.type === 'success').length}
              </span>
            </div>
          )}
        </div>
      )}
    </>
  );
}