'use client';

import React from 'react';
import { Download } from 'lucide-react';

interface DataExportProps {
  data: any[];
  filename: string;
  type?: 'csv' | 'json';
  className?: string;
}

export default function DataExport({ 
  data, 
  filename, 
  type = 'csv', 
  className = '' 
}: DataExportProps) {
  const downloadCSV = (data: any[], filename: string) => {
    if (!data || data.length === 0) return;

    const headers = Object.keys(data[0]);
    const csvContent = [
      headers.join(','),
      ...data.map(row => 
        headers.map(header => 
          typeof row[header] === 'string' && row[header].includes(',') 
            ? `"${row[header]}"` 
            : row[header]
        ).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${filename}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const downloadJSON = (data: any[], filename: string) => {
    const jsonContent = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${filename}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const handleExport = () => {
    if (type === 'csv') {
      downloadCSV(data, filename);
    } else {
      downloadJSON(data, filename);
    }
  };

  return (
    <button
      onClick={handleExport}
      disabled={!data || data.length === 0}
      className={`
        inline-flex items-center px-3 py-2 border border-gray-300 dark:border-gray-600 
        shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 dark:text-gray-300 
        bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 
        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 
        disabled:opacity-50 disabled:cursor-not-allowed transition-colors
        ${className}
      `}
      title={`Export as ${type.toUpperCase()}`}
    >
      <Download className="h-4 w-4 mr-2" />
      Export {type.toUpperCase()}
    </button>
  );
}