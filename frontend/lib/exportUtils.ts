// Export utilities for data export functionality
import { toast } from '@/hooks/useToast'

export interface ExportOptions {
  data: any
  filename: string
  format: 'csv' | 'json' | 'pdf'
  title?: string
}

export const exportToCSV = (data: any[], filename: string) => {
  // Convert data to CSV format
  if (!data || data.length === 0) {
    toast({
      title: 'No data to export',
      description: 'There is no data available for export',
      variant: 'destructive'
    })
    return
  }

  // Get headers from first object
  const headers = Object.keys(data[0])
  const csvContent = [
    headers.join(','),
    ...data.map(row => 
      headers.map(header => {
        const value = row[header]
        // Handle values with commas or quotes
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`
        }
        return value ?? ''
      }).join(',')
    )
  ].join('\n')

  // Create blob and download
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)
  
  link.setAttribute('href', url)
  link.setAttribute('download', `${filename}_${new Date().toISOString().split('T')[0]}.csv`)
  link.style.visibility = 'hidden'
  
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  
  toast({
    title: 'Export successful',
    description: `Data exported to ${filename}.csv`
  })
}

export const exportToJSON = (data: any, filename: string) => {
  const jsonContent = JSON.stringify(data, null, 2)
  
  const blob = new Blob([jsonContent], { type: 'application/json' })
  const link = document.createElement('a')
  const url = URL.createObjectURL(blob)
  
  link.setAttribute('href', url)
  link.setAttribute('download', `${filename}_${new Date().toISOString().split('T')[0]}.json`)
  link.style.visibility = 'hidden'
  
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  
  toast({
    title: 'Export successful',
    description: `Data exported to ${filename}.json`
  })
}

export const exportToPDF = async (data: any, filename: string, title?: string) => {
  // For now, we'll export as formatted JSON in a data URL
  // In production, you'd integrate with a PDF library like jsPDF
  toast({
    title: 'Generating PDF',
    description: 'Creating PDF report...'
  })
  
  // Simulate PDF generation
  setTimeout(() => {
    const content = `
      <html>
        <head>
          <title>${title || filename}</title>
          <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            h1 { color: #333; }
            pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
            .timestamp { color: #666; font-size: 12px; }
          </style>
        </head>
        <body>
          <h1>${title || filename}</h1>
          <p class="timestamp">Generated on ${new Date().toLocaleString()}</p>
          <pre>${JSON.stringify(data, null, 2)}</pre>
        </body>
      </html>
    `
    
    const blob = new Blob([content], { type: 'text/html' })
    const link = document.createElement('a')
    const url = URL.createObjectURL(blob)
    
    link.setAttribute('href', url)
    link.setAttribute('download', `${filename}_${new Date().toISOString().split('T')[0]}.html`)
    link.style.visibility = 'hidden'
    
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    toast({
      title: 'Export successful',
      description: `Report exported to ${filename}.html`
    })
  }, 1000)
}

export const handleExport = (options: ExportOptions) => {
  const { data, filename, format, title } = options
  
  switch (format) {
    case 'csv':
      if (Array.isArray(data)) {
        exportToCSV(data, filename)
      } else {
        exportToCSV([data], filename)
      }
      break
    case 'json':
      exportToJSON(data, filename)
      break
    case 'pdf':
      exportToPDF(data, filename, title)
      break
    default:
      toast({
        title: 'Invalid format',
        description: 'Please select a valid export format',
        variant: 'destructive'
      })
  }
}