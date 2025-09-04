import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import DataExport from '@/components/DataExport';

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = jest.fn(() => 'blob:mock-url');
global.URL.revokeObjectURL = jest.fn();

// Mock Blob
global.Blob = jest.fn((content, options) => ({
  content,
  options,
  size: content[0].length,
  type: options.type
})) as any;

describe('DataExport Component', () => {
  const mockData = [
    { id: 1, name: 'Resource 1', status: 'Active', cost: 100 },
    { id: 2, name: 'Resource 2', status: 'Inactive', cost: 150 },
    { id: 3, name: 'Resource, Special', status: 'Active', cost: 200 }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    // Mock createElement for link
    document.createElement = jest.fn((tagName) => {
      if (tagName === 'a') {
        return {
          href: '',
          download: '',
          click: jest.fn(),
          setAttribute: jest.fn(),
          style: {}
        } as any;
      }
      return document.createElement.call(document, tagName);
    });
  });

  describe('Rendering', () => {
    it('renders export button with CSV as default type', () => {
      render(<DataExport data={mockData} filename="test-export" />);
      
      const button = screen.getByRole('button');
      expect(button).toBeInTheDocument();
      expect(button).toHaveTextContent('Export CSV');
      expect(button).toHaveAttribute('title', 'Export as CSV');
    });

    it('renders export button for JSON type', () => {
      render(<DataExport data={mockData} filename="test-export" type="json" />);
      
      const button = screen.getByRole('button');
      expect(button).toHaveTextContent('Export JSON');
      expect(button).toHaveAttribute('title', 'Export as JSON');
    });

    it('renders with custom className', () => {
      render(
        <DataExport 
          data={mockData} 
          filename="test-export" 
          className="custom-export-class"
        />
      );
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('custom-export-class');
    });

    it('renders with Download icon', () => {
      const { container } = render(<DataExport data={mockData} filename="test-export" />);
      
      const icon = container.querySelector('svg');
      expect(icon).toBeInTheDocument();
      expect(icon).toHaveClass('h-4', 'w-4', 'mr-2');
    });

    it('is disabled when data is empty', () => {
      render(<DataExport data={[]} filename="test-export" />);
      
      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
      expect(button).toHaveClass('disabled:opacity-50', 'disabled:cursor-not-allowed');
    });

    it('is disabled when data is null', () => {
      render(<DataExport data={null as any} filename="test-export" />);
      
      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
    });

    it('is enabled when data has items', () => {
      render(<DataExport data={mockData} filename="test-export" />);
      
      const button = screen.getByRole('button');
      expect(button).toBeEnabled();
    });
  });

  describe('CSV Export', () => {
    it('exports data as CSV when clicked', () => {
      const createElementSpy = jest.spyOn(document, 'createElement');
      render(<DataExport data={mockData} filename="test-export" type="csv" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      // Check that a link was created and clicked
      expect(createElementSpy).toHaveBeenCalledWith('a');
      
      // Check Blob was created with CSV content
      expect(global.Blob).toHaveBeenCalledWith(
        expect.arrayContaining([expect.stringContaining('id,name,status,cost')]),
        { type: 'text/csv;charset=utf-8;' }
      );
      
      // Check URL methods were called
      expect(global.URL.createObjectURL).toHaveBeenCalled();
      expect(global.URL.revokeObjectURL).toHaveBeenCalled();
    });

    it('handles CSV data with commas in values', () => {
      render(<DataExport data={mockData} filename="test-export" type="csv" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      // Check that comma-containing value is quoted
      const blobCall = (global.Blob as jest.Mock).mock.calls[0];
      const csvContent = blobCall[0][0];
      expect(csvContent).toContain('"Resource, Special"');
    });

    it('creates CSV with correct headers', () => {
      render(<DataExport data={mockData} filename="test-export" type="csv" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      const blobCall = (global.Blob as jest.Mock).mock.calls[0];
      const csvContent = blobCall[0][0];
      const lines = csvContent.split('\n');
      
      expect(lines[0]).toBe('id,name,status,cost');
    });

    it('creates CSV with correct data rows', () => {
      render(<DataExport data={mockData} filename="test-export" type="csv" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      const blobCall = (global.Blob as jest.Mock).mock.calls[0];
      const csvContent = blobCall[0][0];
      const lines = csvContent.split('\n');
      
      expect(lines[1]).toBe('1,Resource 1,Active,100');
      expect(lines[2]).toBe('2,Resource 2,Inactive,150');
    });

    it('sets correct filename for CSV download', () => {
      const createElementSpy = jest.spyOn(document, 'createElement');
      render(<DataExport data={mockData} filename="my-data" type="csv" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      const linkElement = createElementSpy.mock.results[0].value;
      expect(linkElement.download).toBe('my-data.csv');
    });

    it('does not export when data is empty', () => {
      render(<DataExport data={[]} filename="test-export" type="csv" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      // Blob should not be created for empty data
      expect(global.Blob).not.toHaveBeenCalled();
    });
  });

  describe('JSON Export', () => {
    it('exports data as JSON when clicked', () => {
      render(<DataExport data={mockData} filename="test-export" type="json" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      // Check Blob was created with JSON content
      expect(global.Blob).toHaveBeenCalledWith(
        [JSON.stringify(mockData, null, 2)],
        { type: 'application/json' }
      );
      
      expect(global.URL.createObjectURL).toHaveBeenCalled();
      expect(global.URL.revokeObjectURL).toHaveBeenCalled();
    });

    it('creates properly formatted JSON', () => {
      render(<DataExport data={mockData} filename="test-export" type="json" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      const blobCall = (global.Blob as jest.Mock).mock.calls[0];
      const jsonContent = blobCall[0][0];
      const parsedJson = JSON.parse(jsonContent);
      
      expect(parsedJson).toEqual(mockData);
    });

    it('sets correct filename for JSON download', () => {
      const createElementSpy = jest.spyOn(document, 'createElement');
      render(<DataExport data={mockData} filename="my-data" type="json" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      const linkElement = createElementSpy.mock.results[0].value;
      expect(linkElement.download).toBe('my-data.json');
    });

    it('handles complex nested data structures', () => {
      const complexData = [
        {
          id: 1,
          nested: { level1: { level2: 'value' } },
          array: [1, 2, 3]
        }
      ];
      
      render(<DataExport data={complexData} filename="complex" type="json" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      const blobCall = (global.Blob as jest.Mock).mock.calls[0];
      const jsonContent = blobCall[0][0];
      const parsedJson = JSON.parse(jsonContent);
      
      expect(parsedJson).toEqual(complexData);
    });
  });

  describe('Edge Cases', () => {
    it('handles data with missing fields', () => {
      const inconsistentData = [
        { id: 1, name: 'Item 1', extra: 'field' },
        { id: 2, different: 'structure' }
      ];
      
      render(<DataExport data={inconsistentData} filename="test" type="csv" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      expect(global.Blob).toHaveBeenCalled();
    });

    it('handles data with null values', () => {
      const dataWithNull = [
        { id: 1, name: null, status: 'Active' },
        { id: 2, name: 'Resource', status: null }
      ];
      
      render(<DataExport data={dataWithNull} filename="test" type="csv" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      const blobCall = (global.Blob as jest.Mock).mock.calls[0];
      const csvContent = blobCall[0][0];
      
      // null should be converted to string 'null' in CSV
      expect(csvContent).toContain('null');
    });

    it('handles data with undefined values', () => {
      const dataWithUndefined = [
        { id: 1, name: undefined, status: 'Active' }
      ];
      
      render(<DataExport data={dataWithUndefined} filename="test" type="json" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      // JSON.stringify should handle undefined properly
      expect(global.Blob).toHaveBeenCalled();
    });

    it('handles special characters in data', () => {
      const specialData = [
        { id: 1, name: 'Item "with" quotes', description: 'Line 1\nLine 2' }
      ];
      
      render(<DataExport data={specialData} filename="test" type="csv" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      expect(global.Blob).toHaveBeenCalled();
    });

    it('handles very large datasets', () => {
      const largeData = Array.from({ length: 1000 }, (_, i) => ({
        id: i,
        name: `Resource ${i}`,
        value: Math.random()
      }));
      
      render(<DataExport data={largeData} filename="large" type="json" />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      
      expect(global.Blob).toHaveBeenCalled();
      const blobCall = (global.Blob as jest.Mock).mock.calls[0];
      const jsonContent = blobCall[0][0];
      const parsedJson = JSON.parse(jsonContent);
      
      expect(parsedJson.length).toBe(1000);
    });
  });

  describe('Styling and Theme Support', () => {
    it('includes dark mode classes', () => {
      render(<DataExport data={mockData} filename="test" />);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass(
        'dark:border-gray-600',
        'dark:text-gray-300',
        'dark:bg-gray-800',
        'dark:hover:bg-gray-700'
      );
    });

    it('has transition classes for smooth interactions', () => {
      render(<DataExport data={mockData} filename="test" />);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass('transition-colors');
    });

    it('has focus styles for accessibility', () => {
      render(<DataExport data={mockData} filename="test" />);
      
      const button = screen.getByRole('button');
      expect(button).toHaveClass(
        'focus:outline-none',
        'focus:ring-2',
        'focus:ring-offset-2',
        'focus:ring-blue-500'
      );
    });
  });

  describe('Accessibility', () => {
    it('has accessible button with descriptive text', () => {
      render(<DataExport data={mockData} filename="test" type="csv" />);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAccessibleName('Export CSV');
    });

    it('provides title attribute for tooltip', () => {
      render(<DataExport data={mockData} filename="test" type="json" />);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('title', 'Export as JSON');
    });

    it('indicates disabled state appropriately', () => {
      render(<DataExport data={[]} filename="test" />);
      
      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('disabled');
      expect(button).toHaveClass('disabled:cursor-not-allowed');
    });

    it('is keyboard accessible', () => {
      render(<DataExport data={mockData} filename="test" />);
      
      const button = screen.getByRole('button');
      expect(button.tagName).toBe('BUTTON');
      expect(button).toBeEnabled();
    });
  });
});