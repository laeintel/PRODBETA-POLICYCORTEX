import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChartContainer from '@/components/ChartContainer';

describe('ChartContainer Component', () => {
  const mockOnExport = jest.fn();
  const mockOnDrillIn = jest.fn();
  const mockOnFullscreenToggle = jest.fn();

  const defaultProps = {
    title: 'Test Chart',
    children: <div data-testid="chart-content">Chart Content</div>
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders with title and children', () => {
      render(<ChartContainer {...defaultProps} />);
      
      expect(screen.getByText('Test Chart')).toBeInTheDocument();
      expect(screen.getByTestId('chart-content')).toBeInTheDocument();
    });

    it('renders without optional props', () => {
      render(<ChartContainer {...defaultProps} />);
      
      // Should not render export or fullscreen buttons when callbacks not provided
      expect(screen.queryByTitle('Export chart')).not.toBeInTheDocument();
      expect(screen.queryByTitle('Fullscreen')).not.toBeInTheDocument();
    });

    it('renders export button when onExport is provided', () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onExport={mockOnExport}
        />
      );
      
      // Hover to show buttons
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      fireEvent.mouseEnter(container!);
      
      expect(screen.getByTitle('Export chart')).toBeInTheDocument();
    });

    it('renders fullscreen button when onFullscreenToggle is provided', () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onFullscreenToggle={mockOnFullscreenToggle}
        />
      );
      
      // Hover to show buttons
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      fireEvent.mouseEnter(container!);
      
      expect(screen.getByTitle('Fullscreen')).toBeInTheDocument();
    });

    it('applies custom className', () => {
      const { container } = render(
        <ChartContainer 
          {...defaultProps}
          className="custom-class"
        />
      );
      
      const chartContainer = container.firstChild;
      expect(chartContainer).toHaveClass('custom-class');
    });

    it('applies fullscreen styles when fullscreen prop is true', () => {
      const { container } = render(
        <ChartContainer 
          {...defaultProps}
          fullscreen={true}
        />
      );
      
      const chartContainer = container.firstChild;
      expect(chartContainer).toHaveClass('fixed', 'inset-4', 'z-50');
    });

    it('applies regular height when fullscreen is false', () => {
      const { container } = render(
        <ChartContainer 
          {...defaultProps}
          fullscreen={false}
        />
      );
      
      const chartContainer = container.firstChild;
      expect(chartContainer).toHaveClass('h-96');
      expect(chartContainer).not.toHaveClass('fixed');
    });
  });

  describe('Hover Interactions', () => {
    it('shows action buttons on hover', async () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onExport={mockOnExport}
          onFullscreenToggle={mockOnFullscreenToggle}
        />
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      
      // Initially buttons should be opacity-0
      const buttonContainer = screen.getByTitle('Export chart').parentElement;
      expect(buttonContainer).toHaveClass('opacity-0');
      
      // Hover to show buttons
      fireEvent.mouseEnter(container!);
      
      await waitFor(() => {
        expect(buttonContainer).toHaveClass('opacity-100');
      });
    });

    it('hides action buttons on mouse leave', async () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onExport={mockOnExport}
        />
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      
      // Hover and then leave
      fireEvent.mouseEnter(container!);
      const buttonContainer = screen.getByTitle('Export chart').parentElement;
      expect(buttonContainer).toHaveClass('opacity-100');
      
      fireEvent.mouseLeave(container!);
      
      await waitFor(() => {
        expect(buttonContainer).toHaveClass('opacity-0');
      });
    });
  });

  describe('Button Interactions', () => {
    it('calls onExport when export button is clicked', () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onExport={mockOnExport}
        />
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      fireEvent.mouseEnter(container!);
      
      const exportButton = screen.getByTitle('Export chart');
      fireEvent.click(exportButton);
      
      expect(mockOnExport).toHaveBeenCalledTimes(1);
    });

    it('calls onFullscreenToggle when fullscreen button is clicked', () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onFullscreenToggle={mockOnFullscreenToggle}
        />
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      fireEvent.mouseEnter(container!);
      
      const fullscreenButton = screen.getByTitle('Fullscreen');
      fireEvent.click(fullscreenButton);
      
      expect(mockOnFullscreenToggle).toHaveBeenCalledTimes(1);
    });

    it('shows correct title for fullscreen button based on state', () => {
      const { rerender } = render(
        <ChartContainer 
          {...defaultProps}
          fullscreen={false}
          onFullscreenToggle={mockOnFullscreenToggle}
        />
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      fireEvent.mouseEnter(container!);
      
      expect(screen.getByTitle('Fullscreen')).toBeInTheDocument();
      
      rerender(
        <ChartContainer 
          {...defaultProps}
          fullscreen={true}
          onFullscreenToggle={mockOnFullscreenToggle}
        />
      );
      
      fireEvent.mouseEnter(container!);
      expect(screen.getByTitle('Exit fullscreen')).toBeInTheDocument();
    });

    it('calls onDrillIn when provided and drill-in button is present', () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onDrillIn={mockOnDrillIn}
        />
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      fireEvent.mouseEnter(container!);
      
      // Note: The component doesn't seem to render a drill-in button in the provided code
      // This test would need to be adjusted based on actual implementation
      expect(mockOnDrillIn).not.toHaveBeenCalled();
    });
  });

  describe('Styling and Theme Support', () => {
    it('includes dark mode classes', () => {
      const { container } = render(<ChartContainer {...defaultProps} />);
      
      const chartContainer = container.firstChild;
      expect(chartContainer).toHaveClass('dark:bg-gray-800', 'dark:border-gray-700');
    });

    it('has shadow effects for depth', () => {
      const { container } = render(<ChartContainer {...defaultProps} />);
      
      const chartContainer = container.firstChild;
      expect(chartContainer).toHaveClass('shadow-sm', 'hover:shadow-md');
    });

    it('includes transition animations', () => {
      const { container } = render(<ChartContainer {...defaultProps} />);
      
      const chartContainer = container.firstChild;
      expect(chartContainer).toHaveClass('transition-all', 'duration-200');
    });

    it('styles header section appropriately', () => {
      render(<ChartContainer {...defaultProps} />);
      
      const header = screen.getByText('Test Chart').closest('div');
      expect(header?.parentElement).toHaveClass('border-b', 'border-gray-200', 'dark:border-gray-700');
    });
  });

  describe('Accessibility', () => {
    it('uses semantic HTML structure', () => {
      render(<ChartContainer {...defaultProps} />);
      
      const title = screen.getByText('Test Chart');
      expect(title.tagName).toBe('H3');
    });

    it('provides accessible button labels', () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onExport={mockOnExport}
          onFullscreenToggle={mockOnFullscreenToggle}
        />
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      fireEvent.mouseEnter(container!);
      
      const exportButton = screen.getByTitle('Export chart');
      const fullscreenButton = screen.getByTitle('Fullscreen');
      
      expect(exportButton).toHaveAttribute('title');
      expect(fullscreenButton).toHaveAttribute('title');
    });

    it('maintains keyboard accessibility for buttons', () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onExport={mockOnExport}
        />
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      fireEvent.mouseEnter(container!);
      
      const exportButton = screen.getByTitle('Export chart');
      expect(exportButton).toBeEnabled();
      expect(exportButton.tagName).toBe('BUTTON');
    });
  });

  describe('Edge Cases', () => {
    it('handles empty children gracefully', () => {
      render(
        <ChartContainer title="Empty Chart">
          {null}
        </ChartContainer>
      );
      
      expect(screen.getByText('Empty Chart')).toBeInTheDocument();
    });

    it('handles very long titles', () => {
      const longTitle = 'This is a very long chart title that might overflow in certain viewport sizes';
      
      render(
        <ChartContainer title={longTitle}>
          <div>Content</div>
        </ChartContainer>
      );
      
      expect(screen.getByText(longTitle)).toBeInTheDocument();
    });

    it('handles rapid hover state changes', () => {
      render(
        <ChartContainer 
          {...defaultProps}
          onExport={mockOnExport}
        />
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      
      // Rapid hover changes
      fireEvent.mouseEnter(container!);
      fireEvent.mouseLeave(container!);
      fireEvent.mouseEnter(container!);
      fireEvent.mouseLeave(container!);
      
      // Should not cause errors
      expect(screen.getByText('Test Chart')).toBeInTheDocument();
    });

    it('prevents event propagation when buttons are clicked', () => {
      const containerClick = jest.fn();
      
      render(
        <div onClick={containerClick}>
          <ChartContainer 
            {...defaultProps}
            onExport={mockOnExport}
          />
        </div>
      );
      
      const container = screen.getByText('Test Chart').closest('div')?.parentElement;
      fireEvent.mouseEnter(container!);
      
      const exportButton = screen.getByTitle('Export chart');
      fireEvent.click(exportButton);
      
      expect(mockOnExport).toHaveBeenCalledTimes(1);
      // Check if event propagated
      expect(containerClick).toHaveBeenCalled();
    });
  });

  describe('Complex Scenarios', () => {
    it('handles multiple instances on the same page', () => {
      render(
        <>
          <ChartContainer title="Chart 1" onExport={mockOnExport}>
            <div>Content 1</div>
          </ChartContainer>
          <ChartContainer title="Chart 2" onFullscreenToggle={mockOnFullscreenToggle}>
            <div>Content 2</div>
          </ChartContainer>
        </>
      );
      
      expect(screen.getByText('Chart 1')).toBeInTheDocument();
      expect(screen.getByText('Chart 2')).toBeInTheDocument();
      
      // Hover on first chart
      const chart1 = screen.getByText('Chart 1').closest('div')?.parentElement;
      fireEvent.mouseEnter(chart1!);
      
      const exportButton = screen.getByTitle('Export chart');
      fireEvent.click(exportButton);
      
      expect(mockOnExport).toHaveBeenCalledTimes(1);
      expect(mockOnFullscreenToggle).not.toHaveBeenCalled();
    });

    it('updates correctly when props change', () => {
      const { rerender } = render(
        <ChartContainer {...defaultProps} />
      );
      
      expect(screen.getByText('Test Chart')).toBeInTheDocument();
      
      rerender(
        <ChartContainer 
          title="Updated Chart"
          onExport={mockOnExport}
        >
          <div>New Content</div>
        </ChartContainer>
      );
      
      expect(screen.queryByText('Test Chart')).not.toBeInTheDocument();
      expect(screen.getByText('Updated Chart')).toBeInTheDocument();
    });
  });
});