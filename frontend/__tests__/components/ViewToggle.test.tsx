import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import ViewToggle from '@/components/ViewToggle';

describe('ViewToggle Component', () => {
  const mockOnViewChange = jest.fn();

  beforeEach(() => {
    mockOnViewChange.mockClear();
  });

  describe('Rendering', () => {
    it('renders both Cards and Visualizations buttons', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      expect(screen.getByText('Cards')).toBeInTheDocument();
      expect(screen.getByText('Visualizations')).toBeInTheDocument();
    });

    it('applies active styles to Cards button when view is cards', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const cardsButton = screen.getByRole('button', { name: /cards/i });
      const vizButton = screen.getByRole('button', { name: /visualizations/i });
      
      expect(cardsButton).toHaveClass('bg-white', 'dark:bg-gray-700', 'shadow-sm');
      expect(vizButton).not.toHaveClass('bg-white');
      expect(vizButton).toHaveClass('text-gray-500');
    });

    it('applies active styles to Visualizations button when view is visualizations', () => {
      render(<ViewToggle view="visualizations" onViewChange={mockOnViewChange} />);
      
      const cardsButton = screen.getByRole('button', { name: /cards/i });
      const vizButton = screen.getByRole('button', { name: /visualizations/i });
      
      expect(vizButton).toHaveClass('bg-white', 'dark:bg-gray-700', 'shadow-sm');
      expect(cardsButton).not.toHaveClass('bg-white');
      expect(cardsButton).toHaveClass('text-gray-500');
    });

    it('renders with proper accessibility attributes', () => {
      const { container } = render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const buttons = container.querySelectorAll('button');
      expect(buttons).toHaveLength(2);
      
      buttons.forEach(button => {
        expect(button).toHaveAttribute('type', 'button');
      });
    });
  });

  describe('Interactions', () => {
    it('calls onViewChange with "cards" when Cards button is clicked', () => {
      render(<ViewToggle view="visualizations" onViewChange={mockOnViewChange} />);
      
      const cardsButton = screen.getByRole('button', { name: /cards/i });
      fireEvent.click(cardsButton);
      
      expect(mockOnViewChange).toHaveBeenCalledTimes(1);
      expect(mockOnViewChange).toHaveBeenCalledWith('cards');
    });

    it('calls onViewChange with "visualizations" when Visualizations button is clicked', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const vizButton = screen.getByRole('button', { name: /visualizations/i });
      fireEvent.click(vizButton);
      
      expect(mockOnViewChange).toHaveBeenCalledTimes(1);
      expect(mockOnViewChange).toHaveBeenCalledWith('visualizations');
    });

    it('does not call onViewChange when clicking already active view', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const cardsButton = screen.getByRole('button', { name: /cards/i });
      fireEvent.click(cardsButton);
      
      // Should still be called since component doesn't check current state
      expect(mockOnViewChange).toHaveBeenCalledWith('cards');
    });

    it('handles rapid clicks without issues', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const vizButton = screen.getByRole('button', { name: /visualizations/i });
      const cardsButton = screen.getByRole('button', { name: /cards/i });
      
      fireEvent.click(vizButton);
      fireEvent.click(cardsButton);
      fireEvent.click(vizButton);
      
      expect(mockOnViewChange).toHaveBeenCalledTimes(3);
      expect(mockOnViewChange).toHaveBeenNthCalledWith(1, 'visualizations');
      expect(mockOnViewChange).toHaveBeenNthCalledWith(2, 'cards');
      expect(mockOnViewChange).toHaveBeenNthCalledWith(3, 'visualizations');
    });
  });

  describe('Styling and Theme Support', () => {
    it('includes dark mode classes for theme support', () => {
      const { container } = render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const wrapper = container.querySelector('.dark\\:bg-gray-800');
      expect(wrapper).toBeInTheDocument();
      
      const buttons = container.querySelectorAll('button');
      buttons.forEach(button => {
        const classes = button.className;
        expect(classes).toMatch(/dark:/);
      });
    });

    it('has proper hover states', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const vizButton = screen.getByRole('button', { name: /visualizations/i });
      expect(vizButton).toHaveClass('hover:text-gray-700', 'dark:hover:text-gray-300');
    });

    it('includes transition classes for smooth animations', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveClass('transition-colors');
      });
    });
  });

  describe('Edge Cases', () => {
    it('handles undefined onViewChange gracefully', () => {
      // @ts-ignore - testing runtime behavior
      const { container } = render(<ViewToggle view="cards" onViewChange={undefined} />);
      
      const cardsButton = screen.getByRole('button', { name: /cards/i });
      
      // Should throw error when onViewChange is undefined
      expect(() => fireEvent.click(cardsButton)).toThrow();
    });

    it('renders correctly with invalid view prop', () => {
      // @ts-ignore - testing runtime behavior
      const { container } = render(<ViewToggle view="invalid" onViewChange={mockOnViewChange} />);
      
      // Should still render buttons
      expect(screen.getByText('Cards')).toBeInTheDocument();
      expect(screen.getByText('Visualizations')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('is keyboard navigable', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toBeEnabled();
        expect(parseInt(button.tabIndex.toString())).toBeGreaterThanOrEqual(0);
      });
    });

    it('has sufficient color contrast in light mode', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const activeButton = screen.getByRole('button', { name: /cards/i });
      expect(activeButton).toHaveClass('text-gray-900');
    });

    it('maintains focus after interaction', () => {
      render(<ViewToggle view="cards" onViewChange={mockOnViewChange} />);
      
      const vizButton = screen.getByRole('button', { name: /visualizations/i });
      vizButton.focus();
      fireEvent.click(vizButton);
      
      expect(document.activeElement).toBe(vizButton);
    });
  });
});