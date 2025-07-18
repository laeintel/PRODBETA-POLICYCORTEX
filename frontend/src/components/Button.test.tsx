import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';

// Simple component test without external dependencies
const SimpleButton = ({ children }: { children: React.ReactNode }) => (
  <button data-testid="simple-button">{children}</button>
);

describe('SimpleButton', () => {
  it('renders button with text', () => {
    render(<SimpleButton>Click me</SimpleButton>);
    expect(screen.getByTestId('simple-button')).toBeInTheDocument();
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  it('renders with children', () => {
    render(<SimpleButton><span>Child content</span></SimpleButton>);
    expect(screen.getByText('Child content')).toBeInTheDocument();
  });
});