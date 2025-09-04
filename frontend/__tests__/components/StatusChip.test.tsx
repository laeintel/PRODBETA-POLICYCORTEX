import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import StatusChip, { 
  IntegrityChip, 
  SignatureChip, 
  MerkleProofChip, 
  ComplianceChip, 
  RiskChip 
} from '@/components/StatusChip';

describe('StatusChip Component', () => {
  const mockOnClick = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('renders with success variant', () => {
      render(<StatusChip variant="success" />);
      
      const chip = screen.getByText('Success');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-green-100', 'text-green-800');
    });

    it('renders with warning variant', () => {
      render(<StatusChip variant="warning" />);
      
      const chip = screen.getByText('Warning');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-yellow-100', 'text-yellow-800');
    });

    it('renders with danger variant', () => {
      render(<StatusChip variant="danger" />);
      
      const chip = screen.getByText('Danger');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-red-100', 'text-red-800');
    });

    it('renders with info variant', () => {
      render(<StatusChip variant="info" />);
      
      const chip = screen.getByText('Info');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-blue-100', 'text-blue-800');
    });

    it('renders with custom label', () => {
      render(<StatusChip variant="success" label="Custom Label" />);
      
      expect(screen.getByText('Custom Label')).toBeInTheDocument();
      expect(screen.queryByText('Success')).not.toBeInTheDocument();
    });

    it('renders with custom className', () => {
      render(<StatusChip variant="info" className="custom-class" />);
      
      const chip = screen.getByText('Info').parentElement;
      expect(chip).toHaveClass('custom-class');
    });
  });

  describe('Integrity Variants', () => {
    it('renders integrity-ok variant', () => {
      render(<StatusChip variant="integrity-ok" />);
      
      const chip = screen.getByText('Integrity OK');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-emerald-100', 'text-emerald-800');
    });

    it('renders integrity-failed variant', () => {
      render(<StatusChip variant="integrity-failed" />);
      
      const chip = screen.getByText('Integrity Failed');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-red-100', 'text-red-800');
    });

    it('renders signature-valid variant', () => {
      render(<StatusChip variant="signature-valid" />);
      
      const chip = screen.getByText('Signed');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-blue-100', 'text-blue-800');
    });

    it('renders merkle-proof variant', () => {
      render(<StatusChip variant="merkle-proof" />);
      
      const chip = screen.getByText('Merkle Proof');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-purple-100', 'text-purple-800');
    });
  });

  describe('Activity Variants', () => {
    it('renders pending variant', () => {
      render(<StatusChip variant="pending" />);
      
      const chip = screen.getByText('Pending');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-gray-100', 'text-gray-800');
    });

    it('renders active variant with animation', () => {
      render(<StatusChip variant="active" />);
      
      const chip = screen.getByText('Active');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-green-100', 'animate-pulse');
    });

    it('renders inactive variant', () => {
      render(<StatusChip variant="inactive" />);
      
      const chip = screen.getByText('Inactive');
      expect(chip).toBeInTheDocument();
      expect(chip.parentElement).toHaveClass('bg-gray-100', 'text-gray-800');
    });
  });

  describe('Size Variants', () => {
    it('renders with xs size', () => {
      render(<StatusChip variant="success" size="xs" />);
      
      const chip = screen.getByText('Success').parentElement;
      expect(chip).toHaveClass('px-1.5', 'py-0.5', 'text-xs');
    });

    it('renders with sm size (default)', () => {
      render(<StatusChip variant="success" size="sm" />);
      
      const chip = screen.getByText('Success').parentElement;
      expect(chip).toHaveClass('px-2', 'py-0.5', 'text-xs');
    });

    it('renders with md size', () => {
      render(<StatusChip variant="success" size="md" />);
      
      const chip = screen.getByText('Success').parentElement;
      expect(chip).toHaveClass('px-2.5', 'py-1', 'text-sm');
    });

    it('renders with lg size', () => {
      render(<StatusChip variant="success" size="lg" />);
      
      const chip = screen.getByText('Success').parentElement;
      expect(chip).toHaveClass('px-3', 'py-1.5', 'text-base');
    });
  });

  describe('Icon Display', () => {
    it('shows icon by default', () => {
      const { container } = render(<StatusChip variant="success" />);
      
      const icon = container.querySelector('svg');
      expect(icon).toBeInTheDocument();
    });

    it('hides icon when showIcon is false', () => {
      const { container } = render(<StatusChip variant="success" showIcon={false} />);
      
      const icon = container.querySelector('svg');
      expect(icon).not.toBeInTheDocument();
    });

    it('renders correct icon size for each size variant', () => {
      const { container: containerXs } = render(<StatusChip variant="success" size="xs" />);
      const { container: containerLg } = render(<StatusChip variant="success" size="lg" />);
      
      const iconXs = containerXs.querySelector('svg');
      const iconLg = containerLg.querySelector('svg');
      
      expect(iconXs).toHaveClass('w-3', 'h-3');
      expect(iconLg).toHaveClass('w-5', 'h-5');
    });
  });

  describe('Click Interactions', () => {
    it('handles click when onClick is provided', () => {
      render(<StatusChip variant="success" onClick={mockOnClick} />);
      
      const chip = screen.getByRole('button');
      fireEvent.click(chip);
      
      expect(mockOnClick).toHaveBeenCalledTimes(1);
    });

    it('does not have button role when onClick is not provided', () => {
      render(<StatusChip variant="success" />);
      
      const chip = screen.getByText('Success').parentElement;
      expect(chip).not.toHaveAttribute('role', 'button');
      expect(chip).not.toHaveAttribute('tabIndex');
    });

    it('has cursor pointer style when clickable', () => {
      render(<StatusChip variant="success" onClick={mockOnClick} />);
      
      const chip = screen.getByRole('button');
      expect(chip).toHaveClass('cursor-pointer', 'hover:opacity-80');
    });

    it('handles Enter key press', () => {
      render(<StatusChip variant="success" onClick={mockOnClick} />);
      
      const chip = screen.getByRole('button');
      fireEvent.keyDown(chip, { key: 'Enter' });
      
      expect(mockOnClick).toHaveBeenCalledTimes(1);
    });

    it('handles Space key press', () => {
      render(<StatusChip variant="success" onClick={mockOnClick} />);
      
      const chip = screen.getByRole('button');
      fireEvent.keyDown(chip, { key: ' ' });
      
      expect(mockOnClick).toHaveBeenCalledTimes(1);
    });

    it('does not trigger onClick for other keys', () => {
      render(<StatusChip variant="success" onClick={mockOnClick} />);
      
      const chip = screen.getByRole('button');
      fireEvent.keyDown(chip, { key: 'a' });
      
      expect(mockOnClick).not.toHaveBeenCalled();
    });
  });

  describe('Compound Components', () => {
    describe('IntegrityChip', () => {
      it('renders as integrity-ok when isValid is true', () => {
        render(<IntegrityChip isValid={true} />);
        
        expect(screen.getByText('Integrity OK')).toBeInTheDocument();
      });

      it('renders as integrity-failed when isValid is false', () => {
        render(<IntegrityChip isValid={false} />);
        
        expect(screen.getByText('Integrity Failed')).toBeInTheDocument();
      });

      it('passes through other props', () => {
        render(<IntegrityChip isValid={true} size="lg" onClick={mockOnClick} />);
        
        const chip = screen.getByRole('button');
        expect(chip).toHaveClass('px-3', 'py-1.5');
        
        fireEvent.click(chip);
        expect(mockOnClick).toHaveBeenCalled();
      });
    });

    describe('SignatureChip', () => {
      it('renders when isSigned is true', () => {
        render(<SignatureChip isSigned={true} />);
        
        expect(screen.getByText('Signed')).toBeInTheDocument();
      });

      it('renders nothing when isSigned is false', () => {
        const { container } = render(<SignatureChip isSigned={false} />);
        
        expect(container.firstChild).toBeNull();
      });
    });

    describe('MerkleProofChip', () => {
      it('renders when hasProof is true', () => {
        render(<MerkleProofChip hasProof={true} />);
        
        expect(screen.getByText('Merkle Proof')).toBeInTheDocument();
      });

      it('renders nothing when hasProof is false', () => {
        const { container } = render(<MerkleProofChip hasProof={false} />);
        
        expect(container.firstChild).toBeNull();
      });
    });

    describe('ComplianceChip', () => {
      it('renders HIGH impact as danger', () => {
        render(<ComplianceChip impact="HIGH" />);
        
        const chip = screen.getByText('HIGH');
        expect(chip).toBeInTheDocument();
        expect(chip.parentElement).toHaveClass('bg-red-100');
      });

      it('renders MEDIUM impact as warning', () => {
        render(<ComplianceChip impact="MEDIUM" />);
        
        const chip = screen.getByText('MEDIUM');
        expect(chip).toBeInTheDocument();
        expect(chip.parentElement).toHaveClass('bg-yellow-100');
      });

      it('renders LOW impact as info', () => {
        render(<ComplianceChip impact="LOW" />);
        
        const chip = screen.getByText('LOW');
        expect(chip).toBeInTheDocument();
        expect(chip.parentElement).toHaveClass('bg-blue-100');
      });

      it('renders NONE impact as success', () => {
        render(<ComplianceChip impact="NONE" />);
        
        const chip = screen.getByText('NONE');
        expect(chip).toBeInTheDocument();
        expect(chip.parentElement).toHaveClass('bg-green-100');
      });
    });

    describe('RiskChip', () => {
      it('renders high risk (≥70) as danger', () => {
        render(<RiskChip score={85} />);
        
        const chip = screen.getByText('85% Risk');
        expect(chip).toBeInTheDocument();
        expect(chip.parentElement).toHaveClass('bg-red-100');
      });

      it('renders medium risk (40-69) as warning', () => {
        render(<RiskChip score={55} />);
        
        const chip = screen.getByText('55% Risk');
        expect(chip).toBeInTheDocument();
        expect(chip.parentElement).toHaveClass('bg-yellow-100');
      });

      it('renders low risk (<40) as success', () => {
        render(<RiskChip score={25} />);
        
        const chip = screen.getByText('25% Risk');
        expect(chip).toBeInTheDocument();
        expect(chip.parentElement).toHaveClass('bg-green-100');
      });

      it('handles edge cases for risk boundaries', () => {
        const { rerender } = render(<RiskChip score={70} />);
        expect(screen.getByText('70% Risk').parentElement).toHaveClass('bg-red-100');
        
        rerender(<RiskChip score={40} />);
        expect(screen.getByText('40% Risk').parentElement).toHaveClass('bg-yellow-100');
        
        rerender(<RiskChip score={39} />);
        expect(screen.getByText('39% Risk').parentElement).toHaveClass('bg-green-100');
      });
    });
  });

  describe('Accessibility', () => {
    it('has aria-label with display label', () => {
      render(<StatusChip variant="success" />);
      
      const chip = screen.getByText('Success').parentElement;
      expect(chip).toHaveAttribute('aria-label', 'Success');
    });

    it('has aria-label with custom label', () => {
      render(<StatusChip variant="success" label="Custom Status" />);
      
      const chip = screen.getByText('Custom Status').parentElement;
      expect(chip).toHaveAttribute('aria-label', 'Custom Status');
    });

    it('is keyboard accessible when clickable', () => {
      render(<StatusChip variant="success" onClick={mockOnClick} />);
      
      const chip = screen.getByRole('button');
      expect(chip).toHaveAttribute('tabIndex', '0');
    });

    it('is not keyboard focusable when not clickable', () => {
      render(<StatusChip variant="success" />);
      
      const chip = screen.getByText('Success').parentElement;
      expect(chip).not.toHaveAttribute('tabIndex');
    });

    it('prevents default action on key press', () => {
      render(<StatusChip variant="success" onClick={mockOnClick} />);
      
      const chip = screen.getByRole('button');
      const event = new KeyboardEvent('keydown', { key: 'Enter' });
      const preventDefaultSpy = jest.spyOn(event, 'preventDefault');
      
      chip.dispatchEvent(event);
      
      expect(preventDefaultSpy).toHaveBeenCalled();
    });
  });

  describe('Dark Mode Support', () => {
    it('includes dark mode classes for all variants', () => {
      const variants = ['success', 'warning', 'danger', 'info'] as const;
      
      variants.forEach(variant => {
        const { container } = render(<StatusChip variant={variant} />);
        const chip = container.firstChild as HTMLElement;
        
        const classes = chip.className;
        expect(classes).toMatch(/dark:/);
      });
    });

    it('has dark mode classes for success variant', () => {
      render(<StatusChip variant="success" />);
      
      const chip = screen.getByText('Success').parentElement;
      expect(chip).toHaveClass('dark:bg-green-900/30', 'dark:text-green-200', 'dark:border-green-800');
    });

    it('has dark mode classes for danger variant', () => {
      render(<StatusChip variant="danger" />);
      
      const chip = screen.getByText('Danger').parentElement;
      expect(chip).toHaveClass('dark:bg-red-900/30', 'dark:text-red-200', 'dark:border-red-800');
    });
  });

  describe('Edge Cases', () => {
    it('handles very long custom labels', () => {
      const longLabel = 'This is a very long label that might cause layout issues';
      render(<StatusChip variant="success" label={longLabel} />);
      
      expect(screen.getByText(longLabel)).toBeInTheDocument();
    });

    it('handles empty string label', () => {
      render(<StatusChip variant="success" label="" />);
      
      // Should fall back to default label
      expect(screen.getByText('Success')).toBeInTheDocument();
    });

    it('handles rapid clicks without issues', () => {
      render(<StatusChip variant="success" onClick={mockOnClick} />);
      
      const chip = screen.getByRole('button');
      
      // Simulate rapid clicks
      for (let i = 0; i < 10; i++) {
        fireEvent.click(chip);
      }
      
      expect(mockOnClick).toHaveBeenCalledTimes(10);
    });

    it('renders multiple chips independently', () => {
      render(
        <>
          <StatusChip variant="success" />
          <StatusChip variant="danger" />
          <StatusChip variant="info" />
        </>
      );
      
      expect(screen.getByText('Success')).toBeInTheDocument();
      expect(screen.getByText('Danger')).toBeInTheDocument();
      expect(screen.getByText('Info')).toBeInTheDocument();
    });
  });
});