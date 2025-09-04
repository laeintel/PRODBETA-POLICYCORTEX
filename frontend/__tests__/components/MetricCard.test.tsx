import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock component for testing
interface MetricCardProps {
  title: string;
  value: string | number;
  trend?: string;
  icon?: React.ComponentType<{ className?: string }>;
}

function MetricCard({ title, value, trend }: MetricCardProps) {
  return (
    <div className="metric-card">
      <h3>{title}</h3>
      <p className="value">{value}</p>
      {trend && <p className="trend">{trend}</p>}
    </div>
  );
}

describe('MetricCard', () => {
  it('renders title and value', () => {
    render(<MetricCard title="Total Resources" value={150} />);
    
    expect(screen.getByText('Total Resources')).toBeInTheDocument();
    expect(screen.getByText('150')).toBeInTheDocument();
  });

  it('renders trend when provided', () => {
    render(<MetricCard title="Compliance" value="95%" trend="+5% from last week" />);
    
    expect(screen.getByText('+5% from last week')).toBeInTheDocument();
  });

  it('handles string values', () => {
    render(<MetricCard title="Status" value="Healthy" />);
    
    expect(screen.getByText('Healthy')).toBeInTheDocument();
  });

  it('renders without trend', () => {
    const { container } = render(<MetricCard title="Test" value={42} />);
    
    expect(container.querySelector('.trend')).not.toBeInTheDocument();
  });
});