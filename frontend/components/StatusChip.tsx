import React from 'react';
import {
  CheckCircle, XCircle, AlertCircle, Clock, Shield,
  Lock, AlertTriangle, Activity, Zap, Info
} from 'lucide-react';

export type StatusVariant = 
  | 'success' 
  | 'warning' 
  | 'danger' 
  | 'info' 
  | 'integrity-ok'
  | 'integrity-failed'
  | 'signature-valid'
  | 'merkle-proof'
  | 'pending'
  | 'active'
  | 'inactive';

interface StatusChipProps {
  variant: StatusVariant;
  label?: string;
  size?: 'xs' | 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  className?: string;
  onClick?: () => void;
}

const variantConfig = {
  success: {
    label: 'Success',
    icon: CheckCircle,
    classes: 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 border-green-200 dark:border-green-800',
  },
  warning: {
    label: 'Warning',
    icon: AlertTriangle,
    classes: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-200 border-yellow-200 dark:border-yellow-800',
  },
  danger: {
    label: 'Danger',
    icon: XCircle,
    classes: 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200 border-red-200 dark:border-red-800',
  },
  info: {
    label: 'Info',
    icon: Info,
    classes: 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 border-blue-200 dark:border-blue-800',
  },
  'integrity-ok': {
    label: 'Integrity OK',
    icon: Shield,
    classes: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-800 dark:text-emerald-200 border-emerald-200 dark:border-emerald-800',
  },
  'integrity-failed': {
    label: 'Integrity Failed',
    icon: Shield,
    classes: 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200 border-red-200 dark:border-red-800',
  },
  'signature-valid': {
    label: 'Signed',
    icon: Lock,
    classes: 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 border-blue-200 dark:border-blue-800',
  },
  'merkle-proof': {
    label: 'Merkle Proof',
    icon: Shield,
    classes: 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-200 border-purple-200 dark:border-purple-800',
  },
  pending: {
    label: 'Pending',
    icon: Clock,
    classes: 'bg-gray-100 dark:bg-gray-900/30 text-gray-800 dark:text-gray-200 border-gray-200 dark:border-gray-800',
  },
  active: {
    label: 'Active',
    icon: Activity,
    classes: 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200 border-green-200 dark:border-green-800 animate-pulse',
  },
  inactive: {
    label: 'Inactive',
    icon: XCircle,
    classes: 'bg-gray-100 dark:bg-gray-900/30 text-gray-800 dark:text-gray-200 border-gray-200 dark:border-gray-800',
  },
};

const sizeConfig = {
  xs: 'px-1.5 py-0.5 text-xs',
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-sm',
  lg: 'px-3 py-1.5 text-base',
};

const iconSizeConfig = {
  xs: 'w-3 h-3',
  sm: 'w-3 h-3',
  md: 'w-4 h-4',
  lg: 'w-5 h-5',
};

export default function StatusChip({
  variant,
  label,
  size = 'sm',
  showIcon = true,
  className = '',
  onClick,
}: StatusChipProps) {
  const config = variantConfig[variant];
  const Icon = config.icon;
  const displayLabel = label || config.label;

  const chipClasses = `
    inline-flex items-center gap-1 rounded-full font-medium border transition-all
    ${sizeConfig[size]}
    ${config.classes}
    ${onClick ? 'cursor-pointer hover:opacity-80' : ''}
    ${className}
  `.trim();

  return (
    <span 
      className={chipClasses}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      aria-label={displayLabel}
    >
      {showIcon && <Icon className={iconSizeConfig[size]} />}
      <span>{displayLabel}</span>
    </span>
  );
}

// Compound components for specific use cases
export function IntegrityChip({ 
  isValid, 
  ...props 
}: Omit<StatusChipProps, 'variant'> & { isValid: boolean }) {
  return (
    <StatusChip 
      variant={isValid ? 'integrity-ok' : 'integrity-failed'} 
      {...props} 
    />
  );
}

export function SignatureChip({ 
  isSigned, 
  ...props 
}: Omit<StatusChipProps, 'variant'> & { isSigned: boolean }) {
  return isSigned ? (
    <StatusChip variant="signature-valid" {...props} />
  ) : null;
}

export function MerkleProofChip({ 
  hasProof, 
  ...props 
}: Omit<StatusChipProps, 'variant'> & { hasProof: boolean }) {
  return hasProof ? (
    <StatusChip variant="merkle-proof" {...props} />
  ) : null;
}

export function ComplianceChip({ 
  impact,
  ...props 
}: Omit<StatusChipProps, 'variant' | 'label'> & { 
  impact: 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE' 
}) {
  const variantMap = {
    HIGH: 'danger' as const,
    MEDIUM: 'warning' as const,
    LOW: 'info' as const,
    NONE: 'success' as const,
  };
  
  return (
    <StatusChip 
      variant={variantMap[impact]} 
      label={impact}
      {...props} 
    />
  );
}

export function RiskChip({ 
  score,
  ...props 
}: Omit<StatusChipProps, 'variant' | 'label'> & { 
  score: number 
}) {
  const variant = score >= 70 ? 'danger' : score >= 40 ? 'warning' : 'success';
  const label = `${score}% Risk`;
  
  return (
    <StatusChip 
      variant={variant} 
      label={label}
      {...props} 
    />
  );
}