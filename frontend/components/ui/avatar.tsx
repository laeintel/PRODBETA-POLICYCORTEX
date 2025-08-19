'use client';

import React from 'react';

interface AvatarProps {
  src?: string;
  alt?: string;
  className?: string;
  size?: number | string;
  fallback?: string;
}

export const Avatar: React.FC<AvatarProps> = ({
  src,
  alt = 'User',
  className = '',
  size = 40,
  fallback,
}) => {
  const dimension = typeof size === 'number' ? `${size}px` : size;
  const initials = fallback || (alt?.trim().split(/\s+/).map(p => p[0]).join('').slice(0, 2).toUpperCase() || 'U');

  return (
    <div
      className={`inline-flex items-center justify-center rounded-full bg-gray-200 text-gray-700 overflow-hidden ${className}`}
      style={{ width: dimension, height: dimension }}
      aria-label={alt}
    >
      {src ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src} alt={alt} className="w-full h-full object-cover" />
      ) : (
        <span className="text-xs font-medium select-none">{initials}</span>
      )}
    </div>
  );
};

export const AvatarImage: React.FC<React.ImgHTMLAttributes<HTMLImageElement>> = (props) => (
  // eslint-disable-next-line @next/next/no-img-element
  <img {...props} />
);

export const AvatarFallback: React.FC<{ children?: React.ReactNode; className?: string }> = ({ children, className = '' }) => (
  <div className={`inline-flex items-center justify-center rounded-full bg-gray-200 text-gray-700 ${className}`}>
    {children}
  </div>
);


