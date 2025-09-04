import { cn, formatBytes, formatCurrency, formatPercentage, formatDate } from '@/lib/utils';

describe('utils', () => {
  describe('cn (className utility)', () => {
    it('merges single class name', () => {
      const result = cn('text-red-500');
      expect(result).toBe('text-red-500');
    });

    it('merges multiple class names', () => {
      const result = cn('text-red-500', 'bg-blue-500');
      expect(result).toBe('text-red-500 bg-blue-500');
    });

    it('handles conditional classes with clsx syntax', () => {
      const result = cn(
        'base-class',
        {
          'active-class': true,
          'inactive-class': false
        }
      );
      expect(result).toBe('base-class active-class');
    });

    it('merges and deduplicates Tailwind classes', () => {
      const result = cn('p-4 text-red-500', 'p-2 text-blue-500');
      // tailwind-merge should keep the last conflicting utility
      expect(result).toBe('p-2 text-blue-500');
    });

    it('handles arrays of class names', () => {
      const result = cn(['text-red-500', 'bg-blue-500']);
      expect(result).toBe('text-red-500 bg-blue-500');
    });

    it('handles undefined and null values', () => {
      const result = cn('text-red-500', undefined, null, 'bg-blue-500');
      expect(result).toBe('text-red-500 bg-blue-500');
    });

    it('handles empty strings', () => {
      const result = cn('text-red-500', '', 'bg-blue-500');
      expect(result).toBe('text-red-500 bg-blue-500');
    });

    it('handles complex Tailwind utility conflicts', () => {
      const result = cn(
        'px-2 py-1 p-3',
        'mt-2 mx-4',
        'hover:bg-red-500 hover:bg-blue-500'
      );
      // Should keep p-3 (overrides px-2 py-1), and hover:bg-blue-500 (last one wins)
      expect(result).toContain('p-3');
      expect(result).toContain('hover:bg-blue-500');
      expect(result).not.toContain('hover:bg-red-500');
    });

    it('preserves non-conflicting classes', () => {
      const result = cn(
        'text-red-500 font-bold',
        'bg-blue-500 italic'
      );
      expect(result).toBe('text-red-500 font-bold bg-blue-500 italic');
    });

    it('handles responsive variants correctly', () => {
      const result = cn(
        'text-red-500 md:text-blue-500',
        'text-green-500 lg:text-purple-500'
      );
      // Order may vary but all classes should be present
      expect(result).toContain('text-green-500');
      expect(result).toContain('md:text-blue-500');
      expect(result).toContain('lg:text-purple-500');
      expect(result).not.toContain('text-red-500'); // Should be overridden
    });
  });

  describe('formatBytes', () => {
    it('formats 0 bytes correctly', () => {
      expect(formatBytes(0)).toBe('0 Bytes');
    });

    it('formats bytes (< 1KB)', () => {
      expect(formatBytes(500)).toBe('500 Bytes');
      expect(formatBytes(1023)).toBe('1023 Bytes');
    });

    it('formats kilobytes', () => {
      expect(formatBytes(1024)).toBe('1 KB');
      expect(formatBytes(1536)).toBe('1.5 KB');
      expect(formatBytes(2048)).toBe('2 KB');
    });

    it('formats megabytes', () => {
      expect(formatBytes(1048576)).toBe('1 MB');
      expect(formatBytes(1572864)).toBe('1.5 MB');
      expect(formatBytes(5242880)).toBe('5 MB');
    });

    it('formats gigabytes', () => {
      expect(formatBytes(1073741824)).toBe('1 GB');
      expect(formatBytes(2147483648)).toBe('2 GB');
    });

    it('formats terabytes', () => {
      expect(formatBytes(1099511627776)).toBe('1 TB');
      expect(formatBytes(2199023255552)).toBe('2 TB');
    });

    it('formats with custom decimal places', () => {
      expect(formatBytes(1536, 0)).toBe('2 KB');
      expect(formatBytes(1536, 1)).toBe('1.5 KB');
      expect(formatBytes(1536, 3)).toBe('1.5 KB');
      expect(formatBytes(1536.789 * 1024, 3)).toBe('1.501 MB');
    });

    it('handles negative decimal places', () => {
      expect(formatBytes(1536, -1)).toBe('2 KB');
    });

    it('formats very large numbers', () => {
      expect(formatBytes(1125899906842624)).toBe('1 PB'); // Petabyte
      expect(formatBytes(1152921504606846976)).toBe('1 EB'); // Exabyte
      expect(formatBytes(1180591620717411303424)).toBe('1 ZB'); // Zettabyte
      expect(formatBytes(1208925819614629174706176)).toBe('1 YB'); // Yottabyte
    });

    it('handles edge cases', () => {
      expect(formatBytes(1)).toBe('1 Bytes');
      expect(formatBytes(1023.999)).toBe('1024 Bytes');
      expect(formatBytes(1024.001)).toBe('1 KB');
    });
  });

  describe('formatCurrency', () => {
    it('formats USD currency by default', () => {
      expect(formatCurrency(100)).toBe('$100.00');
      expect(formatCurrency(1234.56)).toBe('$1,234.56');
    });

    it('formats large amounts with commas', () => {
      expect(formatCurrency(1000000)).toBe('$1,000,000.00');
      expect(formatCurrency(1234567.89)).toBe('$1,234,567.89');
    });

    it('formats negative amounts', () => {
      expect(formatCurrency(-100)).toBe('-$100.00');
      expect(formatCurrency(-1234.56)).toBe('-$1,234.56');
    });

    it('formats zero', () => {
      expect(formatCurrency(0)).toBe('$0.00');
    });

    it('formats with different currencies', () => {
      expect(formatCurrency(100, 'EUR')).toMatch(/100/);
      expect(formatCurrency(100, 'GBP')).toMatch(/100/);
      expect(formatCurrency(100, 'JPY')).toMatch(/100/);
    });

    it('rounds to 2 decimal places', () => {
      expect(formatCurrency(100.999)).toBe('$101.00');
      expect(formatCurrency(100.004)).toBe('$100.00');
      expect(formatCurrency(100.005)).toBe('$100.01');
    });

    it('handles very small amounts', () => {
      expect(formatCurrency(0.01)).toBe('$0.01');
      expect(formatCurrency(0.001)).toBe('$0.00');
    });

    it('handles decimal precision correctly', () => {
      expect(formatCurrency(99.99)).toBe('$99.99');
      expect(formatCurrency(99.999)).toBe('$100.00');
      expect(formatCurrency(0.50)).toBe('$0.50');
    });
  });

  describe('formatPercentage', () => {
    it('formats basic percentages', () => {
      expect(formatPercentage(50)).toBe('50.0%');
      expect(formatPercentage(100)).toBe('100.0%');
      expect(formatPercentage(0)).toBe('0.0%');
    });

    it('formats with default 1 decimal place', () => {
      expect(formatPercentage(50.5)).toBe('50.5%');
      expect(formatPercentage(33.33333)).toBe('33.3%');
    });

    it('formats with custom decimal places', () => {
      expect(formatPercentage(50.5678, 0)).toBe('51%');
      expect(formatPercentage(50.5678, 2)).toBe('50.57%');
      expect(formatPercentage(50.5678, 3)).toBe('50.568%');
      expect(formatPercentage(50.5678, 4)).toBe('50.5678%');
    });

    it('handles edge cases', () => {
      expect(formatPercentage(0.1)).toBe('0.1%');
      expect(formatPercentage(99.99)).toBe('100.0%');
      expect(formatPercentage(99.94)).toBe('99.9%');
      expect(formatPercentage(99.95)).toBe('100.0%');
    });

    it('handles negative percentages', () => {
      expect(formatPercentage(-50)).toBe('-50.0%');
      expect(formatPercentage(-25.5)).toBe('-25.5%');
    });

    it('handles very large percentages', () => {
      expect(formatPercentage(200)).toBe('200.0%');
      expect(formatPercentage(1000)).toBe('1000.0%');
      expect(formatPercentage(9999.99)).toBe('10000.0%');
    });

    it('rounds correctly', () => {
      expect(formatPercentage(33.333, 2)).toBe('33.33%');
      expect(formatPercentage(33.335, 2)).toBe('33.34%');
      expect(formatPercentage(33.334, 2)).toBe('33.33%');
    });
  });

  describe('formatDate', () => {
    // Use fixed dates to avoid timezone issues in tests
    const testDate = new Date('2024-01-15T14:30:00');
    const testDateString = '2024-01-15T14:30:00';

    it('formats Date objects', () => {
      const result = formatDate(testDate);
      expect(result).toContain('Jan');
      expect(result).toContain('15');
      expect(result).toContain('2024');
    });

    it('formats date strings', () => {
      const result = formatDate(testDateString);
      expect(result).toContain('Jan');
      expect(result).toContain('15');
      expect(result).toContain('2024');
    });

    it('formats ISO date strings', () => {
      const isoString = '2024-01-15T14:30:00.000Z';
      const result = formatDate(isoString);
      expect(result).toContain('Jan');
      expect(result).toContain('15');
      expect(result).toContain('2024');
    });

    it('includes time in format', () => {
      const result = formatDate('2024-01-15T14:30:00');
      expect(result).toMatch(/\d{1,2}:\d{2}/); // Matches time format like 2:30
    });

    it('handles different months correctly', () => {
      expect(formatDate('2024-01-01T00:00:00')).toContain('Jan');
      expect(formatDate('2024-02-01T00:00:00')).toContain('Feb');
      expect(formatDate('2024-03-01T00:00:00')).toContain('Mar');
      expect(formatDate('2024-04-01T00:00:00')).toContain('Apr');
      expect(formatDate('2024-05-01T00:00:00')).toContain('May');
      expect(formatDate('2024-06-01T00:00:00')).toContain('Jun');
      expect(formatDate('2024-07-01T00:00:00')).toContain('Jul');
      expect(formatDate('2024-08-01T00:00:00')).toContain('Aug');
      expect(formatDate('2024-09-01T00:00:00')).toContain('Sep');
      expect(formatDate('2024-10-01T00:00:00')).toContain('Oct');
      expect(formatDate('2024-11-01T00:00:00')).toContain('Nov');
      expect(formatDate('2024-12-01T00:00:00')).toContain('Dec');
    });

    it('handles different years', () => {
      expect(formatDate('2020-01-01T00:00:00')).toContain('2020');
      expect(formatDate('2024-01-01T00:00:00')).toContain('2024');
      expect(formatDate('2030-01-01T00:00:00')).toContain('2030');
    });

    it('handles edge dates', () => {
      // Beginning of year
      const newYear = formatDate('2024-01-01T00:00:00');
      expect(newYear).toContain('Jan');
      expect(newYear).toContain('1');
      expect(newYear).toContain('2024');

      // End of year
      const endYear = formatDate('2024-12-31T23:59:59');
      expect(endYear).toContain('Dec');
      expect(endYear).toContain('31');
      expect(endYear).toContain('2024');
    });

    it('handles invalid dates gracefully', () => {
      // Invalid date strings will throw, so we handle that expectation
      const result = formatDate('invalid-date');
      expect(result).toContain('Invalid');
    });

    it('formats timestamps', () => {
      const timestamp = new Date('2024-01-15T14:30:00').getTime();
      const result = formatDate(new Date(timestamp));
      expect(result).toContain('Jan');
      expect(result).toContain('15');
      expect(result).toContain('2024');
    });

    it('maintains consistent format', () => {
      const date1 = formatDate('2024-01-01T09:05:00');
      const date2 = formatDate('2024-12-31T23:59:00');
      
      // Both should have similar format structure
      expect(date1).toMatch(/\w{3} \d{1,2}, \d{4}/); // e.g., "Jan 1, 2024"
      expect(date2).toMatch(/\w{3} \d{1,2}, \d{4}/); // e.g., "Dec 31, 2024"
    });
  });

  describe('Integration tests', () => {
    it('all format functions handle various inputs', () => {
      // Test with zero values
      expect(formatBytes(0)).toBe('0 Bytes');
      expect(formatCurrency(0)).toBe('$0.00');
      expect(formatPercentage(0)).toBe('0.0%');
      
      // Test with valid inputs
      expect(formatBytes(1024)).toBe('1 KB');
      expect(formatCurrency(100)).toBe('$100.00');
      expect(formatPercentage(50)).toBe('50.0%');
      expect(formatDate('2024-01-01T00:00:00')).toContain('2024');
    });

    it('format functions return consistent types', () => {
      expect(typeof formatBytes(1024)).toBe('string');
      expect(typeof formatCurrency(100)).toBe('string');
      expect(typeof formatPercentage(50)).toBe('string');
      expect(typeof formatDate(new Date())).toBe('string');
    });

    it('cn utility works with all Tailwind class types', () => {
      const complexClasses = cn(
        // Colors
        'text-red-500 bg-blue-500',
        // Spacing
        'p-4 m-2',
        // Typography
        'font-bold text-lg',
        // Layout
        'flex items-center justify-between',
        // Responsive
        'sm:p-2 md:p-4 lg:p-6',
        // States
        'hover:bg-gray-100 focus:ring-2',
        // Dark mode
        'dark:text-white dark:bg-black'
      );

      expect(complexClasses).toBeTruthy();
      expect(typeof complexClasses).toBe('string');
    });
  });
});