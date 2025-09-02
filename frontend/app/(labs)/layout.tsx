import { Metadata } from 'next';

export const metadata: Metadata = {
  robots: { index: false, follow: true }, // Keep Labs pages out of search
  other: { 'x-section': 'labs' }
};

export default function LabsLayout({ 
  children 
}: { 
  children: React.ReactNode 
}) {
  return (
    <div data-section="labs">
      {/* Labs badge at the top of all labs pages */}
      <div className="mb-4 mx-4 sm:mx-6 lg:mx-8 mt-4">
        <div className="inline-flex items-center gap-2 rounded-full border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 px-3 py-1 text-xs">
          <span className="font-medium text-amber-800 dark:text-amber-200">Labs</span>
          <span className="opacity-70 text-amber-700 dark:text-amber-300">
            Feature preview — functionality limited
          </span>
        </div>
      </div>
      {children}
    </div>
  );
}