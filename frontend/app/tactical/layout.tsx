import React from 'react';
import TacticalSidebar from '../../components/TacticalSidebar';
import AuthGuard from '../../components/AuthGuard';

export default function TacticalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthGuard requireAuth={true}>
      <div className="min-h-screen bg-gray-950 text-gray-100 flex">
        {/* Persistent Sidebar */}
        <TacticalSidebar />
        
        {/* Main Content Area */}
        <div className="flex-1 flex flex-col">
          {children}
        </div>
      </div>
    </AuthGuard>
  );
}