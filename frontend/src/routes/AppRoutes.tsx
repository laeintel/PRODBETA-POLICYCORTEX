import { Routes, Route, Navigate } from 'react-router-dom'
import { Suspense, lazy } from 'react'
import { LoadingScreen } from '@/components/UI/LoadingScreen'
import { ProtectedRoute } from '@/components/Auth/ProtectedRoute'

// Lazy load pages
const DashboardPage = lazy(() => import('@/pages/Dashboard/DashboardPage'))
const PoliciesPage = lazy(() => import('@/pages/Policies/PoliciesPage'))
const PolicyDetailsPage = lazy(() => import('@/pages/Policies/PolicyDetailsPage'))
const CreatePolicyPage = lazy(() => import('@/pages/Policies/CreatePolicyPage'))
const PolicyTemplatesPage = lazy(() => import('@/pages/Policies/PolicyTemplatesPage'))
const ResourcesPage = lazy(() => import('@/pages/Resources/ResourcesPage'))
const ResourceDetailsPage = lazy(() => import('@/pages/Resources/ResourceDetailsPage'))
const ResourceInventoryPage = lazy(() => import('@/pages/Resources/ResourceInventoryPage'))
const ResourceTopologyPage = lazy(() => import('@/pages/Resources/ResourceTopologyPage'))
const CostsPage = lazy(() => import('@/pages/Costs/CostsPage'))
const CostAnalysisPage = lazy(() => import('@/pages/Costs/CostAnalysisPage'))
const CostBudgetsPage = lazy(() => import('@/pages/Costs/CostBudgetsPage'))
const ConversationPage = lazy(() => import('@/pages/Conversation/ConversationPage'))
const AnalyticsPage = lazy(() => import('@/pages/Analytics/AnalyticsPage'))
const AnalyticsReportsPage = lazy(() => import('@/pages/Analytics/AnalyticsReportsPage'))
const SecurityPage = lazy(() => import('@/pages/Security/SecurityPage'))
const CompliancePage = lazy(() => import('@/pages/Security/CompliancePage'))
const NotificationsPage = lazy(() => import('@/pages/Notifications/NotificationsPage'))
const SettingsPage = lazy(() => import('@/pages/Settings/SettingsPage'))
const ProfilePage = lazy(() => import('@/pages/Profile/ProfilePage'))
const NotFoundPage = lazy(() => import('@/pages/NotFoundPage'))

export const AppRoutes = () => {
  return (
    <Suspense fallback={<LoadingScreen message="Loading page..." />}>
      <Routes>
        {/* Default redirect */}
        <Route path="/" element={<Navigate to="/dashboard" replace />} />

        {/* Dashboard */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          }
        />

        {/* Policies */}
        <Route
          path="/policies"
          element={
            <ProtectedRoute permission="policies:view">
              <PoliciesPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/policies/create"
          element={
            <ProtectedRoute permission="policies:create">
              <CreatePolicyPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/policies/templates"
          element={
            <ProtectedRoute permission="policies:view">
              <PolicyTemplatesPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/policies/:id"
          element={
            <ProtectedRoute permission="policies:view">
              <PolicyDetailsPage />
            </ProtectedRoute>
          }
        />

        {/* Resources */}
        <Route
          path="/resources"
          element={
            <ProtectedRoute permission="resources:view">
              <ResourcesPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/resources/inventory"
          element={
            <ProtectedRoute permission="resources:view">
              <ResourceInventoryPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/resources/topology"
          element={
            <ProtectedRoute permission="resources:view">
              <ResourceTopologyPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/resources/:id"
          element={
            <ProtectedRoute permission="resources:view">
              <ResourceDetailsPage />
            </ProtectedRoute>
          }
        />

        {/* Costs */}
        <Route
          path="/costs"
          element={
            <ProtectedRoute permission="costs:view">
              <CostsPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/costs/analysis"
          element={
            <ProtectedRoute permission="costs:view">
              <CostAnalysisPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/costs/budgets"
          element={
            <ProtectedRoute permission="costs:manage">
              <CostBudgetsPage />
            </ProtectedRoute>
          }
        />

        {/* Conversation */}
        <Route
          path="/conversation"
          element={
            <ProtectedRoute>
              <ConversationPage />
            </ProtectedRoute>
          }
        />

        {/* Analytics */}
        <Route
          path="/analytics"
          element={
            <ProtectedRoute permission="analytics:view">
              <AnalyticsPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/analytics/reports"
          element={
            <ProtectedRoute permission="analytics:view">
              <AnalyticsReportsPage />
            </ProtectedRoute>
          }
        />

        {/* Security */}
        <Route
          path="/security"
          element={
            <ProtectedRoute permission="security:view">
              <SecurityPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/security/compliance"
          element={
            <ProtectedRoute permission="security:view">
              <CompliancePage />
            </ProtectedRoute>
          }
        />

        {/* Notifications */}
        <Route
          path="/notifications"
          element={
            <ProtectedRoute>
              <NotificationsPage />
            </ProtectedRoute>
          }
        />

        {/* Settings */}
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <SettingsPage />
            </ProtectedRoute>
          }
        />

        {/* Profile */}
        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <ProfilePage />
            </ProtectedRoute>
          }
        />

        {/* 404 */}
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </Suspense>
  )
}