// Base types
export interface BaseEntity {
  id: string
  createdAt: string
  updatedAt: string
  createdBy?: string
  updatedBy?: string
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
    hasNext: boolean
    hasPrev: boolean
  }
}

export interface ApiResponse<T = unknown> {
  success: boolean
  data?: T
  message?: string
  error?: string
  code?: string
  timestamp?: string
}

export interface ApiError {
  message: string
  code?: string
  details?: Record<string, any>
  timestamp?: string
}

// Authentication types
export interface User extends BaseEntity {
  email: string
  firstName: string
  lastName: string
  displayName: string
  avatar?: string
  role: UserRole
  permissions: Permission[]
  preferences: UserPreferences
  lastLoginAt?: string
  isActive: boolean
  tenantId: string
  departments: string[]
  jobTitle?: string
  manager?: string
  phone?: string
  location?: string
}

export interface UserRole {
  id: string
  name: string
  description: string
  permissions: Permission[]
  isDefault: boolean
}

export interface Permission {
  id: string
  name: string
  resource: string
  action: string
  description: string
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system'
  language: string
  timezone: string
  currency: string
  dateFormat: string
  timeFormat: string
  notifications: NotificationPreferences
  dashboard: DashboardPreferences
  accessibility: AccessibilityPreferences
}

export interface NotificationPreferences {
  email: boolean
  push: boolean
  sms: boolean
  inApp: boolean
  frequency: 'instant' | 'daily' | 'weekly'
  categories: string[]
}

export interface DashboardPreferences {
  layout: 'grid' | 'list'
  widgets: DashboardWidget[]
  refreshInterval: number
  showWelcome: boolean
  compactMode: boolean
}

export interface AccessibilityPreferences {
  highContrast: boolean
  reducedMotion: boolean
  screenReader: boolean
  fontSize: 'small' | 'medium' | 'large'
  keyboardNavigation: boolean
}

// Dashboard types
export interface DashboardData {
  overview: DashboardOverview
  metrics: DashboardMetrics
  charts: DashboardCharts
  alerts: Alert[]
  recentActivity: Activity[]
}

export interface DashboardOverview {
  totalResources: number
  totalPolicies: number
  complianceScore: number
  monthlyCost: number
  costTrend: number
  alertCount: number
  recommendations: number
}

export interface DashboardMetrics {
  resourcesByType: Record<string, number>
  resourcesByRegion: Record<string, number>
  resourcesBySubscription: Record<string, number>
  costBySubscription: Record<string, number>
  costByResourceType: Record<string, number>
  complianceByPolicy: Record<string, number>
}

export interface DashboardCharts {
  costTrend: ChartData[]
  resourceGrowth: ChartData[]
  complianceScore: ChartData[]
  alertTrend: ChartData[]
}

export interface ChartData {
  date: string
  value: number
  label?: string
  category?: string
}

export interface DashboardWidget {
  id: string
  type: string
  title: string
  position: {
    x: number
    y: number
    width: number
    height: number
  }
  config: Record<string, any>
  isVisible: boolean
}

// Policy types
export interface Policy extends BaseEntity {
  name: string
  description: string
  type: PolicyType
  category: PolicyCategory
  definition: PolicyDefinition
  parameters: PolicyParameter[]
  compliance: PolicyCompliance
  assignment: PolicyAssignment
  isEnabled: boolean
  version: string
  tags: Record<string, string>
  metadata: PolicyMetadata
}

export interface PolicyType {
  id: string
  name: string
  description: string
  category: string
  schema: Record<string, any>
}

export interface PolicyCategory {
  id: string
  name: string
  description: string
  color: string
  icon: string
}

export interface PolicyDefinition {
  if: Record<string, any>
  then: Record<string, any>
  else?: Record<string, any>
}

export interface PolicyParameter {
  name: string
  type: string
  description: string
  defaultValue?: any
  allowedValues?: any[]
  required: boolean
  metadata?: Record<string, any>
}

export interface PolicyCompliance {
  score: number
  compliantResources: number
  nonCompliantResources: number
  totalResources: number
  lastEvaluated: string
  violations: PolicyViolation[]
}

export interface PolicyViolation {
  id: string
  resourceId: string
  resourceName: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  detectedAt: string
  remediation?: string
  status: 'active' | 'resolved' | 'ignored'
}

export interface PolicyAssignment {
  id: string
  scope: string
  scopeType: 'subscription' | 'resourceGroup' | 'resource'
  enforcementMode: 'default' | 'doNotEnforce'
  exclusions: string[]
  assignedAt: string
  assignedBy: string
}

export interface PolicyMetadata {
  source: string
  version: string
  documentation?: string
  category: string
  severity: string
  remediation?: string
}

// Resource types
export interface AzureResource extends BaseEntity {
  name: string
  type: string
  resourceGroup: string
  subscription: string
  location: string
  tags: Record<string, string>
  properties: Record<string, any>
  sku?: ResourceSku
  identity?: ResourceIdentity
  plan?: ResourcePlan
  compliance: ResourceCompliance
  cost: ResourceCost
  recommendations: ResourceRecommendation[]
  dependencies: ResourceDependency[]
}

export interface ResourceSku {
  name: string
  tier: string
  size?: string
  family?: string
  capacity?: number
}

export interface ResourceIdentity {
  type: string
  principalId?: string
  tenantId?: string
  userAssignedIdentities?: Record<string, any>
}

export interface ResourcePlan {
  name: string
  publisher: string
  product: string
  promotionCode?: string
  version?: string
}

export interface ResourceCompliance {
  score: number
  policies: PolicyCompliance[]
  violations: PolicyViolation[]
  lastEvaluated: string
}

export interface ResourceCost {
  dailyCost: number
  monthlyCost: number
  currency: string
  trend: number
  forecast: number
  breakdown: CostBreakdown[]
}

export interface CostBreakdown {
  category: string
  amount: number
  percentage: number
  trend: number
}

export interface ResourceRecommendation {
  id: string
  type: string
  title: string
  description: string
  impact: 'low' | 'medium' | 'high'
  category: string
  potentialSavings?: number
  effort: 'low' | 'medium' | 'high'
  status: 'active' | 'implemented' | 'dismissed'
  createdAt: string
}

export interface ResourceDependency {
  id: string
  name: string
  type: string
  relationship: string
  direction: 'inbound' | 'outbound'
}

// Notification types
export interface Notification extends BaseEntity {
  title: string
  message: string
  type: NotificationType
  category: NotificationCategory
  severity: 'info' | 'warning' | 'error' | 'success'
  isRead: boolean
  readAt?: string
  actionUrl?: string
  actionLabel?: string
  data?: Record<string, any>
  expiresAt?: string
  userId: string
  source: string
}

export interface NotificationType {
  id: string
  name: string
  description: string
  icon: string
  color: string
  template: string
}

export interface NotificationCategory {
  id: string
  name: string
  description: string
  icon: string
  color: string
  isDefault: boolean
}

// Conversation types
export interface Conversation extends BaseEntity {
  title: string
  participants: ConversationParticipant[]
  messages: Message[]
  status: 'active' | 'archived' | 'closed'
  context: ConversationContext
  metadata: Record<string, any>
}

export interface ConversationParticipant {
  id: string
  type: 'user' | 'ai' | 'system'
  name: string
  avatar?: string
  isActive: boolean
  joinedAt: string
  lastSeenAt?: string
}

export interface Message extends BaseEntity {
  content: string
  type: 'text' | 'image' | 'file' | 'code' | 'chart' | 'card'
  sender: ConversationParticipant
  replyTo?: string
  reactions: MessageReaction[]
  metadata: MessageMetadata
  isEdited: boolean
  editedAt?: string
}

export interface MessageReaction {
  emoji: string
  count: number
  users: string[]
}

export interface MessageMetadata {
  attachments?: MessageAttachment[]
  mentions?: string[]
  intent?: string
  entities?: Record<string, any>
  confidence?: number
  processingTime?: number
}

export interface MessageAttachment {
  id: string
  name: string
  type: string
  size: number
  url: string
  thumbnail?: string
}

export interface ConversationContext {
  topic?: string
  scope?: string
  filters?: Record<string, any>
  relatedResources?: string[]
  sessionId?: string
}

// Alert types
export interface Alert extends BaseEntity {
  title: string
  description: string
  severity: 'info' | 'warning' | 'error' | 'critical'
  status: 'active' | 'acknowledged' | 'resolved' | 'suppressed'
  category: AlertCategory
  source: string
  resourceId?: string
  resourceName?: string
  conditions: AlertCondition[]
  actions: AlertAction[]
  assignedTo?: string
  acknowledgedBy?: string
  acknowledgedAt?: string
  resolvedBy?: string
  resolvedAt?: string
  firstOccurrence: string
  lastOccurrence: string
  occurrenceCount: number
  suppressUntil?: string
  metadata: Record<string, any>
}

export interface AlertCategory {
  id: string
  name: string
  description: string
  icon: string
  color: string
  defaultSeverity: string
}

export interface AlertCondition {
  field: string
  operator: string
  value: any
  threshold?: number
  duration?: number
}

export interface AlertAction {
  id: string
  type: string
  name: string
  description: string
  config: Record<string, any>
  isEnabled: boolean
  lastExecuted?: string
  executionCount: number
}

// Activity types
export interface Activity extends BaseEntity {
  type: string
  action: string
  description: string
  actor: ActivityActor
  target: ActivityTarget
  result: 'success' | 'failure' | 'pending'
  metadata: Record<string, any>
  ipAddress?: string
  userAgent?: string
  location?: string
  duration?: number
}

export interface ActivityActor {
  id: string
  type: 'user' | 'system' | 'service'
  name: string
  email?: string
}

export interface ActivityTarget {
  id: string
  type: string
  name: string
  metadata?: Record<string, any>
}

// WebSocket types
export interface WebSocketMessage {
  id: string
  type: string
  event: string
  data: any
  timestamp: string
  userId?: string
  sessionId?: string
}

export interface WebSocketState {
  isConnected: boolean
  connectionId?: string
  lastPing?: string
  reconnectAttempts: number
  subscriptions: string[]
}

// Settings types
export interface Settings extends BaseEntity {
  userId: string
  general: GeneralSettings
  security: SecuritySettings
  notifications: NotificationSettings
  integrations: IntegrationSettings
  appearance: AppearanceSettings
}

export interface GeneralSettings {
  language: string
  timezone: string
  currency: string
  dateFormat: string
  timeFormat: string
  defaultSubscription?: string
  defaultResourceGroup?: string
}

export interface SecuritySettings {
  twoFactorEnabled: boolean
  sessionTimeout: number
  allowedIpAddresses: string[]
  apiKeys: ApiKey[]
  auditLog: boolean
}

export interface ApiKey {
  id: string
  name: string
  key: string
  permissions: string[]
  expiresAt?: string
  lastUsed?: string
  isActive: boolean
}

export interface NotificationSettings {
  email: boolean
  push: boolean
  sms: boolean
  inApp: boolean
  frequency: 'instant' | 'daily' | 'weekly'
  categories: NotificationCategorySettings[]
}

export interface NotificationCategorySettings {
  categoryId: string
  enabled: boolean
  channels: string[]
  threshold?: string
}

export interface IntegrationSettings {
  azure: AzureIntegrationSettings
  thirdParty: ThirdPartyIntegrationSettings
}

export interface AzureIntegrationSettings {
  tenantId: string
  subscriptions: string[]
  resourceGroups: string[]
  syncFrequency: number
  enabledServices: string[]
}

export interface ThirdPartyIntegrationSettings {
  slack?: SlackIntegrationSettings
  teams?: TeamsIntegrationSettings
  jira?: JiraIntegrationSettings
  servicenow?: ServiceNowIntegrationSettings
}

export interface SlackIntegrationSettings {
  webhookUrl: string
  channel: string
  enabled: boolean
}

export interface TeamsIntegrationSettings {
  webhookUrl: string
  enabled: boolean
}

export interface JiraIntegrationSettings {
  url: string
  username: string
  apiToken: string
  project: string
  enabled: boolean
}

export interface ServiceNowIntegrationSettings {
  instanceUrl: string
  username: string
  password: string
  enabled: boolean
}

export interface AppearanceSettings {
  theme: 'light' | 'dark' | 'system'
  primaryColor: string
  secondaryColor: string
  fontFamily: string
  fontSize: 'small' | 'medium' | 'large'
  compactMode: boolean
  showAnimations: boolean
  showTooltips: boolean
}

// Filter and search types
export interface FilterOptions {
  search?: string
  categories?: string[]
  tags?: string[]
  dateRange?: DateRange
  status?: string[]
  severity?: string[]
  assignedTo?: string[]
  sortBy?: string
  sortDirection?: 'asc' | 'desc'
  page?: number
  limit?: number
}

export interface DateRange {
  start: string
  end: string
}

export interface SearchResult<T = any> {
  item: T
  score: number
  highlights: string[]
  metadata?: Record<string, any>
}

// Form types
export interface FormField {
  name: string
  type: string
  label: string
  placeholder?: string
  required?: boolean
  validation?: ValidationRule[]
  options?: SelectOption[]
  defaultValue?: any
  disabled?: boolean
  readonly?: boolean
  description?: string
  tooltip?: string
}

export interface ValidationRule {
  type: string
  value?: any
  message: string
}

export interface SelectOption {
  value: any
  label: string
  description?: string
  disabled?: boolean
  group?: string
}

// Upload types
export interface FileUpload {
  id: string
  name: string
  type: string
  size: number
  status: 'pending' | 'uploading' | 'completed' | 'failed'
  progress: number
  url?: string
  error?: string
  metadata?: Record<string, any>
}

// Export types
export interface ExportOptions {
  format: 'csv' | 'excel' | 'json' | 'pdf'
  fields?: string[]
  filters?: FilterOptions
  template?: string
  includeHeaders?: boolean
  includeMetadata?: boolean
}

export interface ExportResult {
  id: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  url?: string
  error?: string
  createdAt: string
  expiresAt?: string
}