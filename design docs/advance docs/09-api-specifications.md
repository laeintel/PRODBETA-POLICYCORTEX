# API Specifications

## Table of Contents
1. [OpenAPI Overview](#openapi-overview)
2. [Core API Endpoints](#core-api-endpoints)
3. [GraphQL Schema](#graphql-schema)
4. [Authentication & Authorization](#authentication--authorization)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [API Versioning](#api-versioning)
8. [Request/Response Examples](#requestresponse-examples)

## OpenAPI Overview

PolicyCortex provides comprehensive REST and GraphQL APIs for all platform functionality. The APIs are designed following OpenAPI 3.0 specification with full documentation, type safety, and automated client generation.

### API Architecture

```yaml
# api/openapi.yaml
openapi: 3.0.3
info:
  title: PolicyCortex API
  description: |
    PolicyCortex AI-powered Azure governance platform API.
    
    Features four patented technologies:
    1. Unified AI-Driven Cloud Governance Platform
    2. Predictive Policy Compliance Engine  
    3. Conversational Governance Intelligence System
    4. Cross-Domain Governance Correlation Engine
    
    ## Authentication
    
    All API endpoints require authentication via Bearer token:
    ```
    Authorization: Bearer <your-jwt-token>
    ```
    
    ## Rate Limiting
    
    API requests are limited to:
    - 1000 requests per hour for standard users
    - 5000 requests per hour for premium users
    - 10000 requests per hour for enterprise users
    
    ## Error Handling
    
    All errors follow RFC 7807 Problem Details format.
    
  version: 2.0.0
  contact:
    name: PolicyCortex API Support
    url: https://policycortex.com/support
    email: api-support@policycortex.com
  license:
    name: Proprietary
    url: https://policycortex.com/license
  termsOfService: https://policycortex.com/terms

servers:
  - url: https://api.policycortex.com/v1
    description: Production server
  - url: https://staging-api.policycortex.com/v1
    description: Staging server
  - url: http://localhost:8080/api/v1
    description: Local development server

security:
  - BearerAuth: []
  - ApiKeyAuth: []

paths:
  # Resources
  /resources:
    get:
      summary: List resources
      description: Retrieve a paginated list of Azure resources with optional filtering
      operationId: listResources
      tags:
        - Resources
      parameters:
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/LimitParam'
        - $ref: '#/components/parameters/SubscriptionIdFilter'
        - $ref: '#/components/parameters/ResourceTypeFilter'
        - $ref: '#/components/parameters/LocationFilter'
        - name: resourceGroupName
          in: query
          description: Filter by resource group name
          schema:
            type: string
        - name: tags
          in: query
          description: Filter by tags (key:value format)
          schema:
            type: array
            items:
              type: string
        - name: complianceStatus
          in: query
          description: Filter by compliance status
          schema:
            $ref: '#/components/schemas/EvaluationResult'
      responses:
        '200':
          description: List of resources
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ResourceListResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'
    post:
      summary: Create resource
      description: Create a new resource (typically from Azure sync)
      operationId: createResource
      tags:
        - Resources
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateResourceRequest'
      responses:
        '201':
          description: Resource created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Resource'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '409':
          $ref: '#/components/responses/Conflict'
        '500':
          $ref: '#/components/responses/InternalServerError'

  /resources/{resourceId}:
    parameters:
      - $ref: '#/components/parameters/ResourceIdPath'
    get:
      summary: Get resource
      description: Retrieve detailed information about a specific resource
      operationId: getResource
      tags:
        - Resources
      responses:
        '200':
          description: Resource details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Resource'
        '404':
          $ref: '#/components/responses/NotFound'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'
    put:
      summary: Update resource
      description: Update resource information
      operationId: updateResource
      tags:
        - Resources
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateResourceRequest'
      responses:
        '200':
          description: Resource updated successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Resource'
        '400':
          $ref: '#/components/responses/BadRequest'
        '404':
          $ref: '#/components/responses/NotFound'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'
    delete:
      summary: Delete resource
      description: Soft delete a resource
      operationId: deleteResource
      tags:
        - Resources
      responses:
        '204':
          description: Resource deleted successfully
        '404':
          $ref: '#/components/responses/NotFound'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  /resources/{resourceId}/compliance:
    parameters:
      - $ref: '#/components/parameters/ResourceIdPath'
    get:
      summary: Get resource compliance status
      description: Retrieve current compliance status for a resource
      operationId: getResourceComplianceStatus
      tags:
        - Compliance
      responses:
        '200':
          description: Compliance status
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ComplianceStatus'
        '404':
          $ref: '#/components/responses/NotFound'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  /resources/{resourceId}/insights:
    parameters:
      - $ref: '#/components/parameters/ResourceIdPath'
    get:
      summary: Get AI insights for resource
      description: Retrieve AI-generated insights and recommendations
      operationId: getResourceInsights
      tags:
        - AI Insights
      responses:
        '200':
          description: AI insights
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AIInsights'
        '404':
          $ref: '#/components/responses/NotFound'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  # Policies
  /policies:
    get:
      summary: List policies
      description: Retrieve a paginated list of governance policies
      operationId: listPolicies
      tags:
        - Policies
      parameters:
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/LimitParam'
        - name: type
          in: query
          description: Filter by policy type
          schema:
            $ref: '#/components/schemas/PolicyType'
        - name: category
          in: query
          description: Filter by policy category
          schema:
            type: string
        - name: severity
          in: query
          description: Filter by severity level
          schema:
            $ref: '#/components/schemas/SeverityLevel'
        - name: enabled
          in: query
          description: Filter by enabled status
          schema:
            type: boolean
      responses:
        '200':
          description: List of policies
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PolicyListResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'
    post:
      summary: Create policy
      description: Create a new governance policy
      operationId: createPolicy
      tags:
        - Policies
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreatePolicyRequest'
      responses:
        '201':
          description: Policy created successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Policy'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  /policies/{policyId}:
    parameters:
      - $ref: '#/components/parameters/PolicyIdPath'
    get:
      summary: Get policy
      description: Retrieve detailed information about a specific policy
      operationId: getPolicy
      tags:
        - Policies
      responses:
        '200':
          description: Policy details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Policy'
        '404':
          $ref: '#/components/responses/NotFound'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'
    put:
      summary: Update policy
      description: Update policy definition
      operationId: updatePolicy
      tags:
        - Policies
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdatePolicyRequest'
      responses:
        '200':
          description: Policy updated successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Policy'
        '400':
          $ref: '#/components/responses/BadRequest'
        '404':
          $ref: '#/components/responses/NotFound'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  /policies/{policyId}/evaluate:
    parameters:
      - $ref: '#/components/parameters/PolicyIdPath'
    post:
      summary: Evaluate policy
      description: Evaluate a policy against one or more resources
      operationId: evaluatePolicy
      tags:
        - Policy Evaluation
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/EvaluatePolicyRequest'
      responses:
        '202':
          description: Policy evaluation started
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PolicyEvaluationJob'
        '400':
          $ref: '#/components/responses/BadRequest'
        '404':
          $ref: '#/components/responses/NotFound'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  # Patent Feature APIs
  /ai/generate-policy:
    post:
      summary: Generate policy using AI (Patent Feature)
      description: |
        Generate governance policy using the Conversational Governance Intelligence System.
        This endpoint implements patent #2 technology.
      operationId: generatePolicy
      tags:
        - AI Patent Features
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/GeneratePolicyRequest'
      responses:
        '200':
          description: Policy generated successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/GeneratedPolicy'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  /ai/conversational:
    post:
      summary: Conversational governance query (Patent Feature)
      description: |
        Process natural language governance queries using the Conversational Governance Intelligence System.
        This endpoint implements patent #2 technology.
      operationId: conversationalQuery
      tags:
        - AI Patent Features
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ConversationalQueryRequest'
      responses:
        '200':
          description: Query processed successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ConversationalResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  /predictions:
    get:
      summary: Get compliance predictions (Patent Feature)
      description: |
        Retrieve predictive compliance insights using the Predictive Policy Compliance Engine.
        This endpoint implements patent #4 technology.
      operationId: getCompliancePredictions
      tags:
        - AI Patent Features
      parameters:
        - name: resourceId
          in: query
          description: Filter predictions for specific resource
          schema:
            type: string
            format: uuid
        - name: policyId
          in: query
          description: Filter predictions for specific policy
          schema:
            type: string
            format: uuid
        - name: timeHorizon
          in: query
          description: Prediction time horizon in days
          schema:
            type: integer
            minimum: 1
            maximum: 365
            default: 30
      responses:
        '200':
          description: Compliance predictions
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CompliancePredictions'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  /correlations:
    get:
      summary: Get cross-domain correlations (Patent Feature)
      description: |
        Retrieve cross-domain governance correlations using the Cross-Domain Governance Correlation Engine.
        This endpoint implements patent #1 technology.
      operationId: getCrossDomainCorrelations
      tags:
        - AI Patent Features
      parameters:
        - name: domains
          in: query
          description: Domains to analyze for correlations
          schema:
            type: array
            items:
              type: string
              enum: [security, compliance, cost, performance, governance]
        - name: timeRange
          in: query
          description: Time range for correlation analysis
          schema:
            type: string
            enum: [1h, 6h, 24h, 7d, 30d]
            default: 24h
      responses:
        '200':
          description: Cross-domain correlations
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CrossDomainCorrelations'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  /metrics/unified:
    get:
      summary: Get unified governance metrics (Patent Feature)
      description: |
        Retrieve unified metrics across all governance domains using the Unified AI-Driven Cloud Governance Platform.
        This endpoint implements patent #3 technology.
      operationId: getUnifiedMetrics
      tags:
        - AI Patent Features
      parameters:
        - name: subscriptionId
          in: query
          description: Filter by subscription ID
          schema:
            type: string
        - name: aggregation
          in: query
          description: Metric aggregation level
          schema:
            type: string
            enum: [resource, resourceGroup, subscription, tenant]
            default: subscription
        - name: timeRange
          in: query
          description: Time range for metrics
          schema:
            type: string
            enum: [1h, 6h, 24h, 7d, 30d]
            default: 24h
      responses:
        '200':
          description: Unified governance metrics
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UnifiedMetrics'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  # Actions/Remediation
  /actions:
    get:
      summary: List actions
      description: Retrieve a paginated list of remediation actions
      operationId: listActions
      tags:
        - Actions
      parameters:
        - $ref: '#/components/parameters/PageParam'
        - $ref: '#/components/parameters/LimitParam'
        - name: resourceId
          in: query
          description: Filter by resource ID
          schema:
            type: string
            format: uuid
        - name: status
          in: query
          description: Filter by execution status
          schema:
            $ref: '#/components/schemas/ExecutionStatus'
        - name: actionType
          in: query
          description: Filter by action type
          schema:
            type: string
      responses:
        '200':
          description: List of actions
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ActionListResponse'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'
    post:
      summary: Execute action
      description: Execute a remediation action on a resource
      operationId: executeAction
      tags:
        - Actions
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ExecuteActionRequest'
      responses:
        '202':
          description: Action execution started
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ActionExecution'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'
        '403':
          $ref: '#/components/responses/Forbidden'
        '500':
          $ref: '#/components/responses/InternalServerError'

  # Health and Monitoring
  /health:
    get:
      summary: Health check
      description: Check API health status
      operationId: healthCheck
      tags:
        - Health
      security: []  # No authentication required
      responses:
        '200':
          description: API is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthStatus'
        '503':
          description: API is unhealthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthStatus'

  /health/ready:
    get:
      summary: Readiness check
      description: Check if API is ready to serve requests
      operationId: readinessCheck
      tags:
        - Health
      security: []  # No authentication required
      responses:
        '200':
          description: API is ready
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ReadinessStatus'
        '503':
          description: API is not ready
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ReadinessStatus'

  /metrics:
    get:
      summary: Get metrics
      description: Retrieve Prometheus-formatted metrics
      operationId: getMetrics
      tags:
        - Monitoring
      security: []  # No authentication required for metrics
      responses:
        '200':
          description: Prometheus metrics
          content:
            text/plain:
              schema:
                type: string

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT bearer token authentication
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: API key authentication

  parameters:
    PageParam:
      name: page
      in: query
      description: Page number (1-based)
      schema:
        type: integer
        minimum: 1
        default: 1
    
    LimitParam:
      name: limit
      in: query
      description: Number of items per page
      schema:
        type: integer
        minimum: 1
        maximum: 100
        default: 20

    ResourceIdPath:
      name: resourceId
      in: path
      required: true
      description: Resource UUID
      schema:
        type: string
        format: uuid

    PolicyIdPath:
      name: policyId
      in: path
      required: true
      description: Policy UUID
      schema:
        type: string
        format: uuid

    SubscriptionIdFilter:
      name: subscriptionId
      in: query
      description: Filter by Azure subscription ID
      schema:
        type: string
        pattern: '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'

    ResourceTypeFilter:
      name: resourceType
      in: query
      description: Filter by resource type
      schema:
        $ref: '#/components/schemas/ResourceType'

    LocationFilter:
      name: location
      in: query
      description: Filter by Azure region
      schema:
        type: string

  schemas:
    # Core entities
    Resource:
      type: object
      required:
        - id
        - tenantId
        - azureResourceId
        - subscriptionId
        - resourceGroupName
        - name
        - type
        - location
        - createdAt
        - updatedAt
      properties:
        id:
          type: string
          format: uuid
          description: Unique resource identifier
          readOnly: true
        tenantId:
          type: string
          format: uuid
          description: Tenant identifier
          readOnly: true
        azureResourceId:
          type: string
          description: Azure resource ID
          maxLength: 500
        subscriptionId:
          type: string
          description: Azure subscription ID
          pattern: '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        resourceGroupName:
          type: string
          description: Azure resource group name
          maxLength: 255
        name:
          type: string
          description: Resource name
          maxLength: 255
        type:
          $ref: '#/components/schemas/ResourceType'
        location:
          type: string
          description: Azure region
          maxLength: 100
        properties:
          type: object
          description: Resource-specific properties
          additionalProperties: true
        tags:
          type: object
          description: Resource tags
          additionalProperties:
            type: string
        createdAt:
          type: string
          format: date-time
          description: Creation timestamp
          readOnly: true
        updatedAt:
          type: string
          format: date-time
          description: Last update timestamp
          readOnly: true
        syncedAt:
          type: string
          format: date-time
          description: Last sync timestamp
          readOnly: true

    Policy:
      type: object
      required:
        - id
        - tenantId
        - name
        - type
        - definition
        - severity
        - enabled
        - version
        - createdAt
        - updatedAt
      properties:
        id:
          type: string
          format: uuid
          description: Unique policy identifier
          readOnly: true
        tenantId:
          type: string
          format: uuid
          description: Tenant identifier
          readOnly: true
        name:
          type: string
          description: Policy name
          maxLength: 255
        displayName:
          type: string
          description: Human-readable policy name
          maxLength: 255
        description:
          type: string
          description: Policy description
        type:
          $ref: '#/components/schemas/PolicyType'
        category:
          type: string
          description: Policy category
          maxLength: 100
        severity:
          $ref: '#/components/schemas/SeverityLevel'
        definition:
          type: object
          description: Policy definition (JSON schema or rules)
          additionalProperties: true
        metadata:
          type: object
          description: Additional policy metadata
          additionalProperties: true
        enabled:
          type: boolean
          description: Whether policy is enabled
          default: true
        version:
          type: integer
          description: Policy version
          readOnly: true
        createdBy:
          type: string
          format: uuid
          description: User who created the policy
          readOnly: true
        createdAt:
          type: string
          format: date-time
          description: Creation timestamp
          readOnly: true
        updatedAt:
          type: string
          format: date-time
          description: Last update timestamp
          readOnly: true

    PolicyEvaluation:
      type: object
      required:
        - id
        - tenantId
        - policyId
        - resourceId
        - status
        - createdAt
      properties:
        id:
          type: string
          format: uuid
          description: Unique evaluation identifier
          readOnly: true
        tenantId:
          type: string
          format: uuid
          description: Tenant identifier
          readOnly: true
        policyId:
          type: string
          format: uuid
          description: Policy identifier
        resourceId:
          type: string
          format: uuid
          description: Resource identifier
        status:
          $ref: '#/components/schemas/EvaluationStatus'
        result:
          $ref: '#/components/schemas/EvaluationResult'
        score:
          type: number
          format: float
          minimum: 0
          maximum: 100
          description: Compliance score (0-100)
        evidence:
          type: array
          description: Evidence supporting the evaluation
          items:
            $ref: '#/components/schemas/Evidence'
        recommendations:
          type: array
          description: Remediation recommendations
          items:
            type: string
        errorMessage:
          type: string
          description: Error message if evaluation failed
        executionTimeMs:
          type: integer
          description: Evaluation execution time in milliseconds
        evaluatedAt:
          type: string
          format: date-time
          description: Evaluation completion timestamp
        createdAt:
          type: string
          format: date-time
          description: Evaluation start timestamp
          readOnly: true

    ComplianceStatus:
      type: object
      required:
        - resourceId
        - overallStatus
        - complianceScore
        - policyEvaluations
        - lastEvaluated
      properties:
        resourceId:
          type: string
          format: uuid
          description: Resource identifier
        overallStatus:
          $ref: '#/components/schemas/EvaluationResult'
        complianceScore:
          type: number
          format: float
          minimum: 0
          maximum: 100
          description: Overall compliance score (0-100)
        policyEvaluations:
          type: array
          description: Individual policy evaluations
          items:
            $ref: '#/components/schemas/PolicyEvaluation'
        lastEvaluated:
          type: string
          format: date-time
          description: Last evaluation timestamp

    ActionExecution:
      type: object
      required:
        - id
        - tenantId
        - resourceId
        - name
        - actionType
        - status
        - dryRun
        - createdAt
      properties:
        id:
          type: string
          format: uuid
          description: Unique action identifier
          readOnly: true
        tenantId:
          type: string
          format: uuid
          description: Tenant identifier
          readOnly: true
        resourceId:
          type: string
          format: uuid
          description: Resource identifier
        policyEvaluationId:
          type: string
          format: uuid
          description: Related policy evaluation ID
        name:
          type: string
          description: Action name
          maxLength: 255
        description:
          type: string
          description: Action description
        actionType:
          type: string
          description: Type of action
          maxLength: 100
        parameters:
          type: object
          description: Action parameters
          additionalProperties: true
        status:
          $ref: '#/components/schemas/ExecutionStatus'
        dryRun:
          type: boolean
          description: Whether this is a dry run
        result:
          type: object
          description: Action execution result
          additionalProperties: true
        errorMessage:
          type: string
          description: Error message if execution failed
        executionTimeMs:
          type: integer
          description: Execution time in milliseconds
        executedBy:
          type: string
          format: uuid
          description: User who executed the action
        approvedBy:
          type: string
          format: uuid
          description: User who approved the action
        createdAt:
          type: string
          format: date-time
          description: Creation timestamp
          readOnly: true
        startedAt:
          type: string
          format: date-time
          description: Execution start timestamp
        completedAt:
          type: string
          format: date-time
          description: Execution completion timestamp

    # AI Patent Feature Schemas
    AIInsights:
      type: object
      required:
        - resourceId
        - securityScore
        - complianceScore
        - costOptimizationScore
        - performanceScore
        - recommendations
        - risks
        - opportunities
        - generatedAt
      properties:
        resourceId:
          type: string
          format: uuid
          description: Resource identifier
        securityScore:
          type: number
          format: float
          minimum: 0
          maximum: 100
          description: Security score (0-100)
        complianceScore:
          type: number
          format: float
          minimum: 0
          maximum: 100
          description: Compliance score (0-100)
        costOptimizationScore:
          type: number
          format: float
          minimum: 0
          maximum: 100
          description: Cost optimization score (0-100)
        performanceScore:
          type: number
          format: float
          minimum: 0
          maximum: 100
          description: Performance score (0-100)
        recommendations:
          type: array
          description: AI-generated recommendations
          items:
            $ref: '#/components/schemas/Recommendation'
        risks:
          type: array
          description: Identified risks
          items:
            $ref: '#/components/schemas/Risk'
        opportunities:
          type: array
          description: Optimization opportunities
          items:
            $ref: '#/components/schemas/Opportunity'
        generatedAt:
          type: string
          format: date-time
          description: Insights generation timestamp
        expiresAt:
          type: string
          format: date-time
          description: Insights expiration timestamp

    GeneratedPolicy:
      type: object
      required:
        - name
        - type
        - category
        - definition
        - rationale
        - affectedResourceTypes
        - estimatedImpact
      properties:
        name:
          type: string
          description: Generated policy name
        displayName:
          type: string
          description: Human-readable policy name
        type:
          $ref: '#/components/schemas/PolicyType'
        category:
          type: string
          description: Policy category
        severity:
          $ref: '#/components/schemas/SeverityLevel'
        definition:
          type: object
          description: Policy definition
          additionalProperties: true
        rationale:
          type: string
          description: AI-generated rationale for the policy
        affectedResourceTypes:
          type: array
          description: Resource types affected by this policy
          items:
            $ref: '#/components/schemas/ResourceType'
        estimatedImpact:
          $ref: '#/components/schemas/PolicyImpact'
        confidence:
          type: number
          format: float
          minimum: 0
          maximum: 1
          description: AI confidence in the generated policy

    ConversationalResponse:
      type: object
      required:
        - query
        - response
        - confidence
        - sources
        - suggestedActions
        - relatedQuestions
      properties:
        query:
          type: string
          description: Original user query
        response:
          type: string
          description: AI-generated response
        confidence:
          type: number
          format: float
          minimum: 0
          maximum: 1
          description: Confidence in the response
        sources:
          type: array
          description: Information sources used
          items:
            type: string
        suggestedActions:
          type: array
          description: Suggested follow-up actions
          items:
            type: string
        relatedQuestions:
          type: array
          description: Related questions user might ask
          items:
            type: string
        contextId:
          type: string
          format: uuid
          description: Context ID for follow-up questions

    CompliancePredictions:
      type: object
      required:
        - predictions
        - timeHorizon
        - confidence
        - factors
        - generatedAt
      properties:
        predictions:
          type: array
          description: Compliance predictions
          items:
            $ref: '#/components/schemas/CompliancePrediction'
        timeHorizon:
          type: integer
          description: Prediction time horizon in days
        confidence:
          type: number
          format: float
          minimum: 0
          maximum: 1
          description: Overall prediction confidence
        factors:
          type: array
          description: Factors influencing predictions
          items:
            $ref: '#/components/schemas/PredictionFactor'
        generatedAt:
          type: string
          format: date-time
          description: Prediction generation timestamp

    CompliancePrediction:
      type: object
      required:
        - resourceId
        - policyId
        - predictedStatus
        - confidence
        - timeHorizon
        - factors
        - mitigation
        - generatedAt
      properties:
        resourceId:
          type: string
          format: uuid
          description: Resource identifier
        policyId:
          type: string
          format: uuid
          description: Policy identifier
        predictedStatus:
          $ref: '#/components/schemas/EvaluationResult'
        confidence:
          type: number
          format: float
          minimum: 0
          maximum: 1
          description: Prediction confidence
        timeHorizon:
          type: integer
          description: Prediction horizon in days
        factors:
          type: array
          description: Factors influencing prediction
          items:
            $ref: '#/components/schemas/PredictionFactor'
        mitigation:
          type: array
          description: Mitigation strategies
          items:
            type: string
        generatedAt:
          type: string
          format: date-time
          description: Prediction generation timestamp

    CrossDomainCorrelations:
      type: object
      required:
        - correlations
        - domains
        - timeRange
        - confidence
        - generatedAt
      properties:
        correlations:
          type: array
          description: Identified correlations
          items:
            $ref: '#/components/schemas/DomainCorrelation'
        domains:
          type: array
          description: Analyzed domains
          items:
            type: string
            enum: [security, compliance, cost, performance, governance]
        timeRange:
          type: string
          description: Analysis time range
        confidence:
          type: number
          format: float
          minimum: 0
          maximum: 1
          description: Overall correlation confidence
        generatedAt:
          type: string
          format: date-time
          description: Analysis generation timestamp

    UnifiedMetrics:
      type: object
      required:
        - securityMetrics
        - complianceMetrics
        - costMetrics
        - performanceMetrics
        - governanceMetrics
        - aggregationLevel
        - timeRange
        - generatedAt
      properties:
        securityMetrics:
          $ref: '#/components/schemas/SecurityMetrics'
        complianceMetrics:
          $ref: '#/components/schemas/ComplianceMetrics'
        costMetrics:
          $ref: '#/components/schemas/CostMetrics'
        performanceMetrics:
          $ref: '#/components/schemas/PerformanceMetrics'
        governanceMetrics:
          $ref: '#/components/schemas/GovernanceMetrics'
        aggregationLevel:
          type: string
          enum: [resource, resourceGroup, subscription, tenant]
        timeRange:
          type: string
        generatedAt:
          type: string
          format: date-time

    # Enum types
    ResourceType:
      type: string
      enum:
        - virtual_machine
        - storage_account
        - sql_database
        - app_service
        - function_app
        - key_vault
        - network_security_group
        - load_balancer
        - virtual_network
        - public_ip
        - network_interface
        - disk
        - availability_set
        - scale_set

    PolicyType:
      type: string
      enum:
        - security
        - compliance
        - cost_optimization
        - performance
        - governance
        - finops

    SeverityLevel:
      type: string
      enum:
        - critical
        - high
        - medium
        - low
        - info

    EvaluationStatus:
      type: string
      enum:
        - pending
        - evaluating
        - completed
        - failed
        - skipped

    EvaluationResult:
      type: string
      enum:
        - compliant
        - non_compliant
        - warning
        - unknown
        - not_applicable

    ExecutionStatus:
      type: string
      enum:
        - pending
        - running
        - completed
        - failed
        - cancelled
        - rolled_back

    # Supporting schemas
    Evidence:
      type: object
      required:
        - type
        - description
        - severity
        - data
      properties:
        type:
          type: string
          description: Evidence type
        description:
          type: string
          description: Evidence description
        severity:
          $ref: '#/components/schemas/SeverityLevel'
        data:
          type: object
          description: Evidence data
          additionalProperties: true

    Recommendation:
      type: object
      required:
        - id
        - type
        - title
        - description
        - priority
        - category
        - estimatedImpact
        - implementationEffort
      properties:
        id:
          type: string
          format: uuid
        type:
          type: string
          enum:
            - security_improvement
            - compliance_fix
            - cost_optimization
            - performance_enhancement
            - governance_alignment
        title:
          type: string
        description:
          type: string
        priority:
          type: string
          enum: [low, medium, high, critical]
        category:
          type: string
        estimatedImpact:
          type: string
        implementationEffort:
          type: string
        relatedPolicies:
          type: array
          items:
            type: string
            format: uuid

    Risk:
      type: object
      required:
        - id
        - type
        - title
        - description
        - severity
        - likelihood
        - impact
        - mitigation
      properties:
        id:
          type: string
          format: uuid
        type:
          type: string
        title:
          type: string
        description:
          type: string
        severity:
          $ref: '#/components/schemas/SeverityLevel'
        likelihood:
          type: number
          format: float
          minimum: 0
          maximum: 1
        impact:
          type: number
          format: float
          minimum: 0
          maximum: 1
        mitigation:
          type: array
          items:
            type: string

    Opportunity:
      type: object
      required:
        - id
        - type
        - title
        - description
        - value
        - effort
        - timeframe
      properties:
        id:
          type: string
          format: uuid
        type:
          type: string
        title:
          type: string
        description:
          type: string
        value:
          type: number
          format: float
        effort:
          type: string
        timeframe:
          type: string

    PolicyImpact:
      type: object
      required:
        - securityImprovement
        - complianceImprovement
        - costImpact
        - performanceImpact
      properties:
        securityImprovement:
          type: number
          format: float
          minimum: 0
          maximum: 100
        complianceImprovement:
          type: number
          format: float
          minimum: 0
          maximum: 100
        costImpact:
          type: number
          format: float
          description: Cost impact (negative for savings, positive for cost increase)
        performanceImpact:
          type: number
          format: float
          minimum: 0
          maximum: 100

    PredictionFactor:
      type: object
      required:
        - name
        - impact
        - description
      properties:
        name:
          type: string
        impact:
          type: number
          format: float
          minimum: -1
          maximum: 1
          description: Factor impact (-1 to 1)
        description:
          type: string

    DomainCorrelation:
      type: object
      required:
        - domains
        - correlationStrength
        - description
        - evidence
        - recommendation
      properties:
        domains:
          type: array
          items:
            type: string
        correlationStrength:
          type: number
          format: float
          minimum: -1
          maximum: 1
        description:
          type: string
        evidence:
          type: array
          items:
            type: object
            additionalProperties: true
        recommendation:
          type: string

    SecurityMetrics:
      type: object
      properties:
        overallSecurityScore:
          type: number
          format: float
          minimum: 0
          maximum: 100
        criticalVulnerabilities:
          type: integer
        highVulnerabilities:
          type: integer
        mediumVulnerabilities:
          type: integer
        lowVulnerabilities:
          type: integer
        patchComplianceRate:
          type: number
          format: float
          minimum: 0
          maximum: 100

    ComplianceMetrics:
      type: object
      properties:
        overallComplianceScore:
          type: number
          format: float
          minimum: 0
          maximum: 100
        compliantResources:
          type: integer
        nonCompliantResources:
          type: integer
        complianceByFramework:
          type: object
          additionalProperties:
            type: number
            format: float

    CostMetrics:
      type: object
      properties:
        totalMonthlyCost:
          type: number
          format: float
        costTrend:
          type: number
          format: float
        potentialSavings:
          type: number
          format: float
        costByService:
          type: object
          additionalProperties:
            type: number
            format: float

    PerformanceMetrics:
      type: object
      properties:
        overallPerformanceScore:
          type: number
          format: float
          minimum: 0
          maximum: 100
        averageResponseTime:
          type: number
          format: float
        availability:
          type: number
          format: float
          minimum: 0
          maximum: 100
        throughput:
          type: number
          format: float

    GovernanceMetrics:
      type: object
      properties:
        overallGovernanceScore:
          type: number
          format: float
          minimum: 0
          maximum: 100
        policiesEvaluated:
          type: integer
        policiesCompliant:
          type: integer
        governanceGaps:
          type: integer

    # Request/Response schemas
    ResourceListResponse:
      type: object
      required:
        - data
        - pagination
      properties:
        data:
          type: array
          items:
            $ref: '#/components/schemas/Resource'
        pagination:
          $ref: '#/components/schemas/PaginationInfo'

    PolicyListResponse:
      type: object
      required:
        - data
        - pagination
      properties:
        data:
          type: array
          items:
            $ref: '#/components/schemas/Policy'
        pagination:
          $ref: '#/components/schemas/PaginationInfo'

    ActionListResponse:
      type: object
      required:
        - data
        - pagination
      properties:
        data:
          type: array
          items:
            $ref: '#/components/schemas/ActionExecution'
        pagination:
          $ref: '#/components/schemas/PaginationInfo'

    PaginationInfo:
      type: object
      required:
        - page
        - limit
        - totalItems
        - totalPages
        - hasNextPage
        - hasPreviousPage
      properties:
        page:
          type: integer
          minimum: 1
        limit:
          type: integer
          minimum: 1
        totalItems:
          type: integer
          minimum: 0
        totalPages:
          type: integer
          minimum: 0
        hasNextPage:
          type: boolean
        hasPreviousPage:
          type: boolean

    CreateResourceRequest:
      type: object
      required:
        - azureResourceId
        - subscriptionId
        - resourceGroupName
        - name
        - type
        - location
      properties:
        azureResourceId:
          type: string
          maxLength: 500
        subscriptionId:
          type: string
          pattern: '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        resourceGroupName:
          type: string
          maxLength: 255
        name:
          type: string
          maxLength: 255
        type:
          $ref: '#/components/schemas/ResourceType'
        location:
          type: string
          maxLength: 100
        properties:
          type: object
          additionalProperties: true
        tags:
          type: object
          additionalProperties:
            type: string

    UpdateResourceRequest:
      type: object
      properties:
        name:
          type: string
          maxLength: 255
        properties:
          type: object
          additionalProperties: true
        tags:
          type: object
          additionalProperties:
            type: string

    CreatePolicyRequest:
      type: object
      required:
        - name
        - type
        - definition
        - severity
      properties:
        name:
          type: string
          maxLength: 255
        displayName:
          type: string
          maxLength: 255
        description:
          type: string
        type:
          $ref: '#/components/schemas/PolicyType'
        category:
          type: string
          maxLength: 100
        severity:
          $ref: '#/components/schemas/SeverityLevel'
        definition:
          type: object
          additionalProperties: true
        metadata:
          type: object
          additionalProperties: true
        enabled:
          type: boolean
          default: true

    UpdatePolicyRequest:
      type: object
      properties:
        name:
          type: string
          maxLength: 255
        displayName:
          type: string
          maxLength: 255
        description:
          type: string
        category:
          type: string
          maxLength: 100
        severity:
          $ref: '#/components/schemas/SeverityLevel'
        definition:
          type: object
          additionalProperties: true
        metadata:
          type: object
          additionalProperties: true
        enabled:
          type: boolean

    EvaluatePolicyRequest:
      type: object
      properties:
        resourceIds:
          type: array
          items:
            type: string
            format: uuid
          description: Specific resources to evaluate (if empty, evaluates all applicable resources)
        async:
          type: boolean
          default: true
          description: Whether to run evaluation asynchronously

    PolicyEvaluationJob:
      type: object
      required:
        - jobId
        - status
        - createdAt
      properties:
        jobId:
          type: string
          format: uuid
        status:
          type: string
          enum: [pending, running, completed, failed]
        progress:
          type: number
          format: float
          minimum: 0
          maximum: 100
        totalResources:
          type: integer
        processedResources:
          type: integer
        failedResources:
          type: integer
        createdAt:
          type: string
          format: date-time
        startedAt:
          type: string
          format: date-time
        completedAt:
          type: string
          format: date-time
        errorMessage:
          type: string

    ExecuteActionRequest:
      type: object
      required:
        - resourceId
        - actionType
      properties:
        resourceId:
          type: string
          format: uuid
        policyEvaluationId:
          type: string
          format: uuid
        actionType:
          type: string
        name:
          type: string
          maxLength: 255
        description:
          type: string
        parameters:
          type: object
          additionalProperties: true
        dryRun:
          type: boolean
          default: false
        requireApproval:
          type: boolean
          default: false

    GeneratePolicyRequest:
      type: object
      required:
        - resourceType
        - category
        - requirements
      properties:
        resourceType:
          $ref: '#/components/schemas/ResourceType'
        category:
          type: string
        requirements:
          type: array
          items:
            type: string
        existingPolicies:
          type: array
          items:
            type: string
            format: uuid
        constraints:
          type: object
          additionalProperties: true

    ConversationalQueryRequest:
      type: object
      required:
        - query
      properties:
        query:
          type: string
          maxLength: 1000
        contextId:
          type: string
          format: uuid
          description: Context ID for follow-up questions
        includeResources:
          type: boolean
          default: false
          description: Include specific resource information in response
        includePolicies:
          type: boolean
          default: false
          description: Include policy information in response

    # Health and status schemas
    HealthStatus:
      type: object
      required:
        - status
        - timestamp
      properties:
        status:
          type: string
          enum: [healthy, unhealthy]
        timestamp:
          type: string
          format: date-time
        version:
          type: string
        services:
          type: object
          additionalProperties:
            type: object
            properties:
              status:
                type: string
                enum: [healthy, unhealthy]
              responseTime:
                type: number
                format: float
              lastCheck:
                type: string
                format: date-time

    ReadinessStatus:
      type: object
      required:
        - ready
        - timestamp
      properties:
        ready:
          type: boolean
        timestamp:
          type: string
          format: date-time
        checks:
          type: object
          additionalProperties:
            type: object
            properties:
              ready:
                type: boolean
              message:
                type: string

    # Error schemas
    ErrorResponse:
      type: object
      required:
        - type
        - title
        - status
      properties:
        type:
          type: string
          format: uri
          description: A URI reference that identifies the problem type
        title:
          type: string
          description: A short, human-readable summary of the problem type
        status:
          type: integer
          description: The HTTP status code
        detail:
          type: string
          description: A human-readable explanation specific to this occurrence
        instance:
          type: string
          format: uri
          description: A URI reference that identifies the specific occurrence
        correlationId:
          type: string
          format: uuid
          description: Unique identifier for this request
        timestamp:
          type: string
          format: date-time
          description: When the error occurred
        errors:
          type: array
          description: Detailed validation errors
          items:
            $ref: '#/components/schemas/ValidationError'

    ValidationError:
      type: object
      required:
        - field
        - message
      properties:
        field:
          type: string
          description: Field that failed validation
        message:
          type: string
          description: Validation error message
        code:
          type: string
          description: Error code
        rejectedValue:
          description: The rejected value

  responses:
    BadRequest:
      description: Bad request - invalid input parameters
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            type: "https://policycortex.com/problems/bad-request"
            title: "Bad Request"
            status: 400
            detail: "The request is invalid"
            correlationId: "550e8400-e29b-41d4-a716-446655440000"
            timestamp: "2024-01-15T10:30:00Z"

    Unauthorized:
      description: Unauthorized - authentication required
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            type: "https://policycortex.com/problems/unauthorized"
            title: "Unauthorized"
            status: 401
            detail: "Authentication is required"
            correlationId: "550e8400-e29b-41d4-a716-446655440000"
            timestamp: "2024-01-15T10:30:00Z"

    Forbidden:
      description: Forbidden - insufficient permissions
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            type: "https://policycortex.com/problems/forbidden"
            title: "Forbidden"
            status: 403
            detail: "Insufficient permissions to perform this operation"
            correlationId: "550e8400-e29b-41d4-a716-446655440000"
            timestamp: "2024-01-15T10:30:00Z"

    NotFound:
      description: Not found - resource does not exist
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            type: "https://policycortex.com/problems/not-found"
            title: "Not Found"
            status: 404
            detail: "The requested resource was not found"
            correlationId: "550e8400-e29b-41d4-a716-446655440000"
            timestamp: "2024-01-15T10:30:00Z"

    Conflict:
      description: Conflict - resource already exists
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            type: "https://policycortex.com/problems/conflict"
            title: "Conflict"
            status: 409
            detail: "Resource already exists"
            correlationId: "550e8400-e29b-41d4-a716-446655440000"
            timestamp: "2024-01-15T10:30:00Z"

    TooManyRequests:
      description: Too many requests - rate limit exceeded
      headers:
        Retry-After:
          description: Number of seconds to wait before retrying
          schema:
            type: integer
        X-RateLimit-Limit:
          description: Request limit per hour
          schema:
            type: integer
        X-RateLimit-Remaining:
          description: Remaining requests in current window
          schema:
            type: integer
        X-RateLimit-Reset:
          description: Unix timestamp when rate limit resets
          schema:
            type: integer
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            type: "https://policycortex.com/problems/rate-limit"
            title: "Too Many Requests"
            status: 429
            detail: "Rate limit exceeded"
            correlationId: "550e8400-e29b-41d4-a716-446655440000"
            timestamp: "2024-01-15T10:30:00Z"

    InternalServerError:
      description: Internal server error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            type: "https://policycortex.com/problems/internal-error"
            title: "Internal Server Error"
            status: 500
            detail: "An unexpected error occurred"
            correlationId: "550e8400-e29b-41d4-a716-446655440000"
            timestamp: "2024-01-15T10:30:00Z"

tags:
  - name: Resources
    description: Azure resource management
  - name: Policies
    description: Governance policy management
  - name: Policy Evaluation
    description: Policy evaluation and compliance checking
  - name: Compliance
    description: Compliance status and reporting
  - name: Actions
    description: Remediation action execution
  - name: AI Patent Features
    description: Patented AI-powered governance features
  - name: AI Insights
    description: AI-generated insights and recommendations
  - name: Health
    description: API health monitoring
  - name: Monitoring
    description: System monitoring and metrics

externalDocs:
  description: PolicyCortex Documentation
  url: https://docs.policycortex.com
```

## Core API Endpoints

### Resource Management API

```rust
// core/src/api/resources.rs
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::AppState;

#[derive(Debug, Deserialize)]
pub struct ListResourcesQuery {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub subscription_id: Option<String>,
    pub resource_type: Option<String>,
    pub location: Option<String>,
    pub resource_group_name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub compliance_status: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResourceListResponse {
    pub data: Vec<Resource>,
    pub pagination: PaginationInfo,
}

#[derive(Debug, Serialize)]
pub struct PaginationInfo {
    pub page: u32,
    pub limit: u32,
    pub total_items: u64,
    pub total_pages: u32,
    pub has_next_page: bool,
    pub has_previous_page: bool,
}

// GET /api/v1/resources
pub async fn list_resources(
    Query(params): Query<ListResourcesQuery>,
    State(state): State<AppState>,
) -> Result<Json<ResourceListResponse>, (StatusCode, String)> {
    let page = params.page.unwrap_or(1);
    let limit = params.limit.unwrap_or(20).min(100);
    let offset = (page - 1) * limit;

    let mut query = sqlx::QueryBuilder::new(
        "SELECT r.*, COUNT(*) OVER() as total_count FROM resources r WHERE r.deleted_at IS NULL"
    );

    // Add filters
    if let Some(subscription_id) = &params.subscription_id {
        query.push(" AND r.subscription_id = ");
        query.push_bind(subscription_id);
    }

    if let Some(resource_type) = &params.resource_type {
        query.push(" AND r.type = ");
        query.push_bind(resource_type);
    }

    if let Some(location) = &params.location {
        query.push(" AND r.location = ");
        query.push_bind(location);
    }

    if let Some(resource_group) = &params.resource_group_name {
        query.push(" AND r.resource_group_name = ");
        query.push_bind(resource_group);
    }

    // Tag filtering
    if let Some(tags) = &params.tags {
        for tag in tags {
            if let Some((key, value)) = tag.split_once(':') {
                query.push(" AND r.tags->>");
                query.push_bind(key);
                query.push(" = ");
                query.push_bind(value);
            } else {
                query.push(" AND r.tags ? ");
                query.push_bind(tag);
            }
        }
    }

    // Compliance status filtering
    if let Some(compliance_status) = &params.compliance_status {
        query.push(" AND r.id IN (
            SELECT DISTINCT pe.resource_id 
            FROM policy_evaluations pe 
            WHERE pe.result = ");
        query.push_bind(compliance_status);
        query.push(" AND pe.evaluated_at = (
            SELECT MAX(evaluated_at) 
            FROM policy_evaluations 
            WHERE resource_id = pe.resource_id AND policy_id = pe.policy_id
        ))");
    }

    query.push(" ORDER BY r.updated_at DESC");
    query.push(" LIMIT ");
    query.push_bind(limit);
    query.push(" OFFSET ");
    query.push_bind(offset);

    let rows = query
        .build()
        .fetch_all(&state.database)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if rows.is_empty() {
        return Ok(Json(ResourceListResponse {
            data: vec![],
            pagination: PaginationInfo {
                page,
                limit,
                total_items: 0,
                total_pages: 0,
                has_next_page: false,
                has_previous_page: false,
            },
        }));
    }

    let total_count: i64 = rows[0].get("total_count");
    let total_pages = ((total_count as f64) / (limit as f64)).ceil() as u32;

    let resources: Vec<Resource> = rows
        .into_iter()
        .map(|row| Resource::from_row(&row))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(ResourceListResponse {
        data: resources,
        pagination: PaginationInfo {
            page,
            limit,
            total_items: total_count as u64,
            total_pages,
            has_next_page: page < total_pages,
            has_previous_page: page > 1,
        },
    }))
}

// GET /api/v1/resources/{id}
pub async fn get_resource(
    Path(resource_id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<Resource>, (StatusCode, String)> {
    let resource = sqlx::query_as::<_, Resource>(
        "SELECT * FROM resources WHERE id = $1 AND deleted_at IS NULL"
    )
    .bind(resource_id)
    .fetch_optional(&state.database)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .ok_or((StatusCode::NOT_FOUND, "Resource not found".to_string()))?;

    Ok(Json(resource))
}

// POST /api/v1/resources
pub async fn create_resource(
    State(state): State<AppState>,
    Json(payload): Json<CreateResourceRequest>,
) -> Result<(StatusCode, Json<Resource>), (StatusCode, String)> {
    // Check if resource already exists
    let existing = sqlx::query!(
        "SELECT id FROM resources WHERE azure_resource_id = $1",
        payload.azure_resource_id
    )
    .fetch_optional(&state.database)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if existing.is_some() {
        return Err((StatusCode::CONFLICT, "Resource already exists".to_string()));
    }

    let resource_id = Uuid::new_v4();
    let now = chrono::Utc::now();

    let resource = sqlx::query_as::<_, Resource>(
        r#"
        INSERT INTO resources (
            id, tenant_id, azure_resource_id, subscription_id, resource_group_name,
            name, type, location, properties, tags, created_at, updated_at, synced_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING *
        "#
    )
    .bind(resource_id)
    .bind(state.tenant_id) // From authenticated context
    .bind(&payload.azure_resource_id)
    .bind(&payload.subscription_id)
    .bind(&payload.resource_group_name)
    .bind(&payload.name)
    .bind(&payload.resource_type)
    .bind(&payload.location)
    .bind(&payload.properties)
    .bind(&payload.tags)
    .bind(now)
    .bind(now)
    .bind(now)
    .fetch_one(&state.database)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Trigger policy evaluations for new resource
    tokio::spawn(async move {
        if let Err(e) = trigger_policy_evaluations(resource_id, &state).await {
            eprintln!("Failed to trigger policy evaluations: {}", e);
        }
    });

    Ok((StatusCode::CREATED, Json(resource)))
}

// PUT /api/v1/resources/{id}
pub async fn update_resource(
    Path(resource_id): Path<Uuid>,
    State(state): State<AppState>,
    Json(payload): Json<UpdateResourceRequest>,
) -> Result<Json<Resource>, (StatusCode, String)> {
    let mut query = sqlx::QueryBuilder::new("UPDATE resources SET updated_at = NOW()");
    let mut has_updates = false;

    if let Some(name) = &payload.name {
        query.push(", name = ");
        query.push_bind(name);
        has_updates = true;
    }

    if let Some(properties) = &payload.properties {
        query.push(", properties = properties || ");
        query.push_bind(properties);
        has_updates = true;
    }

    if let Some(tags) = &payload.tags {
        query.push(", tags = tags || ");
        query.push_bind(tags);
        has_updates = true;
    }

    if !has_updates {
        return Err((StatusCode::BAD_REQUEST, "No updates provided".to_string()));
    }

    query.push(" WHERE id = ");
    query.push_bind(resource_id);
    query.push(" AND deleted_at IS NULL RETURNING *");

    let resource = query
        .build_query_as::<Resource>()
        .fetch_optional(&state.database)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "Resource not found".to_string()))?;

    Ok(Json(resource))
}

// DELETE /api/v1/resources/{id}
pub async fn delete_resource(
    Path(resource_id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<StatusCode, (StatusCode, String)> {
    let affected = sqlx::query!(
        "UPDATE resources SET deleted_at = NOW() WHERE id = $1 AND deleted_at IS NULL",
        resource_id
    )
    .execute(&state.database)
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .rows_affected();

    if affected == 0 {
        return Err((StatusCode::NOT_FOUND, "Resource not found".to_string()));
    }

    Ok(StatusCode::NO_CONTENT)
}

async fn trigger_policy_evaluations(
    resource_id: Uuid,
    state: &AppState,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get applicable policies for this resource
    let policies = sqlx::query!(
        "SELECT id FROM policies WHERE enabled = true AND deleted_at IS NULL"
    )
    .fetch_all(&state.database)
    .await?;

    // Trigger evaluation for each policy
    for policy in policies {
        let evaluation_id = Uuid::new_v4();
        sqlx::query!(
            r#"
            INSERT INTO policy_evaluations (
                id, tenant_id, policy_id, resource_id, status, created_at
            ) VALUES ($1, $2, $3, $4, 'pending', NOW())
            "#,
            evaluation_id,
            state.tenant_id,
            policy.id,
            resource_id
        )
        .execute(&state.database)
        .await?;

        // Send to evaluation queue
        // Implementation would depend on your message queue system
    }

    Ok(())
}

#[derive(Debug, Deserialize)]
pub struct CreateResourceRequest {
    pub azure_resource_id: String,
    pub subscription_id: String,
    pub resource_group_name: String,
    pub name: String,
    pub resource_type: String,
    pub location: String,
    pub properties: serde_json::Value,
    pub tags: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct UpdateResourceRequest {
    pub name: Option<String>,
    pub properties: Option<serde_json::Value>,
    pub tags: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct Resource {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub azure_resource_id: String,
    pub subscription_id: String,
    pub resource_group_name: String,
    pub name: String,
    pub r#type: String,
    pub location: String,
    pub properties: serde_json::Value,
    pub tags: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub synced_at: Option<chrono::DateTime<chrono::Utc>>,
}
```

This comprehensive API specification provides complete OpenAPI documentation for PolicyCortex's REST APIs, including all patent features, detailed schemas, authentication, error handling, and example implementations. The specification supports automatic client generation and provides a complete reference for API consumers.