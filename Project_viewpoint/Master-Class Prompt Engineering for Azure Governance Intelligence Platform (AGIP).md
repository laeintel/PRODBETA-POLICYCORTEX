# Master-Class Prompt Engineering for Azure Governance Intelligence Platform (policyCortex)

## Executive Summary

This document provides comprehensive, master-class prompt engineering instructions for AI Claude to build the Azure Governance Intelligence Platform (policyCortex). The prompts are meticulously designed to minimize cost while maximizing output quality, ensuring efficient development of a production-ready AI-powered Azure governance solution.

The prompt engineering approach follows advanced techniques including chain-of-thought reasoning, role-based prompting, modular prompt architecture, and iterative refinement strategies. Each prompt is optimized for specific development phases and includes comprehensive error handling, edge case management, and quality assurance mechanisms.

## 1. Prompt Engineering Principles and Optimization Strategy

### 1.1 Cost Optimization Principles

**Token Efficiency Maximization**
The prompt engineering strategy prioritizes token efficiency through several key techniques. First, we employ concise yet comprehensive language that conveys maximum information density per token. This involves using technical terminology appropriately while avoiding unnecessary verbosity. Second, we implement modular prompt structures that allow for reusable components, reducing the need to repeat context across multiple interactions. Third, we utilize progressive disclosure techniques where complex requirements are broken down into manageable chunks that build upon previous outputs.

**Context Window Optimization**
Given the constraints of AI model context windows, the prompts are designed to maximize the effective use of available context space. This includes strategic use of reference materials, efficient prompt chaining techniques, and intelligent context management that prioritizes the most critical information for each development phase. The prompts include explicit instructions for maintaining context continuity across multiple interactions while minimizing redundant information transfer.

**Iterative Development Approach**
The prompt structure supports iterative development cycles that allow for incremental progress validation and course correction without requiring complete regeneration of previous work. This approach significantly reduces token consumption by enabling focused refinements rather than wholesale recreations of code or documentation.

### 1.2 Quality Maximization Strategies

**Comprehensive Requirement Specification**
Each prompt includes detailed requirement specifications that eliminate ambiguity and reduce the likelihood of misinterpretation. This includes explicit functional requirements, non-functional requirements, technical constraints, and quality criteria. The prompts specify expected input formats, output formats, error handling requirements, and performance expectations.

**Multi-Layer Validation Framework**
The prompts incorporate multiple validation layers including syntax validation, functional validation, integration validation, and business logic validation. Each layer includes specific criteria and test cases that ensure the generated code meets production-quality standards.

**Best Practice Integration**
The prompts embed industry best practices for software development, cloud architecture, AI/ML implementation, and Azure-specific development patterns. This ensures that generated code follows established conventions and maintains high quality standards without requiring extensive post-generation refinement.

### 1.3 Modular Prompt Architecture

**Component-Based Prompt Design**
The prompt architecture follows a component-based design where individual prompts focus on specific functional areas while maintaining clear interfaces with other components. This approach enables parallel development, easier maintenance, and more efficient debugging when issues arise.

**Reusable Prompt Templates**
Common patterns and structures are abstracted into reusable prompt templates that can be instantiated with specific parameters for different use cases. This reduces prompt development overhead and ensures consistency across the entire platform implementation.

**Progressive Complexity Management**
The prompts are structured to handle progressive complexity, starting with foundational components and gradually building more sophisticated features. This approach ensures that each development phase has a solid foundation and reduces the risk of architectural issues that could require significant rework.

## 2. Phase 1: Infrastructure and Foundation Setup

### 2.1 Azure Infrastructure Deployment Prompt

```
You are an expert Azure cloud architect and DevOps engineer tasked with implementing the infrastructure foundation for the Azure Governance Intelligence Platform (policyCortex). Your goal is to create production-ready infrastructure using Infrastructure as Code (IaC) principles.

CONTEXT:
policyCortex is an AI-powered Azure governance platform that provides intelligent policy management, RBAC optimization, network security analysis, and cost optimization. The platform requires scalable, secure, and cost-optimized infrastructure that can handle real-time data processing, machine learning workloads, and conversational AI interactions.

REQUIREMENTS:
1. Create comprehensive Terraform modules for all required Azure services
2. Implement multi-environment support (dev, staging, production)
3. Follow Azure Well-Architected Framework principles
4. Ensure security best practices and compliance requirements
5. Optimize for cost efficiency while maintaining performance
6. Include monitoring, logging, and alerting configurations
7. Implement disaster recovery and backup strategies

TECHNICAL SPECIFICATIONS:
- Azure Kubernetes Service (AKS) with auto-scaling node pools
- Azure Machine Learning workspace with compute clusters
- Azure Data Lake Storage Gen2 with hierarchical namespace
- Azure SQL Database with elastic pools
- Azure Cosmos DB with multi-region replication
- Azure Key Vault for secrets management
- Azure Container Registry for container images
- Azure Application Gateway with WAF
- Azure Monitor and Log Analytics workspace
- Azure Service Bus for message queuing

DELIVERABLES:
1. Complete Terraform module structure with all required resources
2. Environment-specific variable files (dev, staging, prod)
3. CI/CD pipeline configuration for infrastructure deployment
4. Security configuration including RBAC, network security groups, and private endpoints
5. Monitoring and alerting configuration
6. Documentation for infrastructure management and troubleshooting

CONSTRAINTS:
- Must support multi-tenant architecture for MSP scenarios
- Infrastructure costs should be optimized for different usage tiers
- All resources must be deployed in specified Azure regions
- Compliance with SOC 2, ISO 27001, and GDPR requirements
- Support for both Azure Commercial and Azure Government clouds

OUTPUT FORMAT:
Provide the complete infrastructure code organized in the following structure:
```
terraform/
├── modules/
│   ├── aks-cluster/
│   ├── data-services/
│   ├── ai-services/
│   ├── networking/
│   └── monitoring/
├── environments/
│   ├── dev/
│   ├── staging/
│   └── prod/
└── shared/
```

Include detailed comments explaining design decisions, security considerations, and optimization strategies. Provide deployment instructions and troubleshooting guidance.

QUALITY CRITERIA:
- All Terraform code must pass validation and formatting checks
- Security configurations must follow Azure security baselines
- Cost optimization features must be properly configured
- All resources must have appropriate tags for governance
- Documentation must be comprehensive and actionable

Begin with the main Terraform configuration and module structure, then provide each module implementation with detailed explanations.
```

### 2.2 Core Application Architecture Prompt

```
You are a senior software architect specializing in cloud-native applications and AI/ML systems. Your task is to implement the core application architecture for the Azure Governance Intelligence Platform (policyCortex) using microservices patterns and modern development practices.

CONTEXT:
policyCortex requires a scalable, maintainable application architecture that can handle multiple concurrent users, process large volumes of Azure governance data, and provide real-time AI-powered insights. The architecture must support both synchronous and asynchronous processing patterns.

ARCHITECTURAL REQUIREMENTS:
1. Microservices architecture with clear service boundaries
2. Event-driven communication patterns using Azure Service Bus
3. API Gateway pattern for external communication
4. CQRS pattern for data operations where appropriate
5. Circuit breaker and retry patterns for resilience
6. Comprehensive logging, monitoring, and distributed tracing
7. Authentication and authorization using Azure AD
8. Multi-tenant support with data isolation

SERVICE DESIGN:
Design and implement the following core services:

1. **API Gateway Service**
   - Request routing and load balancing
   - Authentication and authorization
   - Rate limiting and throttling
   - Request/response transformation
   - API versioning support

2. **Azure Integration Service**
   - Azure Policy API integration
   - Azure RBAC API integration
   - Azure Network API integration
   - Azure Cost Management API integration
   - Caching and rate limiting
   - Error handling and retry logic

3. **AI Engine Service**
   - Machine learning model serving
   - Natural language processing
   - Predictive analytics
   - Anomaly detection
   - Model training orchestration

4. **Data Processing Service**
   - Real-time data ingestion
   - Batch data processing
   - Data transformation and enrichment
   - Data quality validation
   - Event stream processing

5. **Conversation Service**
   - Natural language query processing
   - Context management
   - Response generation
   - Multi-turn conversation handling
   - Intent recognition and entity extraction

6. **Notification Service**
   - Alert generation and delivery
   - Email and SMS notifications
   - Webhook integrations
   - Notification preferences management
   - Delivery status tracking

TECHNICAL STACK:
- Programming Language: Python 3.11+ with FastAPI framework
- Container Runtime: Docker with multi-stage builds
- Orchestration: Kubernetes with Helm charts
- Database: Azure SQL Database and Azure Cosmos DB
- Message Queue: Azure Service Bus
- Caching: Redis for session and application caching
- Monitoring: Prometheus, Grafana, and Azure Monitor
- Logging: Structured logging with correlation IDs

DELIVERABLES:
1. Complete microservices implementation with FastAPI
2. Docker configurations and Kubernetes manifests
3. Service-to-service communication patterns
4. Authentication and authorization implementation
5. Error handling and resilience patterns
6. Comprehensive testing suite (unit, integration, end-to-end)
7. API documentation with OpenAPI specifications
8. Deployment and scaling guidelines

QUALITY REQUIREMENTS:
- All services must achieve 99.9% uptime SLA
- API response times must be under 200ms for 95th percentile
- Services must handle graceful degradation under load
- All code must have minimum 80% test coverage
- Security vulnerabilities must be addressed (OWASP Top 10)
- Performance must scale linearly with resource allocation

OUTPUT FORMAT:
Provide the complete application code organized by service, including:
- Service implementation with all endpoints
- Data models and schemas
- Business logic and algorithms
- Integration patterns and communication protocols
- Configuration management
- Testing strategies and test implementations

Begin with the overall architecture overview, then implement each service with detailed explanations of design decisions and implementation patterns.
```

## 3. Phase 2: AI Engine Implementation

### 3.1 Machine Learning Models Development Prompt

```
You are a senior machine learning engineer and data scientist with expertise in cloud governance, time series forecasting, and natural language processing. Your task is to implement the complete AI engine for the Azure Governance Intelligence Platform (policyCortex).

CONTEXT:
policyCortex requires sophisticated AI models that can analyze Azure governance data, predict future trends, detect anomalies, and provide intelligent recommendations. The models must be production-ready, scalable, and capable of continuous learning from new data.

ML MODEL REQUIREMENTS:

1. **Policy Compliance Prediction Model**
   - Predict future policy compliance states
   - Identify resources at risk of non-compliance
   - Recommend preventive actions
   - Handle time series data with seasonal patterns
   - Support multiple policy types and organizational contexts

2. **Cost Optimization Model**
   - Forecast future Azure costs with confidence intervals
   - Identify cost optimization opportunities
   - Recommend resource right-sizing actions
   - Analyze usage patterns and trends
   - Support multiple cost optimization strategies

3. **RBAC Analysis Model**
   - Detect anomalous access patterns
   - Identify over-privileged accounts
   - Recommend role optimizations
   - Analyze access request patterns
   - Support principle of least privilege enforcement

4. **Network Security Model**
   - Analyze network traffic patterns
   - Detect security anomalies and threats
   - Recommend network security improvements
   - Identify optimization opportunities
   - Support real-time threat detection

5. **Natural Language Processing Model**
   - Process governance-related queries
   - Extract intent and entities from user input
   - Generate contextual responses
   - Support multi-turn conversations
   - Handle domain-specific terminology

TECHNICAL SPECIFICATIONS:
- Framework: PyTorch and Scikit-learn for model development
- MLOps: Azure Machine Learning for model lifecycle management
- Data Processing: Pandas, NumPy for data manipulation
- Feature Engineering: Automated feature selection and engineering
- Model Serving: FastAPI with async endpoints
- Model Monitoring: Continuous performance monitoring and drift detection
- Scalability: Support for distributed training and inference

MODEL ARCHITECTURE REQUIREMENTS:
1. Ensemble methods combining multiple algorithms
2. Time series models (LSTM, ARIMA, Prophet) for forecasting
3. Anomaly detection using isolation forests and autoencoders
4. Transformer models for natural language processing
5. Graph neural networks for network analysis
6. Reinforcement learning for optimization recommendations

DATA PIPELINE REQUIREMENTS:
1. Real-time data ingestion from Azure APIs
2. Data validation and quality checks
3. Feature engineering and transformation
4. Model training automation
5. A/B testing framework for model comparison
6. Automated model deployment and rollback

DELIVERABLES:
1. Complete ML model implementations with training pipelines
2. Feature engineering and data preprocessing modules
3. Model evaluation and validation frameworks
4. Real-time inference endpoints
5. Model monitoring and alerting systems
6. Automated retraining and deployment pipelines
7. Comprehensive model documentation and performance metrics

PERFORMANCE REQUIREMENTS:
- Model inference latency < 100ms for 95th percentile
- Model accuracy > 85% for classification tasks
- Forecast accuracy within 10% MAPE for cost predictions
- Support for 1000+ concurrent inference requests
- Model training completion within 4 hours for full dataset
- Automated model updates without service interruption

QUALITY CRITERIA:
- All models must pass statistical validation tests
- Cross-validation scores must meet minimum thresholds
- Model explanability and interpretability requirements
- Bias detection and fairness validation
- Comprehensive unit and integration testing
- Performance benchmarking against baseline models

OUTPUT FORMAT:
Provide complete ML implementation including:
```
ml_engine/
├── models/
│   ├── compliance_predictor/
│   ├── cost_optimizer/
│   ├── rbac_analyzer/
│   ├── network_security/
│   └── nlp_processor/
├── data_pipeline/
├── training/
├── inference/
├── monitoring/
└── utils/
```

Include detailed model architectures, training procedures, evaluation metrics, and deployment strategies. Provide comprehensive documentation for model maintenance and improvement.

Begin with the overall ML architecture and data flow, then implement each model with detailed explanations of algorithms, feature engineering, and optimization strategies.
```

### 3.2 Natural Language Processing Implementation Prompt

```
You are an expert NLP engineer and conversational AI specialist. Your task is to implement a sophisticated natural language processing system for the Azure Governance Intelligence Platform (policyCortex) that enables natural language interaction with complex Azure governance data and operations.

CONTEXT:
policyCortex requires advanced NLP capabilities that can understand governance-specific queries, maintain conversation context, extract relevant entities, and generate intelligent responses. The system must handle technical terminology, multi-turn conversations, and provide explanations suitable for different user expertise levels.

NLP SYSTEM REQUIREMENTS:

1. **Intent Recognition and Classification**
   - Classify user queries into governance domains (Policy, RBAC, Network, Cost)
   - Support complex, multi-intent queries
   - Handle ambiguous and incomplete queries
   - Provide confidence scores for classifications
   - Support continuous learning from user feedback

2. **Named Entity Recognition (NER)**
   - Extract Azure-specific entities (subscriptions, resource groups, policies, roles)
   - Recognize temporal expressions and cost thresholds
   - Handle nested and overlapping entities
   - Support custom entity types for organizational terminology
   - Provide entity linking to Azure resources

3. **Conversation Management**
   - Maintain context across multi-turn conversations
   - Handle conversation flow and state management
   - Support conversation branching and topic switching
   - Implement conversation memory and history
   - Provide conversation summarization capabilities

4. **Response Generation**
   - Generate contextual, informative responses
   - Adapt language complexity to user expertise level
   - Include actionable recommendations and next steps
   - Support multiple response formats (text, structured data, visualizations)
   - Provide source attribution and confidence indicators

5. **Query Understanding and Transformation**
   - Transform natural language queries into API calls
   - Handle complex filtering and aggregation requests
   - Support temporal queries and comparisons
   - Generate appropriate database queries
   - Validate query feasibility and constraints

TECHNICAL ARCHITECTURE:
- Base Models: Fine-tuned transformer models (BERT, GPT, T5)
- Framework: Hugging Face Transformers with PyTorch
- Deployment: FastAPI with async processing
- Context Storage: Redis for conversation state
- Knowledge Base: Vector database for governance knowledge
- Training Data: Synthetic and real governance conversations

IMPLEMENTATION COMPONENTS:

1. **Intent Classification Pipeline**
   ```python
   class IntentClassifier:
       def __init__(self, model_path: str):
           # Load pre-trained model
           # Initialize tokenizer and configuration
           
       async def classify_intent(self, query: str, context: Dict) -> IntentResult:
           # Tokenize and encode input
           # Generate predictions with confidence scores
           # Apply context-aware post-processing
           # Return structured intent result
   ```

2. **Entity Extraction Pipeline**
   ```python
   class EntityExtractor:
       def __init__(self, ner_model_path: str, entity_linker_path: str):
           # Initialize NER model and entity linker
           
       async def extract_entities(self, query: str, intent: str) -> List[Entity]:
           # Extract named entities
           # Link entities to Azure resources
           # Validate entity relationships
           # Return structured entity list
   ```

3. **Conversation Manager**
   ```python
   class ConversationManager:
       def __init__(self, context_store: Redis):
           # Initialize context storage and state management
           
       async def process_turn(self, user_input: str, session_id: str) -> ConversationTurn:
           # Retrieve conversation context
           # Update conversation state
           # Generate contextual response
           # Store updated context
   ```

4. **Response Generator**
   ```python
   class ResponseGenerator:
       def __init__(self, generation_model: str, knowledge_base: VectorDB):
           # Initialize generation model and knowledge retrieval
           
       async def generate_response(self, intent: IntentResult, entities: List[Entity], 
                                 context: ConversationContext) -> Response:
           # Retrieve relevant knowledge
           # Generate contextual response
           # Include actionable recommendations
           # Format for user presentation
   ```

TRAINING AND FINE-TUNING:
1. Create domain-specific training datasets
2. Fine-tune pre-trained models on governance data
3. Implement active learning for continuous improvement
4. Develop evaluation metrics for conversation quality
5. Create automated testing and validation pipelines

DELIVERABLES:
1. Complete NLP pipeline implementation
2. Fine-tuned models for governance domain
3. Conversation management system
4. Training and evaluation frameworks
5. API endpoints for NLP services
6. Comprehensive testing suite
7. Performance monitoring and analytics

PERFORMANCE REQUIREMENTS:
- Intent classification accuracy > 90%
- Entity extraction F1 score > 85%
- Response generation latency < 500ms
- Conversation context retention across sessions
- Support for 100+ concurrent conversations
- Multi-language support (English, Spanish, French)

QUALITY CRITERIA:
- Responses must be factually accurate and helpful
- Conversation flow must feel natural and intuitive
- Error handling must be graceful and informative
- System must handle edge cases and ambiguous inputs
- Privacy and security requirements must be met
- Comprehensive logging and monitoring

OUTPUT FORMAT:
Provide complete NLP implementation with:
- Model architectures and training procedures
- Data preprocessing and feature engineering
- Conversation flow management
- Response generation strategies
- Evaluation and testing frameworks
- Deployment and scaling configurations

Begin with the overall NLP architecture, then implement each component with detailed explanations of algorithms, training strategies, and optimization techniques.
```

## 4. Phase 3: Azure Service Integration

### 4.1 Azure Policy Integration Prompt

```
You are an expert Azure cloud engineer with deep expertise in Azure Policy, governance automation, and API integration. Your task is to implement comprehensive Azure Policy integration for the Azure Governance Intelligence Platform (policyCortex).

CONTEXT:
policyCortex requires sophisticated integration with Azure Policy APIs to provide intelligent policy management, compliance monitoring, and automated remediation capabilities. The integration must handle large-scale policy operations, real-time compliance monitoring, and intelligent policy optimization.

INTEGRATION REQUIREMENTS:

1. **Policy Data Collection and Analysis**
   - Retrieve policy definitions, assignments, and compliance states
   - Monitor policy evaluation results in real-time
   - Collect policy exemptions and their justifications
   - Track policy changes and their impact over time
   - Aggregate compliance data across multiple subscriptions

2. **Intelligent Policy Management**
   - Analyze policy effectiveness and compliance rates
   - Identify redundant or conflicting policies
   - Recommend policy optimizations and consolidations
   - Generate custom policies based on organizational requirements
   - Implement policy testing and validation frameworks

3. **Automated Compliance Monitoring**
   - Real-time compliance state monitoring
   - Automated compliance reporting and dashboards
   - Proactive compliance risk identification
   - Compliance trend analysis and forecasting
   - Integration with organizational compliance frameworks

4. **Policy Remediation and Automation**
   - Automated remediation for common policy violations
   - Intelligent remediation recommendation engine
   - Bulk policy operations and management
   - Policy deployment automation with approval workflows
   - Rollback and recovery mechanisms for policy changes

TECHNICAL IMPLEMENTATION:

```python
import asyncio
import aiohttp
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class PolicyInsight:
    policy_id: str
    compliance_rate: float
    violation_count: int
    affected_resources: List[str]
    trend_direction: str
    risk_score: float
    recommendations: List[str]

class AzurePolicyIntelligenceEngine:
    def __init__(self, credential: DefaultAzureCredential):
        self.credential = credential
        self.policy_client = AzurePolicyClient(credential)
        self.compliance_analyzer = PolicyComplianceAnalyzer()
        self.remediation_engine = PolicyRemediationEngine()
        self.optimization_engine = PolicyOptimizationEngine()
        
    async def analyze_policy_landscape(self, scope: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of policy landscape for a given scope.
        """
        # Collect all policy data
        policies = await self.policy_client.get_all_policies(scope)
        assignments = await self.policy_client.get_policy_assignments(scope)
        compliance_data = await self.policy_client.get_compliance_states(scope)
        
        # Analyze policy effectiveness
        effectiveness_analysis = self.compliance_analyzer.analyze_effectiveness(
            policies, assignments, compliance_data
        )
        
        # Identify optimization opportunities
        optimization_opportunities = self.optimization_engine.identify_optimizations(
            policies, assignments, compliance_data
        )
        
        # Generate remediation recommendations
        remediation_recommendations = self.remediation_engine.generate_recommendations(
            compliance_data, policies
        )
        
        # Create comprehensive insights
        insights = self.generate_policy_insights(
            effectiveness_analysis, optimization_opportunities, remediation_recommendations
        )
        
        return {
            'scope': scope,
            'analysis_timestamp': datetime.now().isoformat(),
            'policy_summary': {
                'total_policies': len(policies),
                'total_assignments': len(assignments),
                'overall_compliance_rate': effectiveness_analysis['overall_compliance_rate']
            },
            'effectiveness_analysis': effectiveness_analysis,
            'optimization_opportunities': optimization_opportunities,
            'remediation_recommendations': remediation_recommendations,
            'insights': insights
        }
    
    async def implement_intelligent_remediation(self, scope: str, 
                                              auto_remediate: bool = False) -> Dict[str, Any]:
        """
        Implement intelligent policy remediation based on AI analysis.
        """
        # Get current compliance violations
        violations = await self.policy_client.get_compliance_violations(scope)
        
        # Analyze violations and generate remediation plans
        remediation_plans = []
        
        for violation in violations:
            plan = await self.remediation_engine.create_remediation_plan(violation)
            
            if auto_remediate and plan['automation_confidence'] > 0.8:
                # Execute automated remediation
                result = await self.execute_remediation(plan)
                plan['execution_result'] = result
            
            remediation_plans.append(plan)
        
        return {
            'scope': scope,
            'total_violations': len(violations),
            'remediation_plans': remediation_plans,
            'auto_remediated': len([p for p in remediation_plans if p.get('execution_result')])
        }
    
    async def optimize_policy_configuration(self, scope: str) -> Dict[str, Any]:
        """
        Optimize policy configuration based on AI analysis.
        """
        # Analyze current policy configuration
        current_config = await self.analyze_policy_landscape(scope)
        
        # Generate optimization recommendations
        optimizations = self.optimization_engine.generate_optimizations(current_config)
        
        # Create implementation plan
        implementation_plan = self.create_optimization_implementation_plan(optimizations)
        
        return {
            'scope': scope,
            'current_configuration_analysis': current_config,
            'optimization_recommendations': optimizations,
            'implementation_plan': implementation_plan,
            'estimated_improvements': self.calculate_optimization_impact(optimizations)
        }

class PolicyComplianceAnalyzer:
    def analyze_effectiveness(self, policies: List[Dict], assignments: List[Dict], 
                            compliance_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze policy effectiveness using advanced analytics.
        """
        # Calculate compliance rates by policy
        policy_compliance = {}
        
        for policy in policies:
            policy_id = policy['id']
            policy_violations = [c for c in compliance_data 
                               if c['policy_definition_id'] == policy_id and 
                               c['compliance_state'] != 'Compliant']
            
            total_evaluations = len([c for c in compliance_data 
                                   if c['policy_definition_id'] == policy_id])
            
            compliance_rate = 1.0 - (len(policy_violations) / total_evaluations) if total_evaluations > 0 else 1.0
            
            policy_compliance[policy_id] = {
                'compliance_rate': compliance_rate,
                'total_evaluations': total_evaluations,
                'violations': len(policy_violations),
                'policy_name': policy.get('properties', {}).get('displayName', 'Unknown')
            }
        
        # Identify trends and patterns
        overall_compliance_rate = sum(p['compliance_rate'] for p in policy_compliance.values()) / len(policy_compliance) if policy_compliance else 1.0
        
        # Find problematic policies
        problematic_policies = [
            policy_id for policy_id, data in policy_compliance.items()
            if data['compliance_rate'] < 0.8
        ]
        
        return {
            'overall_compliance_rate': overall_compliance_rate,
            'policy_compliance_details': policy_compliance,
            'problematic_policies': problematic_policies,
            'compliance_trends': self.analyze_compliance_trends(compliance_data)
        }
    
    def analyze_compliance_trends(self, compliance_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze compliance trends over time.
        """
        # Group compliance data by date
        from collections import defaultdict
        daily_compliance = defaultdict(list)
        
        for record in compliance_data:
            date = record['timestamp'][:10]  # Extract date part
            daily_compliance[date].append(record['compliance_state'] == 'Compliant')
        
        # Calculate daily compliance rates
        daily_rates = {}
        for date, compliance_states in daily_compliance.items():
            daily_rates[date] = sum(compliance_states) / len(compliance_states)
        
        # Determine trend direction
        if len(daily_rates) >= 2:
            dates = sorted(daily_rates.keys())
            recent_rate = daily_rates[dates[-1]]
            previous_rate = daily_rates[dates[-2]] if len(dates) > 1 else recent_rate
            
            trend_direction = 'improving' if recent_rate > previous_rate else 'declining' if recent_rate < previous_rate else 'stable'
        else:
            trend_direction = 'insufficient_data'
        
        return {
            'daily_compliance_rates': daily_rates,
            'trend_direction': trend_direction,
            'average_compliance_rate': sum(daily_rates.values()) / len(daily_rates) if daily_rates else 0
        }

class PolicyOptimizationEngine:
    def identify_optimizations(self, policies: List[Dict], assignments: List[Dict], 
                             compliance_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Identify policy optimization opportunities using AI analysis.
        """
        optimizations = []
        
        # Find duplicate or overlapping policies
        duplicate_policies = self.find_duplicate_policies(policies)
        if duplicate_policies:
            optimizations.append({
                'type': 'policy_consolidation',
                'description': 'Consolidate duplicate or overlapping policies',
                'affected_policies': duplicate_policies,
                'estimated_impact': 'Reduced management overhead and improved clarity',
                'implementation_complexity': 'Medium'
            })
        
        # Find unused policy assignments
        unused_assignments = self.find_unused_assignments(assignments, compliance_data)
        if unused_assignments:
            optimizations.append({
                'type': 'unused_assignment_cleanup',
                'description': 'Remove unused policy assignments',
                'affected_assignments': unused_assignments,
                'estimated_impact': 'Reduced evaluation overhead and improved performance',
                'implementation_complexity': 'Low'
            })
        
        # Find overly restrictive policies
        restrictive_policies = self.find_overly_restrictive_policies(policies, compliance_data)
        if restrictive_policies:
            optimizations.append({
                'type': 'policy_relaxation',
                'description': 'Adjust overly restrictive policies',
                'affected_policies': restrictive_policies,
                'estimated_impact': 'Improved compliance rates and reduced friction',
                'implementation_complexity': 'High'
            })
        
        return optimizations
    
    def find_duplicate_policies(self, policies: List[Dict]) -> List[Dict[str, Any]]:
        """
        Find duplicate or very similar policies.
        """
        duplicates = []
        
        for i, policy1 in enumerate(policies):
            for j, policy2 in enumerate(policies[i+1:], i+1):
                similarity = self.calculate_policy_similarity(policy1, policy2)
                
                if similarity > 0.8:  # 80% similarity threshold
                    duplicates.append({
                        'policy1': policy1['id'],
                        'policy2': policy2['id'],
                        'similarity_score': similarity,
                        'recommendation': 'Consider consolidating these similar policies'
                    })
        
        return duplicates
    
    def calculate_policy_similarity(self, policy1: Dict, policy2: Dict) -> float:
        """
        Calculate similarity between two policies.
        """
        # Simple similarity calculation based on policy rules
        # In a real implementation, this would use more sophisticated NLP techniques
        
        rules1 = str(policy1.get('properties', {}).get('policyRule', {}))
        rules2 = str(policy2.get('properties', {}).get('policyRule', {}))
        
        # Basic string similarity (would be replaced with semantic similarity)
        common_chars = len(set(rules1) & set(rules2))
        total_chars = len(set(rules1) | set(rules2))
        
        return common_chars / total_chars if total_chars > 0 else 0
```

DELIVERABLES:
1. Complete Azure Policy integration implementation
2. Policy compliance monitoring and analysis system
3. Intelligent policy optimization engine
4. Automated remediation framework
5. Policy testing and validation tools
6. Comprehensive API documentation
7. Performance monitoring and alerting

PERFORMANCE REQUIREMENTS:
- Policy data retrieval latency < 2 seconds for 1000 policies
- Compliance analysis completion < 30 seconds for large subscriptions
- Real-time compliance monitoring with < 5 minute delay
- Support for 10,000+ policy evaluations per minute
- Automated remediation execution < 60 seconds
- 99.9% API availability and reliability

QUALITY CRITERIA:
- All policy operations must maintain audit trails
- Remediation actions must be reversible
- Policy recommendations must be validated before implementation
- Integration must handle Azure API rate limits gracefully
- Error handling must be comprehensive and informative
- Security and compliance requirements must be maintained

OUTPUT FORMAT:
Provide complete implementation including:
- Azure Policy API client with all required operations
- Policy analysis and optimization algorithms
- Automated remediation workflows
- Testing and validation frameworks
- Monitoring and alerting configurations
- Comprehensive documentation and examples

Begin with the overall integration architecture, then implement each component with detailed explanations of Azure Policy concepts, API usage patterns, and optimization strategies.
```

This master-class prompt engineering document provides comprehensive, production-ready prompts for implementing the Azure Governance Intelligence Platform. Each prompt is carefully crafted to maximize output quality while minimizing token consumption, ensuring efficient development of a sophisticated AI-powered governance solution.


### 4.2 Azure RBAC Integration Prompt

```
You are an expert Azure security engineer with deep expertise in Azure RBAC, identity management, and access control automation. Your task is to implement comprehensive Azure RBAC integration for the Azure Governance Intelligence Platform (policyCortex) with advanced AI-powered access analysis and optimization capabilities.

CONTEXT:
policyCortex requires sophisticated RBAC integration that can analyze access patterns, detect security risks, optimize role assignments, and provide intelligent access management recommendations. The system must handle complex organizational structures, multi-tenant scenarios, and compliance requirements.

RBAC INTEGRATION REQUIREMENTS:

1. **Access Pattern Analysis and Intelligence**
   - Analyze user access patterns and behaviors
   - Detect anomalous access activities and potential security risks
   - Identify over-privileged accounts and unused permissions
   - Track access request patterns and approval workflows
   - Generate access analytics and insights dashboards

2. **Intelligent Role Optimization**
   - Recommend optimal role assignments based on actual usage
   - Identify opportunities for custom role creation
   - Suggest role consolidation and simplification
   - Implement principle of least privilege automation
   - Provide role effectiveness scoring and metrics

3. **Automated Access Reviews and Compliance**
   - Automate periodic access reviews with AI-powered recommendations
   - Generate compliance reports for various frameworks (SOX, PCI, GDPR)
   - Track access certification and attestation processes
   - Implement automated access cleanup for inactive accounts
   - Provide audit trails and access history analysis

4. **Proactive Security and Risk Management**
   - Real-time monitoring of privileged access activities
   - Automated detection of privilege escalation attempts
   - Risk scoring for access assignments and changes
   - Integration with security incident response workflows
   - Predictive analysis for access-related security risks

TECHNICAL IMPLEMENTATION:

```python
import asyncio
import aiohttp
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

@dataclass
class AccessRisk:
    principal_id: str
    principal_type: str
    risk_score: float
    risk_factors: List[str]
    recommended_actions: List[str]
    severity: str

@dataclass
class RoleOptimization:
    principal_id: str
    current_roles: List[str]
    recommended_roles: List[str]
    optimization_type: str
    estimated_risk_reduction: float
    implementation_complexity: str

class AzureRBACIntelligenceEngine:
    def __init__(self, credential: DefaultAzureCredential):
        self.credential = credential
        self.rbac_client = AzureRBACClient(credential)
        self.access_analyzer = AccessPatternAnalyzer()
        self.risk_engine = AccessRiskEngine()
        self.optimization_engine = RoleOptimizationEngine()
        self.compliance_manager = AccessComplianceManager()
        
    async def analyze_access_landscape(self, scope: str, analysis_period_days: int = 90) -> Dict[str, Any]:
        """
        Comprehensive analysis of RBAC landscape with AI-powered insights.
        """
        # Collect RBAC data
        role_assignments = await self.rbac_client.get_role_assignments(scope)
        role_definitions = await self.rbac_client.get_role_definitions(scope)
        access_logs = await self.rbac_client.get_access_logs(scope, analysis_period_days)
        
        # Analyze access patterns
        access_patterns = self.access_analyzer.analyze_patterns(
            role_assignments, access_logs, analysis_period_days
        )
        
        # Assess security risks
        security_risks = await self.risk_engine.assess_risks(
            role_assignments, role_definitions, access_logs
        )
        
        # Generate optimization recommendations
        optimizations = self.optimization_engine.generate_optimizations(
            role_assignments, role_definitions, access_patterns
        )
        
        # Compliance analysis
        compliance_status = self.compliance_manager.analyze_compliance(
            role_assignments, role_definitions, access_patterns
        )
        
        return {
            'scope': scope,
            'analysis_period_days': analysis_period_days,
            'analysis_timestamp': datetime.now().isoformat(),
            'rbac_summary': {
                'total_role_assignments': len(role_assignments),
                'unique_principals': len(set(ra.principal_id for ra in role_assignments)),
                'unique_roles': len(set(ra.role_definition_id for ra in role_assignments)),
                'privileged_assignments': len([ra for ra in role_assignments if self.is_privileged_role(ra)])
            },
            'access_patterns': access_patterns,
            'security_risks': security_risks,
            'optimization_recommendations': optimizations,
            'compliance_status': compliance_status
        }
    
    async def implement_intelligent_access_optimization(self, scope: str, 
                                                       auto_implement: bool = False) -> Dict[str, Any]:
        """
        Implement AI-driven access optimization with safety controls.
        """
        # Analyze current access configuration
        analysis = await self.analyze_access_landscape(scope)
        
        # Generate detailed optimization plan
        optimization_plan = self.optimization_engine.create_implementation_plan(
            analysis['optimization_recommendations']
        )
        
        # Implement optimizations with approval workflow
        implementation_results = []
        
        for optimization in optimization_plan:
            if auto_implement and optimization['risk_level'] == 'Low':
                # Execute low-risk optimizations automatically
                result = await self.execute_optimization(optimization)
                implementation_results.append(result)
            else:
                # Queue for manual approval
                approval_request = self.create_approval_request(optimization)
                implementation_results.append(approval_request)
        
        return {
            'scope': scope,
            'optimization_plan': optimization_plan,
            'implementation_results': implementation_results,
            'auto_implemented': len([r for r in implementation_results if r.get('status') == 'completed']),
            'pending_approval': len([r for r in implementation_results if r.get('status') == 'pending_approval'])
        }
    
    async def monitor_access_security(self, scope: str) -> Dict[str, Any]:
        """
        Real-time access security monitoring with AI-powered threat detection.
        """
        # Get recent access activities
        recent_activities = await self.rbac_client.get_recent_access_activities(scope, hours=24)
        
        # Detect anomalous activities
        anomalies = self.risk_engine.detect_access_anomalies(recent_activities)
        
        # Assess current security posture
        security_posture = await self.risk_engine.assess_current_security_posture(scope)
        
        # Generate security alerts
        security_alerts = self.generate_security_alerts(anomalies, security_posture)
        
        return {
            'scope': scope,
            'monitoring_timestamp': datetime.now().isoformat(),
            'recent_activities_count': len(recent_activities),
            'detected_anomalies': anomalies,
            'security_posture': security_posture,
            'security_alerts': security_alerts,
            'recommended_actions': self.generate_security_recommendations(anomalies, security_posture)
        }

class AccessPatternAnalyzer:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    def analyze_patterns(self, role_assignments: List, access_logs: List, 
                        analysis_period_days: int) -> Dict[str, Any]:
        """
        Analyze access patterns using machine learning techniques.
        """
        # Aggregate access data by principal
        principal_patterns = self.aggregate_access_by_principal(access_logs)
        
        # Analyze temporal patterns
        temporal_patterns = self.analyze_temporal_patterns(access_logs)
        
        # Detect usage anomalies
        usage_anomalies = self.detect_usage_anomalies(principal_patterns)
        
        # Analyze role effectiveness
        role_effectiveness = self.analyze_role_effectiveness(role_assignments, access_logs)
        
        return {
            'principal_patterns': principal_patterns,
            'temporal_patterns': temporal_patterns,
            'usage_anomalies': usage_anomalies,
            'role_effectiveness': role_effectiveness,
            'pattern_insights': self.generate_pattern_insights(
                principal_patterns, temporal_patterns, usage_anomalies
            )
        }
    
    def aggregate_access_by_principal(self, access_logs: List) -> Dict[str, Any]:
        """
        Aggregate access patterns by principal for analysis.
        """
        principal_stats = {}
        
        for log in access_logs:
            principal_id = log.get('principal_id')
            if principal_id not in principal_stats:
                principal_stats[principal_id] = {
                    'total_accesses': 0,
                    'unique_resources': set(),
                    'access_times': [],
                    'operations': [],
                    'success_rate': 0,
                    'last_access': None
                }
            
            stats = principal_stats[principal_id]
            stats['total_accesses'] += 1
            stats['unique_resources'].add(log.get('resource_id', ''))
            stats['access_times'].append(log.get('timestamp'))
            stats['operations'].append(log.get('operation'))
            
            if log.get('result') == 'Success':
                stats['success_rate'] += 1
            
            # Update last access time
            access_time = datetime.fromisoformat(log.get('timestamp', ''))
            if not stats['last_access'] or access_time > stats['last_access']:
                stats['last_access'] = access_time
        
        # Calculate final statistics
        for principal_id, stats in principal_stats.items():
            stats['unique_resources'] = len(stats['unique_resources'])
            stats['success_rate'] = stats['success_rate'] / stats['total_accesses'] if stats['total_accesses'] > 0 else 0
            stats['days_since_last_access'] = (datetime.now() - stats['last_access']).days if stats['last_access'] else 999
        
        return principal_stats
    
    def detect_usage_anomalies(self, principal_patterns: Dict) -> List[Dict[str, Any]]:
        """
        Detect anomalous usage patterns using machine learning.
        """
        if len(principal_patterns) < 10:  # Need sufficient data for anomaly detection
            return []
        
        # Prepare features for anomaly detection
        features = []
        principal_ids = []
        
        for principal_id, stats in principal_patterns.items():
            features.append([
                stats['total_accesses'],
                stats['unique_resources'],
                stats['success_rate'],
                stats['days_since_last_access']
            ])
            principal_ids.append(principal_id)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Detect anomalies
        anomaly_scores = self.anomaly_detector.fit_predict(features_scaled)
        
        # Identify anomalous principals
        anomalies = []
        for i, score in enumerate(anomaly_scores):
            if score == -1:  # Anomaly detected
                principal_id = principal_ids[i]
                stats = principal_patterns[principal_id]
                
                anomalies.append({
                    'principal_id': principal_id,
                    'anomaly_type': self.classify_anomaly_type(stats),
                    'anomaly_score': float(self.anomaly_detector.score_samples([features_scaled[i]])[0]),
                    'principal_stats': stats,
                    'recommended_action': self.recommend_anomaly_action(stats)
                })
        
        return anomalies
    
    def classify_anomaly_type(self, stats: Dict) -> str:
        """
        Classify the type of anomaly based on access patterns.
        """
        if stats['days_since_last_access'] > 90:
            return 'inactive_account'
        elif stats['total_accesses'] > 1000:
            return 'high_activity'
        elif stats['success_rate'] < 0.5:
            return 'high_failure_rate'
        elif stats['unique_resources'] > 100:
            return 'broad_access_pattern'
        else:
            return 'general_anomaly'

class AccessRiskEngine:
    def __init__(self):
        self.risk_weights = {
            'privileged_role': 0.3,
            'inactive_account': 0.2,
            'broad_permissions': 0.2,
            'recent_changes': 0.15,
            'external_principal': 0.15
        }
    
    async def assess_risks(self, role_assignments: List, role_definitions: List, 
                          access_logs: List) -> List[AccessRisk]:
        """
        Assess access-related security risks using AI analysis.
        """
        risks = []
        
        # Group assignments by principal
        principal_assignments = {}
        for assignment in role_assignments:
            principal_id = assignment.principal_id
            if principal_id not in principal_assignments:
                principal_assignments[principal_id] = []
            principal_assignments[principal_id].append(assignment)
        
        # Assess risk for each principal
        for principal_id, assignments in principal_assignments.items():
            risk = await self.assess_principal_risk(principal_id, assignments, role_definitions, access_logs)
            if risk.risk_score > 0.5:  # Only include medium to high risk principals
                risks.append(risk)
        
        return sorted(risks, key=lambda x: x.risk_score, reverse=True)
    
    async def assess_principal_risk(self, principal_id: str, assignments: List, 
                                   role_definitions: List, access_logs: List) -> AccessRisk:
        """
        Assess risk for a specific principal.
        """
        risk_factors = []
        risk_score = 0.0
        
        # Check for privileged roles
        privileged_roles = self.identify_privileged_roles(assignments, role_definitions)
        if privileged_roles:
            risk_factors.append(f"Has {len(privileged_roles)} privileged role(s)")
            risk_score += self.risk_weights['privileged_role'] * len(privileged_roles) / 10
        
        # Check for inactive accounts
        principal_logs = [log for log in access_logs if log.get('principal_id') == principal_id]
        if not principal_logs:
            risk_factors.append("No recent access activity")
            risk_score += self.risk_weights['inactive_account']
        else:
            last_access = max(datetime.fromisoformat(log.get('timestamp', '')) for log in principal_logs)
            days_inactive = (datetime.now() - last_access).days
            if days_inactive > 30:
                risk_factors.append(f"Inactive for {days_inactive} days")
                risk_score += self.risk_weights['inactive_account'] * min(days_inactive / 90, 1.0)
        
        # Check for broad permissions
        total_permissions = sum(len(self.get_role_permissions(assignment, role_definitions)) 
                               for assignment in assignments)
        if total_permissions > 50:
            risk_factors.append(f"Has {total_permissions} total permissions")
            risk_score += self.risk_weights['broad_permissions'] * min(total_permissions / 200, 1.0)
        
        # Check for recent role changes
        recent_assignments = [a for a in assignments 
                             if (datetime.now() - a.created_on).days < 7]
        if recent_assignments:
            risk_factors.append(f"{len(recent_assignments)} recent role assignment(s)")
            risk_score += self.risk_weights['recent_changes'] * len(recent_assignments) / 5
        
        # Determine severity
        if risk_score > 0.8:
            severity = 'Critical'
        elif risk_score > 0.6:
            severity = 'High'
        elif risk_score > 0.4:
            severity = 'Medium'
        else:
            severity = 'Low'
        
        # Generate recommendations
        recommendations = self.generate_risk_recommendations(risk_factors, assignments)
        
        return AccessRisk(
            principal_id=principal_id,
            principal_type=assignments[0].principal_type if assignments else 'Unknown',
            risk_score=min(risk_score, 1.0),
            risk_factors=risk_factors,
            recommended_actions=recommendations,
            severity=severity
        )
    
    def identify_privileged_roles(self, assignments: List, role_definitions: List) -> List[str]:
        """
        Identify privileged roles from assignments.
        """
        privileged_role_names = [
            'Owner', 'Contributor', 'User Access Administrator',
            'Security Administrator', 'Global Administrator', 'Privileged Role Administrator'
        ]
        
        privileged_roles = []
        role_def_map = {rd['id']: rd for rd in role_definitions}
        
        for assignment in assignments:
            role_def = role_def_map.get(assignment.role_definition_id)
            if role_def:
                role_name = role_def.get('properties', {}).get('roleName', '')
                if role_name in privileged_role_names:
                    privileged_roles.append(role_name)
                
                # Check for wildcard permissions
                permissions = role_def.get('properties', {}).get('permissions', [])
                for permission in permissions:
                    if '*' in permission.get('actions', []):
                        privileged_roles.append(f"{role_name} (wildcard permissions)")
        
        return privileged_roles
    
    def generate_risk_recommendations(self, risk_factors: List[str], assignments: List) -> List[str]:
        """
        Generate specific recommendations based on identified risk factors.
        """
        recommendations = []
        
        if any('privileged role' in factor for factor in risk_factors):
            recommendations.append("Review necessity of privileged role assignments")
            recommendations.append("Implement just-in-time access for administrative roles")
        
        if any('Inactive' in factor for factor in risk_factors):
            recommendations.append("Remove or suspend inactive account access")
            recommendations.append("Implement automated access review for inactive accounts")
        
        if any('permissions' in factor for factor in risk_factors):
            recommendations.append("Apply principle of least privilege")
            recommendations.append("Consider role consolidation or custom role creation")
        
        if any('recent' in factor for factor in risk_factors):
            recommendations.append("Verify recent role changes are authorized")
            recommendations.append("Implement approval workflow for role assignments")
        
        return recommendations

class RoleOptimizationEngine:
    def generate_optimizations(self, role_assignments: List, role_definitions: List, 
                             access_patterns: Dict) -> List[RoleOptimization]:
        """
        Generate role optimization recommendations using AI analysis.
        """
        optimizations = []
        
        # Group assignments by principal
        principal_assignments = {}
        for assignment in role_assignments:
            principal_id = assignment.principal_id
            if principal_id not in principal_assignments:
                principal_assignments[principal_id] = []
            principal_assignments[principal_id].append(assignment)
        
        # Analyze each principal for optimization opportunities
        for principal_id, assignments in principal_assignments.items():
            optimization = self.analyze_principal_optimization(
                principal_id, assignments, role_definitions, access_patterns
            )
            if optimization:
                optimizations.append(optimization)
        
        return optimizations
    
    def analyze_principal_optimization(self, principal_id: str, assignments: List, 
                                     role_definitions: List, access_patterns: Dict) -> Optional[RoleOptimization]:
        """
        Analyze optimization opportunities for a specific principal.
        """
        current_roles = [self.get_role_name(assignment, role_definitions) for assignment in assignments]
        principal_pattern = access_patterns.get('principal_patterns', {}).get(principal_id, {})
        
        # Determine optimization type and recommendations
        if len(assignments) > 3:
            # Multiple role assignments - consider consolidation
            recommended_roles = self.recommend_consolidated_roles(assignments, role_definitions, principal_pattern)
            optimization_type = 'role_consolidation'
            estimated_risk_reduction = 0.3
            implementation_complexity = 'Medium'
        
        elif self.has_excessive_permissions(assignments, role_definitions, principal_pattern):
            # Over-privileged - recommend right-sizing
            recommended_roles = self.recommend_rightsized_roles(assignments, role_definitions, principal_pattern)
            optimization_type = 'privilege_reduction'
            estimated_risk_reduction = 0.5
            implementation_complexity = 'High'
        
        elif self.has_unused_permissions(assignments, role_definitions, principal_pattern):
            # Unused permissions - recommend cleanup
            recommended_roles = self.recommend_minimal_roles(assignments, role_definitions, principal_pattern)
            optimization_type = 'permission_cleanup'
            estimated_risk_reduction = 0.2
            implementation_complexity = 'Low'
        
        else:
            # No optimization needed
            return None
        
        return RoleOptimization(
            principal_id=principal_id,
            current_roles=current_roles,
            recommended_roles=recommended_roles,
            optimization_type=optimization_type,
            estimated_risk_reduction=estimated_risk_reduction,
            implementation_complexity=implementation_complexity
        )
    
    def recommend_consolidated_roles(self, assignments: List, role_definitions: List, 
                                   access_pattern: Dict) -> List[str]:
        """
        Recommend consolidated roles for principals with multiple assignments.
        """
        # Analyze actual usage patterns to recommend appropriate consolidated roles
        # This is a simplified implementation - real implementation would use more sophisticated analysis
        
        if access_pattern.get('unique_resources', 0) > 20:
            return ['Contributor']  # Broad access needed
        elif access_pattern.get('total_accesses', 0) > 100:
            return ['Reader', 'Specific Contributor Role']  # Active user with specific needs
        else:
            return ['Reader']  # Limited access needed
    
    def has_excessive_permissions(self, assignments: List, role_definitions: List, 
                                access_pattern: Dict) -> bool:
        """
        Determine if principal has excessive permissions based on usage patterns.
        """
        total_permissions = sum(len(self.get_role_permissions(assignment, role_definitions)) 
                               for assignment in assignments)
        actual_usage = access_pattern.get('total_accesses', 0)
        
        # Simple heuristic: if permissions significantly exceed usage, consider excessive
        return total_permissions > 20 and actual_usage < 10
    
    def get_role_permissions(self, assignment, role_definitions: List) -> List[str]:
        """
        Get permissions for a role assignment.
        """
        role_def = next((rd for rd in role_definitions if rd['id'] == assignment.role_definition_id), None)
        if not role_def:
            return []
        
        permissions = []
        for permission in role_def.get('properties', {}).get('permissions', []):
            permissions.extend(permission.get('actions', []))
        
        return permissions
```

DELIVERABLES:
1. Complete Azure RBAC integration with AI-powered analysis
2. Access pattern analysis and anomaly detection system
3. Intelligent role optimization engine
4. Automated access review and compliance framework
5. Real-time security monitoring and alerting
6. Comprehensive testing and validation suite
7. API documentation and integration guides

PERFORMANCE REQUIREMENTS:
- RBAC data analysis completion < 60 seconds for 10,000 role assignments
- Real-time anomaly detection with < 2 minute latency
- Risk assessment processing < 30 seconds for 1,000 principals
- Support for 100,000+ role assignments across multiple tenants
- Optimization recommendations generation < 45 seconds
- 99.9% system availability with automated failover

QUALITY CRITERIA:
- Access risk assessment accuracy > 90%
- Anomaly detection false positive rate < 5%
- Role optimization recommendations must be validated and safe
- All access changes must maintain comprehensive audit trails
- Integration must respect Azure AD rate limits and quotas
- Security and privacy requirements must be strictly maintained

OUTPUT FORMAT:
Provide complete RBAC integration implementation including:
- Azure RBAC API client with comprehensive operations
- AI-powered access analysis and risk assessment algorithms
- Role optimization and recommendation engines
- Automated compliance and review frameworks
- Real-time monitoring and alerting systems
- Testing, validation, and deployment configurations

Begin with the overall RBAC integration architecture, then implement each component with detailed explanations of access control concepts, security best practices, and AI optimization techniques.
```

### 4.3 Deployment and Scaling Strategy Prompt

```
You are an expert DevOps engineer and cloud architect specializing in large-scale Azure deployments, Kubernetes orchestration, and AI/ML system operations. Your task is to implement comprehensive deployment and scaling strategies for the Azure Governance Intelligence Platform (policyCortex).

CONTEXT:
policyCortex is a sophisticated AI-powered platform that requires enterprise-grade deployment, scaling, and operational capabilities. The platform must support multiple deployment scenarios, from single-tenant installations to large-scale multi-tenant SaaS operations, while maintaining high availability, security, and performance.

DEPLOYMENT REQUIREMENTS:

1. **Multi-Environment Deployment Pipeline**
   - Automated CI/CD pipelines for development, staging, and production
   - Infrastructure as Code (IaC) with Terraform and Helm
   - Blue-green and canary deployment strategies
   - Automated testing and validation at each stage
   - Rollback and disaster recovery capabilities

2. **Kubernetes-Native Architecture**
   - Microservices deployment on Azure Kubernetes Service (AKS)
   - Auto-scaling based on CPU, memory, and custom metrics
   - Service mesh implementation for secure communication
   - Ingress controllers with SSL termination and load balancing
   - Persistent storage for stateful components

3. **Multi-Tenant SaaS Architecture**
   - Tenant isolation and data segregation
   - Dynamic tenant provisioning and deprovisioning
   - Resource quotas and limits per tenant
   - Tenant-specific configuration management
   - Billing and usage tracking integration

4. **Monitoring and Observability**
   - Comprehensive monitoring with Prometheus and Grafana
   - Distributed tracing with Jaeger or Azure Application Insights
   - Centralized logging with ELK stack or Azure Monitor
   - Custom metrics for AI model performance
   - Automated alerting and incident response

TECHNICAL IMPLEMENTATION:

```yaml
# Kubernetes Deployment Configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: policyCortex-api-gateway
  namespace: policyCortex-production
  labels:
    app: policyCortex-api-gateway
    version: v1.0.0
    tier: frontend
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: policyCortex-api-gateway
  template:
    metadata:
      labels:
        app: policyCortex-api-gateway
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: policyCortex-api-gateway
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: api-gateway
        image: policyCortexregistry.azurecr.io/policyCortex-api-gateway:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8443
          name: https
        env:
        - name: AZURE_CLIENT_ID
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: client-id
        - name: AZURE_CLIENT_SECRET
          valueFrom:
            secretKeyRef:
              name: azure-credentials
              key: client-secret
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: connection-string
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: tls-certs
          mountPath: /app/certs
          readOnly: true
      volumes:
      - name: config-volume
        configMap:
          name: policyCortex-config
      - name: tls-certs
        secret:
          secretName: policyCortex-tls-certs
      imagePullSecrets:
      - name: acr-credentials

---
apiVersion: v1
kind: Service
metadata:
  name: policyCortex-api-gateway-service
  namespace: policyCortex-production
  labels:
    app: policyCortex-api-gateway
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  - port: 443
    targetPort: 8443
    protocol: TCP
    name: https
  selector:
    app: policyCortex-api-gateway

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: policyCortex-api-gateway-hpa
  namespace: policyCortex-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: policyCortex-api-gateway
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

```python
# Deployment Automation Script
import asyncio
import subprocess
import yaml
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

@dataclass
class DeploymentConfig:
    environment: str
    namespace: str
    image_tag: str
    replicas: int
    resource_limits: Dict[str, str]
    environment_variables: Dict[str, str]

class policyCortexDeploymentManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
    async def deploy_platform(self, environment: str, deployment_strategy: str = "rolling") -> Dict[str, Any]:
        """
        Deploy policyCortex platform to specified environment.
        """
        self.logger.info(f"Starting deployment to {environment} using {deployment_strategy} strategy")
        
        # Load deployment configuration
        config = self.load_deployment_config(environment)
        
        # Validate prerequisites
        await self.validate_prerequisites(config)
        
        # Deploy infrastructure
        infrastructure_result = await self.deploy_infrastructure(config)
        
        # Deploy applications
        application_result = await self.deploy_applications(config, deployment_strategy)
        
        # Run post-deployment validation
        validation_result = await self.validate_deployment(config)
        
        # Update monitoring and alerting
        monitoring_result = await self.configure_monitoring(config)
        
        deployment_result = {
            'environment': environment,
            'deployment_strategy': deployment_strategy,
            'deployment_timestamp': datetime.now().isoformat(),
            'infrastructure_deployment': infrastructure_result,
            'application_deployment': application_result,
            'validation_result': validation_result,
            'monitoring_configuration': monitoring_result,
            'status': 'success' if all([
                infrastructure_result['success'],
                application_result['success'],
                validation_result['success']
            ]) else 'failed'
        }
        
        self.logger.info(f"Deployment completed with status: {deployment_result['status']}")
        return deployment_result
    
    async def deploy_infrastructure(self, config: DeploymentConfig) -> Dict[str, Any]:
        """
        Deploy infrastructure using Terraform.
        """
        self.logger.info("Deploying infrastructure with Terraform")
        
        try:
            # Initialize Terraform
            await self.run_command([
                "terraform", "init",
                f"-backend-config=key={config.environment}/terraform.tfstate"
            ], cwd="terraform/environments/" + config.environment)
            
            # Plan infrastructure changes
            plan_result = await self.run_command([
                "terraform", "plan",
                f"-var-file={config.environment}.tfvars",
                "-out=tfplan"
            ], cwd="terraform/environments/" + config.environment)
            
            # Apply infrastructure changes
            apply_result = await self.run_command([
                "terraform", "apply",
                "-auto-approve",
                "tfplan"
            ], cwd="terraform/environments/" + config.environment)
            
            # Get infrastructure outputs
            outputs = await self.get_terraform_outputs(config.environment)
            
            return {
                'success': True,
                'outputs': outputs,
                'plan_result': plan_result,
                'apply_result': apply_result
            }
            
        except Exception as e:
            self.logger.error(f"Infrastructure deployment failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def deploy_applications(self, config: DeploymentConfig, strategy: str) -> Dict[str, Any]:
        """
        Deploy applications using Kubernetes and Helm.
        """
        self.logger.info(f"Deploying applications using {strategy} strategy")
        
        try:
            # Create namespace if it doesn't exist
            await self.run_command([
                "kubectl", "create", "namespace", config.namespace, "--dry-run=client", "-o", "yaml"
            ])
            await self.run_command([
                "kubectl", "apply", "-f", "-"
            ])
            
            # Deploy applications based on strategy
            if strategy == "blue-green":
                result = await self.deploy_blue_green(config)
            elif strategy == "canary":
                result = await self.deploy_canary(config)
            else:
                result = await self.deploy_rolling(config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Application deployment failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def deploy_rolling(self, config: DeploymentConfig) -> Dict[str, Any]:
        """
        Deploy using rolling update strategy.
        """
        deployment_results = []
        
        # Deploy each service
        services = [
            'api-gateway',
            'azure-integration',
            'ai-engine',
            'data-processing',
            'conversation-service',
            'notification-service'
        ]
        
        for service in services:
            self.logger.info(f"Deploying {service}")
            
            # Update Helm values
            helm_values = self.generate_helm_values(config, service)
            
            # Deploy with Helm
            result = await self.run_command([
                "helm", "upgrade", "--install",
                f"policyCortex-{service}",
                f"./helm-charts/{service}",
                f"--namespace={config.namespace}",
                f"--values={helm_values}",
                "--wait",
                "--timeout=600s"
            ])
            
            deployment_results.append({
                'service': service,
                'result': result,
                'success': result['returncode'] == 0
            })
        
        return {
            'success': all(r['success'] for r in deployment_results),
            'service_results': deployment_results
        }
    
    async def deploy_blue_green(self, config: DeploymentConfig) -> Dict[str, Any]:
        """
        Deploy using blue-green strategy.
        """
        self.logger.info("Implementing blue-green deployment")
        
        # Determine current and new environments
        current_env = await self.get_current_environment(config.namespace)
        new_env = "green" if current_env == "blue" else "blue"
        
        # Deploy to new environment
        new_config = config
        new_config.namespace = f"{config.namespace}-{new_env}"
        
        deployment_result = await self.deploy_rolling(new_config)
        
        if deployment_result['success']:
            # Run smoke tests on new environment
            smoke_test_result = await self.run_smoke_tests(new_config)
            
            if smoke_test_result['success']:
                # Switch traffic to new environment
                await self.switch_traffic(config.namespace, new_env)
                
                # Clean up old environment
                await self.cleanup_old_environment(config.namespace, current_env)
                
                return {
                    'success': True,
                    'strategy': 'blue-green',
                    'switched_from': current_env,
                    'switched_to': new_env,
                    'deployment_result': deployment_result,
                    'smoke_test_result': smoke_test_result
                }
            else:
                # Rollback new environment
                await self.cleanup_old_environment(config.namespace, new_env)
                return {
                    'success': False,
                    'error': 'Smoke tests failed',
                    'smoke_test_result': smoke_test_result
                }
        else:
            return deployment_result
    
    async def validate_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """
        Validate deployment health and functionality.
        """
        self.logger.info("Validating deployment")
        
        validation_results = []
        
        # Check pod health
        pod_health = await self.check_pod_health(config.namespace)
        validation_results.append(pod_health)
        
        # Check service connectivity
        service_connectivity = await self.check_service_connectivity(config.namespace)
        validation_results.append(service_connectivity)
        
        # Run integration tests
        integration_tests = await self.run_integration_tests(config)
        validation_results.append(integration_tests)
        
        # Check AI model endpoints
        model_health = await self.check_ai_model_health(config.namespace)
        validation_results.append(model_health)
        
        return {
            'success': all(r['success'] for r in validation_results),
            'validation_results': validation_results
        }
    
    async def configure_monitoring(self, config: DeploymentConfig) -> Dict[str, Any]:
        """
        Configure monitoring and alerting for the deployment.
        """
        self.logger.info("Configuring monitoring and alerting")
        
        try:
            # Deploy Prometheus monitoring
            prometheus_result = await self.deploy_prometheus(config)
            
            # Deploy Grafana dashboards
            grafana_result = await self.deploy_grafana_dashboards(config)
            
            # Configure alerting rules
            alerting_result = await self.configure_alerting_rules(config)
            
            # Set up log aggregation
            logging_result = await self.configure_logging(config)
            
            return {
                'success': True,
                'prometheus': prometheus_result,
                'grafana': grafana_result,
                'alerting': alerting_result,
                'logging': logging_result
            }
            
        except Exception as e:
            self.logger.error(f"Monitoring configuration failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def run_command(self, command: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        Run shell command asynchronously.
        """
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        stdout, stderr = await process.communicate()
        
        return {
            'command': ' '.join(command),
            'returncode': process.returncode,
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        }

class TenantManager:
    def __init__(self, k8s_client, database_client):
        self.k8s_client = k8s_client
        self.database_client = database_client
        
    async def provision_tenant(self, tenant_id: str, tenant_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provision a new tenant with isolated resources.
        """
        try:
            # Create tenant namespace
            namespace_result = await self.create_tenant_namespace(tenant_id)
            
            # Set up resource quotas
            quota_result = await self.create_resource_quotas(tenant_id, tenant_config)
            
            # Deploy tenant-specific configurations
            config_result = await self.deploy_tenant_config(tenant_id, tenant_config)
            
            # Initialize tenant database
            database_result = await self.initialize_tenant_database(tenant_id)
            
            # Configure tenant monitoring
            monitoring_result = await self.configure_tenant_monitoring(tenant_id)
            
            return {
                'tenant_id': tenant_id,
                'provisioning_timestamp': datetime.now().isoformat(),
                'namespace': namespace_result,
                'quotas': quota_result,
                'configuration': config_result,
                'database': database_result,
                'monitoring': monitoring_result,
                'status': 'provisioned'
            }
            
        except Exception as e:
            # Cleanup partial provisioning
            await self.cleanup_tenant(tenant_id)
            raise e
    
    async def scale_tenant_resources(self, tenant_id: str, scaling_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scale tenant resources based on usage and requirements.
        """
        # Update resource quotas
        quota_update = await self.update_resource_quotas(tenant_id, scaling_config)
        
        # Scale application replicas
        replica_scaling = await self.scale_application_replicas(tenant_id, scaling_config)
        
        # Update database resources
        database_scaling = await self.scale_database_resources(tenant_id, scaling_config)
        
        return {
            'tenant_id': tenant_id,
            'scaling_timestamp': datetime.now().isoformat(),
            'quota_updates': quota_update,
            'replica_scaling': replica_scaling,
            'database_scaling': database_scaling
        }
```

DELIVERABLES:
1. Complete CI/CD pipeline implementation with Azure DevOps
2. Kubernetes deployment manifests and Helm charts
3. Multi-tenant architecture with resource isolation
4. Auto-scaling configurations for all components
5. Monitoring and observability stack deployment
6. Disaster recovery and backup strategies
7. Security hardening and compliance configurations
8. Performance testing and optimization guidelines

PERFORMANCE REQUIREMENTS:
- Deployment pipeline execution < 30 minutes for full platform
- Zero-downtime deployments with < 1 second traffic interruption
- Auto-scaling response time < 60 seconds for load increases
- Support for 1000+ concurrent tenants with resource isolation
- 99.99% platform availability with automated failover
- Recovery time objective (RTO) < 15 minutes for disasters

QUALITY CRITERIA:
- All deployments must pass automated testing and validation
- Infrastructure must be fully defined as code
- Security scanning must be integrated into CI/CD pipeline
- Monitoring must provide comprehensive observability
- Documentation must be complete and actionable
- Disaster recovery procedures must be tested and validated

OUTPUT FORMAT:
Provide complete deployment and scaling implementation including:
- CI/CD pipeline configurations and scripts
- Kubernetes manifests and Helm charts for all services
- Infrastructure as Code (Terraform) modules
- Monitoring and observability configurations
- Multi-tenant provisioning and management systems
- Testing, validation, and quality assurance frameworks

Begin with the overall deployment architecture and strategy, then implement each component with detailed explanations of DevOps best practices, Kubernetes patterns, and scaling strategies.
```

## 5. Phase 4: Testing and Quality Assurance

### 5.1 Comprehensive Testing Strategy Prompt

```
You are an expert software quality engineer and test automation specialist with deep expertise in testing AI/ML systems, cloud platforms, and enterprise applications. Your task is to implement a comprehensive testing strategy for the Azure Governance Intelligence Platform (policyCortex) that ensures production-ready quality and reliability.

CONTEXT:
policyCortex is a complex AI-powered platform that requires sophisticated testing approaches covering functional testing, AI model validation, performance testing, security testing, and integration testing. The testing strategy must ensure reliability, accuracy, and security across all platform components.

TESTING REQUIREMENTS:

1. **AI Model Testing and Validation**
   - Model accuracy and performance validation
   - Bias detection and fairness testing
   - Model drift monitoring and alerting
   - A/B testing framework for model comparison
   - Adversarial testing for robustness

2. **Functional and Integration Testing**
   - End-to-end workflow testing
   - API testing and contract validation
   - Database integration testing
   - Azure service integration testing
   - Multi-tenant functionality testing

3. **Performance and Load Testing**
   - Scalability testing under various load conditions
   - Stress testing for breaking point identification
   - Latency and throughput optimization
   - Resource utilization monitoring
   - Capacity planning validation

4. **Security and Compliance Testing**
   - Penetration testing and vulnerability assessment
   - Authentication and authorization testing
   - Data privacy and encryption validation
   - Compliance framework testing (SOC 2, GDPR, etc.)
   - Security incident response testing

TECHNICAL IMPLEMENTATION:

```python
import pytest
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import requests
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import logging

@dataclass
class TestResult:
    test_name: str
    status: str
    execution_time: float
    details: Dict[str, Any]
    timestamp: datetime

class policyCortexTestSuite:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.test_results = []
        
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite covering all aspects of policyCortex.
        """
        self.logger.info("Starting comprehensive test suite execution")
        
        # AI Model Testing
        model_test_results = await self.run_ai_model_tests()
        
        # Functional Testing
        functional_test_results = await self.run_functional_tests()
        
        # Integration Testing
        integration_test_results = await self.run_integration_tests()
        
        # Performance Testing
        performance_test_results = await self.run_performance_tests()
        
        # Security Testing
        security_test_results = await self.run_security_tests()
        
        # Compliance Testing
        compliance_test_results = await self.run_compliance_tests()
        
        # Generate comprehensive report
        test_report = self.generate_test_report([
            model_test_results,
            functional_test_results,
            integration_test_results,
            performance_test_results,
            security_test_results,
            compliance_test_results
        ])
        
        return test_report
    
    async def run_ai_model_tests(self) -> Dict[str, Any]:
        """
        Comprehensive AI model testing and validation.
        """
        self.logger.info("Running AI model tests")
        
        model_tests = [
            self.test_policy_compliance_model(),
            self.test_cost_optimization_model(),
            self.test_rbac_analysis_model(),
            self.test_network_security_model(),
            self.test_nlp_model(),
            self.test_model_bias_and_fairness(),
            self.test_model_robustness(),
            self.test_model_drift_detection()
        ]
        
        results = await asyncio.gather(*model_tests, return_exceptions=True)
        
        return {
            'category': 'ai_model_testing',
            'total_tests': len(model_tests),
            'passed': len([r for r in results if isinstance(r, TestResult) and r.status == 'passed']),
            'failed': len([r for r in results if isinstance(r, TestResult) and r.status == 'failed']),
            'errors': len([r for r in results if isinstance(r, Exception)]),
            'test_results': [r for r in results if isinstance(r, TestResult)],
            'execution_time': sum(r.execution_time for r in results if isinstance(r, TestResult))
        }
    
    async def test_policy_compliance_model(self) -> TestResult:
        """
        Test policy compliance prediction model.
        """
        start_time = datetime.now()
        
        try:
            # Load test data
            test_data = self.load_test_data('policy_compliance')
            X_test, y_test = test_data['features'], test_data['labels']
            
            # Load trained model
            model = self.load_model('policy_compliance_predictor')
            
            # Generate predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            f1 = f1_score(y_test, predictions, average='weighted')
            
            # Validate performance thresholds
            performance_thresholds = {
                'accuracy': 0.85,
                'precision': 0.80,
                'recall': 0.80,
                'f1': 0.80
            }
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Check if all metrics meet thresholds
            passed = all(metrics[metric] >= threshold for metric, threshold in performance_thresholds.items())
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name='policy_compliance_model_accuracy',
                status='passed' if passed else 'failed',
                execution_time=execution_time,
                details={
                    'metrics': metrics,
                    'thresholds': performance_thresholds,
                    'test_samples': len(X_test),
                    'model_version': model.version if hasattr(model, 'version') else 'unknown'
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name='policy_compliance_model_accuracy',
                status='error',
                execution_time=execution_time,
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    async def test_model_bias_and_fairness(self) -> TestResult:
        """
        Test AI models for bias and fairness across different groups.
        """
        start_time = datetime.now()
        
        try:
            # Load test data with demographic information
            test_data = self.load_test_data('bias_fairness')
            
            bias_results = {}
            
            # Test each model for bias
            models = ['policy_compliance', 'cost_optimization', 'rbac_analysis']
            
            for model_name in models:
                model = self.load_model(f'{model_name}_predictor')
                
                # Test for demographic parity
                demographic_parity = self.test_demographic_parity(model, test_data)
                
                # Test for equalized odds
                equalized_odds = self.test_equalized_odds(model, test_data)
                
                # Test for individual fairness
                individual_fairness = self.test_individual_fairness(model, test_data)
                
                bias_results[model_name] = {
                    'demographic_parity': demographic_parity,
                    'equalized_odds': equalized_odds,
                    'individual_fairness': individual_fairness
                }
            
            # Determine overall bias status
            bias_threshold = 0.1  # Maximum acceptable bias
            overall_bias = max(
                max(result['demographic_parity'], result['equalized_odds'], result['individual_fairness'])
                for result in bias_results.values()
            )
            
            passed = overall_bias <= bias_threshold
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name='model_bias_and_fairness',
                status='passed' if passed else 'failed',
                execution_time=execution_time,
                details={
                    'bias_results': bias_results,
                    'overall_bias': overall_bias,
                    'bias_threshold': bias_threshold,
                    'recommendation': 'Models pass fairness criteria' if passed else 'Models require bias mitigation'
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name='model_bias_and_fairness',
                status='error',
                execution_time=execution_time,
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    async def run_performance_tests(self) -> Dict[str, Any]:
        """
        Comprehensive performance testing suite.
        """
        self.logger.info("Running performance tests")
        
        performance_tests = [
            self.test_api_response_times(),
            self.test_concurrent_user_load(),
            self.test_database_performance(),
            self.test_ai_model_inference_speed(),
            self.test_memory_usage(),
            self.test_cpu_utilization(),
            self.test_scalability_limits()
        ]
        
        results = await asyncio.gather(*performance_tests, return_exceptions=True)
        
        return {
            'category': 'performance_testing',
            'total_tests': len(performance_tests),
            'passed': len([r for r in results if isinstance(r, TestResult) and r.status == 'passed']),
            'failed': len([r for r in results if isinstance(r, TestResult) and r.status == 'failed']),
            'errors': len([r for r in results if isinstance(r, Exception)]),
            'test_results': [r for r in results if isinstance(r, TestResult)],
            'execution_time': sum(r.execution_time for r in results if isinstance(r, TestResult))
        }
    
    async def test_api_response_times(self) -> TestResult:
        """
        Test API response times under normal load.
        """
        start_time = datetime.now()
        
        try:
            api_endpoints = [
                '/api/v1/policy/analyze',
                '/api/v1/rbac/analyze',
                '/api/v1/cost/optimize',
                '/api/v1/network/analyze',
                '/api/v1/conversation/query'
            ]
            
            response_times = {}
            
            for endpoint in api_endpoints:
                # Test multiple requests to get average response time
                times = []
                for _ in range(10):
                    request_start = datetime.now()
                    response = await self.make_api_request(endpoint)
                    request_time = (datetime.now() - request_start).total_seconds() * 1000  # Convert to ms
                    times.append(request_time)
                
                response_times[endpoint] = {
                    'average': np.mean(times),
                    'p95': np.percentile(times, 95),
                    'p99': np.percentile(times, 99),
                    'max': np.max(times)
                }
            
            # Check performance thresholds
            performance_thresholds = {
                'average': 200,  # 200ms average
                'p95': 500,      # 500ms 95th percentile
                'p99': 1000      # 1000ms 99th percentile
            }
            
            passed = all(
                all(response_times[endpoint][metric] <= threshold 
                    for metric, threshold in performance_thresholds.items())
                for endpoint in api_endpoints
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name='api_response_times',
                status='passed' if passed else 'failed',
                execution_time=execution_time,
                details={
                    'response_times': response_times,
                    'thresholds': performance_thresholds,
                    'endpoints_tested': len(api_endpoints)
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name='api_response_times',
                status='error',
                execution_time=execution_time,
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    async def test_concurrent_user_load(self) -> TestResult:
        """
        Test system performance under concurrent user load.
        """
        start_time = datetime.now()
        
        try:
            # Simulate concurrent users
            concurrent_users = [50, 100, 200, 500, 1000]
            load_test_results = {}
            
            for user_count in concurrent_users:
                self.logger.info(f"Testing with {user_count} concurrent users")
                
                # Create concurrent tasks
                tasks = []
                for _ in range(user_count):
                    task = self.simulate_user_session()
                    tasks.append(task)
                
                # Execute concurrent requests
                session_start = datetime.now()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                session_duration = (datetime.now() - session_start).total_seconds()
                
                # Analyze results
                successful_requests = len([r for r in results if not isinstance(r, Exception)])
                failed_requests = len([r for r in results if isinstance(r, Exception)])
                success_rate = successful_requests / len(results) * 100
                
                load_test_results[user_count] = {
                    'successful_requests': successful_requests,
                    'failed_requests': failed_requests,
                    'success_rate': success_rate,
                    'duration': session_duration,
                    'requests_per_second': len(results) / session_duration
                }
            
            # Determine if load test passed
            min_success_rate = 95  # 95% success rate required
            passed = all(result['success_rate'] >= min_success_rate for result in load_test_results.values())
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name='concurrent_user_load',
                status='passed' if passed else 'failed',
                execution_time=execution_time,
                details={
                    'load_test_results': load_test_results,
                    'min_success_rate': min_success_rate,
                    'max_concurrent_users': max(concurrent_users)
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name='concurrent_user_load',
                status='error',
                execution_time=execution_time,
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    async def run_security_tests(self) -> Dict[str, Any]:
        """
        Comprehensive security testing suite.
        """
        self.logger.info("Running security tests")
        
        security_tests = [
            self.test_authentication_security(),
            self.test_authorization_controls(),
            self.test_data_encryption(),
            self.test_input_validation(),
            self.test_sql_injection_protection(),
            self.test_xss_protection(),
            self.test_csrf_protection(),
            self.test_rate_limiting(),
            self.test_session_management()
        ]
        
        results = await asyncio.gather(*security_tests, return_exceptions=True)
        
        return {
            'category': 'security_testing',
            'total_tests': len(security_tests),
            'passed': len([r for r in results if isinstance(r, TestResult) and r.status == 'passed']),
            'failed': len([r for r in results if isinstance(r, TestResult) and r.status == 'failed']),
            'errors': len([r for r in results if isinstance(r, Exception)]),
            'test_results': [r for r in results if isinstance(r, TestResult)],
            'execution_time': sum(r.execution_time for r in results if isinstance(r, TestResult))
        }
    
    async def test_authentication_security(self) -> TestResult:
        """
        Test authentication security mechanisms.
        """
        start_time = datetime.now()
        
        try:
            security_checks = {}
            
            # Test invalid credentials
            invalid_auth_response = await self.test_invalid_authentication()
            security_checks['invalid_credentials'] = invalid_auth_response['status'] == 401
            
            # Test token expiration
            token_expiration_response = await self.test_token_expiration()
            security_checks['token_expiration'] = token_expiration_response['handled_correctly']
            
            # Test password complexity requirements
            password_complexity = await self.test_password_complexity()
            security_checks['password_complexity'] = password_complexity['enforced']
            
            # Test multi-factor authentication
            mfa_test = await self.test_mfa_enforcement()
            security_checks['mfa_enforcement'] = mfa_test['enforced']
            
            # Test account lockout
            account_lockout = await self.test_account_lockout()
            security_checks['account_lockout'] = account_lockout['working']
            
            passed = all(security_checks.values())
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TestResult(
                test_name='authentication_security',
                status='passed' if passed else 'failed',
                execution_time=execution_time,
                details={
                    'security_checks': security_checks,
                    'recommendations': self.generate_auth_security_recommendations(security_checks)
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestResult(
                test_name='authentication_security',
                status='error',
                execution_time=execution_time,
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def generate_test_report(self, test_category_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive test report.
        """
        total_tests = sum(result['total_tests'] for result in test_category_results)
        total_passed = sum(result['passed'] for result in test_category_results)
        total_failed = sum(result['failed'] for result in test_category_results)
        total_errors = sum(result['errors'] for result in test_category_results)
        total_execution_time = sum(result['execution_time'] for result in test_category_results)
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall status
        if total_errors > 0:
            overall_status = 'error'
        elif total_failed > 0:
            overall_status = 'failed'
        else:
            overall_status = 'passed'
        
        return {
            'test_report_timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'summary': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_failed,
                'errors': total_errors,
                'success_rate': overall_success_rate,
                'total_execution_time': total_execution_time
            },
            'category_results': test_category_results,
            'recommendations': self.generate_overall_recommendations(test_category_results),
            'quality_gates': self.evaluate_quality_gates(test_category_results)
        }
```

DELIVERABLES:
1. Comprehensive test automation framework
2. AI model validation and bias testing suite
3. Performance and load testing infrastructure
4. Security testing and vulnerability assessment tools
5. Integration and end-to-end testing scenarios
6. Compliance testing for regulatory frameworks
7. Continuous testing pipeline integration
8. Test reporting and analytics dashboard

PERFORMANCE REQUIREMENTS:
- Test suite execution completion < 60 minutes for full suite
- AI model accuracy validation > 85% for all models
- Performance tests must validate < 200ms API response times
- Load testing must support 1000+ concurrent users
- Security tests must cover OWASP Top 10 vulnerabilities
- Test automation coverage > 80% for all code paths

QUALITY CRITERIA:
- All tests must be deterministic and repeatable
- Test data must be representative and unbiased
- Security tests must follow industry best practices
- Performance benchmarks must be realistic and achievable
- Test results must be comprehensive and actionable
- Continuous integration must include automated testing

OUTPUT FORMAT:
Provide complete testing implementation including:
- Test automation framework with all test categories
- AI model validation and performance testing
- Security and compliance testing suites
- Performance and scalability testing tools
- Integration testing scenarios and data
- Test reporting and analytics capabilities

Begin with the overall testing strategy and framework, then implement each testing category with detailed explanations of testing methodologies, validation criteria, and quality assurance practices.
```

This comprehensive master-class prompt engineering document provides detailed, production-ready prompts for implementing the Azure Governance Intelligence Platform. Each prompt is carefully crafted to maximize output quality while minimizing token consumption, ensuring efficient development of a sophisticated AI-powered governance solution that meets enterprise requirements for security, scalability, and reliability.


## 6. Cost Optimization and Token Efficiency Guidelines

### 6.1 Advanced Token Optimization Strategies

**Prompt Compression Techniques**
The most effective approach to minimizing token consumption while maintaining output quality involves strategic prompt compression without sacrificing essential information. This includes using domain-specific abbreviations consistently throughout prompts, eliminating redundant phrases and filler words, and structuring prompts with clear hierarchical information flow. The key is to maintain semantic richness while reducing syntactic overhead.

**Context Window Management**
Effective context window utilization requires careful planning of information density and relevance scoring. Implement progressive disclosure patterns where the most critical information appears early in prompts, followed by supporting details and edge cases. Use explicit context boundaries to help the AI model understand when to prioritize recent information over historical context.

**Modular Prompt Architecture Benefits**
The modular approach significantly reduces token overhead by enabling reusable prompt components that can be combined dynamically based on specific requirements. This eliminates the need to repeat common instructions, technical specifications, and quality criteria across multiple prompts. Each module should be self-contained yet designed for seamless integration with other modules.

### 6.2 Quality Assurance Through Prompt Engineering

**Validation and Error Handling Integration**
Every prompt should include explicit validation criteria and error handling instructions that guide the AI toward self-correction and quality improvement. This includes specifying expected output formats, validation checkpoints, and recovery strategies for common failure modes. The prompts should encourage the AI to validate its own outputs against specified criteria before finalizing responses.

**Iterative Refinement Mechanisms**
Build iterative improvement capabilities directly into prompts by including instructions for self-assessment, output evaluation, and incremental enhancement. This approach ensures that initial outputs serve as foundations for progressive refinement rather than final deliverables, leading to higher quality results with minimal additional token consumption.

**Domain Expertise Simulation**
The prompts leverage role-based instruction techniques that simulate domain expertise across multiple disciplines simultaneously. This approach ensures that the AI applies appropriate knowledge and methodologies from cloud engineering, AI/ML development, security, and business strategy without requiring separate expert consultations.

### 6.3 Implementation Success Metrics

**Development Velocity Indicators**
Success metrics for the prompt engineering approach include development velocity measurements such as time-to-first-working-prototype, feature completion rates, and defect density in generated code. The prompts are designed to accelerate development by providing comprehensive guidance that reduces iteration cycles and rework requirements.

**Quality Assurance Metrics**
Quality metrics encompass code quality scores, security vulnerability counts, performance benchmark achievements, and compliance validation results. The prompts include specific quality gates and acceptance criteria that ensure generated outputs meet production-ready standards without extensive manual review and correction.

**Cost Efficiency Measurements**
Cost efficiency is measured through token consumption per feature delivered, development time reduction compared to traditional approaches, and total cost of ownership for the implemented solution. The prompt engineering strategy aims to achieve 60-80% reduction in development time while maintaining or improving output quality.

## 7. Implementation Execution Strategy

### 7.1 Phased Development Approach

**Phase 1: Foundation and Infrastructure (Weeks 1-4)**
Begin implementation with infrastructure setup and core service development using the provided infrastructure and application architecture prompts. Focus on establishing the foundational components including Azure service integrations, database schemas, and basic API endpoints. This phase should result in a working development environment with core services deployed and basic functionality validated.

**Phase 2: AI Engine Development (Weeks 5-8)**
Implement the AI engine components using the machine learning and NLP prompts. This includes model training, validation, and deployment pipelines. Focus on achieving baseline performance metrics for all AI models before proceeding to advanced features. Implement comprehensive testing and validation frameworks to ensure model reliability and accuracy.

**Phase 3: Advanced Features and Integration (Weeks 9-12)**
Develop advanced features including conversational AI, automated remediation, and intelligent optimization using the Azure service integration prompts. Focus on seamless integration between all platform components and comprehensive end-to-end workflow validation. Implement advanced security features and compliance frameworks.

**Phase 4: Testing and Optimization (Weeks 13-16)**
Execute comprehensive testing using the testing strategy prompts, including performance optimization, security validation, and user acceptance testing. Focus on achieving production-ready quality standards and performance benchmarks. Implement monitoring, alerting, and operational procedures.

**Phase 5: Deployment and Launch (Weeks 17-20)**
Deploy the platform using the deployment and scaling strategy prompts. Focus on production deployment, user onboarding, and operational stability. Implement customer success metrics and continuous improvement processes.

### 7.2 Risk Mitigation and Contingency Planning

**Technical Risk Management**
Key technical risks include AI model performance degradation, Azure API rate limiting, and scalability bottlenecks. Mitigation strategies include comprehensive testing frameworks, fallback mechanisms for AI models, and proactive capacity planning. Each prompt includes specific guidance for handling common failure scenarios and implementing robust error recovery mechanisms.

**Business Risk Considerations**
Business risks encompass market competition, customer adoption challenges, and regulatory compliance requirements. The solution design includes competitive differentiation strategies, user experience optimization, and comprehensive compliance frameworks. Regular market validation and customer feedback integration ensure alignment with market needs and expectations.

**Operational Risk Controls**
Operational risks include security vulnerabilities, data privacy concerns, and service availability issues. The implementation includes comprehensive security frameworks, privacy-by-design principles, and high-availability architecture patterns. Continuous monitoring and incident response procedures ensure rapid detection and resolution of operational issues.

### 7.3 Success Validation and Metrics

**Technical Success Criteria**
Technical success is measured through performance benchmarks including API response times under 200ms, AI model accuracy above 85%, system availability above 99.9%, and successful handling of 1000+ concurrent users. The implementation must demonstrate scalability, reliability, and maintainability according to enterprise standards.

**Business Success Indicators**
Business success indicators include customer acquisition rates, user engagement metrics, revenue growth, and market penetration. The platform should demonstrate clear value proposition delivery, customer satisfaction scores above 4.5/5, and positive return on investment within 12 months of deployment.

**Market Validation Metrics**
Market validation encompasses competitive positioning, customer feedback scores, industry recognition, and partnership development. The solution should achieve differentiated market position, positive analyst coverage, and strategic partnership opportunities within the first year of operation.

## 8. Conclusion and Next Steps

### 8.1 Strategic Implementation Roadmap

The Azure Governance Intelligence Platform represents a significant market opportunity that combines proven cloud governance needs with cutting-edge AI capabilities. The comprehensive prompt engineering approach provided in this document ensures efficient development of a production-ready solution that delivers measurable business value while maintaining competitive differentiation.

The implementation strategy balances technical sophistication with practical business considerations, ensuring that the resulting platform meets both immediate customer needs and long-term market evolution. The modular architecture and comprehensive testing frameworks provide flexibility for future enhancements and market expansion.

### 8.2 Competitive Advantage Sustainability

The platform's competitive advantage stems from its unique combination of comprehensive Azure service integration, advanced AI capabilities, and user-centric design. The prompt engineering approach ensures rapid development and deployment while maintaining high quality standards that differentiate the solution from existing market offerings.

Continuous innovation through AI model improvement, feature enhancement, and market expansion ensures sustainable competitive positioning. The platform's architecture supports rapid adaptation to changing market conditions and customer requirements.

### 8.3 Long-term Vision and Expansion

The long-term vision encompasses expansion beyond Azure to multi-cloud governance, integration with additional AI capabilities, and development of industry-specific solutions. The foundational architecture and prompt engineering approach support these expansion opportunities while maintaining core platform stability and performance.

Future development phases should focus on advanced AI capabilities, expanded cloud platform support, and vertical market specialization. The comprehensive documentation and implementation guidance provided ensure successful execution of both immediate development goals and long-term strategic objectives.

---

**Document Metadata:**
- **Author:** Manus AI
- **Version:** 1.0
- **Date:** January 2025
- **Classification:** Technical Implementation Guide
- **Target Audience:** AI Development Teams, Cloud Engineers, Technical Leadership

This master-class prompt engineering document provides the comprehensive foundation for implementing the Azure Governance Intelligence Platform efficiently and effectively, ensuring both technical excellence and business success.

