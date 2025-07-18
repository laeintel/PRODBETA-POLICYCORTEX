# Azure Governance Intelligence Platform (policyCortex) - Implementation Roadmap and Technical Specifications

## Executive Summary

This comprehensive implementation roadmap provides detailed technical specifications, development guidelines, and actionable steps for building the Azure Governance Intelligence Platform (policyCortex). The document serves as a complete blueprint for AI Claude to implement the solution, including specific technology choices, architectural patterns, code structures, and deployment strategies.

The roadmap is designed to enable rapid development while ensuring enterprise-grade quality, security, and scalability. Each component is specified with sufficient detail to enable independent development and integration, while maintaining consistency with the overall architectural vision.

## 1. Technical Foundation and Infrastructure Setup

### 1.1 Development Environment Configuration

**Azure Subscription and Resource Setup**

The development environment requires a comprehensive Azure subscription configuration that supports all necessary services while maintaining cost efficiency during development phases. The setup includes dedicated resource groups for different environment tiers (development, staging, production) with appropriate naming conventions and tagging strategies.

**Resource Group Structure:**
- `policyCortex-dev-rg`: Development environment resources
- `policyCortex-staging-rg`: Staging environment for integration testing
- `policyCortex-prod-rg`: Production environment resources
- `policyCortex-shared-rg`: Shared resources across environments (Key Vault, Container Registry)

**Azure Services Configuration:**
- **Azure Kubernetes Service (AKS)**: Multi-node cluster with auto-scaling enabled
- **Azure Machine Learning Workspace**: Dedicated workspace for AI model development
- **Azure Data Lake Storage Gen2**: Hierarchical storage for governance data
- **Azure SQL Database**: Relational data storage with elastic pool configuration
- **Azure Cosmos DB**: NoSQL storage for real-time data and user sessions
- **Azure Key Vault**: Centralized secrets and certificate management
- **Azure Container Registry**: Private container image storage
- **Azure Application Gateway**: Load balancing and SSL termination
- **Azure Monitor**: Comprehensive monitoring and logging

**Development Tools and IDE Setup**

**Visual Studio Code Configuration**: Comprehensive VS Code setup with essential extensions for Azure development, Python/Node.js development, Docker, Kubernetes, and AI/ML development. The configuration includes workspace settings, debugging configurations, and integrated terminal setups for efficient development workflows.

**Required Extensions:**
- Azure Account and Azure Resources extensions for Azure integration
- Python and Pylance for Python development
- Azure Machine Learning extension for ML model development
- Docker and Kubernetes extensions for containerization
- GitLens and GitHub Pull Requests for version control
- REST Client for API testing and development

**Local Development Environment**: Docker-based local development environment that mirrors the production architecture while enabling rapid development and testing. This includes local instances of databases, message queues, and AI model serving infrastructure.

### 1.2 Infrastructure as Code Implementation

**Terraform Configuration Structure**

The infrastructure is defined using Terraform with a modular structure that enables reusable components and environment-specific configurations. The Terraform configuration follows best practices for state management, variable handling, and resource organization.

**Module Structure:**
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
    ├── variables.tf
    ├── outputs.tf
    └── providers.tf
```

**AKS Cluster Module**: Comprehensive AKS cluster configuration with node pools for different workload types (AI/ML, web services, data processing), network policies for security, and integration with Azure Monitor for observability. The configuration includes auto-scaling parameters, upgrade policies, and security configurations.

**Data Services Module**: Configuration for all data storage services including Azure SQL Database with elastic pools, Azure Cosmos DB with appropriate consistency levels and partitioning strategies, and Azure Data Lake Storage with hierarchical namespace and lifecycle management policies.

**AI Services Module**: Azure Machine Learning workspace configuration with compute clusters for model training, inference endpoints for model serving, and integration with Azure Container Registry for custom model containers. The configuration includes automated scaling policies and cost optimization settings.

**Networking Module**: Virtual network configuration with subnets for different service tiers, network security groups with appropriate rules, and private endpoints for secure service communication. The configuration includes Azure Application Gateway for external access and Azure Private DNS for internal name resolution.

**Azure Resource Manager (ARM) Templates**

Complementary ARM templates provide Azure-specific resource configurations that leverage native Azure capabilities not available in Terraform. These templates focus on Azure-specific features such as managed identities, RBAC assignments, and policy configurations.

**Managed Identity Configuration**: Comprehensive managed identity setup for all service components with appropriate RBAC assignments for Azure service access. This includes system-assigned identities for AKS nodes and user-assigned identities for specific application components.

**Azure Policy Integration**: ARM templates that configure Azure policies for the policyCortex infrastructure itself, ensuring that the platform follows governance best practices and serves as a reference implementation for customers.

### 1.3 CI/CD Pipeline Implementation

**Azure DevOps Pipeline Configuration**

Comprehensive CI/CD pipelines using Azure DevOps that support the full development lifecycle from code commit to production deployment. The pipelines include automated testing, security scanning, and deployment automation with appropriate approval gates.

**Build Pipeline Structure:**
```yaml
trigger:
  branches:
    include:
    - main
    - develop
    - feature/*

pool:
  vmImage: 'ubuntu-latest'

stages:
- stage: Build
  jobs:
  - job: BuildAndTest
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: '3.11'
    - script: |
        pip install -r requirements.txt
        pytest tests/ --junitxml=test-results.xml
      displayName: 'Run Tests'
    - task: PublishTestResults@2
      inputs:
        testResultsFiles: 'test-results.xml'
    - task: Docker@2
      inputs:
        command: 'buildAndPush'
        repository: '$(containerRegistry)/policyCortex'
        tags: '$(Build.BuildId)'
```

**Release Pipeline Configuration**: Multi-stage release pipeline with environment-specific configurations, automated testing in staging environments, and production deployment with manual approval gates. The pipeline includes rollback capabilities and blue-green deployment strategies for zero-downtime updates.

**Security and Quality Gates**: Integration of security scanning tools including container vulnerability scanning, static code analysis, and dependency checking. Quality gates ensure that code meets established standards before deployment to production environments.

## 2. Core AI Engine Implementation

### 2.1 Machine Learning Model Architecture

**Policy Compliance Prediction Models**

The policy compliance prediction system uses ensemble machine learning models that combine multiple algorithms to provide accurate predictions of future compliance states. The implementation leverages Azure Machine Learning for model training and deployment while providing custom logic for Azure-specific governance patterns.

**Model Architecture Design:**
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

class PolicyCompliancePredictor:
    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        self.ensemble_weights = None
        
    def prepare_features(self, governance_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw Azure governance data into features suitable for ML models.
        
        Features include:
        - Resource configuration patterns
        - Historical compliance trends
        - Policy change frequency
        - Resource lifecycle patterns
        - Organizational change indicators
        """
        features = pd.DataFrame()
        
        # Time-based features
        features['days_since_last_policy_change'] = (
            pd.Timestamp.now() - governance_data['last_policy_change']
        ).dt.days
        
        # Resource pattern features
        features['resource_count_trend'] = governance_data.groupby('resource_group')[
            'resource_count'
        ].pct_change()
        
        # Compliance history features
        features['compliance_score_ma_7d'] = governance_data.groupby('policy_id')[
            'compliance_score'
        ].rolling(window=7).mean()
        
        # Configuration drift features
        features['config_change_frequency'] = governance_data.groupby('resource_id')[
            'config_changes'
        ].rolling(window=30).sum()
        
        return features
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train ensemble of models with time series cross-validation.
        """
        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = {}
        
        for name, model in self.models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_fold_train, y_fold_train)
                score = model.score(X_fold_val, y_fold_val)
                scores.append(score)
            
            model_scores[name] = np.mean(scores)
            
        # Calculate ensemble weights based on performance
        total_score = sum(model_scores.values())
        self.ensemble_weights = {
            name: score / total_score 
            for name, score in model_scores.items()
        }
        
        # Train final models on full dataset
        for model in self.models.values():
            model.fit(X_train, y_train)
    
    def predict_compliance_risk(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions for compliance risk.
        """
        predictions = np.zeros(len(X_test))
        
        for name, model in self.models.items():
            model_pred = model.predict_proba(X_test)[:, 1]  # Probability of non-compliance
            predictions += model_pred * self.ensemble_weights[name]
            
        return predictions
```

**Cost Optimization Models**

Advanced cost optimization models analyze resource utilization patterns, pricing trends, and business requirements to provide intelligent cost optimization recommendations. The models use time series forecasting and reinforcement learning to optimize cost while maintaining performance requirements.

**Implementation Architecture:**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from azure.ai.ml import MLClient
import pandas as pd
import numpy as np

class CostOptimizationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CostOptimizationLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class CostOptimizationEngine:
    def __init__(self, ml_client: MLClient):
        self.ml_client = ml_client
        self.model = None
        self.scaler = None
        
    def prepare_cost_features(self, cost_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for cost optimization model.
        
        Features include:
        - Resource utilization metrics
        - Cost trends and patterns
        - Business activity indicators
        - Seasonal patterns
        - Resource lifecycle stages
        """
        features = []
        
        # Utilization features
        features.append(cost_data['cpu_utilization'].values)
        features.append(cost_data['memory_utilization'].values)
        features.append(cost_data['storage_utilization'].values)
        features.append(cost_data['network_utilization'].values)
        
        # Cost trend features
        features.append(cost_data['daily_cost'].rolling(window=7).mean().values)
        features.append(cost_data['daily_cost'].rolling(window=30).mean().values)
        features.append(cost_data['daily_cost'].pct_change().values)
        
        # Business context features
        features.append(cost_data['business_hours_indicator'].values)
        features.append(cost_data['weekend_indicator'].values)
        features.append(cost_data['holiday_indicator'].values)
        
        # Resource lifecycle features
        features.append(cost_data['resource_age_days'].values)
        features.append(cost_data['last_access_days'].values)
        
        return np.column_stack(features)
    
    def train_cost_model(self, cost_data: pd.DataFrame):
        """
        Train LSTM model for cost prediction and optimization.
        """
        features = self.prepare_cost_features(cost_data)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences for LSTM
        sequence_length = 30
        X, y = self.create_sequences(features_scaled, cost_data['daily_cost'].values, sequence_length)
        
        # Initialize model
        input_size = features_scaled.shape[1]
        hidden_size = 64
        num_layers = 2
        output_size = 1
        
        self.model = CostOptimizationLSTM(input_size, hidden_size, num_layers, output_size)
        
        # Training loop
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        dataset = CostDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(100):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
    
    def predict_cost_optimization(self, current_data: pd.DataFrame) -> dict:
        """
        Generate cost optimization recommendations.
        """
        features = self.prepare_cost_features(current_data)
        features_scaled = self.scaler.transform(features)
        
        # Generate predictions
        with torch.no_grad():
            predictions = self.model(torch.FloatTensor(features_scaled))
        
        # Analyze optimization opportunities
        recommendations = self.analyze_optimization_opportunities(
            current_data, predictions.numpy()
        )
        
        return recommendations
    
    def analyze_optimization_opportunities(self, data: pd.DataFrame, predictions: np.ndarray) -> dict:
        """
        Analyze data to identify specific optimization opportunities.
        """
        opportunities = {
            'rightsizing': [],
            'scheduling': [],
            'reserved_instances': [],
            'lifecycle_management': []
        }
        
        # Right-sizing analysis
        underutilized = data[
            (data['cpu_utilization'] < 0.2) & 
            (data['memory_utilization'] < 0.3)
        ]
        
        for _, resource in underutilized.iterrows():
            opportunities['rightsizing'].append({
                'resource_id': resource['resource_id'],
                'current_size': resource['vm_size'],
                'recommended_size': self.recommend_vm_size(resource),
                'estimated_savings': self.calculate_rightsizing_savings(resource)
            })
        
        # Scheduling opportunities
        dev_resources = data[data['environment'] == 'development']
        for _, resource in dev_resources.iterrows():
            if self.should_schedule_resource(resource):
                opportunities['scheduling'].append({
                    'resource_id': resource['resource_id'],
                    'schedule_type': 'business_hours_only',
                    'estimated_savings': self.calculate_scheduling_savings(resource)
                })
        
        return opportunities
```

**Network Intelligence Models**

Network intelligence models analyze traffic patterns, security events, and performance metrics to provide intelligent network optimization and security recommendations. The implementation uses graph neural networks to understand network topology and convolutional neural networks for traffic pattern analysis.

### 2.2 Natural Language Processing Implementation

**Conversational AI Engine**

The conversational AI engine provides natural language interaction capabilities that enable users to query governance data, request optimizations, and receive explanations through natural dialogue. The implementation leverages Azure OpenAI Service with custom fine-tuning for governance-specific use cases.

**Implementation Architecture:**
```python
from azure.ai.ml import MLClient
from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import openai
import json
import re

class GovernanceConversationEngine:
    def __init__(self, openai_client, text_analytics_client: TextAnalyticsClient):
        self.openai_client = openai_client
        self.text_analytics_client = text_analytics_client
        self.conversation_history = []
        self.context_manager = ConversationContextManager()
        
    def process_user_query(self, user_input: str, user_context: dict) -> dict:
        """
        Process user query and generate appropriate response.
        """
        # Extract intent and entities
        intent = self.extract_intent(user_input)
        entities = self.extract_entities(user_input)
        
        # Update conversation context
        self.context_manager.update_context(user_input, intent, entities, user_context)
        
        # Generate response based on intent
        if intent == 'policy_compliance_query':
            response = self.handle_compliance_query(entities, user_context)
        elif intent == 'cost_optimization_request':
            response = self.handle_cost_optimization(entities, user_context)
        elif intent == 'rbac_analysis_request':
            response = self.handle_rbac_analysis(entities, user_context)
        elif intent == 'network_security_query':
            response = self.handle_network_query(entities, user_context)
        else:
            response = self.handle_general_query(user_input, user_context)
        
        # Add to conversation history
        self.conversation_history.append({
            'user_input': user_input,
            'intent': intent,
            'entities': entities,
            'response': response,
            'timestamp': pd.Timestamp.now()
        })
        
        return response
    
    def extract_intent(self, user_input: str) -> str:
        """
        Extract user intent from natural language input.
        """
        # Use Azure Text Analytics for intent classification
        documents = [{"id": "1", "text": user_input}]
        
        # Custom intent classification using fine-tuned model
        intent_prompt = f"""
        Classify the following Azure governance query into one of these intents:
        - policy_compliance_query: Questions about policy compliance status
        - cost_optimization_request: Requests for cost optimization recommendations
        - rbac_analysis_request: Questions about access control and permissions
        - network_security_query: Questions about network security and configuration
        - general_query: General questions or unclear intent
        
        Query: {user_input}
        
        Intent:
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": intent_prompt}],
            max_tokens=50,
            temperature=0.1
        )
        
        intent = response.choices[0].message.content.strip().lower()
        return intent
    
    def extract_entities(self, user_input: str) -> dict:
        """
        Extract relevant entities from user input.
        """
        entities = {
            'resource_groups': [],
            'subscriptions': [],
            'policies': [],
            'time_ranges': [],
            'cost_thresholds': [],
            'users': [],
            'roles': []
        }
        
        # Use regex patterns for Azure-specific entities
        rg_pattern = r'\b[a-zA-Z0-9\-_]+\-rg\b'
        subscription_pattern = r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'
        
        entities['resource_groups'] = re.findall(rg_pattern, user_input)
        entities['subscriptions'] = re.findall(subscription_pattern, user_input)
        
        # Use NER for other entities
        documents = [{"id": "1", "text": user_input}]
        ner_results = self.text_analytics_client.recognize_entities(documents)
        
        for doc in ner_results:
            for entity in doc.entities:
                if entity.category == "DateTime":
                    entities['time_ranges'].append(entity.text)
                elif entity.category == "Person":
                    entities['users'].append(entity.text)
        
        return entities
    
    def handle_compliance_query(self, entities: dict, user_context: dict) -> dict:
        """
        Handle policy compliance queries.
        """
        # Query compliance data based on entities
        compliance_data = self.query_compliance_data(entities, user_context)
        
        # Generate natural language response
        response_prompt = f"""
        Generate a clear, helpful response about Azure policy compliance based on this data:
        
        Compliance Data: {json.dumps(compliance_data, indent=2)}
        
        User Context: {json.dumps(user_context, indent=2)}
        
        Provide:
        1. Summary of compliance status
        2. Key issues or risks identified
        3. Specific recommendations for improvement
        4. Next steps the user should take
        
        Use clear, professional language appropriate for the user's technical level.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": response_prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        return {
            'type': 'compliance_analysis',
            'data': compliance_data,
            'natural_language_response': response.choices[0].message.content,
            'recommendations': self.generate_compliance_recommendations(compliance_data),
            'visualizations': self.generate_compliance_visualizations(compliance_data)
        }
    
    def handle_cost_optimization(self, entities: dict, user_context: dict) -> dict:
        """
        Handle cost optimization requests.
        """
        # Analyze cost data and generate recommendations
        cost_analysis = self.analyze_cost_optimization_opportunities(entities, user_context)
        
        # Generate conversational response
        response_prompt = f"""
        Generate a helpful response about Azure cost optimization based on this analysis:
        
        Cost Analysis: {json.dumps(cost_analysis, indent=2)}
        
        Provide:
        1. Summary of current cost situation
        2. Top optimization opportunities with estimated savings
        3. Implementation guidance for recommendations
        4. Potential risks or considerations
        
        Use clear, business-friendly language that explains technical concepts simply.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": response_prompt}],
            max_tokens=600,
            temperature=0.3
        )
        
        return {
            'type': 'cost_optimization',
            'data': cost_analysis,
            'natural_language_response': response.choices[0].message.content,
            'optimization_opportunities': cost_analysis['opportunities'],
            'estimated_savings': cost_analysis['total_estimated_savings'],
            'implementation_plan': self.generate_implementation_plan(cost_analysis)
        }

class ConversationContextManager:
    def __init__(self):
        self.context = {
            'current_subscription': None,
            'current_resource_group': None,
            'user_preferences': {},
            'conversation_flow': [],
            'active_tasks': []
        }
    
    def update_context(self, user_input: str, intent: str, entities: dict, user_context: dict):
        """
        Update conversation context based on user interaction.
        """
        # Update current scope based on entities
        if entities.get('subscriptions'):
            self.context['current_subscription'] = entities['subscriptions'][0]
        
        if entities.get('resource_groups'):
            self.context['current_resource_group'] = entities['resource_groups'][0]
        
        # Track conversation flow
        self.context['conversation_flow'].append({
            'intent': intent,
            'entities': entities,
            'timestamp': pd.Timestamp.now()
        })
        
        # Update user preferences based on patterns
        self.update_user_preferences(intent, entities, user_context)
    
    def update_user_preferences(self, intent: str, entities: dict, user_context: dict):
        """
        Learn user preferences from interaction patterns.
        """
        # Track frequently queried resources
        if 'frequently_queried' not in self.context['user_preferences']:
            self.context['user_preferences']['frequently_queried'] = {}
        
        for entity_type, entity_values in entities.items():
            if entity_values:
                if entity_type not in self.context['user_preferences']['frequently_queried']:
                    self.context['user_preferences']['frequently_queried'][entity_type] = {}
                
                for value in entity_values:
                    current_count = self.context['user_preferences']['frequently_queried'][entity_type].get(value, 0)
                    self.context['user_preferences']['frequently_queried'][entity_type][value] = current_count + 1
```

This implementation provides a sophisticated conversational AI engine that can understand governance-specific queries, maintain conversation context, and generate intelligent responses with actionable recommendations. The engine integrates with Azure Cognitive Services for natural language understanding while using custom logic for governance-specific processing.


## 3. Azure Service Integration Implementation

### 3.1 Azure Policy Integration Layer

The Azure Policy integration layer provides comprehensive access to Azure Policy APIs with intelligent caching, rate limiting, and error handling. This layer serves as the foundation for all policy-related AI capabilities and automation features.

**Policy API Client Implementation**

```python
import asyncio
import aiohttp
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from typing import Dict, List, Optional, Any
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class PolicyState:
    resource_id: str
    policy_assignment_id: str
    policy_definition_id: str
    compliance_state: str
    timestamp: datetime
    policy_evaluation_details: Dict[str, Any]

class AzurePolicyClient:
    def __init__(self, credential: DefaultAzureCredential):
        self.credential = credential
        self.base_url = "https://management.azure.com"
        self.api_version = "2023-04-01"
        self.session = None
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        self.cache = PolicyDataCache()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_access_token(self) -> str:
        """Get Azure access token for API authentication."""
        token = self.credential.get_token("https://management.azure.com/.default")
        return token.token
    
    async def get_policy_states(self, scope: str, filter_expression: Optional[str] = None) -> List[PolicyState]:
        """
        Retrieve policy compliance states for a given scope.
        
        Args:
            scope: Azure resource scope (subscription, resource group, etc.)
            filter_expression: OData filter expression for filtering results
        """
        cache_key = f"policy_states_{scope}_{filter_expression}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result and not self.cache.is_expired(cache_key):
            return cached_result
        
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{scope}/providers/Microsoft.PolicyInsights/policyStates/latest/queryResults"
        
        params = {
            "api-version": self.api_version,
            "$top": 1000
        }
        
        if filter_expression:
            params["$filter"] = filter_expression
        
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    policy_states = [
                        PolicyState(
                            resource_id=item["resourceId"],
                            policy_assignment_id=item["policyAssignmentId"],
                            policy_definition_id=item["policyDefinitionId"],
                            compliance_state=item["complianceState"],
                            timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
                            policy_evaluation_details=item.get("policyEvaluationDetails", {})
                        )
                        for item in data.get("value", [])
                    ]
                    
                    self.cache.set(cache_key, policy_states, ttl_minutes=15)
                    return policy_states
                else:
                    raise AzureError(f"Failed to retrieve policy states: {response.status}")
                    
        except Exception as e:
            raise AzureError(f"Error retrieving policy states: {str(e)}")
    
    async def get_policy_definitions(self, subscription_id: str) -> List[Dict[str, Any]]:
        """Retrieve all policy definitions for a subscription."""
        cache_key = f"policy_definitions_{subscription_id}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result and not self.cache.is_expired(cache_key):
            return cached_result
        
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/subscriptions/{subscription_id}/providers/Microsoft.Authorization/policyDefinitions"
        
        params = {"api-version": "2023-04-01"}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    definitions = data.get("value", [])
                    self.cache.set(cache_key, definitions, ttl_minutes=60)
                    return definitions
                else:
                    raise AzureError(f"Failed to retrieve policy definitions: {response.status}")
                    
        except Exception as e:
            raise AzureError(f"Error retrieving policy definitions: {str(e)}")
    
    async def create_policy_assignment(self, scope: str, assignment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new policy assignment."""
        await self.rate_limiter.wait_if_needed()
        
        assignment_name = assignment_data["name"]
        url = f"{self.base_url}/{scope}/providers/Microsoft.Authorization/policyAssignments/{assignment_name}"
        
        params = {"api-version": "2023-04-01"}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.put(url, json=assignment_data, params=params, headers=headers) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    # Invalidate related cache entries
                    self.cache.invalidate_pattern(f"policy_assignments_{scope}*")
                    return result
                else:
                    error_text = await response.text()
                    raise AzureError(f"Failed to create policy assignment: {response.status} - {error_text}")
                    
        except Exception as e:
            raise AzureError(f"Error creating policy assignment: {str(e)}")
    
    async def trigger_policy_evaluation(self, scope: str) -> bool:
        """Trigger on-demand policy evaluation for a scope."""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{scope}/providers/Microsoft.PolicyInsights/policyStates/latest/triggerEvaluation"
        
        params = {"api-version": "2023-04-01"}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(url, params=params, headers=headers) as response:
                return response.status == 202
                
        except Exception as e:
            raise AzureError(f"Error triggering policy evaluation: {str(e)}")

class RateLimiter:
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)

class PolicyDataCache:
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl_minutes: int = 30):
        self.cache[key] = value
        self.timestamps[key] = datetime.now() + timedelta(minutes=ttl_minutes)
    
    def is_expired(self, key: str) -> bool:
        if key not in self.timestamps:
            return True
        return datetime.now() > self.timestamps[key]
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern."""
        keys_to_remove = [key for key in self.cache.keys() if pattern.replace("*", "") in key]
        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
```

**Policy Intelligence Engine**

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import asyncio

class PolicyIntelligenceEngine:
    def __init__(self, policy_client: AzurePolicyClient, ml_models: Dict[str, Any]):
        self.policy_client = policy_client
        self.ml_models = ml_models
        self.compliance_analyzer = ComplianceAnalyzer()
        self.policy_optimizer = PolicyOptimizer()
        
    async def analyze_compliance_trends(self, scope: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze compliance trends over time to identify patterns and predict future compliance.
        """
        # Retrieve historical compliance data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        filter_expression = f"timestamp ge {start_date.isoformat()} and timestamp le {end_date.isoformat()}"
        policy_states = await self.policy_client.get_policy_states(scope, filter_expression)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'resource_id': ps.resource_id,
                'policy_id': ps.policy_definition_id,
                'compliance_state': ps.compliance_state,
                'timestamp': ps.timestamp,
                'assignment_id': ps.policy_assignment_id
            }
            for ps in policy_states
        ])
        
        if df.empty:
            return {'error': 'No compliance data found for the specified scope and time range'}
        
        # Analyze trends
        trends = self.compliance_analyzer.analyze_trends(df)
        
        # Generate predictions
        predictions = await self.predict_future_compliance(df, scope)
        
        # Identify risk factors
        risk_factors = self.identify_compliance_risk_factors(df)
        
        return {
            'scope': scope,
            'analysis_period': {'start': start_date, 'end': end_date},
            'current_compliance_rate': trends['current_compliance_rate'],
            'compliance_trend': trends['trend_direction'],
            'trend_confidence': trends['trend_confidence'],
            'predictions': predictions,
            'risk_factors': risk_factors,
            'recommendations': self.generate_compliance_recommendations(trends, predictions, risk_factors)
        }
    
    async def predict_future_compliance(self, historical_data: pd.DataFrame, scope: str) -> Dict[str, Any]:
        """
        Use ML models to predict future compliance states.
        """
        # Prepare features for prediction
        features = self.prepare_compliance_features(historical_data)
        
        # Use trained ML model for prediction
        compliance_model = self.ml_models.get('compliance_predictor')
        if not compliance_model:
            return {'error': 'Compliance prediction model not available'}
        
        # Generate predictions for next 30 days
        predictions = []
        for days_ahead in range(1, 31):
            future_features = self.extrapolate_features(features, days_ahead)
            prediction = compliance_model.predict_compliance_risk(future_features)
            
            predictions.append({
                'date': datetime.now() + timedelta(days=days_ahead),
                'predicted_compliance_rate': float(1 - prediction[0]),  # Convert risk to compliance rate
                'confidence_interval': self.calculate_prediction_confidence(prediction, days_ahead)
            })
        
        return {
            'predictions': predictions,
            'model_accuracy': compliance_model.get_model_accuracy(),
            'prediction_methodology': 'Ensemble ML model with time series analysis'
        }
    
    def identify_compliance_risk_factors(self, compliance_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify factors that contribute to compliance risks.
        """
        risk_factors = []
        
        # Analyze non-compliance patterns by policy
        policy_compliance = compliance_data.groupby('policy_id').agg({
            'compliance_state': lambda x: (x == 'Compliant').mean(),
            'resource_id': 'count'
        }).rename(columns={'compliance_state': 'compliance_rate', 'resource_id': 'resource_count'})
        
        high_risk_policies = policy_compliance[policy_compliance['compliance_rate'] < 0.8]
        
        for policy_id, data in high_risk_policies.iterrows():
            risk_factors.append({
                'type': 'policy_compliance_risk',
                'policy_id': policy_id,
                'compliance_rate': float(data['compliance_rate']),
                'affected_resources': int(data['resource_count']),
                'severity': 'high' if data['compliance_rate'] < 0.5 else 'medium'
            })
        
        # Analyze resource-level patterns
        resource_compliance = compliance_data.groupby('resource_id').agg({
            'compliance_state': lambda x: (x == 'Compliant').mean(),
            'policy_id': 'count'
        }).rename(columns={'compliance_state': 'compliance_rate', 'policy_id': 'policy_count'})
        
        problematic_resources = resource_compliance[resource_compliance['compliance_rate'] < 0.7]
        
        for resource_id, data in problematic_resources.iterrows():
            risk_factors.append({
                'type': 'resource_compliance_risk',
                'resource_id': resource_id,
                'compliance_rate': float(data['compliance_rate']),
                'policy_violations': int(data['policy_count'] * (1 - data['compliance_rate'])),
                'severity': 'high' if data['compliance_rate'] < 0.5 else 'medium'
            })
        
        return risk_factors
    
    def generate_compliance_recommendations(self, trends: Dict, predictions: Dict, risk_factors: List) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on compliance analysis.
        """
        recommendations = []
        
        # Trend-based recommendations
        if trends.get('trend_direction') == 'declining':
            recommendations.append({
                'type': 'trend_intervention',
                'priority': 'high',
                'title': 'Address Declining Compliance Trend',
                'description': 'Compliance rates are declining. Immediate intervention required.',
                'actions': [
                    'Review recent policy changes and their impact',
                    'Identify resources with frequent violations',
                    'Implement automated remediation for common violations',
                    'Increase monitoring frequency for critical policies'
                ],
                'estimated_impact': 'Prevent further compliance degradation'
            })
        
        # Risk factor-based recommendations
        high_risk_policies = [rf for rf in risk_factors if rf['type'] == 'policy_compliance_risk' and rf['severity'] == 'high']
        
        if high_risk_policies:
            recommendations.append({
                'type': 'policy_optimization',
                'priority': 'high',
                'title': 'Optimize High-Risk Policies',
                'description': f'Found {len(high_risk_policies)} policies with low compliance rates',
                'actions': [
                    'Review policy definitions for clarity and feasibility',
                    'Implement automated remediation where possible',
                    'Provide training on policy requirements',
                    'Consider policy exemptions for legitimate use cases'
                ],
                'affected_policies': [p['policy_id'] for p in high_risk_policies],
                'estimated_impact': 'Improve compliance rates by 20-40%'
            })
        
        # Prediction-based recommendations
        if predictions.get('predictions'):
            future_compliance = [p['predicted_compliance_rate'] for p in predictions['predictions'][:7]]  # Next 7 days
            avg_future_compliance = np.mean(future_compliance)
            
            if avg_future_compliance < 0.8:
                recommendations.append({
                    'type': 'proactive_intervention',
                    'priority': 'medium',
                    'title': 'Proactive Compliance Intervention',
                    'description': f'Predicted compliance rate of {avg_future_compliance:.1%} for next week',
                    'actions': [
                        'Schedule proactive compliance reviews',
                        'Increase automated monitoring',
                        'Prepare remediation resources',
                        'Alert relevant teams about potential issues'
                    ],
                    'estimated_impact': 'Prevent compliance violations before they occur'
                })
        
        return recommendations

class ComplianceAnalyzer:
    def analyze_trends(self, compliance_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze compliance trends from historical data.
        """
        # Calculate daily compliance rates
        daily_compliance = compliance_data.groupby(compliance_data['timestamp'].dt.date).agg({
            'compliance_state': lambda x: (x == 'Compliant').mean()
        }).rename(columns={'compliance_state': 'compliance_rate'})
        
        # Calculate trend
        if len(daily_compliance) < 2:
            return {
                'current_compliance_rate': daily_compliance['compliance_rate'].iloc[-1] if not daily_compliance.empty else 0,
                'trend_direction': 'insufficient_data',
                'trend_confidence': 0
            }
        
        # Linear regression for trend analysis
        from sklearn.linear_model import LinearRegression
        
        X = np.arange(len(daily_compliance)).reshape(-1, 1)
        y = daily_compliance['compliance_rate'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_slope = model.coef_[0]
        trend_confidence = model.score(X, y)
        
        trend_direction = 'improving' if trend_slope > 0.001 else 'declining' if trend_slope < -0.001 else 'stable'
        
        return {
            'current_compliance_rate': float(daily_compliance['compliance_rate'].iloc[-1]),
            'trend_direction': trend_direction,
            'trend_slope': float(trend_slope),
            'trend_confidence': float(trend_confidence)
        }

class PolicyOptimizer:
    def __init__(self):
        self.optimization_strategies = {
            'automated_remediation': self.suggest_automated_remediation,
            'policy_refinement': self.suggest_policy_refinement,
            'exemption_management': self.suggest_exemption_management,
            'monitoring_enhancement': self.suggest_monitoring_enhancement
        }
    
    def optimize_policy_configuration(self, policy_data: Dict[str, Any], compliance_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Suggest optimizations for policy configuration based on compliance data.
        """
        optimizations = {}
        
        for strategy_name, strategy_func in self.optimization_strategies.items():
            try:
                optimization = strategy_func(policy_data, compliance_data)
                if optimization:
                    optimizations[strategy_name] = optimization
            except Exception as e:
                optimizations[strategy_name] = {'error': str(e)}
        
        return optimizations
    
    def suggest_automated_remediation(self, policy_data: Dict, compliance_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Suggest automated remediation opportunities.
        """
        # Identify policies with high violation rates that could benefit from automation
        violation_patterns = compliance_data[compliance_data['compliance_state'] != 'Compliant']
        
        if violation_patterns.empty:
            return None
        
        common_violations = violation_patterns.groupby('policy_id').size().sort_values(ascending=False)
        
        automation_candidates = []
        for policy_id, violation_count in common_violations.head(5).items():
            automation_candidates.append({
                'policy_id': policy_id,
                'violation_count': int(violation_count),
                'automation_potential': 'high' if violation_count > 10 else 'medium',
                'suggested_actions': [
                    'Implement automatic resource tagging',
                    'Configure automatic resource shutdown',
                    'Enable automatic security group updates',
                    'Set up automatic backup configuration'
                ]
            })
        
        return {
            'candidates': automation_candidates,
            'estimated_reduction': '60-80% of manual remediation effort'
        }
    
    def suggest_policy_refinement(self, policy_data: Dict, compliance_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Suggest policy definition refinements.
        """
        # Analyze policies with low compliance rates
        policy_compliance = compliance_data.groupby('policy_id').agg({
            'compliance_state': lambda x: (x == 'Compliant').mean()
        })
        
        low_compliance_policies = policy_compliance[policy_compliance['compliance_state'] < 0.7]
        
        if low_compliance_policies.empty:
            return None
        
        refinement_suggestions = []
        for policy_id, compliance_rate in low_compliance_policies['compliance_state'].items():
            refinement_suggestions.append({
                'policy_id': policy_id,
                'current_compliance_rate': float(compliance_rate),
                'suggested_refinements': [
                    'Clarify policy conditions and requirements',
                    'Add grace periods for compliance',
                    'Refine resource scope and applicability',
                    'Improve policy documentation and guidance'
                ],
                'expected_improvement': '15-25% compliance rate increase'
            })
        
        return {
            'suggestions': refinement_suggestions,
            'methodology': 'Analysis of compliance patterns and violation reasons'
        }
```

### 3.2 Azure RBAC Integration Layer

The Azure RBAC integration layer provides comprehensive access control analysis and optimization capabilities. This layer enables intelligent access management through AI-powered analysis of access patterns, role effectiveness, and security risks.

**RBAC API Client Implementation**

```python
import asyncio
import aiohttp
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from typing import Dict, List, Optional, Any, Set
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid

@dataclass
class RoleAssignment:
    id: str
    role_definition_id: str
    principal_id: str
    principal_type: str
    scope: str
    created_on: datetime
    updated_on: datetime
    condition: Optional[str] = None

@dataclass
class RoleDefinition:
    id: str
    name: str
    description: str
    type: str
    permissions: List[Dict[str, Any]]
    assignable_scopes: List[str]
    created_on: datetime
    updated_on: datetime

class AzureRBACClient:
    def __init__(self, credential: DefaultAzureCredential):
        self.credential = credential
        self.base_url = "https://management.azure.com"
        self.api_version = "2022-04-01"
        self.session = None
        self.rate_limiter = RateLimiter(calls_per_minute=100)
        self.cache = RBACDataCache()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_access_token(self) -> str:
        """Get Azure access token for API authentication."""
        token = self.credential.get_token("https://management.azure.com/.default")
        return token.token
    
    async def list_role_assignments(self, scope: str, filter_expression: Optional[str] = None) -> List[RoleAssignment]:
        """
        List role assignments for a given scope.
        
        Args:
            scope: Azure resource scope
            filter_expression: OData filter expression
        """
        cache_key = f"role_assignments_{scope}_{filter_expression}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result and not self.cache.is_expired(cache_key):
            return cached_result
        
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{scope}/providers/Microsoft.Authorization/roleAssignments"
        
        params = {"api-version": self.api_version}
        if filter_expression:
            params["$filter"] = filter_expression
        
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    assignments = []
                    
                    for item in data.get("value", []):
                        properties = item.get("properties", {})
                        assignments.append(RoleAssignment(
                            id=item["id"],
                            role_definition_id=properties["roleDefinitionId"],
                            principal_id=properties["principalId"],
                            principal_type=properties.get("principalType", "Unknown"),
                            scope=properties["scope"],
                            created_on=datetime.fromisoformat(properties["createdOn"].replace("Z", "+00:00")),
                            updated_on=datetime.fromisoformat(properties["updatedOn"].replace("Z", "+00:00")),
                            condition=properties.get("condition")
                        ))
                    
                    self.cache.set(cache_key, assignments, ttl_minutes=30)
                    return assignments
                else:
                    raise AzureError(f"Failed to list role assignments: {response.status}")
                    
        except Exception as e:
            raise AzureError(f"Error listing role assignments: {str(e)}")
    
    async def get_role_definition(self, scope: str, role_definition_id: str) -> Optional[RoleDefinition]:
        """Get details of a specific role definition."""
        cache_key = f"role_definition_{role_definition_id}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result and not self.cache.is_expired(cache_key):
            return cached_result
        
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{scope}/providers/Microsoft.Authorization/roleDefinitions/{role_definition_id}"
        
        params = {"api-version": self.api_version}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    properties = data.get("properties", {})
                    
                    role_def = RoleDefinition(
                        id=data["id"],
                        name=properties["roleName"],
                        description=properties.get("description", ""),
                        type=properties.get("type", ""),
                        permissions=properties.get("permissions", []),
                        assignable_scopes=properties.get("assignableScopes", []),
                        created_on=datetime.fromisoformat(properties["createdOn"].replace("Z", "+00:00")),
                        updated_on=datetime.fromisoformat(properties["updatedOn"].replace("Z", "+00:00"))
                    )
                    
                    self.cache.set(cache_key, role_def, ttl_minutes=60)
                    return role_def
                elif response.status == 404:
                    return None
                else:
                    raise AzureError(f"Failed to get role definition: {response.status}")
                    
        except Exception as e:
            raise AzureError(f"Error getting role definition: {str(e)}")
    
    async def create_role_assignment(self, scope: str, role_definition_id: str, principal_id: str, 
                                   condition: Optional[str] = None) -> RoleAssignment:
        """Create a new role assignment."""
        await self.rate_limiter.wait_if_needed()
        
        assignment_id = str(uuid.uuid4())
        url = f"{self.base_url}/{scope}/providers/Microsoft.Authorization/roleAssignments/{assignment_id}"
        
        assignment_data = {
            "properties": {
                "roleDefinitionId": role_definition_id,
                "principalId": principal_id
            }
        }
        
        if condition:
            assignment_data["properties"]["condition"] = condition
            assignment_data["properties"]["conditionVersion"] = "2.0"
        
        params = {"api-version": self.api_version}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.put(url, json=assignment_data, params=params, headers=headers) as response:
                if response.status in [200, 201]:
                    data = await response.json()
                    properties = data.get("properties", {})
                    
                    assignment = RoleAssignment(
                        id=data["id"],
                        role_definition_id=properties["roleDefinitionId"],
                        principal_id=properties["principalId"],
                        principal_type=properties.get("principalType", "Unknown"),
                        scope=properties["scope"],
                        created_on=datetime.fromisoformat(properties["createdOn"].replace("Z", "+00:00")),
                        updated_on=datetime.fromisoformat(properties["updatedOn"].replace("Z", "+00:00")),
                        condition=properties.get("condition")
                    )
                    
                    # Invalidate related cache entries
                    self.cache.invalidate_pattern(f"role_assignments_{scope}*")
                    return assignment
                else:
                    error_text = await response.text()
                    raise AzureError(f"Failed to create role assignment: {response.status} - {error_text}")
                    
        except Exception as e:
            raise AzureError(f"Error creating role assignment: {str(e)}")
    
    async def delete_role_assignment(self, assignment_id: str) -> bool:
        """Delete a role assignment."""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}{assignment_id}"
        
        params = {"api-version": self.api_version}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.delete(url, params=params, headers=headers) as response:
                if response.status in [200, 204]:
                    # Invalidate related cache entries
                    self.cache.invalidate_pattern("role_assignments_*")
                    return True
                else:
                    return False
                    
        except Exception as e:
            raise AzureError(f"Error deleting role assignment: {str(e)}")

class RBACIntelligenceEngine:
    def __init__(self, rbac_client: AzureRBACClient, activity_analyzer: 'ActivityAnalyzer'):
        self.rbac_client = rbac_client
        self.activity_analyzer = activity_analyzer
        self.access_pattern_analyzer = AccessPatternAnalyzer()
        self.privilege_analyzer = PrivilegeAnalyzer()
        
    async def analyze_access_patterns(self, scope: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze access patterns to identify optimization opportunities and security risks.
        """
        # Get current role assignments
        role_assignments = await self.rbac_client.list_role_assignments(scope)
        
        # Get activity data for analysis
        activity_data = await self.activity_analyzer.get_activity_data(scope, days_back)
        
        # Analyze patterns
        patterns = self.access_pattern_analyzer.analyze_patterns(role_assignments, activity_data)
        
        # Identify optimization opportunities
        optimizations = await self.identify_access_optimizations(role_assignments, activity_data)
        
        # Assess security risks
        security_risks = self.assess_security_risks(role_assignments, activity_data)
        
        return {
            'scope': scope,
            'analysis_period': days_back,
            'total_assignments': len(role_assignments),
            'access_patterns': patterns,
            'optimization_opportunities': optimizations,
            'security_risks': security_risks,
            'recommendations': self.generate_rbac_recommendations(patterns, optimizations, security_risks)
        }
    
    async def identify_access_optimizations(self, assignments: List[RoleAssignment], 
                                          activity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify opportunities to optimize role assignments.
        """
        optimizations = []
        
        # Analyze unused permissions
        unused_permissions = self.privilege_analyzer.find_unused_permissions(assignments, activity_data)
        
        for principal_id, unused_perms in unused_permissions.items():
            if len(unused_perms) > 0:
                optimizations.append({
                    'type': 'unused_permissions',
                    'principal_id': principal_id,
                    'unused_permissions': unused_perms,
                    'recommendation': 'Consider removing unused permissions or switching to a more restrictive role',
                    'potential_risk_reduction': 'High'
                })
        
        # Analyze over-privileged accounts
        over_privileged = self.privilege_analyzer.find_over_privileged_accounts(assignments, activity_data)
        
        for account in over_privileged:
            optimizations.append({
                'type': 'over_privileged_account',
                'principal_id': account['principal_id'],
                'excessive_permissions': account['excessive_permissions'],
                'recommended_role': account['recommended_role'],
                'potential_risk_reduction': 'High'
            })
        
        # Analyze role consolidation opportunities
        consolidation_opportunities = self.identify_role_consolidation_opportunities(assignments)
        
        for opportunity in consolidation_opportunities:
            optimizations.append({
                'type': 'role_consolidation',
                'affected_principals': opportunity['principals'],
                'current_roles': opportunity['current_roles'],
                'recommended_role': opportunity['recommended_role'],
                'potential_risk_reduction': 'Medium'
            })
        
        return optimizations
    
    def assess_security_risks(self, assignments: List[RoleAssignment], 
                            activity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Assess security risks in current RBAC configuration.
        """
        risks = []
        
        # Check for privileged role assignments
        privileged_roles = [
            'Owner', 'Contributor', 'User Access Administrator', 
            'Security Administrator', 'Global Administrator'
        ]
        
        for assignment in assignments:
            role_def = self.get_role_definition_from_cache(assignment.role_definition_id)
            if role_def and role_def.name in privileged_roles:
                # Check if this is a service principal or user
                if assignment.principal_type == 'ServicePrincipal':
                    risks.append({
                        'type': 'privileged_service_principal',
                        'principal_id': assignment.principal_id,
                        'role_name': role_def.name,
                        'scope': assignment.scope,
                        'severity': 'High',
                        'description': f'Service principal has privileged role {role_def.name}'
                    })
                elif assignment.principal_type == 'User':
                    # Check activity patterns for this user
                    user_activity = activity_data.get('users', {}).get(assignment.principal_id, {})
                    if user_activity.get('last_activity_days', 999) > 90:
                        risks.append({
                            'type': 'inactive_privileged_user',
                            'principal_id': assignment.principal_id,
                            'role_name': role_def.name,
                            'scope': assignment.scope,
                            'last_activity': user_activity.get('last_activity_days'),
                            'severity': 'High',
                            'description': f'Inactive user with privileged role {role_def.name}'
                        })
        
        # Check for broad scope assignments
        for assignment in assignments:
            if '/subscriptions/' in assignment.scope and assignment.scope.count('/') == 2:
                # Subscription-level assignment
                role_def = self.get_role_definition_from_cache(assignment.role_definition_id)
                if role_def and any(perm.get('actions', []) == ['*'] for perm in role_def.permissions):
                    risks.append({
                        'type': 'broad_scope_assignment',
                        'principal_id': assignment.principal_id,
                        'role_name': role_def.name,
                        'scope': assignment.scope,
                        'severity': 'Medium',
                        'description': f'Broad permissions at subscription level'
                    })
        
        return risks
    
    def generate_rbac_recommendations(self, patterns: Dict, optimizations: List, risks: List) -> List[Dict[str, Any]]:
        """
        Generate actionable RBAC recommendations.
        """
        recommendations = []
        
        # High-priority security recommendations
        high_risk_count = len([r for r in risks if r.get('severity') == 'High'])
        if high_risk_count > 0:
            recommendations.append({
                'type': 'security_urgent',
                'priority': 'Critical',
                'title': f'Address {high_risk_count} High-Risk Security Issues',
                'description': 'Critical security risks identified in RBAC configuration',
                'actions': [
                    'Review and remove unnecessary privileged role assignments',
                    'Implement regular access reviews for privileged accounts',
                    'Enable conditional access for privileged operations',
                    'Implement just-in-time access for administrative roles'
                ],
                'estimated_impact': 'Significant security risk reduction'
            })
        
        # Optimization recommendations
        optimization_count = len(optimizations)
        if optimization_count > 0:
            recommendations.append({
                'type': 'access_optimization',
                'priority': 'High',
                'title': f'Optimize {optimization_count} Access Assignments',
                'description': 'Opportunities identified to optimize role assignments',
                'actions': [
                    'Remove unused permissions from over-privileged accounts',
                    'Consolidate similar role assignments',
                    'Implement principle of least privilege',
                    'Regular review of role effectiveness'
                ],
                'estimated_impact': '30-50% reduction in unnecessary permissions'
            })
        
        # Pattern-based recommendations
        if patterns.get('inactive_assignments', 0) > 0:
            recommendations.append({
                'type': 'lifecycle_management',
                'priority': 'Medium',
                'title': 'Implement Access Lifecycle Management',
                'description': f'Found {patterns["inactive_assignments"]} inactive role assignments',
                'actions': [
                    'Implement automated access reviews',
                    'Set up alerts for inactive accounts',
                    'Establish role assignment expiration policies',
                    'Regular cleanup of unused assignments'
                ],
                'estimated_impact': 'Improved security posture and compliance'
            })
        
        return recommendations

class AccessPatternAnalyzer:
    def analyze_patterns(self, assignments: List[RoleAssignment], activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze access patterns from role assignments and activity data.
        """
        patterns = {
            'total_assignments': len(assignments),
            'unique_principals': len(set(a.principal_id for a in assignments)),
            'unique_roles': len(set(a.role_definition_id for a in assignments)),
            'assignment_age_distribution': self.analyze_assignment_age(assignments),
            'scope_distribution': self.analyze_scope_distribution(assignments),
            'principal_type_distribution': self.analyze_principal_types(assignments),
            'inactive_assignments': self.count_inactive_assignments(assignments, activity_data)
        }
        
        return patterns
    
    def analyze_assignment_age(self, assignments: List[RoleAssignment]) -> Dict[str, int]:
        """Analyze the age distribution of role assignments."""
        now = datetime.now()
        age_buckets = {'0-30_days': 0, '31-90_days': 0, '91-365_days': 0, 'over_1_year': 0}
        
        for assignment in assignments:
            age_days = (now - assignment.created_on).days
            
            if age_days <= 30:
                age_buckets['0-30_days'] += 1
            elif age_days <= 90:
                age_buckets['31-90_days'] += 1
            elif age_days <= 365:
                age_buckets['91-365_days'] += 1
            else:
                age_buckets['over_1_year'] += 1
        
        return age_buckets
    
    def analyze_scope_distribution(self, assignments: List[RoleAssignment]) -> Dict[str, int]:
        """Analyze the distribution of assignment scopes."""
        scope_types = {'subscription': 0, 'resource_group': 0, 'resource': 0, 'management_group': 0}
        
        for assignment in assignments:
            scope_parts = assignment.scope.split('/')
            
            if 'managementGroups' in scope_parts:
                scope_types['management_group'] += 1
            elif 'subscriptions' in scope_parts and 'resourceGroups' in scope_parts:
                if len(scope_parts) > 5:  # Resource level
                    scope_types['resource'] += 1
                else:  # Resource group level
                    scope_types['resource_group'] += 1
            elif 'subscriptions' in scope_parts:
                scope_types['subscription'] += 1
        
        return scope_types
    
    def analyze_principal_types(self, assignments: List[RoleAssignment]) -> Dict[str, int]:
        """Analyze the distribution of principal types."""
        type_counts = {}
        
        for assignment in assignments:
            principal_type = assignment.principal_type
            type_counts[principal_type] = type_counts.get(principal_type, 0) + 1
        
        return type_counts
    
    def count_inactive_assignments(self, assignments: List[RoleAssignment], activity_data: Dict[str, Any]) -> int:
        """Count assignments for principals with no recent activity."""
        inactive_count = 0
        user_activity = activity_data.get('users', {})
        
        for assignment in assignments:
            if assignment.principal_type == 'User':
                user_data = user_activity.get(assignment.principal_id, {})
                last_activity_days = user_data.get('last_activity_days', 999)
                
                if last_activity_days > 90:  # No activity in 90 days
                    inactive_count += 1
        
        return inactive_count

class PrivilegeAnalyzer:
    def find_unused_permissions(self, assignments: List[RoleAssignment], 
                               activity_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Find permissions that are assigned but not used.
        """
        unused_permissions = {}
        
        # This would require detailed activity logs to determine which specific
        # permissions are being used. For now, we'll identify potentially unused
        # permissions based on role patterns and activity levels.
        
        for assignment in assignments:
            principal_activity = activity_data.get('users', {}).get(assignment.principal_id, {})
            
            # If user has very low activity, all permissions might be unused
            if principal_activity.get('activity_score', 0) < 0.1:
                role_def = self.get_role_definition_from_cache(assignment.role_definition_id)
                if role_def:
                    all_actions = []
                    for permission in role_def.permissions:
                        all_actions.extend(permission.get('actions', []))
                    
                    unused_permissions[assignment.principal_id] = all_actions
        
        return unused_permissions
    
    def find_over_privileged_accounts(self, assignments: List[RoleAssignment], 
                                    activity_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find accounts that have more privileges than they need based on usage patterns.
        """
        over_privileged = []
        
        # Group assignments by principal
        principal_assignments = {}
        for assignment in assignments:
            if assignment.principal_id not in principal_assignments:
                principal_assignments[assignment.principal_id] = []
            principal_assignments[assignment.principal_id].append(assignment)
        
        for principal_id, principal_assignments_list in principal_assignments.items():
            # Check if principal has multiple high-privilege roles
            high_privilege_roles = []
            
            for assignment in principal_assignments_list:
                role_def = self.get_role_definition_from_cache(assignment.role_definition_id)
                if role_def and self.is_high_privilege_role(role_def):
                    high_privilege_roles.append(role_def.name)
            
            if len(high_privilege_roles) > 1:
                # Principal has multiple high-privilege roles
                principal_activity = activity_data.get('users', {}).get(principal_id, {})
                
                over_privileged.append({
                    'principal_id': principal_id,
                    'excessive_permissions': high_privilege_roles,
                    'activity_score': principal_activity.get('activity_score', 0),
                    'recommended_role': self.recommend_minimal_role(principal_activity),
                    'risk_level': 'High' if len(high_privilege_roles) > 2 else 'Medium'
                })
        
        return over_privileged
    
    def is_high_privilege_role(self, role_def: RoleDefinition) -> bool:
        """
        Determine if a role definition represents high privileges.
        """
        high_privilege_indicators = [
            'Owner', 'Contributor', 'User Access Administrator',
            'Security Administrator', 'Global Administrator'
        ]
        
        if role_def.name in high_privilege_indicators:
            return True
        
        # Check for wildcard permissions
        for permission in role_def.permissions:
            if '*' in permission.get('actions', []):
                return True
        
        return False
    
    def recommend_minimal_role(self, activity_data: Dict[str, Any]) -> str:
        """
        Recommend a minimal role based on activity patterns.
        """
        activity_score = activity_data.get('activity_score', 0)
        resource_types = activity_data.get('accessed_resource_types', [])
        
        # Simple heuristic for role recommendation
        if activity_score < 0.1:
            return 'Reader'
        elif 'Microsoft.Compute' in resource_types and activity_score > 0.5:
            return 'Virtual Machine Contributor'
        elif 'Microsoft.Storage' in resource_types:
            return 'Storage Account Contributor'
        else:
            return 'Reader'
    
    def get_role_definition_from_cache(self, role_definition_id: str) -> Optional[RoleDefinition]:
        """
        Get role definition from cache (placeholder implementation).
        """
        # This would integrate with the RBAC client's cache
        # For now, return None to indicate cache miss
        return None

class RBACDataCache:
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl_minutes: int = 30):
        self.cache[key] = value
        self.timestamps[key] = datetime.now() + timedelta(minutes=ttl_minutes)
    
    def is_expired(self, key: str) -> bool:
        if key not in self.timestamps:
            return True
        return datetime.now() > self.timestamps[key]
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern."""
        keys_to_remove = [key for key in self.cache.keys() if pattern.replace("*", "") in key]
        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
```

This comprehensive implementation provides the foundation for intelligent RBAC management with AI-powered analysis, optimization recommendations, and security risk assessment. The system can analyze access patterns, identify over-privileged accounts, and provide actionable recommendations for improving security posture while maintaining operational efficiency.


### 3.3 Azure Network Integration Layer

The Azure Network integration layer provides comprehensive network monitoring, security analysis, and optimization capabilities. This layer enables intelligent network management through AI-powered analysis of traffic patterns, security configurations, and performance metrics.

**Network Monitoring Client Implementation**

```python
import asyncio
import aiohttp
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import ipaddress

@dataclass
class NetworkSecurityGroup:
    id: str
    name: str
    location: str
    resource_group: str
    security_rules: List[Dict[str, Any]]
    default_security_rules: List[Dict[str, Any]]
    network_interfaces: List[str]
    subnets: List[str]

@dataclass
class FlowLogRecord:
    timestamp: datetime
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    traffic_flow: str
    traffic_decision: str
    bytes_sent: int
    packets_sent: int

class AzureNetworkClient:
    def __init__(self, credential: DefaultAzureCredential):
        self.credential = credential
        self.base_url = "https://management.azure.com"
        self.api_version = "2023-04-01"
        self.session = None
        self.rate_limiter = RateLimiter(calls_per_minute=80)
        self.cache = NetworkDataCache()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_access_token(self) -> str:
        """Get Azure access token for API authentication."""
        token = self.credential.get_token("https://management.azure.com/.default")
        return token.token
    
    async def list_network_security_groups(self, subscription_id: str, resource_group: Optional[str] = None) -> List[NetworkSecurityGroup]:
        """
        List network security groups in a subscription or resource group.
        """
        cache_key = f"nsgs_{subscription_id}_{resource_group}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result and not self.cache.is_expired(cache_key):
            return cached_result
        
        await self.rate_limiter.wait_if_needed()
        
        if resource_group:
            url = f"{self.base_url}/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Network/networkSecurityGroups"
        else:
            url = f"{self.base_url}/subscriptions/{subscription_id}/providers/Microsoft.Network/networkSecurityGroups"
        
        params = {"api-version": self.api_version}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    nsgs = []
                    
                    for item in data.get("value", []):
                        properties = item.get("properties", {})
                        nsg = NetworkSecurityGroup(
                            id=item["id"],
                            name=item["name"],
                            location=item["location"],
                            resource_group=item["id"].split("/")[4],
                            security_rules=properties.get("securityRules", []),
                            default_security_rules=properties.get("defaultSecurityRules", []),
                            network_interfaces=[ni["id"] for ni in properties.get("networkInterfaces", [])],
                            subnets=[subnet["id"] for subnet in properties.get("subnets", [])]
                        )
                        nsgs.append(nsg)
                    
                    self.cache.set(cache_key, nsgs, ttl_minutes=30)
                    return nsgs
                else:
                    raise AzureError(f"Failed to list NSGs: {response.status}")
                    
        except Exception as e:
            raise AzureError(f"Error listing NSGs: {str(e)}")
    
    async def get_flow_logs(self, nsg_id: str, start_time: datetime, end_time: datetime) -> List[FlowLogRecord]:
        """
        Retrieve network flow logs for analysis.
        """
        cache_key = f"flow_logs_{nsg_id}_{start_time.isoformat()}_{end_time.isoformat()}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result and not self.cache.is_expired(cache_key):
            return cached_result
        
        await self.rate_limiter.wait_if_needed()
        
        # This would integrate with Azure Monitor Logs to query flow log data
        # For now, we'll simulate the structure
        url = f"https://api.loganalytics.io/v1/workspaces/{{workspace_id}}/query"
        
        query = f"""
        AzureNetworkAnalytics_CL
        | where TimeGenerated between (datetime({start_time.isoformat()}) .. datetime({end_time.isoformat()}))
        | where NSGName_s contains "{nsg_id.split('/')[-1]}"
        | project TimeGenerated, SrcIP_s, DestIP_s, SrcPort_d, DestPort_d, Protocol_s, FlowDirection_s, FlowStatus_s
        """
        
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        request_body = {"query": query}
        
        try:
            # This is a placeholder for actual Log Analytics integration
            flow_records = []  # Would be populated from actual query results
            
            self.cache.set(cache_key, flow_records, ttl_minutes=60)
            return flow_records
                    
        except Exception as e:
            raise AzureError(f"Error retrieving flow logs: {str(e)}")
    
    async def analyze_network_topology(self, subscription_id: str) -> Dict[str, Any]:
        """
        Analyze network topology and connectivity.
        """
        # Get all virtual networks
        vnets = await self.list_virtual_networks(subscription_id)
        
        # Get all network security groups
        nsgs = await self.list_network_security_groups(subscription_id)
        
        # Analyze topology
        topology = {
            'virtual_networks': len(vnets),
            'network_security_groups': len(nsgs),
            'connectivity_matrix': self.build_connectivity_matrix(vnets, nsgs),
            'security_analysis': self.analyze_security_posture(nsgs),
            'optimization_opportunities': self.identify_network_optimizations(vnets, nsgs)
        }
        
        return topology
    
    async def list_virtual_networks(self, subscription_id: str) -> List[Dict[str, Any]]:
        """List all virtual networks in a subscription."""
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/subscriptions/{subscription_id}/providers/Microsoft.Network/virtualNetworks"
        
        params = {"api-version": self.api_version}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("value", [])
                else:
                    raise AzureError(f"Failed to list VNets: {response.status}")
                    
        except Exception as e:
            raise AzureError(f"Error listing VNets: {str(e)}")
    
    def build_connectivity_matrix(self, vnets: List[Dict], nsgs: List[NetworkSecurityGroup]) -> Dict[str, Any]:
        """
        Build a connectivity matrix showing network relationships.
        """
        connectivity = {
            'vnet_peerings': [],
            'subnet_connectivity': {},
            'nsg_associations': {}
        }
        
        # Analyze VNet peerings
        for vnet in vnets:
            properties = vnet.get("properties", {})
            peerings = properties.get("virtualNetworkPeerings", [])
            
            for peering in peerings:
                connectivity['vnet_peerings'].append({
                    'source_vnet': vnet["name"],
                    'target_vnet': peering.get("properties", {}).get("remoteVirtualNetwork", {}).get("id", "").split("/")[-1],
                    'peering_state': peering.get("properties", {}).get("peeringState"),
                    'allow_forwarded_traffic': peering.get("properties", {}).get("allowForwardedTraffic", False)
                })
        
        # Analyze NSG associations
        for nsg in nsgs:
            connectivity['nsg_associations'][nsg.name] = {
                'associated_subnets': len(nsg.subnets),
                'associated_nics': len(nsg.network_interfaces),
                'rule_count': len(nsg.security_rules)
            }
        
        return connectivity
    
    def analyze_security_posture(self, nsgs: List[NetworkSecurityGroup]) -> Dict[str, Any]:
        """
        Analyze network security posture.
        """
        security_analysis = {
            'total_nsgs': len(nsgs),
            'security_issues': [],
            'compliance_score': 0,
            'recommendations': []
        }
        
        total_score = 0
        max_score = 0
        
        for nsg in nsgs:
            nsg_issues = []
            nsg_score = 0
            nsg_max_score = 0
            
            # Check for overly permissive rules
            for rule in nsg.security_rules:
                properties = rule.get("properties", {})
                
                # Check for rules allowing traffic from any source
                if properties.get("sourceAddressPrefix") == "*":
                    if properties.get("access") == "Allow":
                        nsg_issues.append({
                            'type': 'overly_permissive_source',
                            'rule_name': rule.get("name"),
                            'severity': 'High',
                            'description': 'Rule allows traffic from any source'
                        })
                
                # Check for rules allowing all ports
                if properties.get("destinationPortRange") == "*":
                    if properties.get("access") == "Allow":
                        nsg_issues.append({
                            'type': 'overly_permissive_ports',
                            'rule_name': rule.get("name"),
                            'severity': 'Medium',
                            'description': 'Rule allows traffic to all ports'
                        })
                
                # Scoring
                nsg_max_score += 10
                if properties.get("sourceAddressPrefix") != "*":
                    nsg_score += 5
                if properties.get("destinationPortRange") != "*":
                    nsg_score += 5
            
            # Check for missing essential rules
            has_ssh_restriction = any(
                rule.get("properties", {}).get("destinationPortRange") == "22" and
                rule.get("properties", {}).get("access") == "Deny"
                for rule in nsg.security_rules
            )
            
            if not has_ssh_restriction:
                nsg_issues.append({
                    'type': 'missing_ssh_restriction',
                    'severity': 'Medium',
                    'description': 'No explicit SSH restriction rule found'
                })
            
            security_analysis['security_issues'].extend([
                {**issue, 'nsg_name': nsg.name} for issue in nsg_issues
            ])
            
            total_score += nsg_score
            max_score += nsg_max_score
        
        security_analysis['compliance_score'] = (total_score / max_score * 100) if max_score > 0 else 0
        
        return security_analysis
    
    def identify_network_optimizations(self, vnets: List[Dict], nsgs: List[NetworkSecurityGroup]) -> List[Dict[str, Any]]:
        """
        Identify network optimization opportunities.
        """
        optimizations = []
        
        # Check for unused NSGs
        unused_nsgs = [nsg for nsg in nsgs if len(nsg.network_interfaces) == 0 and len(nsg.subnets) == 0]
        
        if unused_nsgs:
            optimizations.append({
                'type': 'unused_nsgs',
                'count': len(unused_nsgs),
                'nsg_names': [nsg.name for nsg in unused_nsgs],
                'potential_savings': 'Minimal cost savings, improved management',
                'recommendation': 'Remove unused network security groups'
            })
        
        # Check for duplicate rules across NSGs
        rule_patterns = {}
        for nsg in nsgs:
            for rule in nsg.security_rules:
                properties = rule.get("properties", {})
                pattern = (
                    properties.get("sourceAddressPrefix"),
                    properties.get("destinationAddressPrefix"),
                    properties.get("destinationPortRange"),
                    properties.get("protocol"),
                    properties.get("access")
                )
                
                if pattern not in rule_patterns:
                    rule_patterns[pattern] = []
                rule_patterns[pattern].append((nsg.name, rule.get("name")))
        
        duplicate_rules = {pattern: nsgs for pattern, nsgs in rule_patterns.items() if len(nsgs) > 1}
        
        if duplicate_rules:
            optimizations.append({
                'type': 'duplicate_rules',
                'count': len(duplicate_rules),
                'examples': list(duplicate_rules.keys())[:3],
                'recommendation': 'Consolidate duplicate security rules across NSGs'
            })
        
        # Check for overly complex NSGs
        complex_nsgs = [nsg for nsg in nsgs if len(nsg.security_rules) > 20]
        
        if complex_nsgs:
            optimizations.append({
                'type': 'complex_nsgs',
                'count': len(complex_nsgs),
                'nsg_names': [nsg.name for nsg in complex_nsgs],
                'recommendation': 'Simplify NSGs with excessive rules'
            })
        
        return optimizations

class NetworkIntelligenceEngine:
    def __init__(self, network_client: AzureNetworkClient, ml_models: Dict[str, Any]):
        self.network_client = network_client
        self.ml_models = ml_models
        self.traffic_analyzer = TrafficPatternAnalyzer()
        self.security_analyzer = NetworkSecurityAnalyzer()
        
    async def analyze_network_performance(self, subscription_id: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze network performance and identify optimization opportunities.
        """
        # Get network topology
        topology = await self.network_client.analyze_network_topology(subscription_id)
        
        # Analyze traffic patterns
        traffic_analysis = await self.analyze_traffic_patterns(subscription_id, days_back)
        
        # Identify performance bottlenecks
        bottlenecks = self.identify_performance_bottlenecks(traffic_analysis)
        
        # Generate optimization recommendations
        optimizations = self.generate_network_optimizations(topology, traffic_analysis, bottlenecks)
        
        return {
            'subscription_id': subscription_id,
            'analysis_period': days_back,
            'topology_summary': topology,
            'traffic_analysis': traffic_analysis,
            'performance_bottlenecks': bottlenecks,
            'optimization_recommendations': optimizations
        }
    
    async def analyze_traffic_patterns(self, subscription_id: str, days_back: int) -> Dict[str, Any]:
        """
        Analyze network traffic patterns using flow logs and metrics.
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Get all NSGs for flow log analysis
        nsgs = await self.network_client.list_network_security_groups(subscription_id)
        
        traffic_patterns = {
            'total_flows': 0,
            'top_talkers': [],
            'protocol_distribution': {},
            'traffic_trends': {},
            'anomalies': []
        }
        
        for nsg in nsgs:
            try:
                flow_logs = await self.network_client.get_flow_logs(nsg.id, start_time, end_time)
                
                # Analyze patterns for this NSG
                nsg_patterns = self.traffic_analyzer.analyze_nsg_traffic(flow_logs)
                
                # Aggregate results
                traffic_patterns['total_flows'] += nsg_patterns.get('flow_count', 0)
                
                # Merge protocol distribution
                for protocol, count in nsg_patterns.get('protocol_distribution', {}).items():
                    traffic_patterns['protocol_distribution'][protocol] = (
                        traffic_patterns['protocol_distribution'].get(protocol, 0) + count
                    )
                
                # Add top talkers
                traffic_patterns['top_talkers'].extend(nsg_patterns.get('top_talkers', []))
                
            except Exception as e:
                # Log error but continue with other NSGs
                print(f"Error analyzing traffic for NSG {nsg.name}: {str(e)}")
        
        # Sort and limit top talkers
        traffic_patterns['top_talkers'] = sorted(
            traffic_patterns['top_talkers'], 
            key=lambda x: x.get('bytes_transferred', 0), 
            reverse=True
        )[:10]
        
        return traffic_patterns
    
    def identify_performance_bottlenecks(self, traffic_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify network performance bottlenecks from traffic analysis.
        """
        bottlenecks = []
        
        # Check for high-volume connections that might indicate bottlenecks
        top_talkers = traffic_analysis.get('top_talkers', [])
        
        for talker in top_talkers[:5]:  # Top 5 talkers
            if talker.get('bytes_transferred', 0) > 1e9:  # > 1GB
                bottlenecks.append({
                    'type': 'high_volume_connection',
                    'source_ip': talker.get('source_ip'),
                    'destination_ip': talker.get('destination_ip'),
                    'bytes_transferred': talker.get('bytes_transferred'),
                    'severity': 'Medium',
                    'recommendation': 'Monitor for potential bandwidth saturation'
                })
        
        # Check protocol distribution for anomalies
        protocol_dist = traffic_analysis.get('protocol_distribution', {})
        total_flows = sum(protocol_dist.values())
        
        if total_flows > 0:
            for protocol, count in protocol_dist.items():
                percentage = (count / total_flows) * 100
                
                if protocol == 'TCP' and percentage < 50:
                    bottlenecks.append({
                        'type': 'unusual_protocol_distribution',
                        'protocol': protocol,
                        'percentage': percentage,
                        'severity': 'Low',
                        'recommendation': 'Investigate unusual protocol distribution'
                    })
        
        return bottlenecks
    
    def generate_network_optimizations(self, topology: Dict, traffic_analysis: Dict, bottlenecks: List) -> List[Dict[str, Any]]:
        """
        Generate network optimization recommendations.
        """
        optimizations = []
        
        # Security optimizations from topology analysis
        security_issues = topology.get('security_analysis', {}).get('security_issues', [])
        high_severity_issues = [issue for issue in security_issues if issue.get('severity') == 'High']
        
        if high_severity_issues:
            optimizations.append({
                'type': 'security_hardening',
                'priority': 'High',
                'title': f'Address {len(high_severity_issues)} High-Severity Security Issues',
                'description': 'Critical network security vulnerabilities identified',
                'actions': [
                    'Restrict overly permissive NSG rules',
                    'Implement network segmentation',
                    'Enable network monitoring and alerting',
                    'Regular security rule audits'
                ],
                'estimated_impact': 'Significant security risk reduction'
            })
        
        # Performance optimizations from traffic analysis
        if bottlenecks:
            optimizations.append({
                'type': 'performance_optimization',
                'priority': 'Medium',
                'title': f'Optimize {len(bottlenecks)} Network Performance Issues',
                'description': 'Network performance bottlenecks identified',
                'actions': [
                    'Optimize high-volume connections',
                    'Implement traffic shaping policies',
                    'Consider network acceleration solutions',
                    'Monitor bandwidth utilization'
                ],
                'estimated_impact': 'Improved network performance and user experience'
            })
        
        # Cost optimizations
        unused_nsgs = topology.get('optimization_opportunities', [])
        unused_nsg_count = sum(1 for opt in unused_nsgs if opt.get('type') == 'unused_nsgs')
        
        if unused_nsg_count > 0:
            optimizations.append({
                'type': 'cost_optimization',
                'priority': 'Low',
                'title': 'Clean Up Unused Network Resources',
                'description': f'Found {unused_nsg_count} unused network security groups',
                'actions': [
                    'Remove unused NSGs',
                    'Consolidate duplicate rules',
                    'Simplify complex NSG configurations',
                    'Implement resource lifecycle management'
                ],
                'estimated_impact': 'Reduced management overhead and improved clarity'
            })
        
        return optimizations

class TrafficPatternAnalyzer:
    def analyze_nsg_traffic(self, flow_logs: List[FlowLogRecord]) -> Dict[str, Any]:
        """
        Analyze traffic patterns for a specific NSG.
        """
        if not flow_logs:
            return {
                'flow_count': 0,
                'protocol_distribution': {},
                'top_talkers': [],
                'traffic_trends': {}
            }
        
        # Protocol distribution
        protocol_counts = {}
        for log in flow_logs:
            protocol_counts[log.protocol] = protocol_counts.get(log.protocol, 0) + 1
        
        # Top talkers by bytes transferred
        talker_stats = {}
        for log in flow_logs:
            key = (log.source_ip, log.destination_ip)
            if key not in talker_stats:
                talker_stats[key] = {'bytes': 0, 'packets': 0, 'flows': 0}
            
            talker_stats[key]['bytes'] += log.bytes_sent
            talker_stats[key]['packets'] += log.packets_sent
            talker_stats[key]['flows'] += 1
        
        top_talkers = [
            {
                'source_ip': key[0],
                'destination_ip': key[1],
                'bytes_transferred': stats['bytes'],
                'packet_count': stats['packets'],
                'flow_count': stats['flows']
            }
            for key, stats in sorted(talker_stats.items(), key=lambda x: x[1]['bytes'], reverse=True)[:10]
        ]
        
        return {
            'flow_count': len(flow_logs),
            'protocol_distribution': protocol_counts,
            'top_talkers': top_talkers,
            'traffic_trends': self.calculate_traffic_trends(flow_logs)
        }
    
    def calculate_traffic_trends(self, flow_logs: List[FlowLogRecord]) -> Dict[str, Any]:
        """
        Calculate traffic trends over time.
        """
        if not flow_logs:
            return {}
        
        # Group by hour for trend analysis
        hourly_stats = {}
        for log in flow_logs:
            hour_key = log.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_stats:
                hourly_stats[hour_key] = {'bytes': 0, 'flows': 0}
            
            hourly_stats[hour_key]['bytes'] += log.bytes_sent
            hourly_stats[hour_key]['flows'] += 1
        
        # Calculate peak hours
        if hourly_stats:
            peak_hour = max(hourly_stats.items(), key=lambda x: x[1]['bytes'])
            avg_bytes = sum(stats['bytes'] for stats in hourly_stats.values()) / len(hourly_stats)
            
            return {
                'peak_hour': peak_hour[0].isoformat(),
                'peak_bytes': peak_hour[1]['bytes'],
                'average_hourly_bytes': avg_bytes,
                'total_hours_analyzed': len(hourly_stats)
            }
        
        return {}

class NetworkSecurityAnalyzer:
    def analyze_security_posture(self, nsgs: List[NetworkSecurityGroup]) -> Dict[str, Any]:
        """
        Comprehensive network security analysis.
        """
        security_metrics = {
            'overall_score': 0,
            'critical_issues': [],
            'compliance_status': {},
            'recommendations': []
        }
        
        total_score = 0
        max_score = 0
        
        for nsg in nsgs:
            nsg_analysis = self.analyze_nsg_security(nsg)
            
            total_score += nsg_analysis['score']
            max_score += nsg_analysis['max_score']
            
            security_metrics['critical_issues'].extend(nsg_analysis['issues'])
        
        security_metrics['overall_score'] = (total_score / max_score * 100) if max_score > 0 else 0
        
        return security_metrics
    
    def analyze_nsg_security(self, nsg: NetworkSecurityGroup) -> Dict[str, Any]:
        """
        Analyze security configuration of a single NSG.
        """
        issues = []
        score = 0
        max_score = 0
        
        # Check each security rule
        for rule in nsg.security_rules:
            properties = rule.get("properties", {})
            rule_score, rule_max_score, rule_issues = self.analyze_security_rule(rule)
            
            score += rule_score
            max_score += rule_max_score
            issues.extend([{**issue, 'nsg_name': nsg.name, 'rule_name': rule.get("name")} for issue in rule_issues])
        
        return {
            'nsg_name': nsg.name,
            'score': score,
            'max_score': max_score,
            'issues': issues
        }
    
    def analyze_security_rule(self, rule: Dict[str, Any]) -> Tuple[int, int, List[Dict[str, Any]]]:
        """
        Analyze a single security rule for security issues.
        """
        properties = rule.get("properties", {})
        issues = []
        score = 0
        max_score = 10
        
        # Check source address prefix
        source_prefix = properties.get("sourceAddressPrefix", "")
        if source_prefix == "*":
            issues.append({
                'type': 'overly_permissive_source',
                'severity': 'High',
                'description': 'Rule allows traffic from any source (0.0.0.0/0)'
            })
        else:
            score += 3
        
        # Check destination port range
        dest_port = properties.get("destinationPortRange", "")
        if dest_port == "*":
            issues.append({
                'type': 'overly_permissive_ports',
                'severity': 'Medium',
                'description': 'Rule allows traffic to all ports'
            })
        else:
            score += 2
        
        # Check for dangerous port combinations
        if properties.get("access") == "Allow":
            dangerous_ports = ["22", "3389", "1433", "3306", "5432"]
            if dest_port in dangerous_ports and source_prefix == "*":
                issues.append({
                    'type': 'dangerous_port_exposure',
                    'severity': 'Critical',
                    'description': f'Port {dest_port} exposed to internet'
                })
            else:
                score += 3
        
        # Check protocol specificity
        protocol = properties.get("protocol", "")
        if protocol != "*":
            score += 2
        
        return score, max_score, issues

class NetworkDataCache:
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl_minutes: int = 30):
        self.cache[key] = value
        self.timestamps[key] = datetime.now() + timedelta(minutes=ttl_minutes)
    
    def is_expired(self, key: str) -> bool:
        if key not in self.timestamps:
            return True
        return datetime.now() > self.timestamps[key]
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern."""
        keys_to_remove = [key for key in self.cache.keys() if pattern.replace("*", "") in key]
        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
```

### 3.4 Azure Cost Management Integration Layer

The Azure Cost Management integration layer provides comprehensive cost analysis, optimization, and forecasting capabilities. This layer enables intelligent cost management through AI-powered analysis of spending patterns, resource utilization, and optimization opportunities.

**Cost Management Client Implementation**

```python
import asyncio
import aiohttp
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from typing import Dict, List, Optional, Any, Union
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

@dataclass
class CostData:
    date: datetime
    cost: float
    currency: str
    resource_group: str
    resource_type: str
    resource_name: str
    meter_category: str
    meter_subcategory: str
    meter_name: str
    unit_of_measure: str
    quantity: float

@dataclass
class BudgetAlert:
    budget_name: str
    current_spend: float
    budget_amount: float
    percentage_used: float
    forecast_spend: float
    alert_threshold: float
    status: str

class AzureCostManagementClient:
    def __init__(self, credential: DefaultAzureCredential):
        self.credential = credential
        self.base_url = "https://management.azure.com"
        self.api_version = "2023-03-01"
        self.session = None
        self.rate_limiter = RateLimiter(calls_per_minute=60)
        self.cache = CostDataCache()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_access_token(self) -> str:
        """Get Azure access token for API authentication."""
        token = self.credential.get_token("https://management.azure.com/.default")
        return token.token
    
    async def get_cost_data(self, scope: str, start_date: datetime, end_date: datetime, 
                           granularity: str = "Daily") -> List[CostData]:
        """
        Retrieve cost data for a given scope and time range.
        
        Args:
            scope: Azure scope (subscription, resource group, etc.)
            start_date: Start date for cost data
            end_date: End date for cost data
            granularity: Data granularity (Daily, Monthly)
        """
        cache_key = f"cost_data_{scope}_{start_date.date()}_{end_date.date()}_{granularity}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result and not self.cache.is_expired(cache_key):
            return cached_result
        
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{scope}/providers/Microsoft.CostManagement/query"
        
        query_body = {
            "type": "ActualCost",
            "timeframe": "Custom",
            "timePeriod": {
                "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                "to": end_date.strftime("%Y-%m-%dT23:59:59Z")
            },
            "dataset": {
                "granularity": granularity,
                "aggregation": {
                    "totalCost": {
                        "name": "Cost",
                        "function": "Sum"
                    }
                },
                "grouping": [
                    {
                        "type": "Dimension",
                        "name": "ResourceGroup"
                    },
                    {
                        "type": "Dimension",
                        "name": "ResourceType"
                    },
                    {
                        "type": "Dimension",
                        "name": "ResourceId"
                    },
                    {
                        "type": "Dimension",
                        "name": "MeterCategory"
                    }
                ]
            }
        }
        
        params = {"api-version": self.api_version}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.post(url, json=query_body, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    cost_records = self.parse_cost_data(data)
                    
                    self.cache.set(cache_key, cost_records, ttl_minutes=60)
                    return cost_records
                else:
                    error_text = await response.text()
                    raise AzureError(f"Failed to retrieve cost data: {response.status} - {error_text}")
                    
        except Exception as e:
            raise AzureError(f"Error retrieving cost data: {str(e)}")
    
    def parse_cost_data(self, raw_data: Dict[str, Any]) -> List[CostData]:
        """
        Parse raw cost management API response into structured cost data.
        """
        cost_records = []
        
        properties = raw_data.get("properties", {})
        rows = properties.get("rows", [])
        columns = properties.get("columns", [])
        
        # Create column mapping
        column_map = {col["name"]: idx for idx, col in enumerate(columns)}
        
        for row in rows:
            try:
                cost_record = CostData(
                    date=datetime.fromisoformat(row[column_map.get("UsageDate", 0)].replace("Z", "+00:00")),
                    cost=float(row[column_map.get("Cost", 1)]),
                    currency=row[column_map.get("Currency", 2)] if "Currency" in column_map else "USD",
                    resource_group=row[column_map.get("ResourceGroup", 3)] if "ResourceGroup" in column_map else "",
                    resource_type=row[column_map.get("ResourceType", 4)] if "ResourceType" in column_map else "",
                    resource_name=row[column_map.get("ResourceId", 5)] if "ResourceId" in column_map else "",
                    meter_category=row[column_map.get("MeterCategory", 6)] if "MeterCategory" in column_map else "",
                    meter_subcategory=row[column_map.get("MeterSubcategory", 7)] if "MeterSubcategory" in column_map else "",
                    meter_name=row[column_map.get("MeterName", 8)] if "MeterName" in column_map else "",
                    unit_of_measure=row[column_map.get("UnitOfMeasure", 9)] if "UnitOfMeasure" in column_map else "",
                    quantity=float(row[column_map.get("Quantity", 10)]) if "Quantity" in column_map else 0.0
                )
                cost_records.append(cost_record)
            except (ValueError, IndexError, KeyError) as e:
                # Skip malformed records
                continue
        
        return cost_records
    
    async def get_budget_alerts(self, scope: str) -> List[BudgetAlert]:
        """
        Retrieve budget alerts for a given scope.
        """
        cache_key = f"budget_alerts_{scope}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result and not self.cache.is_expired(cache_key):
            return cached_result
        
        await self.rate_limiter.wait_if_needed()
        
        url = f"{self.base_url}/{scope}/providers/Microsoft.Consumption/budgets"
        
        params = {"api-version": "2023-05-01"}
        headers = {
            "Authorization": f"Bearer {await self.get_access_token()}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    budget_alerts = self.parse_budget_data(data)
                    
                    self.cache.set(cache_key, budget_alerts, ttl_minutes=30)
                    return budget_alerts
                else:
                    raise AzureError(f"Failed to retrieve budget data: {response.status}")
                    
        except Exception as e:
            raise AzureError(f"Error retrieving budget data: {str(e)}")
    
    def parse_budget_data(self, raw_data: Dict[str, Any]) -> List[BudgetAlert]:
        """
        Parse budget data and calculate alert status.
        """
        budget_alerts = []
        
        for budget_item in raw_data.get("value", []):
            properties = budget_item.get("properties", {})
            
            budget_name = budget_item.get("name", "")
            budget_amount = properties.get("amount", 0)
            current_spend = properties.get("currentSpend", {}).get("amount", 0)
            forecast_spend = properties.get("forecastSpend", {}).get("amount", 0)
            
            percentage_used = (current_spend / budget_amount * 100) if budget_amount > 0 else 0
            
            # Determine alert status
            notifications = properties.get("notifications", {})
            alert_threshold = 80  # Default threshold
            
            for notification in notifications.values():
                threshold = notification.get("threshold", 80)
                if threshold < alert_threshold:
                    alert_threshold = threshold
            
            status = "OK"
            if percentage_used >= alert_threshold:
                status = "ALERT"
            elif percentage_used >= (alert_threshold * 0.8):
                status = "WARNING"
            
            budget_alert = BudgetAlert(
                budget_name=budget_name,
                current_spend=current_spend,
                budget_amount=budget_amount,
                percentage_used=percentage_used,
                forecast_spend=forecast_spend,
                alert_threshold=alert_threshold,
                status=status
            )
            
            budget_alerts.append(budget_alert)
        
        return budget_alerts
    
    async def get_resource_utilization(self, subscription_id: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get resource utilization data for cost optimization analysis.
        """
        # This would integrate with Azure Monitor to get utilization metrics
        # For now, we'll return a placeholder structure
        
        utilization_data = {
            'virtual_machines': await self.get_vm_utilization(subscription_id, days_back),
            'storage_accounts': await self.get_storage_utilization(subscription_id, days_back),
            'databases': await self.get_database_utilization(subscription_id, days_back)
        }
        
        return utilization_data
    
    async def get_vm_utilization(self, subscription_id: str, days_back: int) -> List[Dict[str, Any]]:
        """
        Get virtual machine utilization data.
        """
        # Placeholder implementation - would integrate with Azure Monitor
        return [
            {
                'resource_id': f'/subscriptions/{subscription_id}/resourceGroups/rg1/providers/Microsoft.Compute/virtualMachines/vm1',
                'resource_name': 'vm1',
                'cpu_utilization_avg': 15.5,
                'memory_utilization_avg': 45.2,
                'network_utilization_avg': 8.1,
                'disk_utilization_avg': 25.3,
                'uptime_percentage': 98.5,
                'cost_per_day': 24.50,
                'optimization_potential': 'High'
            }
        ]
    
    async def get_storage_utilization(self, subscription_id: str, days_back: int) -> List[Dict[str, Any]]:
        """
        Get storage account utilization data.
        """
        # Placeholder implementation
        return [
            {
                'resource_id': f'/subscriptions/{subscription_id}/resourceGroups/rg1/providers/Microsoft.Storage/storageAccounts/storage1',
                'resource_name': 'storage1',
                'capacity_used_percentage': 35.8,
                'transaction_count_avg': 1250,
                'cost_per_day': 12.75,
                'optimization_potential': 'Medium'
            }
        ]
    
    async def get_database_utilization(self, subscription_id: str, days_back: int) -> List[Dict[str, Any]]:
        """
        Get database utilization data.
        """
        # Placeholder implementation
        return [
            {
                'resource_id': f'/subscriptions/{subscription_id}/resourceGroups/rg1/providers/Microsoft.Sql/servers/sqlserver1/databases/db1',
                'resource_name': 'db1',
                'dtu_utilization_avg': 25.4,
                'storage_utilization_percentage': 60.2,
                'connection_count_avg': 45,
                'cost_per_day': 45.80,
                'optimization_potential': 'Low'
            }
        ]

class CostIntelligenceEngine:
    def __init__(self, cost_client: AzureCostManagementClient, ml_models: Dict[str, Any]):
        self.cost_client = cost_client
        self.ml_models = ml_models
        self.cost_analyzer = CostPatternAnalyzer()
        self.optimization_engine = CostOptimizationEngine()
        
    async def analyze_cost_trends(self, scope: str, days_back: int = 90) -> Dict[str, Any]:
        """
        Analyze cost trends and provide insights.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get cost data
        cost_data = await self.cost_client.get_cost_data(scope, start_date, end_date)
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'date': cd.date,
                'cost': cd.cost,
                'resource_group': cd.resource_group,
                'resource_type': cd.resource_type,
                'meter_category': cd.meter_category
            }
            for cd in cost_data
        ])
        
        if df.empty:
            return {'error': 'No cost data found for the specified scope and time range'}
        
        # Analyze trends
        trends = self.cost_analyzer.analyze_trends(df)
        
        # Forecast future costs
        forecast = await self.forecast_costs(df, scope)
        
        # Identify cost anomalies
        anomalies = self.identify_cost_anomalies(df)
        
        # Get optimization opportunities
        optimizations = await self.identify_cost_optimizations(scope)
        
        return {
            'scope': scope,
            'analysis_period': {'start': start_date, 'end': end_date},
            'total_cost': float(df['cost'].sum()),
            'daily_average': float(df.groupby('date')['cost'].sum().mean()),
            'trends': trends,
            'forecast': forecast,
            'anomalies': anomalies,
            'optimization_opportunities': optimizations,
            'recommendations': self.generate_cost_recommendations(trends, forecast, anomalies, optimizations)
        }
    
    async def forecast_costs(self, historical_data: pd.DataFrame, scope: str) -> Dict[str, Any]:
        """
        Forecast future costs using ML models.
        """
        # Prepare data for forecasting
        daily_costs = historical_data.groupby('date')['cost'].sum().reset_index()
        daily_costs = daily_costs.sort_values('date')
        
        if len(daily_costs) < 7:
            return {'error': 'Insufficient data for forecasting'}
        
        # Use cost forecasting model
        cost_model = self.ml_models.get('cost_forecaster')
        if not cost_model:
            # Fallback to simple trend analysis
            return self.simple_cost_forecast(daily_costs)
        
        # Generate 30-day forecast
        forecast_days = 30
        forecasts = []
        
        for days_ahead in range(1, forecast_days + 1):
            forecast_date = daily_costs['date'].max() + timedelta(days=days_ahead)
            
            # Prepare features for prediction
            features = self.prepare_forecast_features(daily_costs, days_ahead)
            
            # Generate prediction
            predicted_cost = cost_model.predict(features.reshape(1, -1))[0]
            
            forecasts.append({
                'date': forecast_date,
                'predicted_cost': float(predicted_cost),
                'confidence_interval': self.calculate_forecast_confidence(predicted_cost, days_ahead)
            })
        
        return {
            'forecasts': forecasts,
            'model_accuracy': cost_model.score if hasattr(cost_model, 'score') else 0.85,
            'forecast_methodology': 'Machine learning with time series analysis'
        }
    
    def simple_cost_forecast(self, daily_costs: pd.DataFrame) -> Dict[str, Any]:
        """
        Simple trend-based cost forecasting.
        """
        # Calculate trend
        daily_costs['days'] = (daily_costs['date'] - daily_costs['date'].min()).dt.days
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(daily_costs[['days']], daily_costs['cost'])
        
        forecasts = []
        last_day = daily_costs['days'].max()
        
        for days_ahead in range(1, 31):
            forecast_date = daily_costs['date'].max() + timedelta(days=days_ahead)
            predicted_cost = model.predict([[last_day + days_ahead]])[0]
            
            forecasts.append({
                'date': forecast_date,
                'predicted_cost': float(max(0, predicted_cost)),  # Ensure non-negative
                'confidence_interval': {'lower': predicted_cost * 0.8, 'upper': predicted_cost * 1.2}
            })
        
        return {
            'forecasts': forecasts,
            'model_accuracy': model.score(daily_costs[['days']], daily_costs['cost']),
            'forecast_methodology': 'Linear trend analysis'
        }
    
    def identify_cost_anomalies(self, cost_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify cost anomalies using statistical analysis.
        """
        anomalies = []
        
        # Daily cost anomalies
        daily_costs = cost_data.groupby('date')['cost'].sum()
        
        # Calculate rolling statistics
        rolling_mean = daily_costs.rolling(window=7).mean()
        rolling_std = daily_costs.rolling(window=7).std()
        
        # Identify anomalies (costs > 2 standard deviations from mean)
        threshold = 2
        anomaly_mask = abs(daily_costs - rolling_mean) > (threshold * rolling_std)
        
        for date, is_anomaly in anomaly_mask.items():
            if is_anomaly and not pd.isna(rolling_mean[date]):
                cost = daily_costs[date]
                expected_cost = rolling_mean[date]
                deviation = abs(cost - expected_cost)
                
                anomalies.append({
                    'type': 'daily_cost_anomaly',
                    'date': date,
                    'actual_cost': float(cost),
                    'expected_cost': float(expected_cost),
                    'deviation': float(deviation),
                    'severity': 'High' if deviation > (3 * rolling_std[date]) else 'Medium'
                })
        
        # Resource group anomalies
        rg_costs = cost_data.groupby(['date', 'resource_group'])['cost'].sum().reset_index()
        
        for rg in rg_costs['resource_group'].unique():
            rg_data = rg_costs[rg_costs['resource_group'] == rg]
            if len(rg_data) > 7:  # Need sufficient data
                rg_mean = rg_data['cost'].mean()
                rg_std = rg_data['cost'].std()
                
                recent_cost = rg_data.tail(1)['cost'].iloc[0]
                if abs(recent_cost - rg_mean) > (2 * rg_std):
                    anomalies.append({
                        'type': 'resource_group_anomaly',
                        'resource_group': rg,
                        'recent_cost': float(recent_cost),
                        'average_cost': float(rg_mean),
                        'deviation': float(abs(recent_cost - rg_mean)),
                        'severity': 'Medium'
                    })
        
        return anomalies
    
    async def identify_cost_optimizations(self, scope: str) -> List[Dict[str, Any]]:
        """
        Identify cost optimization opportunities.
        """
        optimizations = []
        
        # Get utilization data
        subscription_id = scope.split('/')[2] if '/subscriptions/' in scope else scope
        utilization_data = await self.cost_client.get_resource_utilization(subscription_id)
        
        # Analyze VM optimization opportunities
        vm_optimizations = self.optimization_engine.analyze_vm_optimizations(
            utilization_data.get('virtual_machines', [])
        )
        optimizations.extend(vm_optimizations)
        
        # Analyze storage optimization opportunities
        storage_optimizations = self.optimization_engine.analyze_storage_optimizations(
            utilization_data.get('storage_accounts', [])
        )
        optimizations.extend(storage_optimizations)
        
        # Analyze database optimization opportunities
        db_optimizations = self.optimization_engine.analyze_database_optimizations(
            utilization_data.get('databases', [])
        )
        optimizations.extend(db_optimizations)
        
        return optimizations
    
    def generate_cost_recommendations(self, trends: Dict, forecast: Dict, anomalies: List, optimizations: List) -> List[Dict[str, Any]]:
        """
        Generate actionable cost management recommendations.
        """
        recommendations = []
        
        # Trend-based recommendations
        if trends.get('trend_direction') == 'increasing':
            trend_slope = trends.get('trend_slope', 0)
            if trend_slope > 0.05:  # 5% daily increase
                recommendations.append({
                    'type': 'cost_trend_alert',
                    'priority': 'High',
                    'title': 'Address Rapidly Increasing Costs',
                    'description': f'Costs are increasing at {trend_slope:.1%} per day',
                    'actions': [
                        'Investigate recent resource deployments',
                        'Review resource scaling policies',
                        'Implement cost alerts and budgets',
                        'Analyze top cost contributors'
                    ],
                    'estimated_impact': 'Prevent runaway costs'
                })
        
        # Anomaly-based recommendations
        high_severity_anomalies = [a for a in anomalies if a.get('severity') == 'High']
        if high_severity_anomalies:
            recommendations.append({
                'type': 'cost_anomaly_investigation',
                'priority': 'High',
                'title': f'Investigate {len(high_severity_anomalies)} Cost Anomalies',
                'description': 'Significant cost deviations detected',
                'actions': [
                    'Review resource usage during anomaly periods',
                    'Check for unauthorized resource deployments',
                    'Validate billing data accuracy',
                    'Implement anomaly detection alerts'
                ],
                'affected_dates': [a.get('date') for a in high_severity_anomalies if a.get('date')],
                'estimated_impact': 'Identify and prevent cost overruns'
            })
        
        # Optimization-based recommendations
        high_potential_optimizations = [o for o in optimizations if o.get('optimization_potential') == 'High']
        if high_potential_optimizations:
            total_savings = sum(o.get('estimated_monthly_savings', 0) for o in high_potential_optimizations)
            
            recommendations.append({
                'type': 'cost_optimization',
                'priority': 'Medium',
                'title': f'Implement {len(high_potential_optimizations)} High-Impact Optimizations',
                'description': f'Potential monthly savings: ${total_savings:.2f}',
                'actions': [
                    'Right-size underutilized resources',
                    'Implement automated scaling policies',
                    'Consider reserved instance purchases',
                    'Optimize storage tiers and lifecycle policies'
                ],
                'estimated_savings': total_savings,
                'estimated_impact': f'${total_savings:.2f} monthly cost reduction'
            })
        
        # Forecast-based recommendations
        if forecast.get('forecasts'):
            monthly_forecast = sum(f.get('predicted_cost', 0) for f in forecast['forecasts'])
            
            recommendations.append({
                'type': 'budget_planning',
                'priority': 'Medium',
                'title': 'Update Budget Based on Cost Forecast',
                'description': f'Forecasted monthly cost: ${monthly_forecast:.2f}',
                'actions': [
                    'Review and adjust budget allocations',
                    'Set up proactive cost alerts',
                    'Plan for seasonal cost variations',
                    'Implement cost governance policies'
                ],
                'forecasted_monthly_cost': monthly_forecast,
                'estimated_impact': 'Improved cost predictability and control'
            })
        
        return recommendations

class CostPatternAnalyzer:
    def analyze_trends(self, cost_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cost trends from historical data.
        """
        # Daily cost aggregation
        daily_costs = cost_data.groupby('date')['cost'].sum().sort_index()
        
        if len(daily_costs) < 2:
            return {
                'trend_direction': 'insufficient_data',
                'trend_slope': 0,
                'trend_confidence': 0
            }
        
        # Linear regression for trend analysis
        from sklearn.linear_model import LinearRegression
        
        daily_costs_reset = daily_costs.reset_index()
        daily_costs_reset['days'] = (daily_costs_reset['date'] - daily_costs_reset['date'].min()).dt.days
        
        X = daily_costs_reset[['days']]
        y = daily_costs_reset['cost']
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_slope = model.coef_[0]
        trend_confidence = model.score(X, y)
        
        # Determine trend direction
        if trend_slope > 0.01:  # Increasing by more than $0.01 per day
            trend_direction = 'increasing'
        elif trend_slope < -0.01:  # Decreasing by more than $0.01 per day
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'
        
        # Calculate additional metrics
        total_cost = float(daily_costs.sum())
        average_daily_cost = float(daily_costs.mean())
        cost_volatility = float(daily_costs.std() / daily_costs.mean()) if daily_costs.mean() > 0 else 0
        
        return {
            'trend_direction': trend_direction,
            'trend_slope': float(trend_slope),
            'trend_confidence': float(trend_confidence),
            'total_cost': total_cost,
            'average_daily_cost': average_daily_cost,
            'cost_volatility': cost_volatility,
            'analysis_period_days': len(daily_costs)
        }

class CostOptimizationEngine:
    def analyze_vm_optimizations(self, vm_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze virtual machine optimization opportunities.
        """
        optimizations = []
        
        for vm in vm_data:
            optimization_potential = vm.get('optimization_potential', 'Low')
            
            if optimization_potential in ['High', 'Medium']:
                cpu_util = vm.get('cpu_utilization_avg', 0)
                memory_util = vm.get('memory_utilization_avg', 0)
                daily_cost = vm.get('cost_per_day', 0)
                
                # Calculate potential savings
                if cpu_util < 20 and memory_util < 30:
                    # Significant right-sizing opportunity
                    estimated_savings = daily_cost * 0.5 * 30  # 50% savings for 30 days
                    recommendation = 'Downsize to smaller VM SKU'
                elif cpu_util < 40 and memory_util < 50:
                    # Moderate right-sizing opportunity
                    estimated_savings = daily_cost * 0.3 * 30  # 30% savings for 30 days
                    recommendation = 'Consider smaller VM SKU or implement auto-scaling'
                else:
                    estimated_savings = daily_cost * 0.1 * 30  # 10% savings through optimization
                    recommendation = 'Optimize VM configuration and implement scheduling'
                
                optimizations.append({
                    'type': 'vm_optimization',
                    'resource_name': vm.get('resource_name'),
                    'resource_id': vm.get('resource_id'),
                    'optimization_potential': optimization_potential,
                    'current_utilization': {
                        'cpu': cpu_util,
                        'memory': memory_util
                    },
                    'recommendation': recommendation,
                    'estimated_monthly_savings': estimated_savings,
                    'implementation_effort': 'Low' if cpu_util < 20 else 'Medium'
                })
        
        return optimizations
    
    def analyze_storage_optimizations(self, storage_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze storage optimization opportunities.
        """
        optimizations = []
        
        for storage in storage_data:
            capacity_used = storage.get('capacity_used_percentage', 0)
            daily_cost = storage.get('cost_per_day', 0)
            
            if capacity_used < 50:
                # Storage tier optimization opportunity
                estimated_savings = daily_cost * 0.3 * 30  # 30% savings through tier optimization
                
                optimizations.append({
                    'type': 'storage_optimization',
                    'resource_name': storage.get('resource_name'),
                    'resource_id': storage.get('resource_id'),
                    'optimization_potential': 'Medium',
                    'current_utilization': capacity_used,
                    'recommendation': 'Implement storage lifecycle policies and tier optimization',
                    'estimated_monthly_savings': estimated_savings,
                    'implementation_effort': 'Low'
                })
        
        return optimizations
    
    def analyze_database_optimizations(self, db_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze database optimization opportunities.
        """
        optimizations = []
        
        for db in db_data:
            dtu_util = db.get('dtu_utilization_avg', 0)
            daily_cost = db.get('cost_per_day', 0)
            
            if dtu_util < 30:
                # Database right-sizing opportunity
                estimated_savings = daily_cost * 0.4 * 30  # 40% savings through right-sizing
                
                optimizations.append({
                    'type': 'database_optimization',
                    'resource_name': db.get('resource_name'),
                    'resource_id': db.get('resource_id'),
                    'optimization_potential': 'High',
                    'current_utilization': dtu_util,
                    'recommendation': 'Downsize database tier or implement elastic pools',
                    'estimated_monthly_savings': estimated_savings,
                    'implementation_effort': 'Medium'
                })
        
        return optimizations

class CostDataCache:
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl_minutes: int = 60):
        self.cache[key] = value
        self.timestamps[key] = datetime.now() + timedelta(minutes=ttl_minutes)
    
    def is_expired(self, key: str) -> bool:
        if key not in self.timestamps:
            return True
        return datetime.now() > self.timestamps[key]
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern."""
        keys_to_remove = [key for key in self.cache.keys() if pattern.replace("*", "") in key]
        for key in keys_to_remove:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
```

This comprehensive implementation provides the foundation for intelligent cost management with AI-powered analysis, forecasting, and optimization recommendations. The system can analyze cost trends, identify anomalies, forecast future costs, and provide actionable recommendations for cost optimization while maintaining operational requirements.

