"""
Advanced Recommendation Engine for Patent #3
Unified AI-Driven Cloud Governance Platform
Generates personalized, ML-based recommendations across all governance domains
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    """Recommendation data structure"""
    id: str
    domain: str  # security, compliance, cost, operations, identity
    title: str
    description: str
    impact: str  # high, medium, low
    effort: str  # high, medium, low
    priority_score: float  # 0-100
    estimated_savings: Optional[float]
    risk_reduction: Optional[float]
    compliance_improvement: Optional[float]
    actions: List[Dict[str, Any]]
    resources_affected: List[str]
    success_probability: float
    implementation_time: str  # e.g., "2 hours", "1 day", "1 week"
    automation_available: bool
    related_recommendations: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'domain': self.domain,
            'title': self.title,
            'description': self.description,
            'impact': self.impact,
            'effort': self.effort,
            'priority_score': self.priority_score,
            'estimated_savings': self.estimated_savings,
            'risk_reduction': self.risk_reduction,
            'compliance_improvement': self.compliance_improvement,
            'actions': self.actions,
            'resources_affected': self.resources_affected,
            'success_probability': self.success_probability,
            'implementation_time': self.implementation_time,
            'automation_available': self.automation_available,
            'related_recommendations': self.related_recommendations,
            'metadata': self.metadata
        }


class RecommendationNN(nn.Module):
    """Neural network for recommendation scoring and personalization"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_domains: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        
        # Domain-specific heads
        self.domain_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 4, 1) for _ in range(num_domains)
        ])
        
        # Priority scorer
        self.priority_head = nn.Linear(hidden_dim // 4, 1)
        
        # Success predictor
        self.success_head = nn.Linear(hidden_dim // 4, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, domain_idx=None):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        
        # Get domain-specific score if domain specified
        domain_scores = []
        if domain_idx is not None:
            domain_score = torch.sigmoid(self.domain_heads[domain_idx](x))
            domain_scores.append(domain_score)
        else:
            for head in self.domain_heads:
                domain_scores.append(torch.sigmoid(head(x)))
        
        priority = torch.sigmoid(self.priority_head(x)) * 100  # Scale to 0-100
        success_prob = torch.sigmoid(self.success_head(x))
        
        return domain_scores, priority, success_prob


class AdvancedRecommendationEngine:
    """Advanced ML-based recommendation engine for cloud governance"""
    
    def __init__(self, organization_profile: Optional[Dict] = None):
        self.organization_profile = organization_profile or {}
        self.recommendation_model = None
        self.impact_model = None
        self.success_tracker = defaultdict(list)
        self.feedback_history = []
        self.personalization_weights = self._initialize_personalization()
        self.domain_models = {}
        self.scaler = StandardScaler()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Domain configurations
        self.domains = ['security', 'compliance', 'cost', 'operations', 'identity']
        
        # Initialize models
        self._initialize_models()
        
        # Recommendation templates
        self.recommendation_templates = self._load_recommendation_templates()
        
    def _initialize_personalization(self) -> Dict[str, float]:
        """Initialize personalization weights based on organization profile"""
        weights = {
            'security': 1.0,
            'compliance': 1.0,
            'cost': 1.0,
            'operations': 1.0,
            'identity': 1.0
        }
        
        # Adjust weights based on organization profile
        if self.organization_profile.get('industry') == 'healthcare':
            weights['compliance'] = 1.5
            weights['security'] = 1.3
        elif self.organization_profile.get('industry') == 'finance':
            weights['security'] = 1.5
            weights['compliance'] = 1.4
        elif self.organization_profile.get('industry') == 'startup':
            weights['cost'] = 1.5
            weights['operations'] = 1.3
            
        if self.organization_profile.get('size') == 'enterprise':
            weights['compliance'] *= 1.2
            weights['security'] *= 1.2
        elif self.organization_profile.get('size') == 'small':
            weights['cost'] *= 1.3
            
        return weights
    
    def _initialize_models(self):
        """Initialize ML models for recommendation generation"""
        # Main recommendation neural network
        self.recommendation_model = RecommendationNN(
            input_dim=50,  # Feature dimension
            hidden_dim=256,
            num_domains=len(self.domains)
        )
        
        # Impact prediction model
        self.impact_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Success prediction model for each domain
        for domain in self.domains:
            self.domain_models[domain] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def _load_recommendation_templates(self) -> Dict[str, List[Dict]]:
        """Load recommendation templates for each domain"""
        templates = {
            'security': [
                {
                    'id': 'sec_001',
                    'title': 'Enable Multi-Factor Authentication',
                    'description': 'Enable MFA for all privileged accounts',
                    'impact': 'high',
                    'effort': 'low',
                    'risk_reduction': 0.7
                },
                {
                    'id': 'sec_002',
                    'title': 'Implement Network Segmentation',
                    'description': 'Segment network to limit blast radius',
                    'impact': 'high',
                    'effort': 'medium',
                    'risk_reduction': 0.6
                },
                {
                    'id': 'sec_003',
                    'title': 'Enable Encryption at Rest',
                    'description': 'Enable encryption for all storage accounts',
                    'impact': 'high',
                    'effort': 'low',
                    'risk_reduction': 0.5
                }
            ],
            'compliance': [
                {
                    'id': 'comp_001',
                    'title': 'Apply Compliance Tags',
                    'description': 'Tag resources with compliance metadata',
                    'impact': 'medium',
                    'effort': 'low',
                    'compliance_improvement': 0.3
                },
                {
                    'id': 'comp_002',
                    'title': 'Enable Audit Logging',
                    'description': 'Enable comprehensive audit logging',
                    'impact': 'high',
                    'effort': 'low',
                    'compliance_improvement': 0.5
                }
            ],
            'cost': [
                {
                    'id': 'cost_001',
                    'title': 'Right-size Underutilized VMs',
                    'description': 'Resize VMs with <20% CPU utilization',
                    'impact': 'high',
                    'effort': 'low',
                    'estimated_savings': 5000.0
                },
                {
                    'id': 'cost_002',
                    'title': 'Delete Orphaned Resources',
                    'description': 'Remove unattached disks and NICs',
                    'impact': 'medium',
                    'effort': 'low',
                    'estimated_savings': 2000.0
                },
                {
                    'id': 'cost_003',
                    'title': 'Implement Auto-shutdown',
                    'description': 'Auto-shutdown non-production resources',
                    'impact': 'medium',
                    'effort': 'medium',
                    'estimated_savings': 3000.0
                }
            ],
            'operations': [
                {
                    'id': 'ops_001',
                    'title': 'Implement Auto-scaling',
                    'description': 'Enable auto-scaling for critical services',
                    'impact': 'high',
                    'effort': 'medium'
                },
                {
                    'id': 'ops_002',
                    'title': 'Set Up Health Monitoring',
                    'description': 'Configure health checks and alerts',
                    'impact': 'high',
                    'effort': 'low'
                }
            ],
            'identity': [
                {
                    'id': 'id_001',
                    'title': 'Review Privileged Access',
                    'description': 'Review and remove unnecessary privileges',
                    'impact': 'high',
                    'effort': 'medium',
                    'risk_reduction': 0.4
                },
                {
                    'id': 'id_002',
                    'title': 'Implement Just-In-Time Access',
                    'description': 'Replace standing privileges with JIT',
                    'impact': 'high',
                    'effort': 'high',
                    'risk_reduction': 0.6
                }
            ]
        }
        return templates
    
    def extract_features(self, resource_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from resource data for ML models"""
        features = []
        
        # Basic resource features
        features.append(len(resource_data.get('violations', [])))
        features.append(resource_data.get('risk_score', 0))
        features.append(resource_data.get('compliance_score', 0))
        features.append(resource_data.get('cost_score', 0))
        features.append(resource_data.get('operational_score', 0))
        
        # Configuration features
        config = resource_data.get('configuration', {})
        features.append(1 if config.get('encryption_enabled') else 0)
        features.append(1 if config.get('mfa_enabled') else 0)
        features.append(1 if config.get('backup_enabled') else 0)
        features.append(1 if config.get('monitoring_enabled') else 0)
        features.append(1 if config.get('public_access') else 0)
        
        # Historical features
        features.append(resource_data.get('days_since_last_update', 0))
        features.append(resource_data.get('change_frequency', 0))
        features.append(resource_data.get('incident_count', 0))
        
        # Resource type encoding (one-hot)
        resource_types = ['vm', 'storage', 'network', 'database', 'container']
        resource_type = resource_data.get('type', 'unknown')
        for rt in resource_types:
            features.append(1 if resource_type == rt else 0)
        
        # Tag features
        tags = resource_data.get('tags', {})
        features.append(1 if 'production' in tags else 0)
        features.append(1 if 'compliance' in tags else 0)
        features.append(1 if 'critical' in tags else 0)
        
        # Cost features
        features.append(resource_data.get('monthly_cost', 0))
        features.append(resource_data.get('cost_trend', 0))
        features.append(resource_data.get('utilization', 0))
        
        # Compliance features
        features.append(resource_data.get('policy_violations', 0))
        features.append(resource_data.get('compliance_frameworks', 0))
        
        # Pad or truncate to fixed size
        feature_vector = np.array(features[:50] + [0] * (50 - len(features)))
        
        return feature_vector
    
    async def generate_recommendations(
        self,
        resource_data: List[Dict[str, Any]],
        context: Optional[Dict] = None
    ) -> List[Recommendation]:
        """Generate personalized recommendations using ML models"""
        recommendations = []
        
        # Process each resource
        for resource in resource_data:
            features = self.extract_features(resource)
            
            # Get ML-based scores
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                domain_scores, priority, success_prob = self.recommendation_model(features_tensor)
            
            # Generate recommendations for each domain
            for domain_idx, domain in enumerate(self.domains):
                domain_score = domain_scores[domain_idx].item() if len(domain_scores) > domain_idx else 0.5
                
                # Skip if domain score is too low
                if domain_score < 0.3:
                    continue
                
                # Get relevant templates
                templates = self.recommendation_templates.get(domain, [])
                
                for template in templates:
                    # Check if recommendation applies
                    if self._recommendation_applies(resource, template):
                        rec = self._create_recommendation(
                            resource=resource,
                            template=template,
                            domain=domain,
                            priority_score=priority.item(),
                            success_probability=success_prob.item(),
                            domain_score=domain_score
                        )
                        recommendations.append(rec)
        
        # Apply personalization
        recommendations = self._apply_personalization(recommendations)
        
        # Rank recommendations
        recommendations = self._rank_recommendations(recommendations)
        
        # Apply business rules
        recommendations = self._apply_business_rules(recommendations, context)
        
        # Add related recommendations
        recommendations = self._add_related_recommendations(recommendations)
        
        return recommendations[:20]  # Return top 20 recommendations
    
    def _recommendation_applies(self, resource: Dict, template: Dict) -> bool:
        """Check if a recommendation template applies to a resource"""
        # Security recommendations
        if template['id'] == 'sec_001':  # MFA
            return not resource.get('configuration', {}).get('mfa_enabled', False)
        elif template['id'] == 'sec_003':  # Encryption
            return not resource.get('configuration', {}).get('encryption_enabled', False)
        
        # Cost recommendations
        elif template['id'] == 'cost_001':  # Right-sizing
            return resource.get('utilization', 100) < 20
        elif template['id'] == 'cost_002':  # Orphaned resources
            return resource.get('status') == 'orphaned'
        
        # Compliance recommendations
        elif template['id'] == 'comp_001':  # Compliance tags
            return 'compliance' not in resource.get('tags', {})
        
        # Default: recommendation applies
        return True
    
    def _create_recommendation(
        self,
        resource: Dict,
        template: Dict,
        domain: str,
        priority_score: float,
        success_probability: float,
        domain_score: float
    ) -> Recommendation:
        """Create a recommendation from template and ML scores"""
        
        # Calculate adjusted priority
        adjusted_priority = priority_score * self.personalization_weights[domain] * domain_score
        
        # Determine implementation time based on effort
        implementation_times = {
            'low': '2 hours',
            'medium': '1 day',
            'high': '1 week'
        }
        
        # Create recommendation
        rec = Recommendation(
            id=f"{template['id']}_{resource.get('id', 'unknown')}",
            domain=domain,
            title=template['title'],
            description=f"{template['description']} for {resource.get('name', 'resource')}",
            impact=template['impact'],
            effort=template['effort'],
            priority_score=min(adjusted_priority, 100),
            estimated_savings=template.get('estimated_savings'),
            risk_reduction=template.get('risk_reduction'),
            compliance_improvement=template.get('compliance_improvement'),
            actions=self._generate_actions(resource, template),
            resources_affected=[resource.get('id', 'unknown')],
            success_probability=success_probability,
            implementation_time=implementation_times.get(template['effort'], '1 day'),
            automation_available=template['effort'] == 'low',
            related_recommendations=[],
            metadata={
                'ml_confidence': domain_score,
                'personalization_weight': self.personalization_weights[domain],
                'template_id': template['id']
            }
        )
        
        return rec
    
    def _generate_actions(self, resource: Dict, template: Dict) -> List[Dict]:
        """Generate specific actions for a recommendation"""
        actions = []
        
        if template['id'] == 'sec_001':  # MFA
            actions = [
                {'step': 1, 'action': 'Navigate to Azure AD'},
                {'step': 2, 'action': 'Select Security > MFA'},
                {'step': 3, 'action': f"Enable MFA for {resource.get('name')}"},
                {'step': 4, 'action': 'Configure MFA methods'},
                {'step': 5, 'action': 'Test MFA login'}
            ]
        elif template['id'] == 'cost_001':  # Right-sizing
            actions = [
                {'step': 1, 'action': f"Stop VM {resource.get('name')}"},
                {'step': 2, 'action': 'Change VM size to recommended tier'},
                {'step': 3, 'action': 'Start VM'},
                {'step': 4, 'action': 'Monitor performance for 24 hours'}
            ]
        # Add more action templates as needed
        
        return actions
    
    def _apply_personalization(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Apply personalization based on organization profile and history"""
        # Boost recommendations that align with organization priorities
        for rec in recommendations:
            # Industry-specific boosting
            if self.organization_profile.get('industry') == 'healthcare':
                if rec.domain == 'compliance':
                    rec.priority_score *= 1.3
                if 'HIPAA' in rec.description:
                    rec.priority_score *= 1.5
                    
            # Size-specific adjustments
            if self.organization_profile.get('size') == 'small':
                if rec.effort == 'high':
                    rec.priority_score *= 0.7  # Reduce priority of high-effort items
                    
            # Historical success adjustments
            if rec.id in self.success_tracker:
                success_rate = np.mean(self.success_tracker[rec.id])
                rec.priority_score *= (0.5 + success_rate)  # Boost successful recommendations
                
        return recommendations
    
    def _rank_recommendations(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Rank recommendations by priority and impact"""
        # Calculate composite score
        for rec in recommendations:
            impact_score = {'high': 3, 'medium': 2, 'low': 1}.get(rec.impact, 1)
            effort_score = {'low': 3, 'medium': 2, 'high': 1}.get(rec.effort, 1)
            
            # Composite score considers priority, impact, effort, and success probability
            rec.priority_score = (
                rec.priority_score * 0.4 +
                impact_score * 20 * 0.3 +
                effort_score * 20 * 0.2 +
                rec.success_probability * 100 * 0.1
            )
            
            # Boost based on potential savings or risk reduction
            if rec.estimated_savings:
                rec.priority_score += min(rec.estimated_savings / 1000, 20)  # Cap at 20 point boost
            if rec.risk_reduction:
                rec.priority_score += rec.risk_reduction * 30
                
        # Sort by priority score
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)
        
        return recommendations
    
    def _apply_business_rules(
        self,
        recommendations: List[Recommendation],
        context: Optional[Dict]
    ) -> List[Recommendation]:
        """Apply business rules and constraints"""
        filtered = []
        
        for rec in recommendations:
            # Budget constraints
            if context and context.get('max_cost'):
                if rec.estimated_savings and rec.estimated_savings < 0:  # Has cost
                    if abs(rec.estimated_savings) > context['max_cost']:
                        continue
                        
            # Compliance requirements
            if context and context.get('compliance_frameworks'):
                if rec.domain == 'compliance':
                    # Boost recommendations for required frameworks
                    for framework in context['compliance_frameworks']:
                        if framework.lower() in rec.description.lower():
                            rec.priority_score *= 1.5
                            
            # Maintenance windows
            if context and context.get('maintenance_window_only'):
                if rec.effort in ['medium', 'high']:
                    rec.metadata['requires_maintenance_window'] = True
                    
            filtered.append(rec)
            
        return filtered
    
    def _add_related_recommendations(
        self,
        recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        """Add related recommendations for better context"""
        rec_dict = {rec.id: rec for rec in recommendations}
        
        for rec in recommendations:
            # Security recommendations often relate to compliance
            if rec.domain == 'security':
                related = [r.id for r in recommendations 
                          if r.domain == 'compliance' and r.id != rec.id]
                rec.related_recommendations = related[:3]
                
            # Cost recommendations may relate to operations
            elif rec.domain == 'cost':
                related = [r.id for r in recommendations 
                          if r.domain == 'operations' and r.id != rec.id]
                rec.related_recommendations = related[:3]
                
        return recommendations
    
    def track_recommendation_success(
        self,
        recommendation_id: str,
        success: bool,
        metadata: Optional[Dict] = None
    ):
        """Track recommendation success for learning"""
        self.success_tracker[recommendation_id].append(1 if success else 0)
        
        # Store feedback for model retraining
        self.feedback_history.append({
            'recommendation_id': recommendation_id,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        
        # Retrain models periodically
        if len(self.feedback_history) >= 100:
            self._retrain_models()
    
    def _retrain_models(self):
        """Retrain models based on feedback"""
        # This would involve retraining the neural network and other models
        # based on accumulated feedback
        logger.info("Retraining recommendation models with feedback")
        # Implementation would go here
        pass
    
    async def get_personalized_dashboard(
        self,
        user_role: str,
        user_preferences: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Get personalized dashboard configuration"""
        dashboard = {
            'widgets': [],
            'recommendations': [],
            'metrics': {},
            'alerts': []
        }
        
        # Role-based widget selection
        if user_role == 'security_admin':
            dashboard['widgets'] = [
                'security_score', 'threat_map', 'compliance_status',
                'recent_incidents', 'vulnerability_trends'
            ]
        elif user_role == 'cost_admin':
            dashboard['widgets'] = [
                'cost_trends', 'budget_status', 'optimization_opportunities',
                'resource_utilization', 'forecast'
            ]
        elif user_role == 'executive':
            dashboard['widgets'] = [
                'executive_summary', 'risk_overview', 'compliance_attestation',
                'cost_summary', 'operational_health'
            ]
        else:  # default
            dashboard['widgets'] = [
                'unified_score', 'top_recommendations', 'recent_changes',
                'alerts_summary', 'quick_actions'
            ]
            
        # Apply user preferences
        if user_preferences:
            if user_preferences.get('widgets'):
                dashboard['widgets'] = user_preferences['widgets']
            if user_preferences.get('theme'):
                dashboard['theme'] = user_preferences['theme']
                
        return dashboard
    
    def export_recommendations(
        self,
        recommendations: List[Recommendation],
        format: str = 'json'
    ) -> Any:
        """Export recommendations in various formats"""
        if format == 'json':
            return json.dumps([rec.to_dict() for rec in recommendations], indent=2)
        elif format == 'csv':
            import csv
            import io
            output = io.StringIO()
            if recommendations:
                fieldnames = recommendations[0].to_dict().keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for rec in recommendations:
                    writer.writerow(rec.to_dict())
            return output.getvalue()
        elif format == 'html':
            # Generate HTML report
            html = "<html><body><h1>Recommendations Report</h1>"
            for rec in recommendations:
                html += f"<div><h2>{rec.title}</h2><p>{rec.description}</p>"
                html += f"<p>Priority: {rec.priority_score:.1f}</p></div>"
            html += "</body></html>"
            return html
        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage
if __name__ == "__main__":
    # Create engine with organization profile
    org_profile = {
        'industry': 'healthcare',
        'size': 'enterprise',
        'compliance_frameworks': ['HIPAA', 'SOC2'],
        'cloud_providers': ['Azure', 'AWS']
    }
    
    engine = AdvancedRecommendationEngine(org_profile)
    
    # Sample resource data
    resources = [
        {
            'id': 'vm-001',
            'name': 'prod-web-server',
            'type': 'vm',
            'configuration': {
                'encryption_enabled': False,
                'mfa_enabled': False,
                'backup_enabled': True
            },
            'utilization': 15,
            'monthly_cost': 500,
            'risk_score': 0.7,
            'compliance_score': 0.6,
            'tags': {'environment': 'production'}
        }
    ]
    
    # Generate recommendations
    async def test():
        recommendations = await engine.generate_recommendations(resources)
        for rec in recommendations[:5]:
            print(f"[{rec.priority_score:.1f}] {rec.title}")
            print(f"  Domain: {rec.domain}, Impact: {rec.impact}, Effort: {rec.effort}")
            print(f"  Success Probability: {rec.success_probability:.2f}")
            print()
    
    asyncio.run(test())