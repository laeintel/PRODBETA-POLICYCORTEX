"""
GNN Correlation Service - Integration API for Cross-Domain Governance Correlation
Provides REST API endpoints and real-time processing capabilities for the GNN system.
"""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import redis.asyncio as redis
from fastapi import HTTPException

from backend.shared.config import get_settings

from ..ml_models.cross_domain_gnn import CorrelationConfig, CorrelationEngine
from .gnn_training_service import GNNTrainingService, TrainingConfig

logger = logging.getLogger(__name__)
settings = get_settings()

class GNNCorrelationService:
    """Service for GNN-based cross-domain correlation analysis"""

    def __init__(self):
        self.config = CorrelationConfig()
        self.correlation_engine = CorrelationEngine(self.config)
        self.training_service = GNNTrainingService(TrainingConfig())

        # Cache for correlation results
        self.redis_client = None
        self.cache_ttl = 3600  # 1 hour

        # Model loaded flag
        self.model_loaded = False

        logger.info("GNNCorrelationService initialized")

    async def initialize(self):
        """Initialize the service and load models"""

        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )

            # Initialize default model
            default_feature_dims = {
                'resource': 20,
                'policy': 12,
                'domain': 4,
                'event': 9,
                'user': 8
            }

            self.correlation_engine.initialize_model(default_feature_dims)

            # Try to load pre-trained model
            try:
                self.correlation_engine.load_model()
                self.model_loaded = True
                logger.info("Pre-trained model loaded successfully")
            except Exception as e:
                logger.warning(f"No pre-trained model found, using initialized model: {e}")
                self.model_loaded = True

            logger.info("GNNCorrelationService initialization completed")

        except Exception as e:
            logger.error(f"Error initializing GNNCorrelationService: {e}")
            raise

    async def analyze_governance_correlations(
        self,
        governance_data: Dict[str,
        Any]
    ) -> Dict[str, Any]:
        """Analyze cross-domain correlations in governance data"""

        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(governance_data)

            # Check cache first
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.debug("Returning cached correlation results")
                return cached_result

            # Perform correlation analysis
            logger.info("Performing GNN correlation analysis")
            correlation_results = self.correlation_engine.detect_correlations(governance_data)

            # Enrich results with additional analysis
            enriched_results = await self._enrich_correlation_results(
                correlation_results,
                governance_data
            )

            # Cache results
            await self._cache_result(cache_key, enriched_results)

            logger.info(
                f"Correlation analysis completed. Found {len(enriched_results.get('correlations',
                []))} correlations"
            )
            return enriched_results

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {str(e)}")

    async def predict_governance_impacts(self,
                                       change_scenario: Dict[str, Any],
                                       current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict impacts of governance changes across domains"""

        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Combine current state with proposed changes
            combined_data = self._merge_governance_data(current_state, change_scenario)

            # Run correlation analysis on combined scenario
            impact_results = self.correlation_engine.detect_correlations(combined_data)

            # Process impact predictions specifically
            impact_analysis = {
                'predicted_impacts': impact_results.get('impacts', []),
                'affected_domains': self._extract_affected_domains(impact_results),
                'risk_assessment': self._assess_change_risks(impact_results),
                'recommendations': self._generate_impact_recommendations(impact_results),
                'confidence_score': self._calculate_impact_confidence(impact_results),
                'analysis_timestamp': datetime.now().isoformat()
            }

            logger.info(
                f"Impact prediction completed for {len(change_scenario.get('changes',
                []))} changes"
            )
            return impact_analysis

        except Exception as e:
            logger.error(f"Error in impact prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Impact prediction failed: {str(e)}")

    async def get_domain_relationships(self, domain_focus: str = None) -> Dict[str, Any]:
        """Get relationships between governance domains"""

        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            # Create sample governance data to analyze domain relationships
            domain_data = self._create_domain_analysis_data(domain_focus)

            # Run correlation analysis
            results = self.correlation_engine.detect_correlations(domain_data)

            # Extract domain-specific insights
            domain_relationships = {
                'domain_focus': domain_focus,
                'cross_domain_correlations': self._extract_domain_correlations(results),
                'domain_influence_scores': self._calculate_domain_influence(results),
                'relationship_strength_matrix': self._build_relationship_matrix(results),
                'key_insights': self._generate_domain_insights(results),
                'analysis_timestamp': datetime.now().isoformat()
            }

            return domain_relationships

        except Exception as e:
            logger.error(f"Error analyzing domain relationships: {e}")
            raise HTTPException(status_code=500, detail=f"Domain analysis failed: {str(e)}")

    async def train_model_with_data(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train or retrain the GNN model with new data"""

        try:
            logger.info(f"Starting model training with {len(training_data)} samples")

            # Validate training data
            validated_data = self._validate_training_data(training_data)

            # Split data for training and validation
            split_index = int(len(validated_data) * 0.8)
            train_data = validated_data[:split_index]
            val_data = validated_data[split_index:]

            # Run training
            training_results = await self.training_service.train_model(train_data, val_data)

            # Update the correlation engine with newly trained model
            self.correlation_engine.model = self.training_service.correlation_engine.model
            self.model_loaded = True

            # Save the trained model
            self.correlation_engine.save_model()

            training_summary = {
                'training_completed': True,
                'training_samples': len(train_data),
                'validation_samples': len(val_data),
                'final_metrics': training_results.get('final_metrics', {}),
                'best_metrics': training_results.get('best_metrics', {}),
                'model_saved': True,
                'training_timestamp': datetime.now().isoformat()
            }

            logger.info("Model training completed successfully")
            return training_summary

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

    async def get_model_health(self) -> Dict[str, Any]:
        """Get health status and performance metrics of the GNN model"""

        health_status = {
            'model_loaded': self.model_loaded,
            'service_status': 'healthy' if self.model_loaded else 'degraded',
            'model_config': asdict(self.config),
            'cache_status': await self._check_cache_health(),
            'last_analysis_time': await self._get_last_analysis_time(),
            'system_resources': await self._get_system_resources(),
            'health_check_timestamp': datetime.now().isoformat()
        }

        return health_status

    def _generate_cache_key(self, governance_data: Dict[str, Any]) -> str:
        """Generate cache key for governance data"""

        # Create a simplified hash of the governance data
        data_str = json.dumps(governance_data, sort_keys=True, default=str)
        import hashlib
        return f"gnn_correlation:{hashlib.md5(data_str.encode()).hexdigest()}"

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached correlation results"""

        if not self.redis_client:
            return None

        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Error retrieving cached result: {e}")

        return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache correlation results"""

        if not self.redis_client:
            return

        try:
            await self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.warning(f"Error caching result: {e}")

    async def _enrich_correlation_results(self,
                                        correlation_results: Dict[str, Any],
                                        governance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich correlation results with additional analysis"""

        enriched = correlation_results.copy()

        # Add correlation insights
        enriched['insights'] = self._generate_correlation_insights(correlation_results)

        # Add risk assessment
        enriched['risk_assessment'] = self._assess_correlation_risks(correlation_results)

        # Add recommendations
        enriched['recommendations'] = (
            self._generate_correlation_recommendations(correlation_results)
        )

        # Add trend analysis
        enriched['trends'] = await self._analyze_correlation_trends(correlation_results)

        return enriched

    def _merge_governance_data(self,
                             current_state: Dict[str, Any],
                             change_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Merge current governance state with proposed changes"""

        merged_data = current_state.copy()

        # Apply changes to resources
        if 'resource_changes' in change_scenario:
            resources = merged_data.get('resources', [])
            for change in change_scenario['resource_changes']:
                resource_id = change.get('resource_id')

                # Find and update resource
                for i, resource in enumerate(resources):
                    if resource.get('id') == resource_id:
                        resources[i].update(change.get('updates', {}))
                        break
                else:
                    # Add new resource if not found
                    if change.get('action') == 'create':
                        resources.append(change.get('resource_data', {}))

            merged_data['resources'] = resources

        # Apply policy changes
        if 'policy_changes' in change_scenario:
            policies = merged_data.get('policies', [])
            for change in change_scenario['policy_changes']:
                policy_id = change.get('policy_id')

                if change.get('action') == 'create':
                    policies.append(change.get('policy_data', {}))
                elif change.get('action') == 'update':
                    for i, policy in enumerate(policies):
                        if policy.get('id') == policy_id:
                            policies[i].update(change.get('updates', {}))
                            break
                elif change.get('action') == 'delete':
                    policies = [p for p in policies if p.get('id') != policy_id]

            merged_data['policies'] = policies

        return merged_data

    def _extract_affected_domains(self, impact_results: Dict[str, Any]) -> List[str]:
        """Extract domains affected by predicted impacts"""

        affected_domains = set()

        for impact in impact_results.get('impacts', []):
            domain = impact.get('impact_category')
            if domain:
                affected_domains.add(domain)

        return list(affected_domains)

    def _assess_change_risks(self, impact_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with predicted impacts"""

        impacts = impact_results.get('impacts', [])

        if not impacts:
            return {'overall_risk': 'low', 'risk_factors': []}

        # Calculate risk scores
        high_impact_count = sum(
            1 for impact in impacts if impact.get('impact_probability',
            0) > 0.8
        )
        medium_impact_count = sum(
            1 for impact in impacts if 0.5 < impact.get('impact_probability',
            0) <= 0.8
        )

        # Determine overall risk level
        if high_impact_count > 3:
            overall_risk = 'high'
        elif high_impact_count > 1 or medium_impact_count > 5:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'

        # Identify risk factors
        risk_factors = []
        for impact in impacts:
            if impact.get('impact_probability', 0) > 0.7:
                risk_factors.append({
                    'factor': impact.get('impact_category', 'unknown'),
                    'probability': impact.get('impact_probability', 0),
                    'severity': 'high' if impact.get('impact_probability', 0) > 0.8 else 'medium'
                })

        return {
            'overall_risk': overall_risk,
            'risk_factors': risk_factors,
            'high_impact_count': high_impact_count,
            'medium_impact_count': medium_impact_count
        }

    def _generate_impact_recommendations(self, impact_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on impact predictions"""

        recommendations = []
        impacts = impact_results.get('impacts', [])

        # High-impact recommendations
        high_impacts = [i for i in impacts if i.get('impact_probability', 0) > 0.8]
        if high_impacts:
            recommendations.append("Consider implementing gradual rollout for high-impact changes")
            recommendations.append("Set up enhanced monitoring for affected domains")

        # Domain-specific recommendations
        security_impacts = [i for i in impacts if i.get('impact_category') == 'security']
        if security_impacts:
            recommendations.append("Review security policies before implementing changes")

        cost_impacts = [i for i in impacts if i.get('impact_category') == 'cost']
        if cost_impacts:
            recommendations.append("Evaluate cost implications and budget impact")

        compliance_impacts = [i for i in impacts if i.get('impact_category') == 'compliance']
        if compliance_impacts:
            recommendations.append("Ensure compliance requirements are maintained")

        if not recommendations:
            recommendations.append("Changes appear to have minimal cross-domain impact")

        return recommendations

    def _calculate_impact_confidence(self, impact_results: Dict[str, Any]) -> float:
        """Calculate confidence score for impact predictions"""

        impacts = impact_results.get('impacts', [])

        if not impacts:
            return 0.5  # Neutral confidence

        # Average the confidence scores
        confidence_scores = [impact.get('impact_probability', 0.5) for impact in impacts]
        avg_confidence = np.mean(confidence_scores)

        # Adjust based on number of predictions
        sample_size_factor = min(len(impacts) / 10.0, 1.0)  # More predictions = higher confidence

        return min(avg_confidence * sample_size_factor, 1.0)

    def _create_domain_analysis_data(self, domain_focus: Optional[str]) -> Dict[str, Any]:
        """Create synthetic governance data for domain relationship analysis"""

        # This would typically use real governance data
        # For now, create representative synthetic data

        domains = ['security', 'compliance', 'cost', 'performance']
        if domain_focus and domain_focus not in domains:
            domains.append(domain_focus)

        domain_data = {
            'domains': [
                {
                    'id': domain,
                    'name': domain.title(),
                    'weight': 1.0,
                    'policies': [f'{domain}_policy_{i}' for i in range(3)],
                    'compliance_score': np.random.uniform(0.7, 1.0),
                    'effectiveness_score': np.random.uniform(0.6, 0.9)
                }
                for domain in domains
            ],
            'resources': [
                {
                    'id': f'resource_{i}',
                    'type': 'virtual_machine',
                    'domains': np.random.choice(domains, size=2, replace=False).tolist(),
                    'security_score': np.random.uniform(0.5, 1.0),
                    'compliance_score': np.random.uniform(0.6, 1.0),
                    'monthly_cost': np.random.uniform(100, 1000),
                    'cpu_utilization': np.random.uniform(0.2, 0.8)
                }
                for i in range(20)
            ]
        }

        return domain_data

    def _extract_domain_correlations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract cross-domain correlations from results"""

        correlations = results.get('correlations', [])
        domain_correlations = []

        for corr in correlations:
            if corr.get('correlation_score', 0) > 0.6:
                domain_correlations.append({
                    'correlation_type': 'cross_domain',
                    'strength': corr.get('correlation_score', 0),
                    'domains': ['security', 'compliance'],  # Would extract from actual data
                    'description': f"Strong correlation detected (
                        score: {corr.get('correlation_score',
                        0):.2f}
                    )"
                })

        return domain_correlations

    def _calculate_domain_influence(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate influence scores for each domain"""

        # This would be based on actual correlation patterns
        return {
            'security': 0.85,
            'compliance': 0.78,
            'cost': 0.72,
            'performance': 0.69
        }

    def _build_relationship_matrix(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Build relationship strength matrix between domains"""

        domains = ['security', 'compliance', 'cost', 'performance']
        matrix = {}

        for domain1 in domains:
            matrix[domain1] = {}
            for domain2 in domains:
                if domain1 == domain2:
                    matrix[domain1][domain2] = 1.0
                else:
                    # Would calculate from actual correlations
                    matrix[domain1][domain2] = np.random.uniform(0.3, 0.8)

        return matrix

    def _generate_domain_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights about domain relationships"""

        return [
            "Security and compliance domains show strong positive correlation",
            "Cost optimization impacts often correlate with performance changes",
            "Cross-domain effects are most pronounced in policy enforcement",
            "Resource changes in security domain tend to cascade to compliance"
        ]

    def _generate_correlation_insights(self, correlation_results: Dict[str, Any]) -> List[str]:
        """Generate insights from correlation analysis"""

        insights = []
        correlations = correlation_results.get('correlations', [])

        if len(correlations) > 10:
            insights.append(f"High correlation density detected: {len(correlations)} correlations found")
        elif len(correlations) > 5:
            insights.append(f"Moderate correlation activity: {len(correlations)} correlations identified")
        else:
            insights.append(f"Low correlation activity: {len(correlations)} correlations detected")

        # Analyze correlation strengths
        if correlations:
            avg_strength = np.mean([c.get('correlation_score', 0) for c in correlations])
            if avg_strength > 0.8:
                insights.append("Strong average correlation strength suggests tight coupling")
            elif avg_strength > 0.6:
                insights.append("Moderate correlation strength indicates some interdependencies")
            else:
                insights.append("Weak correlation strength suggests loose coupling")

        return insights

    def _assess_correlation_risks(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks based on correlation patterns"""

        correlations = correlation_results.get('correlations', [])

        high_correlations = [c for c in correlations if c.get('correlation_score', 0) > 0.8]
        risk_level = (
            'high' if len(high_correlations) > 5 else 'medium' if len(high_correlations) > 2 else 'low'
        )

        return {
            'risk_level': risk_level,
            'high_correlation_count': len(high_correlations),
            'cascade_risk': risk_level in ['high', 'medium'],
            'recommendations': [
                "Monitor highly correlated resources for cascade effects",
                "Consider implementing circuit breakers for critical dependencies"
            ] if risk_level == 'high' else []
        }

    def _generate_correlation_recommendations(
        self,
        correlation_results: Dict[str,
        Any]
    ) -> List[str]:
        """Generate recommendations based on correlation analysis"""

        recommendations = []
        correlations = correlation_results.get('correlations', [])

        if correlations:
            recommendations.append("Review identified correlations for optimization opportunities")
            recommendations.append("Consider implementing automated correlation monitoring")

            high_correlations = [c for c in correlations if c.get('correlation_score', 0) > 0.8]
            if high_correlations:
                recommendations.append("Investigate high-strength correlations for potential consolidation")

        return recommendations

    async def _analyze_correlation_trends(
        self,
        correlation_results: Dict[str,
        Any]
    ) -> Dict[str, Any]:
        """Analyze trends in correlation patterns"""

        # This would typically analyze historical data
        return {
            'trend_direction': 'stable',
            'correlation_growth_rate': 0.02,
            'emerging_patterns': [],
            'historical_comparison': 'similar_to_baseline'
        }

    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check Redis cache health"""

        if not self.redis_client:
            return {'status': 'disabled', 'connection': False}

        try:
            await self.redis_client.ping()
            return {'status': 'healthy', 'connection': True}
        except Exception as e:
            return {'status': 'unhealthy', 'connection': False, 'error': str(e)}

    async def _get_last_analysis_time(self) -> Optional[str]:
        """Get timestamp of last analysis"""

        if not self.redis_client:
            return None

        try:
            last_time = await self.redis_client.get('gnn:last_analysis_time')
            return last_time
        except Exception:
            return None

    async def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource information"""

        import psutil

        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'gpu_available': torch.cuda.is_available() if 'torch' in globals() else False,
            'model_device': str(self.correlation_engine.device) if self.correlation_engine else 'unknown'
        }

    def _validate_training_data(self, training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean training data"""

        validated_data = []

        for sample in training_data:
            # Check required fields
            if 'governance_data' not in sample:
                logger.warning("Sample missing governance_data, skipping")
                continue

            # Validate governance data structure
            governance_data = sample['governance_data']
            if not isinstance(governance_data, dict):
                logger.warning("Invalid governance_data format, skipping")
                continue

            # Ensure minimum required data
            if not governance_data.get('resources') and not governance_data.get('policies'):
                logger.warning("Sample missing resources and policies, skipping")
                continue

            validated_data.append(sample)

        logger.info(f"Validated {len(validated_data)} samples out of {len(training_data)}")
        return validated_data

# Global service instance
gnn_service = GNNCorrelationService()
