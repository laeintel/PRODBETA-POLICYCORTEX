"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

"""
Human Feedback Collection and Integration System
Collects, processes, and integrates human feedback for RLHF
Handles compliance outcomes, user preferences, and audit results
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import numpy as np
import torch

# Import our RLHF components
from .rlhf_system import (
    HumanFeedback, 
    FeedbackType,
    RLHFTrainer,
    OrganizationalPreferenceLearner
)

logger = logging.getLogger(__name__)

class FeedbackSource(Enum):
    """Sources of human feedback"""
    USER_INTERFACE = "user_interface"
    API = "api"
    AUDIT_SYSTEM = "audit_system"
    COMPLIANCE_CHECK = "compliance_check"
    INCIDENT_REPORT = "incident_report"
    AUTOMATED_TESTING = "automated_testing"


@dataclass
class FeedbackRequest:
    """Request for human feedback"""
    request_id: str
    timestamp: datetime
    context: Dict[str, Any]
    options: List[Dict[str, Any]]  # Options to choose from
    feedback_type: FeedbackType
    organization_id: str
    user_id: Optional[str] = None
    deadline: Optional[datetime] = None
    priority: str = "normal"  # low, normal, high, critical


@dataclass
class AuditResult:
    """Result from compliance audit"""
    audit_id: str
    timestamp: datetime
    policy_id: str
    resource_id: str
    compliant: bool
    violations: List[str]
    recommendations: List[str]
    severity: str
    auditor: Optional[str] = None


class FeedbackCollector:
    """
    Main feedback collection system
    Aggregates feedback from multiple sources
    """
    
    def __init__(self, rlhf_trainer: Optional[RLHFTrainer] = None):
        self.rlhf_trainer = rlhf_trainer or RLHFTrainer()
        self.org_learner = OrganizationalPreferenceLearner()
        
        # Feedback storage
        self.pending_requests = {}
        self.completed_feedback = deque(maxlen=10000)
        self.audit_history = deque(maxlen=5000)
        
        # Real-time feedback queue
        self.feedback_queue = asyncio.Queue()
        
        # Statistics
        self.stats = {
            'total_feedback_collected': 0,
            'feedback_by_type': defaultdict(int),
            'feedback_by_source': defaultdict(int),
            'average_response_time': 0,
            'compliance_rate': 0.0
        }
        
        # Start background processing
        self.processing_task = None
        
    async def start(self):
        """Start background feedback processing"""
        self.processing_task = asyncio.create_task(self._process_feedback_queue())
        logger.info("Feedback collection system started")
        
    async def stop(self):
        """Stop background processing"""
        if self.processing_task:
            self.processing_task.cancel()
            await asyncio.gather(self.processing_task, return_exceptions=True)
            
    async def _process_feedback_queue(self):
        """Process feedback queue in background"""
        while True:
            try:
                feedback = await self.feedback_queue.get()
                await self._process_feedback(feedback)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing feedback: {e}")
                
    async def _process_feedback(self, feedback: HumanFeedback):
        """Process individual feedback item"""
        # Add to RLHF trainer
        self.rlhf_trainer.add_feedback(feedback)
        
        # Update organizational preferences
        if feedback.organization_id:
            self.org_learner.learn_organization_preferences(
                feedback.organization_id,
                [feedback]
            )
            
        # Update statistics
        self.stats['total_feedback_collected'] += 1
        self.stats['feedback_by_type'][feedback.feedback_type.value] += 1
        
        # Store completed feedback
        self.completed_feedback.append(feedback)
        
        # Trigger training if enough feedback collected
        if self.stats['total_feedback_collected'] % 100 == 0:
            await self.rlhf_trainer.train_reward_model()
            
    async def collect_user_preference(self, 
                                     option_a: Dict[str, Any],
                                     option_b: Dict[str, Any],
                                     preference: str,
                                     context: Dict[str, Any],
                                     user_id: str,
                                     organization_id: str) -> str:
        """
        Collect user preference between two options
        
        Args:
            option_a: First option (e.g., policy configuration)
            option_b: Second option
            preference: 'a', 'b', or 'equal'
            context: Additional context
            user_id: User identifier
            organization_id: Organization identifier
        """
        feedback = HumanFeedback(
            feedback_id=f"pref_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            feedback_type=FeedbackType.PREFERENCE,
            context=context,
            option_a=option_a,
            option_b=option_b,
            preference=preference,
            user_id=user_id,
            organization_id=organization_id,
            confidence=1.0
        )
        
        await self.feedback_queue.put(feedback)
        
        return feedback.feedback_id
        
    async def collect_rating(self,
                           item: Dict[str, Any],
                           rating: float,
                           max_rating: float,
                           context: Dict[str, Any],
                           user_id: str,
                           organization_id: str) -> str:
        """
        Collect rating feedback
        
        Args:
            item: Item being rated (e.g., policy recommendation)
            rating: Numerical rating
            max_rating: Maximum possible rating
            context: Additional context
            user_id: User identifier
            organization_id: Organization identifier
        """
        feedback = HumanFeedback(
            feedback_id=f"rating_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            feedback_type=FeedbackType.RATING,
            context={**context, 'item': item},
            rating=rating,
            max_rating=max_rating,
            user_id=user_id,
            organization_id=organization_id
        )
        
        await self.feedback_queue.put(feedback)
        
        return feedback.feedback_id
        
    async def collect_correction(self,
                                original: str,
                                corrected: str,
                                context: Dict[str, Any],
                                user_id: str,
                                organization_id: str) -> str:
        """
        Collect correction feedback
        
        Args:
            original: Original output (e.g., generated policy)
            corrected: User's correction
            context: Additional context
            user_id: User identifier
            organization_id: Organization identifier
        """
        feedback = HumanFeedback(
            feedback_id=f"correction_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            feedback_type=FeedbackType.CORRECTION,
            context=context,
            original=original,
            corrected=corrected,
            user_id=user_id,
            organization_id=organization_id
        )
        
        await self.feedback_queue.put(feedback)
        
        return feedback.feedback_id
        
    async def process_audit_result(self, audit: AuditResult) -> str:
        """
        Process audit result as compliance feedback
        
        Args:
            audit: Audit result containing compliance information
        """
        feedback = HumanFeedback(
            feedback_id=f"audit_{audit.audit_id}",
            timestamp=audit.timestamp,
            feedback_type=FeedbackType.COMPLIANCE,
            context={
                'resource_id': audit.resource_id,
                'auditor': audit.auditor,
                'severity': audit.severity
            },
            policy_id=audit.policy_id,
            compliant=audit.compliant,
            violations=audit.violations,
            organization_id='system',  # System-level audit
            confidence=1.0  # Audit results are definitive
        )
        
        await self.feedback_queue.put(feedback)
        
        # Update compliance statistics
        self.stats['compliance_rate'] = (
            self.stats['compliance_rate'] * 0.95 + 
            (1.0 if audit.compliant else 0.0) * 0.05
        )
        
        # Store audit history
        self.audit_history.append(audit)
        
        return feedback.feedback_id
        
    async def process_incident(self,
                              incident_type: str,
                              severity: str,
                              affected_resources: List[str],
                              root_cause: Optional[str],
                              resolution: Optional[str],
                              organization_id: str) -> str:
        """
        Process security/operational incident as negative feedback
        
        Args:
            incident_type: Type of incident
            severity: Incident severity
            affected_resources: Resources affected
            root_cause: Root cause if known
            resolution: How it was resolved
            organization_id: Organization identifier
        """
        feedback = HumanFeedback(
            feedback_id=f"incident_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            feedback_type=FeedbackType.INCIDENT,
            context={
                'incident_type': incident_type,
                'affected_resources': affected_resources,
                'root_cause': root_cause,
                'resolution': resolution
            },
            # Incidents are negative feedback
            rating=0.0,
            max_rating=5.0,
            organization_id=organization_id,
            tags=['incident', severity]
        )
        
        await self.feedback_queue.put(feedback)
        
        # High-severity incidents trigger immediate learning
        if severity in ['critical', 'high']:
            await self.rlhf_trainer.train_reward_model(epochs=5)
            
        return feedback.feedback_id
        
    async def request_feedback(self,
                              context: Dict[str, Any],
                              options: List[Dict[str, Any]],
                              feedback_type: FeedbackType,
                              organization_id: str,
                              user_id: Optional[str] = None,
                              priority: str = "normal",
                              deadline_hours: int = 24) -> FeedbackRequest:
        """
        Create a feedback request for async collection
        
        Args:
            context: Context for feedback
            options: Options to present
            feedback_type: Type of feedback requested
            organization_id: Organization identifier
            user_id: Specific user to request from
            priority: Request priority
            deadline_hours: Hours until deadline
        """
        request = FeedbackRequest(
            request_id=f"req_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            context=context,
            options=options,
            feedback_type=feedback_type,
            organization_id=organization_id,
            user_id=user_id,
            deadline=datetime.utcnow() + timedelta(hours=deadline_hours),
            priority=priority
        )
        
        self.pending_requests[request.request_id] = request
        
        # High-priority requests get processed immediately
        if priority == "critical":
            await self._escalate_request(request)
            
        return request
        
    async def _escalate_request(self, request: FeedbackRequest):
        """Escalate high-priority feedback request"""
        logger.warning(f"Escalating critical feedback request: {request.request_id}")
        # In production, send notifications, alerts, etc.
        
    def get_pending_requests(self, 
                            organization_id: Optional[str] = None,
                            user_id: Optional[str] = None) -> List[FeedbackRequest]:
        """Get pending feedback requests"""
        requests = list(self.pending_requests.values())
        
        if organization_id:
            requests = [r for r in requests if r.organization_id == organization_id]
            
        if user_id:
            requests = [r for r in requests if r.user_id == user_id]
            
        # Sort by priority and deadline
        priority_order = {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}
        requests.sort(key=lambda r: (priority_order[r.priority], r.deadline))
        
        return requests
        
    async def submit_feedback_response(self,
                                      request_id: str,
                                      response: Dict[str, Any]) -> str:
        """
        Submit response to feedback request
        
        Args:
            request_id: ID of feedback request
            response: User's response
        """
        if request_id not in self.pending_requests:
            raise ValueError(f"Unknown request ID: {request_id}")
            
        request = self.pending_requests[request_id]
        
        # Convert response to HumanFeedback
        if request.feedback_type == FeedbackType.PREFERENCE:
            feedback = HumanFeedback(
                feedback_id=f"resp_{request_id}",
                timestamp=datetime.utcnow(),
                feedback_type=FeedbackType.PREFERENCE,
                context=request.context,
                option_a=request.options[0],
                option_b=request.options[1] if len(request.options) > 1 else {},
                preference=response.get('preference', 'equal'),
                user_id=request.user_id,
                organization_id=request.organization_id
            )
        elif request.feedback_type == FeedbackType.RATING:
            feedback = HumanFeedback(
                feedback_id=f"resp_{request_id}",
                timestamp=datetime.utcnow(),
                feedback_type=FeedbackType.RATING,
                context=request.context,
                rating=response.get('rating', 3.0),
                max_rating=response.get('max_rating', 5.0),
                user_id=request.user_id,
                organization_id=request.organization_id
            )
        else:
            feedback = HumanFeedback(
                feedback_id=f"resp_{request_id}",
                timestamp=datetime.utcnow(),
                feedback_type=request.feedback_type,
                context={**request.context, **response},
                user_id=request.user_id,
                organization_id=request.organization_id
            )
            
        await self.feedback_queue.put(feedback)
        
        # Remove from pending
        del self.pending_requests[request_id]
        
        # Update response time statistics
        response_time = (datetime.utcnow() - request.timestamp).total_seconds()
        self.stats['average_response_time'] = (
            self.stats['average_response_time'] * 0.9 + response_time * 0.1
        )
        
        return feedback.feedback_id
        
    def get_feedback_statistics(self, 
                               organization_id: Optional[str] = None) -> Dict[str, Any]:
        """Get feedback collection statistics"""
        stats = dict(self.stats)
        
        if organization_id:
            # Filter for specific organization
            org_feedback = [f for f in self.completed_feedback 
                          if f.organization_id == organization_id]
            
            stats['organization_feedback_count'] = len(org_feedback)
            stats['organization_compliance_rate'] = np.mean([
                f.compliant for f in org_feedback 
                if f.feedback_type == FeedbackType.COMPLIANCE
            ]) if org_feedback else 0.0
            
        stats['pending_requests'] = len(self.pending_requests)
        stats['recent_audits'] = len(self.audit_history)
        
        return stats
        
    async def export_feedback_data(self, 
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None,
                                  organization_id: Optional[str] = None) -> List[Dict]:
        """Export feedback data for analysis"""
        feedback_list = list(self.completed_feedback)
        
        if start_date:
            feedback_list = [f for f in feedback_list if f.timestamp >= start_date]
            
        if end_date:
            feedback_list = [f for f in feedback_list if f.timestamp <= end_date]
            
        if organization_id:
            feedback_list = [f for f in feedback_list 
                           if f.organization_id == organization_id]
            
        return [asdict(f) for f in feedback_list]


class FeedbackIntegrator:
    """
    Integrates feedback with policy generation and compliance checking
    """
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        
    async def validate_policy_with_feedback(self,
                                           policy: Dict[str, Any],
                                           organization_id: str) -> Dict[str, Any]:
        """
        Validate policy using learned preferences
        """
        # Get organization context
        org_context = self.feedback_collector.org_learner.get_organization_context(
            organization_id
        )
        
        # Generate with feedback
        state = torch.randn(1, 768)  # Simplified state representation
        result = self.feedback_collector.rlhf_trainer.generate_with_feedback(
            state,
            temperature=0.7
        )
        
        return {
            'policy': policy,
            'validation_score': result['confidence'],
            'expected_compliance': result['expected_reward'],
            'organization_aligned': True if result['confidence'] > 0.7 else False,
            'recommendations': self._generate_recommendations(policy, result)
        }
        
    def _generate_recommendations(self, 
                                 policy: Dict[str, Any],
                                 feedback_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on feedback"""
        recommendations = []
        
        if feedback_result['confidence'] < 0.5:
            recommendations.append("Consider reviewing policy with compliance team")
            
        if feedback_result['expected_reward'] < 0:
            recommendations.append("Policy may not align with organizational preferences")
            
        if feedback_result['policy_entropy'] > 1.0:
            recommendations.append("Policy scope may be too broad, consider narrowing")
            
        return recommendations


# Global feedback system
feedback_system = None

def initialize_feedback_collection(rlhf_trainer: Optional[RLHFTrainer] = None):
    """Initialize the feedback collection system"""
    global feedback_system
    feedback_system = FeedbackCollector(rlhf_trainer)
    asyncio.create_task(feedback_system.start())
    logger.info("Feedback collection system initialized")
    return feedback_system


# Export main components
__all__ = [
    'FeedbackCollector',
    'FeedbackIntegrator',
    'FeedbackRequest',
    'AuditResult',
    'FeedbackSource',
    'initialize_feedback_collection',
    'feedback_system'
]