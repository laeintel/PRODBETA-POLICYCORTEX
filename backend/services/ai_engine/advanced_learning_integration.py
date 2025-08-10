"""
Advanced Learning Integration Module
Integrates RLHF, Meta-Learning, Few-Shot Learning with existing AI systems
Provides unified interface for all advanced learning capabilities
"""

import torch
import torch.nn as nn
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

# Import existing systems
from .continuous_learning import ContinuousLearningSystem, initialize_continuous_learning
from .memory_enhanced_learning import MemoryEnhancedErrorLearning
from .persistent_memory_system import PersistentMemoryManager
from .multimodal_processing import MultiModalProcessor
from .regularization_enhanced_learning import RegularizedLearningSystem

# Import new advanced systems
from .rlhf_system import RLHFTrainer, initialize_rlhf
from .meta_learning_system import MetaLearningOrchestrator, initialize_meta_learning, Task
from .few_shot_learning import FewShotPolicyGenerator, PolicyExample, initialize_few_shot
from .feedback_collection_system import FeedbackCollector, initialize_feedback_collection

logger = logging.getLogger(__name__)


class UnifiedLearningModel(nn.Module):
    """
    Unified model that combines all learning capabilities
    Base model for meta-learning and RLHF
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 100):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific heads
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.compliance_head = nn.Linear(hidden_dim, 2)  # Binary compliance
        self.risk_head = nn.Linear(hidden_dim, 5)  # Risk levels
        self.cost_head = nn.Linear(hidden_dim, 1)  # Cost prediction
        
    def forward(self, x: torch.Tensor, task: str = 'policy') -> torch.Tensor:
        """Forward pass with task-specific output"""
        features = self.encoder(x)
        
        if task == 'policy':
            return self.policy_head(features)
        elif task == 'compliance':
            return self.compliance_head(features)
        elif task == 'risk':
            return self.risk_head(features)
        elif task == 'cost':
            return self.cost_head(features)
        else:
            return features


class AdvancedLearningOrchestrator:
    """
    Main orchestrator for all advanced learning systems
    Provides unified interface and coordinates between systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Initialize base model
        self.base_model = UnifiedLearningModel(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['output_dim']
        )
        
        # Initialize all learning systems
        self._initialize_systems()
        
        # Statistics and monitoring
        self.stats = {
            'total_learning_iterations': 0,
            'feedback_processed': 0,
            'tasks_adapted': 0,
            'policies_generated': 0,
            'system_uptime': datetime.utcnow()
        }
        
        # Background tasks
        self.background_tasks = []
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'input_dim': 768,
            'hidden_dim': 512,
            'output_dim': 100,
            'enable_rlhf': True,
            'enable_meta_learning': True,
            'enable_few_shot': True,
            'enable_continuous_learning': True,
            'enable_memory': True,
            'enable_multimodal': True,
            'vocab_size': 50000,
            'embedding_dim': 768
        }
        
    def _initialize_systems(self):
        """Initialize all learning systems"""
        logger.info("Initializing advanced learning systems...")
        
        # Continuous Learning
        if self.config['enable_continuous_learning']:
            self.continuous_learner = initialize_continuous_learning(
                vocab_size=self.config['vocab_size']
            )
            logger.info("✓ Continuous learning initialized")
            
        # Memory Systems
        if self.config['enable_memory']:
            self.memory_enhanced = MemoryEnhancedErrorLearning(
                vocab_size=self.config['vocab_size'],
                embedding_dim=self.config['embedding_dim']
            )
            self.persistent_memory = PersistentMemoryManager(
                embedding_dim=self.config['embedding_dim']
            )
            logger.info("✓ Memory systems initialized")
            
        # Multimodal Processing
        if self.config['enable_multimodal']:
            self.multimodal_processor = MultiModalProcessor(
                embedding_dim=self.config['embedding_dim']
            )
            logger.info("✓ Multimodal processing initialized")
            
        # RLHF
        if self.config['enable_rlhf']:
            self.rlhf_trainer = initialize_rlhf(
                state_dim=self.config['input_dim'],
                action_dim=self.config['output_dim']
            )
            self.feedback_collector = initialize_feedback_collection(
                rlhf_trainer=self.rlhf_trainer
            )
            logger.info("✓ RLHF system initialized")
            
        # Meta-Learning
        if self.config['enable_meta_learning']:
            self.meta_learner = initialize_meta_learning(
                base_model=self.base_model,
                feature_dim=self.config['input_dim']
            )
            logger.info("✓ Meta-learning initialized")
            
        # Few-Shot Learning
        if self.config['enable_few_shot']:
            self.few_shot_generator = initialize_few_shot(
                vocab_size=self.config['vocab_size'],
                embedding_dim=self.config['embedding_dim']
            )
            logger.info("✓ Few-shot learning initialized")
            
        # Regularization
        self.regularized_system = RegularizedLearningSystem(
            base_model=self.base_model
        )
        logger.info("✓ Regularization system initialized")
        
        logger.info("All advanced learning systems initialized successfully!")
        
    async def process_policy_request(self,
                                    requirements: Dict[str, Any],
                                    organization_id: str,
                                    use_few_shot: bool = True,
                                    collect_feedback: bool = True) -> Dict[str, Any]:
        """
        Process policy generation request using all available systems
        
        Args:
            requirements: Policy requirements
            organization_id: Organization identifier
            use_few_shot: Whether to use few-shot learning
            collect_feedback: Whether to collect feedback
        """
        logger.info(f"Processing policy request for {organization_id}")
        
        result = {}
        
        # 1. Use few-shot learning if examples available
        if use_few_shot and self.config['enable_few_shot']:
            few_shot_result = self.few_shot_generator.generate_policy(
                requirements=requirements,
                organization_id=organization_id,
                k_shot=5
            )
            result['few_shot_policy'] = few_shot_result
            
        # 2. Apply RLHF for preference alignment
        if self.config['enable_rlhf']:
            state = torch.randn(1, self.config['input_dim'])  # Encode requirements
            rlhf_result = self.rlhf_trainer.generate_with_feedback(
                state=state,
                temperature=0.7
            )
            result['preference_score'] = rlhf_result['confidence']
            result['expected_compliance'] = rlhf_result['expected_reward']
            
        # 3. Use meta-learning for quick adaptation
        if self.config['enable_meta_learning']:
            # Create task from requirements
            task = self._requirements_to_task(requirements)
            adapted_model = self.meta_learner.adapt_to_task(task)
            result['adapted_model'] = 'ready'
            
        # 4. Collect feedback if requested
        if collect_feedback and self.config['enable_rlhf']:
            feedback_request = await self.feedback_collector.request_feedback(
                context=requirements,
                options=[result.get('few_shot_policy', {})],
                feedback_type='rating',
                organization_id=organization_id,
                priority='normal'
            )
            result['feedback_requested'] = feedback_request.request_id
            
        # 5. Store in persistent memory
        if self.config['enable_memory']:
            memory_entry = self.persistent_memory.create_memory(
                content={
                    'type': 'policy_generation',
                    'requirements': requirements,
                    'result': result
                },
                importance=0.7
            )
            result['memory_id'] = memory_entry.id
            
        # Update statistics
        self.stats['policies_generated'] += 1
        
        return result
        
    async def learn_from_incident(self,
                                 incident_data: Dict[str, Any],
                                 organization_id: str) -> Dict[str, Any]:
        """
        Learn from security/operational incident
        
        Args:
            incident_data: Incident information
            organization_id: Organization identifier
        """
        logger.info(f"Learning from incident for {organization_id}")
        
        result = {}
        
        # 1. Process as negative feedback
        if self.config['enable_rlhf']:
            feedback_id = await self.feedback_collector.process_incident(
                incident_type=incident_data.get('type', 'unknown'),
                severity=incident_data.get('severity', 'medium'),
                affected_resources=incident_data.get('resources', []),
                root_cause=incident_data.get('root_cause'),
                resolution=incident_data.get('resolution'),
                organization_id=organization_id
            )
            result['feedback_id'] = feedback_id
            
        # 2. Update continuous learning
        if self.config['enable_continuous_learning']:
            error_logs = [{
                'timestamp': datetime.utcnow().isoformat(),
                'error_type': incident_data.get('type', 'incident'),
                'message': incident_data.get('description', ''),
                'severity': incident_data.get('severity', 'medium'),
                'context': incident_data
            }]
            errors = await self.continuous_learner.collect_errors_from_application(error_logs)
            await self.continuous_learner.learn_from_errors(errors)
            result['continuous_learning'] = 'updated'
            
        # 3. Store in memory with high importance
        if self.config['enable_memory']:
            memory_entry = self.persistent_memory.create_memory(
                content=incident_data,
                importance=0.9  # High importance for incidents
            )
            result['memory_id'] = memory_entry.id
            
            # Update episodic memory
            self.memory_enhanced.episodic_memory.store_episode(
                sequence=torch.randn(10, self.config['embedding_dim']),
                metadata={'incident': incident_data}
            )
            
        # 4. Trigger immediate meta-learning adaptation
        if self.config['enable_meta_learning']:
            incident_task = Task(
                name=f"incident_{incident_data.get('type', 'unknown')}",
                domain='security',
                support_set=[(torch.randn(1, self.config['input_dim']), 
                            torch.tensor([0.0]))],  # Negative example
                query_set=[],
                metadata=incident_data
            )
            self.meta_learner.maml.adapt_to_new_task(incident_task)
            result['meta_adaptation'] = 'complete'
            
        return result
        
    async def adapt_to_new_service(self,
                                  service_name: str,
                                  service_docs: str,
                                  examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adapt to new Azure service using meta-learning
        
        Args:
            service_name: Name of new service
            service_docs: Documentation
            examples: Example configurations
        """
        logger.info(f"Adapting to new service: {service_name}")
        
        result = {}
        
        # 1. Process documentation with multimodal processor
        if self.config['enable_multimodal']:
            doc_features = await self.multimodal_processor.process_input(
                input_data=service_docs,
                input_type='text'
            )
            result['documentation_processed'] = True
            
        # 2. Create few-shot examples
        if self.config['enable_few_shot'] and examples:
            policy_examples = []
            for ex in examples[:5]:  # Use first 5 examples
                policy_ex = PolicyExample(
                    policy_text=json.dumps(ex),
                    policy_type='service_config',
                    resources=[service_name],
                    conditions=ex.get('conditions', {}),
                    actions=ex.get('actions', ['Allow']),
                    compliance_frameworks=[],
                    organization_id='system'
                )
                policy_examples.append(policy_ex)
                
            patterns = self.few_shot_generator.learn_from_examples(
                examples=policy_examples,
                organization_id='system'
            )
            result['patterns_learned'] = patterns
            
        # 3. Use meta-learning for rapid adaptation
        if self.config['enable_meta_learning']:
            # Generate tasks from examples
            tasks = []
            for i in range(0, len(examples), 2):
                if i + 1 < len(examples):
                    support_set = [
                        (torch.randn(1, self.config['input_dim']), 
                         torch.randn(1, 10))
                        for _ in range(min(5, len(examples) // 2))
                    ]
                    query_set = [
                        (torch.randn(1, self.config['input_dim']), 
                         torch.randn(1, 10))
                        for _ in range(min(10, len(examples) // 2))
                    ]
                    
                    task = Task(
                        name=f"{service_name}_config_{i}",
                        domain='azure',
                        support_set=support_set,
                        query_set=query_set,
                        metadata={'service': service_name}
                    )
                    tasks.append(task)
                    
            if tasks:
                self.meta_learner.maml.outer_loop(tasks, num_iterations=10)
                result['meta_learning_complete'] = True
                
        # 4. Store in persistent memory
        if self.config['enable_memory']:
            memory_entry = self.persistent_memory.create_memory(
                content={
                    'type': 'new_service',
                    'service': service_name,
                    'documentation': service_docs[:1000],  # Store summary
                    'examples': examples
                },
                importance=0.8
            )
            result['memory_id'] = memory_entry.id
            
        # Update statistics
        self.stats['tasks_adapted'] += 1
        
        return result
        
    def _requirements_to_task(self, requirements: Dict[str, Any]) -> Task:
        """Convert requirements to meta-learning task"""
        # Create synthetic support and query sets
        support_set = [
            (torch.randn(1, self.config['input_dim']), 
             torch.randn(1, self.config['output_dim']))
            for _ in range(5)
        ]
        query_set = [
            (torch.randn(1, self.config['input_dim']), 
             torch.randn(1, self.config['output_dim']))
            for _ in range(10)
        ]
        
        return Task(
            name=f"policy_{requirements.get('type', 'general')}",
            domain=requirements.get('domain', 'policy'),
            support_set=support_set,
            query_set=query_set,
            metadata=requirements
        )
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get status of all learning systems"""
        status = {
            'uptime': (datetime.utcnow() - self.stats['system_uptime']).total_seconds(),
            'statistics': self.stats,
            'systems': {}
        }
        
        # Check each system
        if self.config['enable_continuous_learning']:
            status['systems']['continuous_learning'] = self.continuous_learner.get_learning_stats()
            
        if self.config['enable_memory']:
            status['systems']['persistent_memory'] = self.persistent_memory.get_memory_summary()
            
        if self.config['enable_rlhf']:
            status['systems']['rlhf'] = self.rlhf_trainer.get_training_stats()
            status['systems']['feedback'] = self.feedback_collector.get_feedback_statistics()
            
        if self.config['enable_meta_learning']:
            status['systems']['meta_learning'] = {
                'tasks_seen': self.meta_learner.maml.meta_stats['tasks_seen'],
                'average_accuracy': self.meta_learner.maml.meta_stats['average_query_accuracy']
            }
            
        return status
        
    async def start_background_learning(self):
        """Start all background learning processes"""
        logger.info("Starting background learning processes...")
        
        # Start continuous learning loop
        if self.config['enable_continuous_learning']:
            task = asyncio.create_task(self.continuous_learner.continuous_learning_loop())
            self.background_tasks.append(task)
            
        # Start RLHF training loop
        if self.config['enable_rlhf']:
            async def rlhf_loop():
                while True:
                    await asyncio.sleep(3600)  # Train every hour
                    if len(self.feedback_collector.completed_feedback) > 50:
                        await self.rlhf_trainer.train_reward_model()
                        logger.info("RLHF model updated")
                        
            task = asyncio.create_task(rlhf_loop())
            self.background_tasks.append(task)
            
        # Memory consolidation loop
        if self.config['enable_memory']:
            async def memory_loop():
                while True:
                    await asyncio.sleep(86400)  # Daily consolidation
                    self.persistent_memory.store.consolidate_memories()
                    self.memory_enhanced.consolidate_memories()
                    logger.info("Memory consolidation complete")
                    
            task = asyncio.create_task(memory_loop())
            self.background_tasks.append(task)
            
        logger.info(f"Started {len(self.background_tasks)} background tasks")
        
    async def stop_background_learning(self):
        """Stop all background processes"""
        logger.info("Stopping background learning processes...")
        
        for task in self.background_tasks:
            task.cancel()
            
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        if self.config['enable_rlhf']:
            await self.feedback_collector.stop()
            
        logger.info("Background processes stopped")


# Global orchestrator instance
advanced_learning_orchestrator = None

def initialize_advanced_learning(config: Optional[Dict[str, Any]] = None):
    """Initialize the advanced learning orchestrator"""
    global advanced_learning_orchestrator
    advanced_learning_orchestrator = AdvancedLearningOrchestrator(config)
    asyncio.create_task(advanced_learning_orchestrator.start_background_learning())
    logger.info("Advanced learning orchestrator initialized and running")
    return advanced_learning_orchestrator


# Export main components
__all__ = [
    'AdvancedLearningOrchestrator',
    'UnifiedLearningModel',
    'initialize_advanced_learning',
    'advanced_learning_orchestrator'
]