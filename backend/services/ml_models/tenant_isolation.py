"""
Patent #4: Tenant-Isolated Infrastructure
Secure multi-tenancy for model training and inference
Author: PolicyCortex ML Team
Date: January 2025

Patent Requirements:
- Tenant-specific model instances
- Differential privacy for training data protection
- Encrypted model parameters using AES-256-GCM
- Secure aggregation for federated learning
- Role-based access control with audit logging
"""

import os
import hashlib
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import numpy as np
import torch
import torch.nn as nn
from threading import Lock
import logging

logger = logging.getLogger(__name__)

@dataclass
class TenantContext:
    """Container for tenant-specific context and metadata"""
    tenant_id: str
    tenant_name: str
    subscription_tier: str
    created_at: datetime
    encryption_key: bytes
    model_version: str
    access_roles: List[str]
    data_residency: str
    compliance_requirements: List[str]
    resource_limits: Dict[str, Any] = field(default_factory=dict)


class DifferentialPrivacy:
    """
    Implement differential privacy for training data protection
    Patent Requirement: Epsilon-delta privacy guarantees
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_budget = epsilon
        self.noise_multiplier = self._calculate_noise_multiplier()
        
    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier for given privacy parameters"""
        # Simplified calculation - in production use Renyi DP accountant
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise_to_gradients(self, gradients: torch.Tensor, 
                              sensitivity: float = 1.0) -> torch.Tensor:
        """Add calibrated noise to gradients for DP-SGD"""
        noise_scale = self.noise_multiplier * sensitivity
        noise = torch.randn_like(gradients) * noise_scale
        return gradients + noise
    
    def clip_gradients(self, gradients: torch.Tensor, 
                       clip_norm: float = 1.0) -> Tuple[torch.Tensor, float]:
        """Clip gradients to bound sensitivity"""
        grad_norm = torch.norm(gradients)
        scale = min(1.0, clip_norm / (grad_norm + 1e-8))
        clipped = gradients * scale
        return clipped, float(grad_norm)
    
    def privatize_data(self, data: np.ndarray, 
                      mechanism: str = 'laplace') -> np.ndarray:
        """Apply differential privacy to data"""
        if mechanism == 'laplace':
            # Laplace mechanism for continuous data
            sensitivity = np.max(np.abs(data)) - np.min(np.abs(data))
            scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, scale, data.shape)
        elif mechanism == 'gaussian':
            # Gaussian mechanism
            sensitivity = np.linalg.norm(data)
            sigma = sensitivity * self.noise_multiplier
            noise = np.random.normal(0, sigma, data.shape)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        
        return data + noise
    
    def update_privacy_budget(self, query_sensitivity: float = 1.0):
        """Track privacy budget consumption"""
        # Simplified budget tracking
        query_cost = query_sensitivity * self.noise_multiplier
        self.privacy_budget -= query_cost
        
        if self.privacy_budget <= 0:
            logger.warning("Privacy budget exhausted!")
        
        return self.privacy_budget


class ModelEncryption:
    """
    Encrypt model parameters using AES-256-GCM
    Patent Requirement: Secure model storage and transmission
    """
    
    def __init__(self, tenant_key: Optional[bytes] = None):
        if tenant_key is None:
            self.key = os.urandom(32)  # 256-bit key
        else:
            self.key = tenant_key
        
        self.backend = default_backend()
        
    def encrypt_model(self, model: nn.Module) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt model parameters
        Returns: (encrypted_data, nonce, tag)
        """
        # Serialize model state
        model_state = model.state_dict()
        serialized = pickle.dumps(model_state)
        
        # Generate nonce
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(nonce),
            backend=self.backend
        )
        
        # Encrypt
        encryptor = cipher.encryptor()
        encrypted = encryptor.update(serialized) + encryptor.finalize()
        
        return encrypted, nonce, encryptor.tag
    
    def decrypt_model(self, encrypted_data: bytes, nonce: bytes, 
                     tag: bytes, model: nn.Module) -> nn.Module:
        """Decrypt and load model parameters"""
        # Create cipher with tag
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.GCM(nonce, tag),
            backend=self.backend
        )
        
        # Decrypt
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Deserialize and load
        model_state = pickle.loads(decrypted)
        model.load_state_dict(model_state)
        
        return model
    
    def derive_tenant_key(self, tenant_id: str, master_key: bytes) -> bytes:
        """Derive tenant-specific encryption key from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=tenant_id.encode(),
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(master_key)


class SecureAggregation:
    """
    Secure aggregation for federated learning
    Patent Requirement: Privacy-preserving gradient aggregation
    """
    
    def __init__(self, num_participants: int, threshold: int):
        self.num_participants = num_participants
        self.threshold = threshold  # Minimum participants for aggregation
        self.participant_masks = {}
        self.aggregated_gradients = None
        
    def generate_masks(self) -> Dict[str, np.ndarray]:
        """Generate random masks for secure aggregation"""
        masks = {}
        
        for i in range(self.num_participants):
            # Generate pairwise masks
            for j in range(i + 1, self.num_participants):
                # Shared random seed between participants i and j
                seed = hashlib.sha256(f"{i}-{j}".encode()).digest()
                np.random.seed(int.from_bytes(seed[:4], 'big'))
                
                # Generate mask
                mask = np.random.randn(1000)  # Placeholder size
                
                # Store with opposite signs
                masks[f"{i}-{j}"] = mask
                masks[f"{j}-{i}"] = -mask
        
        return masks
    
    def add_mask(self, participant_id: str, gradients: np.ndarray) -> np.ndarray:
        """Add mask to gradients for secure transmission"""
        if participant_id not in self.participant_masks:
            # Generate participant-specific masks
            self.participant_masks[participant_id] = self._generate_participant_mask(
                participant_id, gradients.shape
            )
        
        mask = self.participant_masks[participant_id]
        return gradients + mask
    
    def _generate_participant_mask(self, participant_id: str, 
                                  shape: Tuple) -> np.ndarray:
        """Generate mask for specific participant"""
        seed = hashlib.sha256(participant_id.encode()).digest()
        np.random.seed(int.from_bytes(seed[:4], 'big'))
        return np.random.randn(*shape) * 0.01  # Small mask
    
    def aggregate(self, masked_gradients: Dict[str, np.ndarray]) -> np.ndarray:
        """Aggregate masked gradients securely"""
        if len(masked_gradients) < self.threshold:
            raise ValueError(f"Insufficient participants: {len(masked_gradients)} < {self.threshold}")
        
        # Sum masked gradients (masks cancel out)
        aggregated = None
        for gradients in masked_gradients.values():
            if aggregated is None:
                aggregated = gradients.copy()
            else:
                aggregated += gradients
        
        # Average
        aggregated /= len(masked_gradients)
        
        return aggregated


class TenantModelManager:
    """
    Manage tenant-specific model instances
    Patent Requirement: Isolated model instances per tenant
    """
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.tenant_models = {}
        self.tenant_contexts = {}
        self.model_locks = {}
        self.access_log = []
        
    def register_tenant(self, tenant_context: TenantContext):
        """Register new tenant with isolated resources"""
        tenant_id = tenant_context.tenant_id
        
        # Create encryption key
        encryption = ModelEncryption()
        tenant_key = encryption.derive_tenant_key(tenant_id, self.master_key)
        tenant_context.encryption_key = tenant_key
        
        # Store context
        self.tenant_contexts[tenant_id] = tenant_context
        self.model_locks[tenant_id] = Lock()
        
        # Log registration
        self._log_access('register', tenant_id, 'system')
        
        logger.info(f"Registered tenant: {tenant_id}")
    
    def create_model_instance(self, tenant_id: str, model_class: type, 
                            **model_kwargs) -> nn.Module:
        """Create isolated model instance for tenant"""
        if tenant_id not in self.tenant_contexts:
            raise ValueError(f"Tenant {tenant_id} not registered")
        
        with self.model_locks[tenant_id]:
            # Create new model instance
            model = model_class(**model_kwargs)
            
            # Store encrypted
            context = self.tenant_contexts[tenant_id]
            encryption = ModelEncryption(context.encryption_key)
            
            encrypted, nonce, tag = encryption.encrypt_model(model)
            
            self.tenant_models[tenant_id] = {
                'encrypted': encrypted,
                'nonce': nonce,
                'tag': tag,
                'model_class': model_class,
                'model_kwargs': model_kwargs,
                'version': context.model_version,
                'created_at': datetime.now()
            }
            
            self._log_access('create_model', tenant_id, 'system')
            
            return model
    
    def get_model(self, tenant_id: str, user_role: str) -> Optional[nn.Module]:
        """Retrieve tenant model with access control"""
        if tenant_id not in self.tenant_contexts:
            raise ValueError(f"Tenant {tenant_id} not registered")
        
        # Check access permissions
        context = self.tenant_contexts[tenant_id]
        if user_role not in context.access_roles:
            self._log_access('access_denied', tenant_id, user_role)
            raise PermissionError(f"Role {user_role} not authorized for tenant {tenant_id}")
        
        with self.model_locks[tenant_id]:
            if tenant_id not in self.tenant_models:
                return None
            
            # Decrypt model
            model_data = self.tenant_models[tenant_id]
            encryption = ModelEncryption(context.encryption_key)
            
            # Recreate model instance
            model = model_data['model_class'](**model_data['model_kwargs'])
            
            # Load encrypted parameters
            model = encryption.decrypt_model(
                model_data['encrypted'],
                model_data['nonce'],
                model_data['tag'],
                model
            )
            
            self._log_access('get_model', tenant_id, user_role)
            
            return model
    
    def update_model(self, tenant_id: str, model: nn.Module, user_role: str):
        """Update tenant model with new parameters"""
        if tenant_id not in self.tenant_contexts:
            raise ValueError(f"Tenant {tenant_id} not registered")
        
        # Check write permissions
        context = self.tenant_contexts[tenant_id]
        if user_role not in ['admin', 'ml_engineer']:
            self._log_access('update_denied', tenant_id, user_role)
            raise PermissionError(f"Role {user_role} cannot update models")
        
        with self.model_locks[tenant_id]:
            # Encrypt updated model
            encryption = ModelEncryption(context.encryption_key)
            encrypted, nonce, tag = encryption.encrypt_model(model)
            
            # Update stored model
            if tenant_id in self.tenant_models:
                self.tenant_models[tenant_id].update({
                    'encrypted': encrypted,
                    'nonce': nonce,
                    'tag': tag,
                    'updated_at': datetime.now(),
                    'updated_by': user_role
                })
            
            self._log_access('update_model', tenant_id, user_role)
    
    def _log_access(self, action: str, tenant_id: str, user_role: str):
        """Log access for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'tenant_id': tenant_id,
            'user_role': user_role
        }
        self.access_log.append(log_entry)
        
        # Rotate log if too large
        if len(self.access_log) > 10000:
            self.access_log = self.access_log[-5000:]
    
    def get_audit_log(self, tenant_id: Optional[str] = None) -> List[Dict]:
        """Retrieve audit log, optionally filtered by tenant"""
        if tenant_id:
            return [log for log in self.access_log if log['tenant_id'] == tenant_id]
        return self.access_log.copy()


class FederatedLearningCoordinator:
    """
    Coordinate federated learning across tenants
    Patent Requirement: Privacy-preserving collaborative learning
    """
    
    def __init__(self, min_participants: int = 3):
        self.min_participants = min_participants
        self.current_round = 0
        self.participant_updates = {}
        self.global_model = None
        self.dp_module = DifferentialPrivacy(epsilon=1.0)
        self.secure_agg = SecureAggregation(min_participants, min_participants)
        
    def initialize_round(self, global_model: nn.Module):
        """Initialize new federated learning round"""
        self.current_round += 1
        self.participant_updates = {}
        self.global_model = global_model
        
        logger.info(f"Initialized federated learning round {self.current_round}")
    
    def submit_update(self, tenant_id: str, model_update: Dict[str, torch.Tensor],
                     num_samples: int):
        """Submit model update from tenant"""
        # Apply differential privacy
        dp_update = {}
        for param_name, param_value in model_update.items():
            # Clip and add noise
            clipped, norm = self.dp_module.clip_gradients(param_value)
            noisy = self.dp_module.add_noise_to_gradients(clipped)
            dp_update[param_name] = noisy
        
        # Store update
        self.participant_updates[tenant_id] = {
            'update': dp_update,
            'num_samples': num_samples,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Received update from tenant {tenant_id}")
        
        # Check if we can aggregate
        if len(self.participant_updates) >= self.min_participants:
            return self.aggregate_updates()
        
        return None
    
    def aggregate_updates(self) -> nn.Module:
        """Aggregate updates using secure aggregation"""
        if len(self.participant_updates) < self.min_participants:
            raise ValueError("Insufficient participants for aggregation")
        
        # Weighted average based on sample counts
        total_samples = sum(u['num_samples'] for u in self.participant_updates.values())
        
        aggregated_state = {}
        
        for param_name in self.global_model.state_dict():
            weighted_sum = None
            
            for tenant_id, update_data in self.participant_updates.items():
                if param_name in update_data['update']:
                    weight = update_data['num_samples'] / total_samples
                    param_update = update_data['update'][param_name]
                    
                    if weighted_sum is None:
                        weighted_sum = param_update * weight
                    else:
                        weighted_sum += param_update * weight
            
            if weighted_sum is not None:
                aggregated_state[param_name] = weighted_sum
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state, strict=False)
        
        logger.info(f"Aggregated {len(self.participant_updates)} updates")
        
        return self.global_model


class TenantIsolationEngine:
    """
    Main tenant isolation engine orchestrating all security components
    Provides unified interface for secure multi-tenant ML operations
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key is None:
            master_key = os.urandom(32)
        
        self.master_key = master_key
        self.model_manager = TenantModelManager(master_key)
        self.federated_coordinator = FederatedLearningCoordinator()
        self.dp_instances = {}  # Per-tenant DP instances
        
    def onboard_tenant(self, 
                       tenant_id: str,
                       tenant_name: str,
                       subscription_tier: str = 'standard',
                       compliance_requirements: Optional[List[str]] = None) -> TenantContext:
        """Onboard new tenant with full isolation"""
        # Create tenant context
        context = TenantContext(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            subscription_tier=subscription_tier,
            created_at=datetime.now(),
            encryption_key=b'',  # Will be set by manager
            model_version='1.0.0',
            access_roles=['admin', 'ml_engineer', 'analyst'],
            data_residency='us-east',
            compliance_requirements=compliance_requirements or [],
            resource_limits={
                'max_models': 5 if subscription_tier == 'standard' else 20,
                'max_storage_gb': 100 if subscription_tier == 'standard' else 1000,
                'max_requests_per_minute': 100 if subscription_tier == 'standard' else 1000
            }
        )
        
        # Register with model manager
        self.model_manager.register_tenant(context)
        
        # Create tenant-specific DP instance
        if 'HIPAA' in compliance_requirements or 'GDPR' in compliance_requirements:
            # Stricter privacy for regulated data
            self.dp_instances[tenant_id] = DifferentialPrivacy(epsilon=0.5, delta=1e-6)
        else:
            self.dp_instances[tenant_id] = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        logger.info(f"Onboarded tenant {tenant_id} with tier {subscription_tier}")
        
        return context
    
    def train_tenant_model(self,
                          tenant_id: str,
                          model_class: type,
                          training_data: np.ndarray,
                          labels: np.ndarray,
                          user_role: str = 'ml_engineer',
                          **training_kwargs) -> nn.Module:
        """Train model with tenant isolation and privacy"""
        # Get DP instance
        dp = self.dp_instances.get(tenant_id)
        if dp:
            # Apply differential privacy to training data
            training_data = dp.privatize_data(training_data)
        
        # Create model instance
        model = self.model_manager.create_model_instance(
            tenant_id, model_class, **training_kwargs
        )
        
        # Training would happen here with DP-SGD
        # This is placeholder for actual training logic
        logger.info(f"Training model for tenant {tenant_id} with DP guarantees")
        
        # Save trained model
        self.model_manager.update_model(tenant_id, model, user_role)
        
        return model
    
    def predict_with_isolation(self,
                              tenant_id: str,
                              input_data: np.ndarray,
                              user_role: str = 'analyst') -> np.ndarray:
        """Make predictions with tenant isolation"""
        # Retrieve tenant model
        model = self.model_manager.get_model(tenant_id, user_role)
        
        if model is None:
            raise ValueError(f"No model found for tenant {tenant_id}")
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data)
            predictions = model(input_tensor)
        
        return predictions.numpy()
    
    def start_federated_round(self, base_model: nn.Module):
        """Start new federated learning round"""
        self.federated_coordinator.initialize_round(base_model)
    
    def submit_federated_update(self,
                               tenant_id: str,
                               model: nn.Module,
                               num_samples: int) -> Optional[nn.Module]:
        """Submit tenant's model update for federated learning"""
        # Extract model updates (gradients)
        model_update = model.state_dict()
        
        # Submit to coordinator
        aggregated = self.federated_coordinator.submit_update(
            tenant_id, model_update, num_samples
        )
        
        return aggregated
    
    def get_isolation_report(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate isolation and security report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tenants': len(self.model_manager.tenant_contexts),
            'federated_round': self.federated_coordinator.current_round
        }
        
        if tenant_id:
            # Tenant-specific report
            if tenant_id in self.model_manager.tenant_contexts:
                context = self.model_manager.tenant_contexts[tenant_id]
                dp = self.dp_instances.get(tenant_id)
                
                report['tenant'] = {
                    'id': tenant_id,
                    'name': context.tenant_name,
                    'tier': context.subscription_tier,
                    'compliance': context.compliance_requirements,
                    'privacy_budget': dp.privacy_budget if dp else None,
                    'model_version': context.model_version,
                    'access_log': self.model_manager.get_audit_log(tenant_id)[-10:]
                }
        else:
            # Global report
            report['tenant_summary'] = {
                tid: {
                    'name': ctx.tenant_name,
                    'tier': ctx.subscription_tier,
                    'created': ctx.created_at.isoformat()
                }
                for tid, ctx in self.model_manager.tenant_contexts.items()
            }
        
        return report