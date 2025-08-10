"""
Continuous Learning System with Adam Optimizer, Embeddings, and Error-based Learning
Learns from application errors, Stack Overflow, Reddit, and technical channels
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset, random_split
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import aiohttp
import feedparser
from bs4 import BeautifulSoup
import re
from collections import deque
import pickle
import os
import math
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DIM = 512
HIDDEN_DIM = 1024
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1
MAX_SEQUENCE_LENGTH = 512
LEARNING_RATE = 0.001
BATCH_SIZE = 32
CHECKPOINT_DIR = "models/continuous_learning"

@dataclass
class ErrorEvent:
    """Represents an error event from the application or external source"""
    timestamp: datetime
    source: str  # 'application', 'stackoverflow', 'reddit', 'github', etc.
    error_type: str
    error_message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    solution: Optional[str]
    tags: List[str]
    severity: str
    domain: str  # 'cloud', 'network', 'security', etc.


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor"""
        return x + self.pe[:x.size(0), :]


class ErrorLearningModel(nn.Module):
    """Neural network model for learning from errors using transformers with regularization"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = EMBEDDING_DIM):
        super().__init__()
        
        # Embeddings with weight initialization
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.domain_embedding = nn.Embedding(10, embedding_dim // 4)  # Domain types
        self.severity_embedding = nn.Embedding(5, embedding_dim // 4)  # Severity levels
        
        # Initialize embeddings with Xavier normal
        nn.init.xavier_normal_(self.token_embedding.weight)
        nn.init.xavier_normal_(self.domain_embedding.weight)
        nn.init.xavier_normal_(self.severity_embedding.weight)
        
        # Dropout for embeddings (regularization)
        self.embedding_dropout = nn.Dropout(0.2)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, MAX_SEQUENCE_LENGTH)
        
        # Transformer encoder with batch_first=True for better performance
        encoder_layers = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=NUM_HEADS,
            dim_feedforward=HIDDEN_DIM,
            dropout=DROPOUT,
            activation='gelu',
            batch_first=True,  # Fix the warning
            norm_first=True  # Pre-normalization for better stability
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, 
            NUM_LAYERS,
            norm=nn.LayerNorm(embedding_dim)  # Add final layer norm
        )
        
        # Output layers with weight initialization
        self.fc1 = nn.Linear(embedding_dim, HIDDEN_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(HIDDEN_DIM, embedding_dim)
        
        # Initialize FC layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        # Batch normalization for FC layers
        self.batch_norm1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.batch_norm2 = nn.BatchNorm1d(embedding_dim)
        
        # Task-specific heads with weight initialization
        self.error_classifier = nn.Linear(embedding_dim, 100)  # Error categories
        self.solution_generator = nn.Linear(embedding_dim, vocab_size)
        self.severity_predictor = nn.Linear(embedding_dim, 5)
        
        # Initialize task heads
        nn.init.xavier_uniform_(self.error_classifier.weight)
        nn.init.xavier_uniform_(self.solution_generator.weight)
        nn.init.xavier_uniform_(self.severity_predictor.weight)
        
        # Layer normalization (multiple layers for better regularization)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm_final = nn.LayerNorm(embedding_dim)
        
    def forward(self, tokens: torch.Tensor, domain: torch.Tensor, 
                severity: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model with full regularization"""
        
        # Embed inputs with dropout
        token_emb = self.embedding_dropout(self.token_embedding(tokens))
        domain_emb = self.domain_embedding(domain).unsqueeze(1).expand(-1, tokens.size(1), -1)
        severity_emb = self.severity_embedding(severity).unsqueeze(1).expand(-1, tokens.size(1), -1)
        
        # Apply dropout to domain and severity embeddings
        domain_emb = self.embedding_dropout(domain_emb)
        severity_emb = self.embedding_dropout(severity_emb)
        
        # Combine embeddings
        x = torch.cat([token_emb, domain_emb, severity_emb], dim=-1)
        
        # Project to correct dimension with a learnable projection layer
        if not hasattr(self, 'projection'):
            self.projection = nn.Linear(x.size(-1), EMBEDDING_DIM).to(x.device)
            nn.init.xavier_uniform_(self.projection.weight)
        x = self.projection(x)
        
        # Apply first layer normalization
        x = self.layer_norm1(x)
        
        # Add positional encoding (now with batch_first=True, no need to transpose)
        x = self.pos_encoder(x)
        
        # Pass through transformer (batch_first=True)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Apply second layer normalization
        x = self.layer_norm2(x)
        
        # Global pooling with attention weights for better representation
        attention_weights = torch.softmax(x.mean(dim=-1, keepdim=True), dim=1)
        x = (x * attention_weights).sum(dim=1)  # Weighted pooling
        
        # Pass through FC layers with batch normalization
        hidden = self.fc1(x)
        
        # Apply batch normalization if batch size > 1
        if hidden.size(0) > 1:
            hidden = self.batch_norm1(hidden)
        
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        
        features = self.fc2(hidden)
        
        # Apply second batch normalization if batch size > 1
        if features.size(0) > 1:
            features = self.batch_norm2(features)
        
        # Final layer normalization
        features = self.layer_norm_final(features)
        
        # Generate outputs
        outputs = {
            'features': features,
            'error_class': self.error_classifier(features),
            'solution': self.solution_generator(features),
            'severity': self.severity_predictor(features)
        }
        
        return outputs


class AdamOptimizerWithWarmup:
    """Adam optimizer with learning rate warmup, decay, and comprehensive regularization"""
    
    def __init__(self, model: nn.Module, lr: float = LEARNING_RATE, 
                 warmup_steps: int = 1000, decay_factor: float = 0.95,
                 weight_decay: float = 0.01, l1_lambda: float = 1e-5):
        # Use AdamW which decouples weight decay from gradient-based updates
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay  # L2 regularization
        )
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.current_step = 0
        self.base_lr = lr
        self.l1_lambda = l1_lambda  # L1 regularization strength
        self.model = model
        
    def step(self, loss: torch.Tensor):
        """Perform optimization step with learning rate scheduling and L1 regularization"""
        self.current_step += 1
        
        # Add L1 regularization to loss
        if self.l1_lambda > 0:
            l1_loss = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    l1_loss += torch.sum(torch.abs(param))
            loss = loss + self.l1_lambda * l1_loss
        
        # Warmup and decay scheduling
        if self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing with exponential decay
            decay_steps = self.current_step - self.warmup_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * min(1.0, decay_steps / 10000)))
            exponential_factor = self.decay_factor ** (decay_steps / 1000)
            lr = self.base_lr * cosine_factor * exponential_factor
            
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 1.0)
        torch.nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'], 5.0)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return lr


class ErrorDataset(Dataset):
    """Dataset for error events"""
    
    def __init__(self, errors: List[ErrorEvent], tokenizer):
        self.errors = errors
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.errors)
        
    def __getitem__(self, idx):
        error = self.errors[idx]
        
        # Tokenize error message and context
        text = f"{error.error_type}: {error.error_message}"
        if error.stack_trace:
            text += f"\n{error.stack_trace}"
            
        tokens = self.tokenizer.tokenize(text, max_length=MAX_SEQUENCE_LENGTH)
        
        # Convert domain and severity to indices
        domain_map = {'cloud': 0, 'network': 1, 'security': 2, 'other': 3}
        severity_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3, 'emergency': 4}
        
        domain_idx = domain_map.get(error.domain, 3)
        severity_idx = severity_map.get(error.severity, 1)
        
        # Create solution tokens if available
        solution_tokens = []
        if error.solution:
            solution_tokens = self.tokenizer.tokenize(error.solution, max_length=MAX_SEQUENCE_LENGTH)
            
        return {
            'tokens': torch.tensor(tokens),
            'domain': torch.tensor(domain_idx),
            'severity': torch.tensor(severity_idx),
            'solution': torch.tensor(solution_tokens) if solution_tokens else torch.zeros(1),
            'error_type': error.error_type
        }


class TechnicalDataCrawler:
    """Crawls technical sources for error patterns and solutions"""
    
    def __init__(self):
        self.sources = {
            'stackoverflow': 'https://api.stackexchange.com/2.3/questions',
            'reddit': 'https://www.reddit.com/r/{}/new.json',
            'github': 'https://api.github.com/search/issues',
            'hacker_news': 'https://hacker-news.firebaseio.com/v0/topstories.json'
        }
        self.reddit_subs = ['devops', 'aws', 'azure', 'googlecloud', 'kubernetes', 
                           'networking', 'netsec', 'sysadmin']
        
    async def fetch_stackoverflow_errors(self, tags: List[str]) -> List[ErrorEvent]:
        """Fetch error-related questions from Stack Overflow"""
        errors = []
        
        async with aiohttp.ClientSession() as session:
            for tag in tags:
                params = {
                    'order': 'desc',
                    'sort': 'activity',
                    'tagged': f'{tag};error',
                    'site': 'stackoverflow',
                    'filter': 'withbody'
                }
                
                try:
                    async with session.get(self.sources['stackoverflow'], params=params) as response:
                        data = await response.json()
                        
                        for item in data.get('items', [])[:10]:
                            error = ErrorEvent(
                                timestamp=datetime.fromtimestamp(item['creation_date']),
                                source='stackoverflow',
                                error_type='technical_question',
                                error_message=item['title'],
                                stack_trace=item.get('body', ''),
                                context={'tags': item.get('tags', [])},
                                solution=None,  # Will be filled from answers
                                tags=item.get('tags', []),
                                severity='medium',
                                domain=self._classify_domain(item['tags'])
                            )
                            errors.append(error)
                            
                except Exception as e:
                    logger.error(f"Error fetching Stack Overflow data: {e}")
                    
        return errors
        
    async def fetch_reddit_discussions(self) -> List[ErrorEvent]:
        """Fetch error discussions from Reddit technical subreddits"""
        errors = []
        
        async with aiohttp.ClientSession() as session:
            for subreddit in self.reddit_subs:
                url = self.sources['reddit'].format(subreddit)
                headers = {'User-Agent': 'PolicyCortex/1.0'}
                
                try:
                    async with session.get(url, headers=headers) as response:
                        data = await response.json()
                        
                        for post in data.get('data', {}).get('children', [])[:5]:
                            post_data = post['data']
                            
                            # Filter for error-related posts
                            if any(word in post_data['title'].lower() 
                                  for word in ['error', 'issue', 'problem', 'failed', 'broken']):
                                
                                error = ErrorEvent(
                                    timestamp=datetime.fromtimestamp(post_data['created_utc']),
                                    source=f'reddit/{subreddit}',
                                    error_type='community_report',
                                    error_message=post_data['title'],
                                    stack_trace=post_data.get('selftext', ''),
                                    context={'url': post_data['url'], 'score': post_data['score']},
                                    solution=None,
                                    tags=[subreddit],
                                    severity=self._estimate_severity(post_data['score']),
                                    domain=self._classify_domain([subreddit])
                                )
                                errors.append(error)
                                
                except Exception as e:
                    logger.error(f"Error fetching Reddit data from {subreddit}: {e}")
                    
        return errors
        
    async def fetch_github_issues(self, repos: List[str]) -> List[ErrorEvent]:
        """Fetch error-related issues from GitHub repositories"""
        errors = []
        
        async with aiohttp.ClientSession() as session:
            for repo in repos:
                params = {
                    'q': f'repo:{repo} is:issue label:bug',
                    'sort': 'created',
                    'order': 'desc',
                    'per_page': 10
                }
                
                try:
                    async with session.get(self.sources['github'], params=params) as response:
                        data = await response.json()
                        
                        for item in data.get('items', []):
                            error = ErrorEvent(
                                timestamp=datetime.fromisoformat(item['created_at'].replace('Z', '+00:00')),
                                source=f'github/{repo}',
                                error_type='bug_report',
                                error_message=item['title'],
                                stack_trace=item.get('body', ''),
                                context={'labels': [l['name'] for l in item.get('labels', [])]},
                                solution=None,
                                tags=[l['name'] for l in item.get('labels', [])],
                                severity='high' if 'critical' in str(item.get('labels', [])).lower() else 'medium',
                                domain=self._classify_domain([repo.split('/')[-1]])
                            )
                            errors.append(error)
                            
                except Exception as e:
                    logger.error(f"Error fetching GitHub issues from {repo}: {e}")
                    
        return errors
        
    def _classify_domain(self, tags: List[str]) -> str:
        """Classify error domain based on tags"""
        tags_str = ' '.join(tags).lower()
        
        if any(word in tags_str for word in ['aws', 'azure', 'gcp', 'cloud', 'kubernetes', 'docker']):
            return 'cloud'
        elif any(word in tags_str for word in ['network', 'tcp', 'dns', 'firewall', 'routing']):
            return 'network'
        elif any(word in tags_str for word in ['security', 'vulnerability', 'exploit', 'auth']):
            return 'security'
        else:
            return 'other'
            
    def _estimate_severity(self, score: int) -> str:
        """Estimate severity based on community engagement"""
        if score > 100:
            return 'critical'
        elif score > 50:
            return 'high'
        elif score > 10:
            return 'medium'
        else:
            return 'low'


class ContinuousLearningSystem:
    """Main continuous learning system that orchestrates everything"""
    
    def __init__(self, vocab_size: int = 50000, enable_memory: bool = True):
        # Initialize model
        self.model = ErrorLearningModel(vocab_size)
        self.optimizer = AdamOptimizerWithWarmup(self.model)
        
        # Import memory components if available
        self.enable_memory = enable_memory
        if enable_memory:
            try:
                from .memory_enhanced_learning import (
                    LongTermMemoryBank, 
                    EpisodicMemory,
                    TemporalContextPreserver
                )
                self.long_term_memory = LongTermMemoryBank(memory_size=10000, embedding_dim=512)
                self.episodic_memory = EpisodicMemory(embedding_dim=512)
                self.temporal_context = TemporalContextPreserver(context_window=1000)
                logger.info("Memory enhancement modules initialized")
            except ImportError:
                self.enable_memory = False
                logger.warning("Memory enhancement modules not available")
        
        # Initialize tokenizer (simplified)
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()
        
        # Error buffer for continuous learning
        self.error_buffer = deque(maxlen=10000)
        self.learning_history = []
        
        # Data crawler
        self.crawler = TechnicalDataCrawler()
        
        # Training metrics
        self.metrics = {
            'total_errors_processed': 0,
            'total_training_steps': 0,
            'average_loss': 0.0,
            'accuracy': 0.0
        }
        
        # Load checkpoint if exists
        self.load_checkpoint()
        
    def _build_vocab(self):
        """Build vocabulary for tokenization"""
        # Simplified vocabulary building
        common_tokens = ['<PAD>', '<UNK>', '<START>', '<END>'] + \
                       [f'token_{i}' for i in range(self.vocab_size - 4)]
        
        for i, token in enumerate(common_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
            
    def tokenize(self, text: str, max_length: int = MAX_SEQUENCE_LENGTH) -> List[int]:
        """Simple tokenization (in production, use proper tokenizer like BPE)"""
        words = text.lower().split()
        tokens = []
        
        for word in words[:max_length]:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                tokens.append(self.token_to_id['<UNK>'])
                
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.token_to_id['<PAD>'])
            
        return tokens
        
    async def collect_errors_from_application(self, error_logs: List[Dict[str, Any]]) -> List[ErrorEvent]:
        """Collect and process errors from application logs"""
        errors = []
        
        for log in error_logs:
            error = ErrorEvent(
                timestamp=datetime.fromisoformat(log.get('timestamp', datetime.utcnow().isoformat())),
                source='application',
                error_type=log.get('error_type', 'unknown'),
                error_message=log.get('message', ''),
                stack_trace=log.get('stack_trace'),
                context=log.get('context', {}),
                solution=log.get('solution'),
                tags=log.get('tags', []),
                severity=log.get('severity', 'medium'),
                domain=self._detect_domain(log)
            )
            errors.append(error)
            
        return errors
        
    def _detect_domain(self, log: Dict[str, Any]) -> str:
        """Detect domain from log content"""
        message = str(log.get('message', '')).lower()
        
        if any(word in message for word in ['s3', 'ec2', 'lambda', 'azure', 'blob', 'vm']):
            return 'cloud'
        elif any(word in message for word in ['connection', 'timeout', 'dns', 'socket']):
            return 'network'
        elif any(word in message for word in ['authentication', 'authorization', 'permission', 'token']):
            return 'security'
        else:
            return 'other'
            
    async def learn_from_errors(self, errors: List[ErrorEvent]):
        """Train model on new error events with memory enhancement"""
        if not errors:
            return
            
        # Add to buffer
        self.error_buffer.extend(errors)
        
        # Split into train and validation for overfitting prevention
        dataset = ErrorDataset(list(self.error_buffer), self)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # Training loop with validation monitoring
        self.model.train()
        total_train_loss = 0
        total_val_loss = 0
        
        for batch in train_loader:
            # Prepare batch
            tokens = batch['tokens']
            domain = batch['domain']
            severity = batch['severity']
            
            # Memory enhancement: retrieve relevant context
            if self.enable_memory:
                # Get features from current batch
                with torch.no_grad():
                    current_features = self.model(tokens, domain, severity)['features']
                
                # Retrieve from long-term memory
                memory_context, memory_scores = self.long_term_memory.retrieve(
                    current_features.mean(dim=0).unsqueeze(0), k=5
                )
                
                # Get temporal context
                error_dict = {
                    'error_type': batch.get('error_type', ['unknown'])[0],
                    'severity': ['low', 'medium', 'high', 'critical'][severity[0].item()]
                }
                relevant_contexts = self.temporal_context.get_relevant_context(error_dict, k=3)
            
            # Forward pass
            outputs = self.model(tokens, domain, severity)
            
            # Calculate loss with label smoothing for regularization
            target = torch.zeros(tokens.size(0), dtype=torch.long)
            if hasattr(self, 'label_smoothing_loss'):
                loss = self.label_smoothing_loss(outputs['error_class'], target)
            else:
                loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs['error_class'], target)
            
            # Backward pass with Adam optimizer (includes weight decay and L1)
            lr = self.optimizer.step(loss)
            
            # Update memory systems
            if self.enable_memory:
                # Store important patterns in long-term memory
                importance = loss.detach()  # Use loss as importance metric
                self.long_term_memory.update(
                    current_features.detach(),
                    outputs['features'].detach(),
                    importance.unsqueeze(0).expand(current_features.size(0))
                )
                
                # Add to temporal context
                for i in range(min(3, tokens.size(0))):  # Store top 3 from batch
                    self.temporal_context.add_context(
                        error_dict,
                        outputs['features'][i].detach()
                    )
            
            total_train_loss += loss.item()
            self.metrics['total_training_steps'] += 1
        
        # Validation pass
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens']
                domain = batch['domain']
                severity = batch['severity']
                
                outputs = self.model(tokens, domain, severity)
                target = torch.zeros(tokens.size(0), dtype=torch.long)
                val_loss = nn.CrossEntropyLoss()(outputs['error_class'], target)
                total_val_loss += val_loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader) if train_loader else 0
        avg_val_loss = total_val_loss / len(val_loader) if val_loader else 0
        
        # Check for overfitting
        if avg_val_loss > avg_train_loss * 1.5:  # 50% higher validation loss indicates overfitting
            logger.warning(f"Potential overfitting detected: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
            # Increase dropout or reduce learning rate
            if hasattr(self.optimizer, 'decay_factor'):
                self.optimizer.decay_factor *= 0.9  # Increase decay
            
        # Update metrics
        self.metrics['total_errors_processed'] += len(errors)
        self.metrics['average_loss'] = total_loss / len(dataloader) if len(dataloader) > 0 else 0
        
        # Log learning progress
        self.learning_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'errors_processed': len(errors),
            'average_loss': self.metrics['average_loss'],
            'learning_rate': lr
        })
        
        logger.info(f"Learned from {len(errors)} errors. Average loss: {self.metrics['average_loss']:.4f}")
        
    async def continuous_learning_loop(self):
        """Main loop that continuously learns from various sources"""
        while True:
            try:
                # Collect errors from external sources
                errors = []
                
                # Stack Overflow
                so_errors = await self.crawler.fetch_stackoverflow_errors(
                    ['azure', 'aws', 'kubernetes', 'terraform', 'ansible']
                )
                errors.extend(so_errors)
                
                # Reddit
                reddit_errors = await self.crawler.fetch_reddit_discussions()
                errors.extend(reddit_errors)
                
                # GitHub
                github_errors = await self.crawler.fetch_github_issues([
                    'hashicorp/terraform',
                    'kubernetes/kubernetes',
                    'ansible/ansible',
                    'Azure/azure-cli'
                ])
                errors.extend(github_errors)
                
                # Learn from collected errors
                if errors:
                    await self.learn_from_errors(errors)
                    
                # Save checkpoint periodically
                if self.metrics['total_training_steps'] % 100 == 0:
                    self.save_checkpoint()
                    
                # Sleep before next iteration
                await asyncio.sleep(3600)  # Learn every hour
                
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
                
    def predict_solution(self, error_message: str, domain: str = 'other') -> Dict[str, Any]:
        """Predict solution for a given error"""
        self.model.eval()
        
        # Tokenize input
        tokens = torch.tensor([self.tokenize(error_message)])
        domain_map = {'cloud': 0, 'network': 1, 'security': 2, 'other': 3}
        domain_tensor = torch.tensor([domain_map.get(domain, 3)])
        severity_tensor = torch.tensor([1])  # Default medium severity
        
        with torch.no_grad():
            outputs = self.model(tokens, domain_tensor, severity_tensor)
            
        # Get predictions
        error_class = torch.argmax(outputs['error_class'], dim=1).item()
        solution_tokens = torch.argmax(outputs['solution'], dim=1)
        severity_pred = torch.argmax(outputs['severity'], dim=1).item()
        
        # Decode solution (simplified)
        solution_words = [self.id_to_token.get(t.item(), '<UNK>') for t in solution_tokens[:50]]
        solution_text = ' '.join(solution_words)
        
        severity_levels = ['low', 'medium', 'high', 'critical', 'emergency']
        
        return {
            'error_classification': f'error_type_{error_class}',
            'predicted_solution': solution_text,
            'severity': severity_levels[severity_pred],
            'confidence': 0.85,  # In production, calculate actual confidence
            'learned_from': f'{self.metrics["total_errors_processed"]} errors'
        }
        
    def save_checkpoint(self):
        """Save model checkpoint"""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.optimizer.state_dict(),
            'metrics': self.metrics,
            'learning_history': self.learning_history[-100:],  # Keep last 100 entries
            'timestamp': datetime.utcnow().isoformat(),
            'vocab_size': self.vocab_size  # Save vocab size for compatibility check
        }
        
        path = os.path.join(CHECKPOINT_DIR, 'continuous_learning_checkpoint.pt')
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
    def load_checkpoint(self):
        """Load model checkpoint if exists"""
        path = os.path.join(CHECKPOINT_DIR, 'continuous_learning_checkpoint.pt')
        
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                # Check if vocab size matches
                if 'vocab_size' in checkpoint and checkpoint['vocab_size'] != self.vocab_size:
                    logger.warning(f"Vocab size mismatch: checkpoint has {checkpoint['vocab_size']}, model has {self.vocab_size}. Skipping checkpoint.")
                    return
                    
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.metrics = checkpoint['metrics']
                self.learning_history = checkpoint['learning_history']
                logger.info(f"Checkpoint loaded from {checkpoint['timestamp']}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        return {
            'metrics': self.metrics,
            'recent_history': self.learning_history[-10:],
            'buffer_size': len(self.error_buffer),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'domains_covered': ['cloud', 'network', 'security'],
            'sources_active': ['application', 'stackoverflow', 'reddit', 'github']
        }


# Global instance
continuous_learner = None

def initialize_continuous_learning(vocab_size: int = 50000):
    """Initialize the continuous learning system"""
    global continuous_learner
    continuous_learner = ContinuousLearningSystem(vocab_size)
    
    # Start continuous learning in background
    asyncio.create_task(continuous_learner.continuous_learning_loop())
    
    logger.info("Continuous learning system initialized")
    return continuous_learner


# Export main components
__all__ = [
    'ContinuousLearningSystem',
    'ErrorLearningModel',
    'AdamOptimizerWithWarmup',
    'ErrorEvent',
    'TechnicalDataCrawler',
    'initialize_continuous_learning',
    'continuous_learner'
]