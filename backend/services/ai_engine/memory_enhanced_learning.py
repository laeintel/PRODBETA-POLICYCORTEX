"""
Memory-Enhanced Learning System with Long Context Retention
Implements memory mechanisms to prevent context loss during continuous learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)

class LongTermMemoryBank:
    """
    Long-term memory bank for storing important error patterns
    Retains past knowledge to shape future predictions
    """
    
    def __init__(self, memory_size: int = 10000, embedding_dim: int = 512):
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        
        # Memory buffers
        self.key_memory = torch.zeros(memory_size, embedding_dim)
        self.value_memory = torch.zeros(memory_size, embedding_dim)
        self.age_memory = torch.zeros(memory_size)  # Track memory age
        self.importance_scores = torch.zeros(memory_size)  # Track importance
        
        self.current_index = 0
        self.is_full = False
        
    def update(self, keys: torch.Tensor, values: torch.Tensor, 
               importance: Optional[torch.Tensor] = None):
        """Update memory with new experiences"""
        batch_size = keys.size(0)
        
        # Calculate importance if not provided
        if importance is None:
            # Use attention-based importance scoring
            importance = torch.norm(values, dim=1)
        
        for i in range(batch_size):
            # Find slot to update (replace least important old memory)
            if self.is_full:
                # Replace least important memory weighted by age
                age_weight = 1.0 / (1.0 + self.age_memory)
                weighted_importance = self.importance_scores * age_weight
                idx = torch.argmin(weighted_importance).item()
            else:
                idx = self.current_index
                self.current_index = (self.current_index + 1) % self.memory_size
                if self.current_index == 0:
                    self.is_full = True
            
            # Update memory
            self.key_memory[idx] = keys[i]
            self.value_memory[idx] = values[i]
            self.importance_scores[idx] = importance[i]
            self.age_memory[idx] = 0
            
        # Age all memories
        self.age_memory += 1
        
    def retrieve(self, query: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve k most relevant memories for given query"""
        # Calculate similarity scores
        similarity = torch.matmul(query, self.key_memory.T)
        
        # Get top-k memories
        if self.is_full:
            effective_size = self.memory_size
        else:
            effective_size = self.current_index
            
        top_k = min(k, effective_size)
        if top_k == 0:
            return torch.zeros(query.size(0), 0, self.embedding_dim), torch.zeros(query.size(0), 0)
        
        scores, indices = torch.topk(similarity[:, :effective_size], top_k, dim=1)
        
        # Retrieve corresponding values
        batch_size = query.size(0)
        retrieved_values = torch.zeros(batch_size, top_k, self.embedding_dim)
        
        for i in range(batch_size):
            retrieved_values[i] = self.value_memory[indices[i]]
            
        return retrieved_values, scores
    
    def consolidate(self, threshold: float = 0.8):
        """Consolidate similar memories to prevent redundancy"""
        if not self.is_full and self.current_index < 2:
            return
            
        effective_size = self.memory_size if self.is_full else self.current_index
        
        # Calculate pairwise similarities
        similarities = torch.matmul(self.key_memory[:effective_size], 
                                  self.key_memory[:effective_size].T)
        
        # Find highly similar pairs (excluding self-similarity)
        similarities.fill_diagonal_(0)
        high_similarity_pairs = (similarities > threshold).nonzero()
        
        # Merge similar memories
        merged = set()
        for i, j in high_similarity_pairs:
            if i.item() in merged or j.item() in merged:
                continue
                
            # Keep the more important memory
            if self.importance_scores[i] > self.importance_scores[j]:
                keep_idx, remove_idx = i.item(), j.item()
            else:
                keep_idx, remove_idx = j.item(), i.item()
            
            # Merge information
            self.value_memory[keep_idx] = (
                self.value_memory[keep_idx] * self.importance_scores[keep_idx] +
                self.value_memory[remove_idx] * self.importance_scores[remove_idx]
            ) / (self.importance_scores[keep_idx] + self.importance_scores[remove_idx])
            
            # Update importance
            self.importance_scores[keep_idx] += self.importance_scores[remove_idx] * 0.5
            
            # Mark for removal
            merged.add(remove_idx)
            self.importance_scores[remove_idx] = 0


class EpisodicMemory(nn.Module):
    """
    Episodic memory for storing and retrieving specific error episodes
    Helps maintain context over long sequences
    """
    
    def __init__(self, episode_length: int = 100, num_episodes: int = 1000, 
                 embedding_dim: int = 512):
        super().__init__()
        self.episode_length = episode_length
        self.num_episodes = num_episodes
        self.embedding_dim = embedding_dim
        
        # Episode storage
        self.episodes = deque(maxlen=num_episodes)
        
        # Attention mechanism for episode retrieval
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Temporal encoding for episodes
        self.temporal_encoder = nn.LSTM(
            embedding_dim, 
            embedding_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
    def store_episode(self, sequence: torch.Tensor, metadata: Dict[str, Any]):
        """Store an error episode with metadata"""
        episode = {
            'sequence': sequence.detach().cpu(),
            'timestamp': datetime.utcnow(),
            'metadata': metadata,
            'access_count': 0
        }
        self.episodes.append(episode)
        
    def retrieve_relevant_episodes(self, query: torch.Tensor, k: int = 5) -> List[Dict]:
        """Retrieve k most relevant episodes for given query"""
        if not self.episodes:
            return []
        
        # Project query
        query_proj = self.query_projection(query)
        
        # Calculate relevance scores for all episodes
        scores = []
        for episode in self.episodes:
            episode_embedding = episode['sequence'].mean(dim=0)
            if episode_embedding.device != query_proj.device:
                episode_embedding = episode_embedding.to(query_proj.device)
            
            # Calculate attention score
            key_proj = self.key_projection(episode_embedding)
            score = torch.matmul(query_proj, key_proj.T).mean()
            scores.append(score.item())
            
        # Get top-k episodes
        top_indices = np.argsort(scores)[-k:][::-1]
        relevant_episodes = []
        
        for idx in top_indices:
            episode = self.episodes[idx]
            episode['access_count'] += 1
            relevant_episodes.append(episode)
            
        return relevant_episodes
    
    def temporal_attention(self, current_state: torch.Tensor, 
                          episodes: List[Dict]) -> torch.Tensor:
        """Apply temporal attention over retrieved episodes"""
        if not episodes:
            return torch.zeros_like(current_state)
        
        # Encode episodes with temporal information
        episode_tensors = []
        for episode in episodes:
            seq = episode['sequence']
            if seq.device != current_state.device:
                seq = seq.to(current_state.device)
            
            # Add temporal encoding based on age
            age = (datetime.utcnow() - episode['timestamp']).total_seconds() / 3600
            temporal_weight = 1.0 / (1.0 + age * 0.01)  # Decay over time
            
            episode_tensors.append(seq * temporal_weight)
        
        # Stack episodes
        if episode_tensors:
            episodes_tensor = torch.stack(episode_tensors)
            
            # Apply LSTM for temporal processing
            lstm_out, _ = self.temporal_encoder(episodes_tensor)
            
            # Attention over episodes
            attention_scores = torch.matmul(current_state.unsqueeze(0), lstm_out.transpose(1, 2))
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Weighted combination
            context = torch.matmul(attention_weights, lstm_out).squeeze(0)
            
            return context
        
        return torch.zeros_like(current_state)


class WorkingMemory(nn.Module):
    """
    Working memory for maintaining short-term context
    Uses self-attention to preserve relevant information
    """
    
    def __init__(self, memory_slots: int = 32, embedding_dim: int = 512):
        super().__init__()
        self.memory_slots = memory_slots
        self.embedding_dim = embedding_dim
        
        # Memory slots
        self.memory = nn.Parameter(torch.randn(memory_slots, embedding_dim) * 0.01)
        
        # Gating mechanism
        self.input_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.forget_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.output_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Self-attention for memory
        self.self_attention = nn.MultiheadAttention(
            embedding_dim, 
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, input_tensor: torch.Tensor, 
                previous_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Update working memory with new input"""
        batch_size = input_tensor.size(0)
        
        # Initialize or use previous memory
        if previous_memory is None:
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory = previous_memory
        
        # Expand input to match memory slots
        input_expanded = input_tensor.unsqueeze(1).expand(-1, self.memory_slots, -1)
        
        # Concatenate for gating
        combined = torch.cat([memory, input_expanded], dim=-1)
        
        # Apply gates
        i_gate = torch.sigmoid(self.input_gate(combined))
        f_gate = torch.sigmoid(self.forget_gate(combined))
        o_gate = torch.sigmoid(self.output_gate(combined))
        
        # Update memory
        memory = f_gate * memory + i_gate * input_expanded
        
        # Self-attention over memory slots
        attended_memory, _ = self.self_attention(memory, memory, memory)
        
        # Output gating
        output = o_gate * attended_memory
        
        return output


class CompressiveTransformer(nn.Module):
    """
    Compressive Transformer that compresses old memories
    Prevents context loss in long sequences
    """
    
    def __init__(self, d_model: int = 512, nhead: int = 8, 
                 compression_rate: int = 4, memory_size: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.compression_rate = compression_rate
        self.memory_size = memory_size
        
        # Compression network
        self.compressor = nn.Sequential(
            nn.Linear(d_model * compression_rate, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Main transformer layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        # Memory buffer
        self.compressed_memory = None
        self.raw_memory_buffer = []
        
    def compress_memory(self, memory: torch.Tensor) -> torch.Tensor:
        """Compress memory using learned compression"""
        batch_size, seq_len, d_model = memory.shape
        
        # Reshape for compression
        compressed_len = seq_len // self.compression_rate
        memory_reshaped = memory.reshape(batch_size, compressed_len, -1)
        
        # Apply compression
        compressed = self.compressor(memory_reshaped)
        
        return compressed
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with memory compression"""
        batch_size, seq_len, _ = x.shape
        
        # Add to raw memory buffer
        self.raw_memory_buffer.append(x)
        
        # Compress old memories if buffer is full
        if len(self.raw_memory_buffer) * seq_len > self.memory_size:
            # Concatenate buffer
            buffer_tensor = torch.cat(self.raw_memory_buffer[:-1], dim=1)
            
            # Compress
            compressed = self.compress_memory(buffer_tensor)
            
            # Update compressed memory
            if self.compressed_memory is None:
                self.compressed_memory = compressed
            else:
                self.compressed_memory = torch.cat([self.compressed_memory, compressed], dim=1)
                
                # Limit compressed memory size
                if self.compressed_memory.size(1) > self.memory_size // self.compression_rate:
                    self.compressed_memory = self.compressed_memory[:, -self.memory_size // self.compression_rate:]
            
            # Clear buffer except last element
            self.raw_memory_buffer = [self.raw_memory_buffer[-1]]
        
        # Concatenate compressed memory with current input
        if self.compressed_memory is not None:
            full_sequence = torch.cat([self.compressed_memory, x], dim=1)
        else:
            full_sequence = x
        
        # Apply transformer
        output = self.transformer(full_sequence, src_mask=mask)
        
        # Return only the output for current input
        return output[:, -seq_len:]


class MemoryEnhancedErrorLearning(nn.Module):
    """
    Complete memory-enhanced error learning model
    Combines all memory mechanisms for long context retention
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 512):
        super().__init__()
        
        # Base embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Memory components
        self.long_term_memory = LongTermMemoryBank(memory_size=10000, embedding_dim=embedding_dim)
        self.episodic_memory = EpisodicMemory(embedding_dim=embedding_dim)
        self.working_memory = WorkingMemory(memory_slots=32, embedding_dim=embedding_dim)
        
        # Compressive transformer layers
        self.compressive_layers = nn.ModuleList([
            CompressiveTransformer(d_model=embedding_dim, nhead=8)
            for _ in range(6)
        ])
        
        # Memory fusion
        self.memory_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Output heads
        self.classifier = nn.Linear(embedding_dim, 100)
        self.predictor = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, 
                error_metadata: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with full memory enhancement"""
        
        # Embed and encode position
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        
        # Retrieve from long-term memory
        batch_size = x.size(0)
        query = x.mean(dim=1)  # Use mean pooling as query
        ltm_values, ltm_scores = self.long_term_memory.retrieve(query, k=10)
        ltm_context = ltm_values.mean(dim=1) if ltm_values.size(1) > 0 else torch.zeros_like(query)
        
        # Retrieve relevant episodes
        episodes = self.episodic_memory.retrieve_relevant_episodes(query, k=5)
        episodic_context = self.episodic_memory.temporal_attention(query, episodes)
        
        # Update working memory
        working_memory_out = self.working_memory(x.mean(dim=1))
        working_context = working_memory_out.mean(dim=1)
        
        # Process through compressive transformer
        for layer in self.compressive_layers:
            x = layer(x)
        
        # Pool transformer output
        transformer_output = x.mean(dim=1)
        
        # Fuse all memory sources
        combined = torch.cat([
            transformer_output,
            ltm_context,
            episodic_context,
            working_context
        ], dim=-1)
        
        fused = self.memory_fusion(combined)
        
        # Update long-term memory with important patterns
        self.long_term_memory.update(query, fused)
        
        # Store episode if metadata provided
        if error_metadata:
            self.episodic_memory.store_episode(x, error_metadata)
        
        # Generate outputs
        outputs = {
            'features': fused,
            'classification': self.classifier(fused),
            'prediction': self.predictor(fused),
            'memory_scores': {
                'ltm_scores': ltm_scores.mean().item() if ltm_scores.numel() > 0 else 0,
                'episode_count': len(episodes),
                'working_memory_norm': working_context.norm().item()
            }
        }
        
        return outputs
    
    def consolidate_memories(self):
        """Consolidate memories to prevent redundancy"""
        self.long_term_memory.consolidate(threshold=0.8)
        
        # Prune old episodes
        if len(self.episodic_memory.episodes) > self.episodic_memory.num_episodes * 0.9:
            # Remove least accessed episodes
            sorted_episodes = sorted(self.episodic_memory.episodes, 
                                   key=lambda x: x['access_count'])
            num_to_remove = len(self.episodic_memory.episodes) // 10
            for _ in range(num_to_remove):
                self.episodic_memory.episodes.popleft()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalContextPreserver:
    """
    Preserves temporal context across training sessions
    Implements 'retaining the past, shaping the future' principle
    """
    
    def __init__(self, context_window: int = 1000):
        self.context_window = context_window
        self.temporal_buffer = deque(maxlen=context_window)
        self.context_embeddings = []
        self.temporal_patterns = {}
        
    def add_context(self, error_data: Dict, embedding: torch.Tensor):
        """Add new context to temporal buffer"""
        context = {
            'timestamp': datetime.utcnow(),
            'error_data': error_data,
            'embedding': embedding.detach().cpu(),
            'importance': self._calculate_importance(error_data)
        }
        self.temporal_buffer.append(context)
        
        # Extract temporal patterns
        self._extract_patterns()
        
    def _calculate_importance(self, error_data: Dict) -> float:
        """Calculate importance score for context"""
        severity_scores = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
        base_score = severity_scores.get(error_data.get('severity', 'medium'), 0.5)
        
        # Boost score for repeated errors
        error_type = error_data.get('error_type')
        if error_type in self.temporal_patterns:
            frequency = self.temporal_patterns[error_type]['frequency']
            base_score *= (1 + min(frequency * 0.1, 0.5))
            
        return base_score
    
    def _extract_patterns(self):
        """Extract temporal patterns from buffer"""
        if len(self.temporal_buffer) < 10:
            return
            
        # Group by error type
        error_groups = {}
        for context in self.temporal_buffer:
            error_type = context['error_data'].get('error_type')
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(context)
        
        # Identify patterns
        for error_type, contexts in error_groups.items():
            if len(contexts) >= 2:
                # Calculate frequency
                time_diffs = []
                for i in range(1, len(contexts)):
                    diff = (contexts[i]['timestamp'] - contexts[i-1]['timestamp']).total_seconds()
                    time_diffs.append(diff)
                
                avg_interval = np.mean(time_diffs) if time_diffs else 0
                
                self.temporal_patterns[error_type] = {
                    'frequency': len(contexts) / len(self.temporal_buffer),
                    'avg_interval': avg_interval,
                    'last_seen': contexts[-1]['timestamp']
                }
    
    def get_relevant_context(self, current_error: Dict, k: int = 5) -> List[Dict]:
        """Get most relevant historical context"""
        if not self.temporal_buffer:
            return []
        
        # Calculate relevance scores
        scores = []
        current_type = current_error.get('error_type')
        
        for context in self.temporal_buffer:
            score = 0
            
            # Type similarity
            if context['error_data'].get('error_type') == current_type:
                score += 1.0
            
            # Temporal proximity (recent is more relevant)
            age = (datetime.utcnow() - context['timestamp']).total_seconds() / 3600
            score += 1.0 / (1.0 + age * 0.1)
            
            # Importance
            score += context['importance']
            
            scores.append(score)
        
        # Get top-k contexts
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.temporal_buffer[i] for i in top_indices]


# Export components
__all__ = [
    'LongTermMemoryBank',
    'EpisodicMemory',
    'WorkingMemory',
    'CompressiveTransformer',
    'MemoryEnhancedErrorLearning',
    'TemporalContextPreserver'
]