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
Persistent Memory System for Long-term Continuity
Implements dynamic memories that persist across days, months, and years
Enables the model to remember all interactions over extended time periods
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import sqlite3
import json
import pickle
import lmdb
import os
from pathlib import Path
import hashlib
from collections import defaultdict, OrderedDict
import logging
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class MemoryGranularity(Enum):
    """Time granularity levels for memory organization"""
    IMMEDIATE = "immediate"  # Last 24 hours
    DAILY = "daily"         # Last 30 days
    WEEKLY = "weekly"       # Last 3 months
    MONTHLY = "monthly"     # Last year
    YEARLY = "yearly"       # Multiple years
    PERMANENT = "permanent" # Never forget

@dataclass
class PersistentMemoryEntry:
    """Structure for a persistent memory entry"""
    id: str
    timestamp: datetime
    content: Dict[str, Any]
    embedding: Optional[np.ndarray]
    importance_score: float
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    granularity: MemoryGranularity = MemoryGranularity.IMMEDIATE
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)  # Links to related memories
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'importance_score': self.importance_score,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'granularity': self.granularity.value,
            'metadata': self.metadata,
            'relationships': self.relationships
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersistentMemoryEntry':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            content=data['content'],
            embedding=np.array(data['embedding']) if data.get('embedding') else None,
            importance_score=data['importance_score'],
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None,
            granularity=MemoryGranularity(data.get('granularity', 'immediate')),
            metadata=data.get('metadata', {}),
            relationships=data.get('relationships', [])
        )


class HierarchicalMemoryStore:
    """
    Hierarchical storage system for memories at different time scales
    Uses SQLite for metadata and LMDB for embeddings
    """
    
    def __init__(self, db_path: str = "memory_store", embedding_dim: int = 512):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        
        # SQLite for metadata
        self.conn = sqlite3.connect(
            self.db_path / "memories.db",
            check_same_thread=False
        )
        self._init_db()
        
        # LMDB for embeddings (efficient for large arrays)
        self.embedding_db = lmdb.open(
            str(self.db_path / "embeddings"),
            map_size=10 * 1024 * 1024 * 1024  # 10GB
        )
        
        # Memory indices for fast retrieval
        self.temporal_index = defaultdict(list)  # Group by time period
        self.semantic_index = {}  # Semantic similarity index
        self.importance_index = OrderedDict()  # Sorted by importance
        
        # Load existing memories
        self._load_indices()
        
    def _init_db(self):
        """Initialize SQLite database schema"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                content TEXT,
                importance_score REAL,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                granularity TEXT,
                metadata TEXT,
                relationships TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);
            CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance_score);
            CREATE INDEX IF NOT EXISTS idx_granularity ON memories(granularity);
        """)
        
        self.conn.commit()
    
    def _load_indices(self):
        """Load memory indices from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, timestamp, importance_score, granularity FROM memories")
        
        for row in cursor.fetchall():
            memory_id, timestamp, importance, granularity = row
            timestamp = datetime.fromisoformat(timestamp)
            
            # Update temporal index
            period = self._get_time_period(timestamp)
            self.temporal_index[period].append(memory_id)
            
            # Update importance index
            self.importance_index[memory_id] = importance
        
        # Sort importance index
        self.importance_index = OrderedDict(
            sorted(self.importance_index.items(), key=lambda x: x[1], reverse=True)
        )
    
    def _get_time_period(self, timestamp: datetime) -> str:
        """Get time period key for temporal indexing"""
        now = datetime.utcnow()
        age = now - timestamp
        
        if age.days == 0:
            return f"hour_{timestamp.hour}"
        elif age.days <= 7:
            return f"day_{timestamp.date()}"
        elif age.days <= 30:
            return f"week_{timestamp.isocalendar()[1]}_{timestamp.year}"
        elif age.days <= 365:
            return f"month_{timestamp.month}_{timestamp.year}"
        else:
            return f"year_{timestamp.year}"
    
    def store(self, memory: PersistentMemoryEntry):
        """Store a memory entry"""
        # Store metadata in SQLite
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memories 
            (id, timestamp, content, importance_score, access_count, 
             last_accessed, granularity, metadata, relationships)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            memory.timestamp.isoformat(),
            json.dumps(memory.content),
            memory.importance_score,
            memory.access_count,
            memory.last_accessed.isoformat() if memory.last_accessed else None,
            memory.granularity.value,
            json.dumps(memory.metadata),
            json.dumps(memory.relationships)
        ))
        self.conn.commit()
        
        # Store embedding in LMDB
        if memory.embedding is not None:
            with self.embedding_db.begin(write=True) as txn:
                txn.put(
                    memory.id.encode(),
                    pickle.dumps(memory.embedding)
                )
        
        # Update indices
        period = self._get_time_period(memory.timestamp)
        self.temporal_index[period].append(memory.id)
        self.importance_index[memory.id] = memory.importance_score
    
    def retrieve(self, memory_id: str) -> Optional[PersistentMemoryEntry]:
        """Retrieve a specific memory"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Get embedding from LMDB
        embedding = None
        with self.embedding_db.begin() as txn:
            embedding_data = txn.get(memory_id.encode())
            if embedding_data:
                embedding = pickle.loads(embedding_data)
        
        # Update access count and last accessed
        cursor.execute("""
            UPDATE memories 
            SET access_count = access_count + 1, 
                last_accessed = ? 
            WHERE id = ?
        """, (datetime.utcnow().isoformat(), memory_id))
        self.conn.commit()
        
        # Reconstruct memory entry
        return PersistentMemoryEntry(
            id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            content=json.loads(row[2]),
            embedding=embedding,
            importance_score=row[3],
            access_count=row[4] + 1,
            last_accessed=datetime.utcnow(),
            granularity=MemoryGranularity(row[6]),
            metadata=json.loads(row[7]) if row[7] else {},
            relationships=json.loads(row[8]) if row[8] else []
        )
    
    def search_by_time(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Search memories within time range"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id FROM memories 
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """, (start_date.isoformat(), end_date.isoformat()))
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_important_memories(self, k: int = 10) -> List[str]:
        """Get k most important memories"""
        return list(self.importance_index.keys())[:k]
    
    def consolidate_memories(self):
        """Consolidate and compress old memories"""
        now = datetime.utcnow()
        cursor = self.conn.cursor()
        
        # Define consolidation rules
        consolidation_rules = [
            (30, MemoryGranularity.DAILY),    # After 30 days, move to daily
            (90, MemoryGranularity.WEEKLY),   # After 90 days, move to weekly
            (365, MemoryGranularity.MONTHLY), # After 1 year, move to monthly
            (1095, MemoryGranularity.YEARLY)  # After 3 years, move to yearly
        ]
        
        for days_threshold, new_granularity in consolidation_rules:
            threshold_date = now - timedelta(days=days_threshold)
            
            cursor.execute("""
                UPDATE memories
                SET granularity = ?
                WHERE timestamp < ? AND granularity != ?
            """, (new_granularity.value, threshold_date.isoformat(), new_granularity.value))
        
        self.conn.commit()


class DynamicMemoryNetwork(nn.Module):
    """
    Dynamic Memory Network that adapts based on interaction patterns
    Implements attention-based memory updates and retrieval
    """
    
    def __init__(self, embedding_dim: int = 512, memory_hops: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.memory_hops = memory_hops
        
        # Input module
        self.input_gru = nn.GRU(
            embedding_dim, 
            embedding_dim,
            batch_first=True,
            bidirectional=True
        )
        self.input_fusion = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Question module
        self.question_gru = nn.GRU(
            embedding_dim,
            embedding_dim,
            batch_first=True
        )
        
        # Episodic memory module
        self.memory_gru = nn.GRU(
            embedding_dim,
            embedding_dim,
            batch_first=True
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # Answer module
        self.answer_gru = nn.GRU(
            embedding_dim * 2,
            embedding_dim,
            batch_first=True
        )
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, facts: torch.Tensor, question: torch.Tensor, 
                previous_memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dynamic memory network
        
        Args:
            facts: Historical facts/memories (batch, num_facts, embedding_dim)
            question: Current query (batch, seq_len, embedding_dim)
            previous_memory: Previous memory state
            
        Returns:
            answer: Generated answer
            memory: Updated memory state
        """
        batch_size = facts.size(0)
        
        # Input module: encode facts
        facts_encoded, _ = self.input_gru(facts)
        facts_encoded = self.input_fusion(facts_encoded)
        
        # Question module: encode question
        _, question_encoded = self.question_gru(question)
        question_encoded = question_encoded.squeeze(0)
        
        # Episodic memory module: iterative attention
        memory = question_encoded
        for hop in range(self.memory_hops):
            # Calculate attention over facts
            attention_features = []
            for i in range(facts_encoded.size(1)):
                fact = facts_encoded[:, i, :]
                
                # Combine fact, question, memory, and element-wise products
                combined = torch.cat([
                    fact,
                    question_encoded,
                    memory,
                    fact * question_encoded,
                ], dim=-1)
                
                attention_features.append(self.attention_gate(combined))
            
            # Apply attention weights
            attention_weights = torch.cat(attention_features, dim=1)
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Weighted sum of facts
            context = torch.sum(
                facts_encoded * attention_weights.unsqueeze(-1),
                dim=1
            )
            
            # Update memory
            memory_input = torch.cat([context, memory], dim=-1).unsqueeze(1)
            _, memory = self.memory_gru(memory_input)
            memory = memory.squeeze(0)
        
        # Answer module: generate answer
        answer_input = torch.cat([memory, question_encoded], dim=-1).unsqueeze(1)
        answer_output, _ = self.answer_gru(answer_input)
        answer = self.output_projection(answer_output.squeeze(1))
        
        return answer, memory


class PersistentMemoryManager:
    """
    Main manager for persistent memory system
    Coordinates storage, retrieval, and learning across time scales
    """
    
    def __init__(self, storage_path: str = "persistent_memory", 
                 embedding_dim: int = 512):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.store = HierarchicalMemoryStore(storage_path, embedding_dim)
        self.dynamic_network = DynamicMemoryNetwork(embedding_dim)
        
        # Memory statistics
        self.stats = {
            'total_memories': 0,
            'memories_by_granularity': defaultdict(int),
            'average_importance': 0.0,
            'oldest_memory': None,
            'newest_memory': None
        }
        
        self._update_stats()
        
    def _update_stats(self):
        """Update memory statistics"""
        cursor = self.store.conn.cursor()
        
        # Total memories
        cursor.execute("SELECT COUNT(*) FROM memories")
        self.stats['total_memories'] = cursor.fetchone()[0]
        
        # By granularity
        cursor.execute("SELECT granularity, COUNT(*) FROM memories GROUP BY granularity")
        for row in cursor.fetchall():
            self.stats['memories_by_granularity'][row[0]] = row[1]
        
        # Average importance
        cursor.execute("SELECT AVG(importance_score) FROM memories")
        avg_importance = cursor.fetchone()[0]
        self.stats['average_importance'] = avg_importance if avg_importance else 0.0
        
        # Date range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM memories")
        dates = cursor.fetchone()
        if dates[0]:
            self.stats['oldest_memory'] = datetime.fromisoformat(dates[0])
            self.stats['newest_memory'] = datetime.fromisoformat(dates[1])
    
    def create_memory(self, content: Dict[str, Any], 
                     embedding: Optional[np.ndarray] = None,
                     importance: Optional[float] = None) -> PersistentMemoryEntry:
        """Create a new persistent memory"""
        # Generate ID
        memory_id = hashlib.sha256(
            f"{datetime.utcnow().isoformat()}_{json.dumps(content)}".encode()
        ).hexdigest()[:16]
        
        # Calculate importance if not provided
        if importance is None:
            importance = self._calculate_importance(content)
        
        # Determine initial granularity
        granularity = MemoryGranularity.IMMEDIATE
        
        # Create memory entry
        memory = PersistentMemoryEntry(
            id=memory_id,
            timestamp=datetime.utcnow(),
            content=content,
            embedding=embedding,
            importance_score=importance,
            granularity=granularity,
            metadata={
                'source': content.get('source', 'unknown'),
                'domain': content.get('domain', 'general')
            }
        )
        
        # Find relationships with existing memories
        if embedding is not None:
            related_memories = self._find_related_memories(embedding, k=5)
            memory.relationships = related_memories
        
        # Store memory
        self.store.store(memory)
        
        # Update stats
        self.stats['total_memories'] += 1
        self.stats['memories_by_granularity'][granularity.value] += 1
        
        return memory
    
    def _calculate_importance(self, content: Dict[str, Any]) -> float:
        """Calculate importance score for memory"""
        score = 0.5  # Base score
        
        # Adjust based on content
        if content.get('severity') == 'critical':
            score += 0.3
        elif content.get('severity') == 'high':
            score += 0.2
        
        if content.get('error_type'):
            score += 0.1
        
        if content.get('solution'):
            score += 0.2
        
        return min(1.0, score)
    
    def _find_related_memories(self, embedding: np.ndarray, k: int = 5) -> List[str]:
        """Find related memories based on embedding similarity"""
        related = []
        
        # Get all memory IDs with embeddings
        with self.store.embedding_db.begin() as txn:
            cursor = txn.cursor()
            
            similarities = []
            for key, value in cursor:
                memory_id = key.decode()
                stored_embedding = pickle.loads(value)
                
                # Calculate cosine similarity
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )
                similarities.append((memory_id, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem_id for mem_id, _ in similarities[:k]]
    
    def retrieve_context(self, query_embedding: np.ndarray, 
                         time_range: Optional[Tuple[datetime, datetime]] = None,
                         k: int = 10) -> List[PersistentMemoryEntry]:
        """Retrieve relevant context from persistent memory"""
        relevant_memories = []
        
        # Time-based filtering
        if time_range:
            memory_ids = self.store.search_by_time(time_range[0], time_range[1])
        else:
            # Get all memory IDs
            cursor = self.store.conn.cursor()
            cursor.execute("SELECT id FROM memories")
            memory_ids = [row[0] for row in cursor.fetchall()]
        
        # Retrieve and rank by relevance
        memory_scores = []
        for memory_id in memory_ids:
            memory = self.store.retrieve(memory_id)
            if memory and memory.embedding is not None:
                # Calculate relevance score
                similarity = np.dot(query_embedding, memory.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding)
                )
                
                # Combine with importance and recency
                recency_factor = 1.0 / (1.0 + (datetime.utcnow() - memory.timestamp).days * 0.01)
                score = similarity * 0.5 + memory.importance_score * 0.3 + recency_factor * 0.2
                
                memory_scores.append((memory, score))
        
        # Sort and return top k
        memory_scores.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in memory_scores[:k]]
    
    def learn_from_interaction(self, interaction_data: Dict[str, Any],
                              embedding: torch.Tensor) -> torch.Tensor:
        """Learn from new interaction and update memory"""
        # Convert to numpy for storage
        embedding_np = embedding.detach().cpu().numpy()
        
        # Create persistent memory
        memory = self.create_memory(interaction_data, embedding_np)
        
        # Retrieve relevant historical context
        historical_context = self.retrieve_context(embedding_np, k=20)
        
        if historical_context:
            # Prepare facts for dynamic memory network
            facts = []
            for hist_memory in historical_context:
                if hist_memory.embedding is not None:
                    facts.append(torch.tensor(hist_memory.embedding))
            
            if facts:
                facts_tensor = torch.stack(facts).unsqueeze(0)
                question = embedding.unsqueeze(0).unsqueeze(0)
                
                # Process through dynamic memory network
                answer, updated_memory = self.dynamic_network(facts_tensor, question)
                
                # Store updated understanding
                updated_data = {
                    **interaction_data,
                    'learned_from_history': True,
                    'historical_context_size': len(historical_context)
                }
                self.create_memory(updated_data, answer.detach().cpu().numpy(), importance=0.8)
                
                return answer
        
        return embedding
    
    def get_memory_summary(self, time_period: str = 'all') -> Dict[str, Any]:
        """Get summary of memories for a time period"""
        cursor = self.store.conn.cursor()
        
        # Define time ranges
        now = datetime.utcnow()
        time_ranges = {
            'day': now - timedelta(days=1),
            'week': now - timedelta(days=7),
            'month': now - timedelta(days=30),
            'year': now - timedelta(days=365),
            'all': datetime.min
        }
        
        start_date = time_ranges.get(time_period, datetime.min)
        
        # Get statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as count,
                AVG(importance_score) as avg_importance,
                MAX(importance_score) as max_importance,
                SUM(access_count) as total_accesses
            FROM memories
            WHERE timestamp >= ?
        """, (start_date.isoformat(),))
        
        stats = cursor.fetchone()
        
        # Get top memories
        cursor.execute("""
            SELECT id, content, importance_score
            FROM memories
            WHERE timestamp >= ?
            ORDER BY importance_score DESC
            LIMIT 10
        """, (start_date.isoformat(),))
        
        top_memories = []
        for row in cursor.fetchall():
            top_memories.append({
                'id': row[0],
                'content': json.loads(row[1]),
                'importance': row[2]
            })
        
        return {
            'period': time_period,
            'total_memories': stats[0] if stats[0] else 0,
            'average_importance': stats[1] if stats[1] else 0,
            'max_importance': stats[2] if stats[2] else 0,
            'total_accesses': stats[3] if stats[3] else 0,
            'top_memories': top_memories,
            'memory_span_days': (now - self.stats['oldest_memory']).days if self.stats['oldest_memory'] else 0
        }
    
    def cleanup_old_memories(self, retention_days: Dict[str, int] = None):
        """Clean up old memories based on retention policy"""
        if retention_days is None:
            retention_days = {
                MemoryGranularity.IMMEDIATE.value: 7,
                MemoryGranularity.DAILY.value: 30,
                MemoryGranularity.WEEKLY.value: 90,
                MemoryGranularity.MONTHLY.value: 365,
                MemoryGranularity.YEARLY.value: 1825,  # 5 years
                MemoryGranularity.PERMANENT.value: -1  # Never delete
            }
        
        cursor = self.store.conn.cursor()
        now = datetime.utcnow()
        
        for granularity, days in retention_days.items():
            if days > 0:
                cutoff_date = now - timedelta(days=days)
                
                # Delete old memories with low importance
                cursor.execute("""
                    DELETE FROM memories
                    WHERE granularity = ? 
                    AND timestamp < ?
                    AND importance_score < 0.5
                    AND access_count < 5
                """, (granularity, cutoff_date.isoformat()))
        
        self.store.conn.commit()
        
        # Consolidate remaining memories
        self.store.consolidate_memories()
        
        # Update stats
        self._update_stats()


# Export main components
__all__ = [
    'PersistentMemoryEntry',
    'HierarchicalMemoryStore',
    'DynamicMemoryNetwork',
    'PersistentMemoryManager',
    'MemoryGranularity'
]