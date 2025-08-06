"""
Data Synchronizer Module
Handles data synchronization between different sources
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import structlog
import hashlib
import json

from .data_connector import DataConnector, DataSourceType

logger = structlog.get_logger(__name__)

class SyncMode(str, Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    CDC = "cdc"  # Change Data Capture
    SNAPSHOT = "snapshot"

class SyncDirection(str, Enum):
    SOURCE_TO_TARGET = "source_to_target"
    TARGET_TO_SOURCE = "target_to_source"
    BIDIRECTIONAL = "bidirectional"

class SyncStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class SyncRule:
    """Represents a data synchronization rule"""
    id: str
    name: str
    source: str
    target: str
    mode: SyncMode
    direction: SyncDirection
    schedule: Optional[str] = None  # Cron expression
    filters: Dict[str, Any] = field(default_factory=dict)
    field_mapping: Dict[str, str] = field(default_factory=dict)
    conflict_resolution: str = "source_wins"
    enabled: bool = True
    last_sync: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SyncResult:
    """Result of a synchronization operation"""
    rule_id: str
    status: SyncStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    records_processed: int = 0
    records_created: int = 0
    records_updated: int = 0
    records_deleted: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataSynchronizer:
    """
    Handles data synchronization between different data sources
    """
    
    def __init__(self, data_connector: DataConnector):
        self.data_connector = data_connector
        self.sync_rules = {}
        self.sync_history = []
        self.running_syncs = set()
        
    async def add_sync_rule(self, rule: SyncRule) -> bool:
        """Add a synchronization rule"""
        try:
            self.sync_rules[rule.id] = rule
            logger.info(f"Added sync rule: {rule.name} ({rule.id})")
            return True
        except Exception as e:
            logger.error(f"Failed to add sync rule {rule.id}: {e}")
            return False
            
    async def remove_sync_rule(self, rule_id: str) -> bool:
        """Remove a synchronization rule"""
        if rule_id in self.sync_rules:
            del self.sync_rules[rule_id]
            logger.info(f"Removed sync rule: {rule_id}")
            return True
        return False
        
    async def sync_data(self, rule_id: str) -> SyncResult:
        """Execute data synchronization based on rule"""
        
        if rule_id not in self.sync_rules:
            raise ValueError(f"Sync rule {rule_id} not found")
            
        if rule_id in self.running_syncs:
            raise ValueError(f"Sync rule {rule_id} is already running")
            
        rule = self.sync_rules[rule_id]
        
        if not rule.enabled:
            raise ValueError(f"Sync rule {rule_id} is disabled")
            
        result = SyncResult(
            rule_id=rule_id,
            status=SyncStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        self.running_syncs.add(rule_id)
        
        try:
            if rule.mode == SyncMode.FULL:
                result = await self._sync_full(rule, result)
            elif rule.mode == SyncMode.INCREMENTAL:
                result = await self._sync_incremental(rule, result)
            elif rule.mode == SyncMode.CDC:
                result = await self._sync_cdc(rule, result)
            elif rule.mode == SyncMode.SNAPSHOT:
                result = await self._sync_snapshot(rule, result)
                
            result.status = SyncStatus.COMPLETED
            rule.last_sync = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Sync failed for rule {rule_id}: {e}")
            result.status = SyncStatus.FAILED
            result.errors.append(str(e))
            
        finally:
            result.end_time = datetime.utcnow()
            self.running_syncs.discard(rule_id)
            self.sync_history.append(result)
            
        return result
        
    async def _sync_full(self, rule: SyncRule, result: SyncResult) -> SyncResult:
        """Perform full data synchronization"""
        
        # Read all data from source
        source_data = await self.data_connector.read_data(
            rule.source,
            filters=rule.filters
        )
        
        result.records_processed = len(source_data)
        
        # Apply field mapping
        if rule.field_mapping:
            source_data = source_data.rename(columns=rule.field_mapping)
            
        # Write to target (replace mode)
        success = await self.data_connector.write_data(
            rule.target,
            source_data,
            mode='replace'
        )
        
        if success:
            result.records_created = len(source_data)
        else:
            raise Exception("Failed to write data to target")
            
        return result
        
    async def _sync_incremental(self, rule: SyncRule, result: SyncResult) -> SyncResult:
        """Perform incremental data synchronization"""
        
        # Determine incremental filter
        incremental_filter = rule.filters.copy()
        
        if rule.last_sync:
            # Add filter for records modified since last sync
            timestamp_column = rule.filters.get('timestamp_column', 'modified_date')
            incremental_filter['modified_since'] = rule.last_sync.isoformat()
            
        # Read incremental data from source
        source_data = await self.data_connector.read_data(
            rule.source,
            filters=incremental_filter
        )
        
        result.records_processed = len(source_data)
        
        if source_data.empty:
            logger.info(f"No new data to sync for rule {rule.id}")
            return result
            
        # Apply field mapping
        if rule.field_mapping:
            source_data = source_data.rename(columns=rule.field_mapping)
            
        # Determine upsert vs append based on whether we have a primary key
        primary_key = rule.filters.get('primary_key')
        mode = 'upsert' if primary_key else 'append'
        
        # Write to target
        success = await self.data_connector.write_data(
            rule.target,
            source_data,
            mode=mode
        )
        
        if success:
            if mode == 'upsert':
                result.records_updated = len(source_data)
            else:
                result.records_created = len(source_data)
        else:
            raise Exception("Failed to write incremental data to target")
            
        return result
        
    async def _sync_cdc(self, rule: SyncRule, result: SyncResult) -> SyncResult:
        """Perform Change Data Capture synchronization"""
        
        # Read CDC log/changes from source
        cdc_filter = rule.filters.copy()
        
        if rule.last_sync:
            cdc_filter['change_timestamp'] = rule.last_sync.isoformat()
            
        # This would typically connect to a CDC stream or change log
        # For now, we'll simulate by reading changed records
        source_data = await self.data_connector.read_data(
            rule.source,
            filters=cdc_filter
        )
        
        result.records_processed = len(source_data)
        
        if source_data.empty:
            return result
            
        # Process changes based on operation type
        creates = source_data[source_data.get('_operation', 'INSERT') == 'INSERT']
        updates = source_data[source_data.get('_operation', 'INSERT') == 'UPDATE']
        deletes = source_data[source_data.get('_operation', 'INSERT') == 'DELETE']
        
        # Apply field mapping
        if rule.field_mapping:
            creates = creates.rename(columns=rule.field_mapping)
            updates = updates.rename(columns=rule.field_mapping)
            deletes = deletes.rename(columns=rule.field_mapping)
            
        # Process creates
        if not creates.empty:
            success = await self.data_connector.write_data(
                rule.target,
                creates.drop(['_operation'], axis=1, errors='ignore'),
                mode='append'
            )
            if success:
                result.records_created = len(creates)
                
        # Process updates
        if not updates.empty:
            success = await self.data_connector.write_data(
                rule.target,
                updates.drop(['_operation'], axis=1, errors='ignore'),
                mode='upsert'
            )
            if success:
                result.records_updated = len(updates)
                
        # Process deletes (would need special handling per target type)
        if not deletes.empty:
            result.records_deleted = len(deletes)
            # Delete logic would be implemented per target type
            
        return result
        
    async def _sync_snapshot(self, rule: SyncRule, result: SyncResult) -> SyncResult:
        """Perform snapshot synchronization with change detection"""
        
        # Read current data from source
        source_data = await self.data_connector.read_data(
            rule.source,
            filters=rule.filters
        )
        
        result.records_processed = len(source_data)
        
        # Read current data from target for comparison
        try:
            target_data = await self.data_connector.read_data(rule.target)
        except Exception:
            # Target might be empty or not exist
            target_data = pd.DataFrame()
            
        # Apply field mapping to source data
        if rule.field_mapping:
            source_data = source_data.rename(columns=rule.field_mapping)
            
        # Detect changes
        primary_key = rule.filters.get('primary_key', [])
        if not isinstance(primary_key, list):
            primary_key = [primary_key]
            
        if not target_data.empty and primary_key:
            # Merge to identify changes
            merged = source_data.merge(
                target_data,
                on=primary_key,
                how='outer',
                suffixes=('_source', '_target'),
                indicator=True
            )
            
            # New records (only in source)
            new_records = merged[merged['_merge'] == 'left_only']
            new_records = new_records[[col for col in new_records.columns if not col.endswith('_target') and col != '_merge']]
            new_records.columns = [col.replace('_source', '') for col in new_records.columns]
            
            # Deleted records (only in target)
            deleted_records = merged[merged['_merge'] == 'right_only']
            
            # Changed records (in both but different)
            both_records = merged[merged['_merge'] == 'both']
            changed_records = []
            
            for _, row in both_records.iterrows():
                source_cols = [col for col in row.index if col.endswith('_source')]
                target_cols = [col for col in row.index if col.endswith('_target')]
                
                has_changes = False
                for s_col, t_col in zip(source_cols, target_cols):
                    if pd.isna(row[s_col]) != pd.isna(row[t_col]) or (not pd.isna(row[s_col]) and row[s_col] != row[t_col]):
                        has_changes = True
                        break
                        
                if has_changes:
                    changed_record = {}
                    for col in source_cols:
                        base_col = col.replace('_source', '')
                        changed_record[base_col] = row[col]
                    changed_records.append(changed_record)
                    
            changed_records = pd.DataFrame(changed_records)
            
        else:
            # No primary key or target empty - treat all as new
            new_records = source_data
            changed_records = pd.DataFrame()
            deleted_records = pd.DataFrame()
            
        # Apply changes
        if not new_records.empty:
            success = await self.data_connector.write_data(
                rule.target,
                new_records,
                mode='append'
            )
            if success:
                result.records_created = len(new_records)
                
        if not changed_records.empty:
            success = await self.data_connector.write_data(
                rule.target,
                changed_records,
                mode='upsert'
            )
            if success:
                result.records_updated = len(changed_records)
                
        result.records_deleted = len(deleted_records)
        
        return result
        
    async def sync_all_rules(self) -> List[SyncResult]:
        """Execute all enabled synchronization rules"""
        
        results = []
        
        for rule_id, rule in self.sync_rules.items():
            if rule.enabled and rule_id not in self.running_syncs:
                try:
                    result = await self.sync_data(rule_id)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to sync rule {rule_id}: {e}")
                    
        return results
        
    async def schedule_sync(self, rule_id: str, schedule_expression: str) -> bool:
        """Schedule automatic synchronization"""
        
        if rule_id not in self.sync_rules:
            return False
            
        self.sync_rules[rule_id].schedule = schedule_expression
        logger.info(f"Scheduled sync rule {rule_id} with expression: {schedule_expression}")
        
        # In production, this would integrate with a scheduler like APScheduler
        return True
        
    def get_sync_status(self, rule_id: Optional[str] = None) -> Dict[str, Any]:
        """Get synchronization status"""
        
        if rule_id:
            if rule_id not in self.sync_rules:
                return {}
                
            rule = self.sync_rules[rule_id]
            recent_results = [r for r in self.sync_history if r.rule_id == rule_id][-5:]  # Last 5 runs
            
            return {
                'rule': {
                    'id': rule.id,
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'last_sync': rule.last_sync.isoformat() if rule.last_sync else None,
                    'running': rule_id in self.running_syncs
                },
                'recent_results': [
                    {
                        'status': r.status.value,
                        'start_time': r.start_time.isoformat(),
                        'end_time': r.end_time.isoformat() if r.end_time else None,
                        'records_processed': r.records_processed,
                        'records_created': r.records_created,
                        'records_updated': r.records_updated,
                        'records_deleted': r.records_deleted,
                        'errors': r.errors
                    }
                    for r in recent_results
                ]
            }
        else:
            # Return overall status
            total_rules = len(self.sync_rules)
            enabled_rules = sum(1 for r in self.sync_rules.values() if r.enabled)
            running_rules = len(self.running_syncs)
            
            # Recent statistics
            recent_results = [r for r in self.sync_history if r.start_time > datetime.utcnow() - timedelta(hours=24)]
            successful_syncs = sum(1 for r in recent_results if r.status == SyncStatus.COMPLETED)
            failed_syncs = sum(1 for r in recent_results if r.status == SyncStatus.FAILED)
            
            return {
                'total_rules': total_rules,
                'enabled_rules': enabled_rules,
                'running_rules': running_rules,
                'last_24h': {
                    'total_syncs': len(recent_results),
                    'successful_syncs': successful_syncs,
                    'failed_syncs': failed_syncs,
                    'success_rate': successful_syncs / len(recent_results) if recent_results else 0
                }
            }
            
    def get_sync_history(self, 
                        rule_id: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Get synchronization history"""
        
        history = self.sync_history
        
        if rule_id:
            history = [r for r in history if r.rule_id == rule_id]
            
        # Sort by start time descending
        history = sorted(history, key=lambda x: x.start_time, reverse=True)
        
        # Limit results
        history = history[:limit]
        
        return [
            {
                'rule_id': r.rule_id,
                'status': r.status.value,
                'start_time': r.start_time.isoformat(),
                'end_time': r.end_time.isoformat() if r.end_time else None,
                'duration_seconds': (r.end_time - r.start_time).total_seconds() if r.end_time else None,
                'records_processed': r.records_processed,
                'records_created': r.records_created,
                'records_updated': r.records_updated,
                'records_deleted': r.records_deleted,
                'errors': r.errors,
                'metadata': r.metadata
            }
            for r in history
        ]
        
    async def cancel_sync(self, rule_id: str) -> bool:
        """Cancel a running synchronization"""
        
        if rule_id in self.running_syncs:
            # In production, this would signal the sync task to stop
            self.running_syncs.discard(rule_id)
            
            # Create a cancelled result
            result = SyncResult(
                rule_id=rule_id,
                status=SyncStatus.CANCELLED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )
            self.sync_history.append(result)
            
            logger.info(f"Cancelled sync for rule {rule_id}")
            return True
            
        return False
        
    def create_sync_rule(self,
                        rule_id: str,
                        name: str,
                        source: str,
                        target: str,
                        mode: SyncMode = SyncMode.INCREMENTAL,
                        direction: SyncDirection = SyncDirection.SOURCE_TO_TARGET,
                        **kwargs) -> SyncRule:
        """Create a new synchronization rule"""
        
        return SyncRule(
            id=rule_id,
            name=name,
            source=source,
            target=target,
            mode=mode,
            direction=direction,
            **kwargs
        )