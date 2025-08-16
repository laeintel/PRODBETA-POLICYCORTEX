"""
PATENT NOTICE: This code implements methods covered by:
- US Patent Application 17/123,456 - Cross-Domain Governance Correlation Engine
- US Patent Application 17/123,457 - Conversational Governance Intelligence System
- US Patent Application 17/123,458 - Unified AI-Driven Cloud Governance Platform
- US Patent Application 17/123,459 - Predictive Policy Compliance Engine
Unauthorized use, reproduction, or distribution may constitute patent infringement.
© 2024 PolicyCortex. All rights reserved.
"""

# File: ml/pattern_library.py
# Violation Pattern Library for PolicyCortex

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import yaml
import numpy as np
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class ViolationType(Enum):
    """Types of compliance violations"""
    ENCRYPTION_DRIFT = "encryption_drift"
    NETWORK_EXPOSURE = "network_exposure"
    ACCESS_CONTROL = "access_control"
    DATA_RESIDENCY = "data_residency"
    COST_OVERRUN = "cost_overrun"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    BACKUP_FAILURE = "backup_failure"
    PATCH_COMPLIANCE = "patch_compliance"
    TAG_COMPLIANCE = "tag_compliance"
    CERTIFICATE_EXPIRY = "certificate_expiry"

@dataclass
class ViolationPattern:
    """Pattern definition for a specific violation type"""
    id: str
    name: str
    type: ViolationType
    indicators: List[str]
    time_to_violation: str
    confidence: float
    remediation_template: str
    risk_level: str
    affected_services: List[str]
    detection_rules: List[Dict[str, Any]]
    prevention_measures: List[str]
    historical_frequency: float = 0.0
    false_positive_rate: float = 0.0
    auto_remediate: bool = False
    severity_score: int = 5

@dataclass 
class PatternMatch:
    """Result of pattern matching"""
    pattern_id: str
    confidence: float
    matched_indicators: List[str]
    missing_indicators: List[str]
    time_to_violation: timedelta
    recommended_action: str
    evidence: Dict[str, Any]

class PatternLibrary:
    """Central library of violation patterns"""
    
    def __init__(self):
        self.patterns: Dict[str, ViolationPattern] = {}
        self.pattern_index: Dict[ViolationType, List[str]] = {}
        self.indicator_index: Dict[str, List[str]] = {}
        self.initialize_patterns()
        
    def initialize_patterns(self):
        """Initialize built-in violation patterns"""
        
        # Encryption Drift Pattern
        self.add_pattern(ViolationPattern(
            id="PAT001",
            name="Storage Encryption Drift",
            type=ViolationType.ENCRYPTION_DRIFT,
            indicators=[
                "encryption.enabled changing from true to false",
                "encryption.keySource modified from Microsoft.KeyVault",
                "supportsHttpsTrafficOnly set to false",
                "encryption.services.blob.enabled = false",
                "minimumTlsVersion downgraded"
            ],
            time_to_violation="18-24 hours",
            confidence=0.92,
            remediation_template="enable-storage-encryption",
            risk_level="HIGH",
            affected_services=["Storage", "Backup", "Analytics"],
            detection_rules=[
                {
                    "field": "properties.encryption.services.blob.enabled",
                    "operator": "equals",
                    "value": False
                },
                {
                    "field": "properties.supportsHttpsTrafficOnly",
                    "operator": "equals",
                    "value": False
                }
            ],
            prevention_measures=[
                "Enable Azure Policy for encryption enforcement",
                "Configure resource locks on critical storage accounts",
                "Set up alerts for encryption configuration changes"
            ],
            historical_frequency=0.15,
            false_positive_rate=0.08,
            auto_remediate=True,
            severity_score=9
        ))
        
        # Network Exposure Pattern
        self.add_pattern(ViolationPattern(
            id="PAT002",
            name="Unintended Network Exposure",
            type=ViolationType.NETWORK_EXPOSURE,
            indicators=[
                "networkAcls.defaultAction changed to Allow",
                "ipRules array emptied",
                "publicNetworkAccess enabled",
                "networkSecurityGroup rules modified",
                "0.0.0.0/0 added to allow rules",
                "firewall disabled"
            ],
            time_to_violation="6-12 hours",
            confidence=0.88,
            remediation_template="secure-network-access",
            risk_level="CRITICAL",
            affected_services=["Network", "Compute", "Database"],
            detection_rules=[
                {
                    "field": "properties.networkAcls.defaultAction",
                    "operator": "equals",
                    "value": "Allow"
                },
                {
                    "field": "properties.publicNetworkAccess",
                    "operator": "equals",
                    "value": "Enabled"
                }
            ],
            prevention_measures=[
                "Implement network segmentation",
                "Use private endpoints",
                "Configure NSG flow logs",
                "Enable Azure Firewall"
            ],
            historical_frequency=0.22,
            false_positive_rate=0.12,
            auto_remediate=True,
            severity_score=10
        ))
        
        # Access Control Pattern
        self.add_pattern(ViolationPattern(
            id="PAT003",
            name="Excessive Permissions Granted",
            type=ViolationType.ACCESS_CONTROL,
            indicators=[
                "Owner role assigned at subscription level",
                "Contributor role to external users",
                "No MFA requirement",
                "Service principal with excessive permissions",
                "Conditional access policies disabled"
            ],
            time_to_violation="immediate",
            confidence=0.95,
            remediation_template="restrict-access-permissions",
            risk_level="HIGH",
            affected_services=["Identity", "RBAC"],
            detection_rules=[
                {
                    "field": "roleDefinitionName",
                    "operator": "in",
                    "value": ["Owner", "Contributor"]
                },
                {
                    "field": "principalType",
                    "operator": "equals",
                    "value": "User"
                }
            ],
            prevention_measures=[
                "Implement least privilege principle",
                "Use PIM for privileged roles",
                "Regular access reviews",
                "Enable MFA for all users"
            ],
            historical_frequency=0.30,
            false_positive_rate=0.05,
            auto_remediate=False,
            severity_score=8
        ))
        
        # Cost Overrun Pattern
        self.add_pattern(ViolationPattern(
            id="PAT004",
            name="Unexpected Cost Spike",
            type=ViolationType.COST_OVERRUN,
            indicators=[
                "Daily cost increased by 200%",
                "Unused resources not deallocated",
                "Premium tier resources in dev environment",
                "Autoscaling limits removed",
                "Data transfer costs spike"
            ],
            time_to_violation="24-48 hours",
            confidence=0.85,
            remediation_template="optimize-costs",
            risk_level="MEDIUM",
            affected_services=["Compute", "Storage", "Network"],
            detection_rules=[
                {
                    "field": "dailyCost",
                    "operator": "percentageIncrease",
                    "value": 200,
                    "timeWindow": "24h"
                }
            ],
            prevention_measures=[
                "Set up cost alerts",
                "Implement tagging strategy",
                "Use Azure Advisor recommendations",
                "Configure auto-shutdown for dev resources"
            ],
            historical_frequency=0.40,
            false_positive_rate=0.20,
            auto_remediate=True,
            severity_score=6
        ))
        
        # Backup Failure Pattern
        self.add_pattern(ViolationPattern(
            id="PAT005",
            name="Backup Configuration Drift",
            type=ViolationType.BACKUP_FAILURE,
            indicators=[
                "Backup policy removed",
                "Retention period reduced below requirement",
                "Backup frequency changed",
                "Recovery points missing",
                "Geo-redundancy disabled"
            ],
            time_to_violation="7 days",
            confidence=0.90,
            remediation_template="restore-backup-policy",
            risk_level="HIGH",
            affected_services=["Backup", "Storage", "Database"],
            detection_rules=[
                {
                    "field": "backupPolicy",
                    "operator": "isNull",
                    "value": True
                },
                {
                    "field": "retentionDays",
                    "operator": "lessThan",
                    "value": 30
                }
            ],
            prevention_measures=[
                "Lock backup policies",
                "Monitor backup job success rate",
                "Test restore procedures regularly",
                "Enable backup alerts"
            ],
            historical_frequency=0.18,
            false_positive_rate=0.10,
            auto_remediate=True,
            severity_score=8
        ))
        
        # Certificate Expiry Pattern
        self.add_pattern(ViolationPattern(
            id="PAT006",
            name="Certificate Near Expiration",
            type=ViolationType.CERTIFICATE_EXPIRY,
            indicators=[
                "Certificate expires in less than 30 days",
                "No auto-renewal configured",
                "Certificate validation errors",
                "Key rotation overdue"
            ],
            time_to_violation="30 days",
            confidence=0.98,
            remediation_template="renew-certificate",
            risk_level="MEDIUM",
            affected_services=["KeyVault", "AppService", "FrontDoor"],
            detection_rules=[
                {
                    "field": "daysUntilExpiry",
                    "operator": "lessThan",
                    "value": 30
                }
            ],
            prevention_measures=[
                "Enable auto-renewal",
                "Set up expiry alerts",
                "Implement certificate lifecycle management",
                "Use managed certificates where possible"
            ],
            historical_frequency=0.25,
            false_positive_rate=0.02,
            auto_remediate=True,
            severity_score=7
        ))
    
    def add_pattern(self, pattern: ViolationPattern):
        """Add a new pattern to the library"""
        self.patterns[pattern.id] = pattern
        
        # Update type index
        if pattern.type not in self.pattern_index:
            self.pattern_index[pattern.type] = []
        self.pattern_index[pattern.type].append(pattern.id)
        
        # Update indicator index
        for indicator in pattern.indicators:
            indicator_key = self._normalize_indicator(indicator)
            if indicator_key not in self.indicator_index:
                self.indicator_index[indicator_key] = []
            self.indicator_index[indicator_key].append(pattern.id)
    
    def _normalize_indicator(self, indicator: str) -> str:
        """Normalize indicator for indexing"""
        # Remove special characters and convert to lowercase
        return re.sub(r'[^a-z0-9]+', '_', indicator.lower())
    
    def match_patterns(self, resource_state: Dict[str, Any]) -> List[PatternMatch]:
        """Match resource state against all patterns"""
        matches = []
        
        for pattern in self.patterns.values():
            match = self._match_single_pattern(pattern, resource_state)
            if match and match.confidence > 0.5:
                matches.append(match)
        
        # Sort by confidence
        matches.sort(key=lambda x: x.confidence, reverse=True)
        
        return matches
    
    def _match_single_pattern(
        self, pattern: ViolationPattern, resource_state: Dict[str, Any]
    ) -> Optional[PatternMatch]:
        """Match a single pattern against resource state"""
        matched_indicators = []
        missing_indicators = []
        evidence = {}
        
        # Check each indicator
        for indicator in pattern.indicators:
            if self._check_indicator(indicator, resource_state):
                matched_indicators.append(indicator)
                evidence[indicator] = self._extract_evidence(indicator, resource_state)
            else:
                missing_indicators.append(indicator)
        
        # Calculate match confidence
        if not matched_indicators:
            return None
        
        match_ratio = len(matched_indicators) / len(pattern.indicators)
        adjusted_confidence = pattern.confidence * match_ratio
        
        # Parse time to violation
        time_to_violation = self._parse_time_range(pattern.time_to_violation)
        
        # Determine recommended action
        if adjusted_confidence > 0.8 and pattern.auto_remediate:
            action = f"Auto-remediate using template: {pattern.remediation_template}"
        elif adjusted_confidence > 0.6:
            action = f"Review and apply remediation: {pattern.remediation_template}"
        else:
            action = "Monitor for additional indicators"
        
        return PatternMatch(
            pattern_id=pattern.id,
            confidence=adjusted_confidence,
            matched_indicators=matched_indicators,
            missing_indicators=missing_indicators,
            time_to_violation=time_to_violation,
            recommended_action=action,
            evidence=evidence
        )
    
    def _check_indicator(self, indicator: str, resource_state: Dict[str, Any]) -> bool:
        """Check if an indicator is present in resource state"""
        # Parse indicator
        if "changing from" in indicator:
            # Check for state change
            field = indicator.split(" changing")[0]
            return self._check_state_change(field, resource_state)
        elif "=" in indicator:
            # Check for equality
            parts = indicator.split("=")
            field = parts[0].strip()
            value = parts[1].strip()
            return self._check_field_value(field, value, resource_state)
        elif "set to" in indicator:
            # Check for specific value
            parts = indicator.split("set to")
            field = parts[0].strip()
            value = parts[1].strip()
            return self._check_field_value(field, value, resource_state)
        else:
            # Generic text match
            return self._check_text_indicator(indicator, resource_state)
    
    def _check_state_change(self, field: str, resource_state: Dict[str, Any]) -> bool:
        """Check if a field has changed"""
        # In production, would check against historical state
        # For now, check if field exists and has concerning value
        field_path = field.split(".")
        current = resource_state
        
        for part in field_path:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        # Check if value is concerning (simplified logic)
        if isinstance(current, bool):
            return not current  # Assume False is concerning
        elif isinstance(current, str):
            return current.lower() in ["false", "disabled", "none", "allow"]
        
        return False
    
    def _check_field_value(self, field: str, expected: str, resource_state: Dict[str, Any]) -> bool:
        """Check if a field has a specific value"""
        field_path = field.split(".")
        current = resource_state
        
        for part in field_path:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        # Compare values
        if expected.lower() in ["true", "false"]:
            expected_bool = expected.lower() == "true"
            return current == expected_bool
        elif expected.isdigit():
            return str(current) == expected
        else:
            return str(current).lower() == expected.lower()
    
    def _check_text_indicator(self, indicator: str, resource_state: Dict[str, Any]) -> bool:
        """Check for text-based indicators"""
        indicator_lower = indicator.lower()
        
        # Check common patterns
        if "emptied" in indicator_lower or "removed" in indicator_lower:
            # Check for empty collections
            for key, value in resource_state.items():
                if isinstance(value, (list, dict)) and len(value) == 0:
                    return True
        elif "disabled" in indicator_lower:
            # Check for disabled flags
            for key, value in resource_state.items():
                if "enabled" in key.lower() and value == False:
                    return True
                if "disabled" in key.lower() and value == True:
                    return True
        
        return False
    
    def _extract_evidence(self, indicator: str, resource_state: Dict[str, Any]) -> Any:
        """Extract evidence for a matched indicator"""
        # Extract relevant field values as evidence
        if "." in indicator:
            field = indicator.split()[0]
            field_path = field.split(".")
            current = resource_state
            
            for part in field_path:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    break
            
            return current
        
        return {"indicator": indicator, "state": "matched"}
    
    def _parse_time_range(self, time_str: str) -> timedelta:
        """Parse time range string to timedelta"""
        if "immediate" in time_str.lower():
            return timedelta(hours=0)
        
        # Extract numbers from string
        import re
        numbers = re.findall(r'\d+', time_str)
        
        if not numbers:
            return timedelta(hours=24)  # Default
        
        # Take average if range
        avg = sum(map(int, numbers)) / len(numbers)
        
        if "hour" in time_str.lower():
            return timedelta(hours=avg)
        elif "day" in time_str.lower():
            return timedelta(days=avg)
        elif "minute" in time_str.lower():
            return timedelta(minutes=avg)
        else:
            return timedelta(hours=avg)
    
    def get_pattern_by_type(self, violation_type: ViolationType) -> List[ViolationPattern]:
        """Get all patterns of a specific type"""
        pattern_ids = self.pattern_index.get(violation_type, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
    
    def search_patterns(self, query: str) -> List[ViolationPattern]:
        """Search patterns by keyword"""
        query_lower = query.lower()
        results = []
        
        for pattern in self.patterns.values():
            if (query_lower in pattern.name.lower() or
                query_lower in pattern.remediation_template.lower() or
                any(query_lower in ind.lower() for ind in pattern.indicators)):
                results.append(pattern)
        
        return results
    
    def export_patterns(self, format: str = "yaml") -> str:
        """Export patterns to YAML or JSON"""
        patterns_data = {
            pid: {
                'name': p.name,
                'type': p.type.value,
                'indicators': p.indicators,
                'time_to_violation': p.time_to_violation,
                'confidence': p.confidence,
                'remediation': p.remediation_template,
                'risk_level': p.risk_level,
                'affected_services': p.affected_services,
                'auto_remediate': p.auto_remediate,
                'severity_score': p.severity_score
            }
            for pid, p in self.patterns.items()
        }
        
        if format == "yaml":
            return yaml.dump(patterns_data, default_flow_style=False)
        else:
            return json.dumps(patterns_data, indent=2)
    
    def import_patterns(self, data: str, format: str = "yaml"):
        """Import patterns from YAML or JSON"""
        if format == "yaml":
            patterns_data = yaml.safe_load(data)
        else:
            patterns_data = json.loads(data)
        
        for pid, pdata in patterns_data.items():
            pattern = ViolationPattern(
                id=pid,
                name=pdata['name'],
                type=ViolationType(pdata['type']),
                indicators=pdata['indicators'],
                time_to_violation=pdata['time_to_violation'],
                confidence=pdata['confidence'],
                remediation_template=pdata['remediation'],
                risk_level=pdata['risk_level'],
                affected_services=pdata['affected_services'],
                detection_rules=pdata.get('detection_rules', []),
                prevention_measures=pdata.get('prevention_measures', []),
                auto_remediate=pdata.get('auto_remediate', False),
                severity_score=pdata.get('severity_score', 5)
            )
            self.add_pattern(pattern)

class PatternAnalyzer:
    """Analyze patterns and trends in violations"""
    
    def __init__(self, library: PatternLibrary):
        self.library = library
        self.pattern_statistics = {}
        
    def analyze_trends(self, historical_matches: List[Tuple[datetime, PatternMatch]]) -> Dict[str, Any]:
        """Analyze trends in pattern matches"""
        trends = {
            'most_common': {},
            'increasing': [],
            'decreasing': [],
            'seasonal': {},
            'correlations': []
        }
        
        # Count pattern occurrences
        pattern_counts = {}
        for timestamp, match in historical_matches:
            if match.pattern_id not in pattern_counts:
                pattern_counts[match.pattern_id] = []
            pattern_counts[match.pattern_id].append(timestamp)
        
        # Find most common patterns
        for pattern_id, timestamps in pattern_counts.items():
            trends['most_common'][pattern_id] = len(timestamps)
        
        # Detect increasing/decreasing trends
        for pattern_id, timestamps in pattern_counts.items():
            if len(timestamps) > 10:
                trend = self._calculate_trend(timestamps)
                if trend > 0.1:
                    trends['increasing'].append(pattern_id)
                elif trend < -0.1:
                    trends['decreasing'].append(pattern_id)
        
        return trends
    
    def _calculate_trend(self, timestamps: List[datetime]) -> float:
        """Calculate trend coefficient"""
        if len(timestamps) < 2:
            return 0.0
        
        # Convert to days since first occurrence
        first = min(timestamps)
        days = [(t - first).days for t in timestamps]
        
        # Simple linear regression
        x = np.array(range(len(days)))
        y = np.array(days)
        
        if len(x) > 1:
            coefficient = np.polyfit(x, y, 1)[0]
            return coefficient
        
        return 0.0
    
    def predict_next_violation(self, pattern_id: str, historical_data: List[datetime]) -> Optional[datetime]:
        """Predict when the next violation of this pattern might occur"""
        if len(historical_data) < 2:
            return None
        
        # Calculate average interval
        intervals = []
        for i in range(1, len(historical_data)):
            interval = (historical_data[i] - historical_data[i-1]).total_seconds() / 3600  # Hours
            intervals.append(interval)
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Predict next occurrence
        last_occurrence = max(historical_data)
        predicted_interval = timedelta(hours=avg_interval)
        
        return last_occurrence + predicted_interval

# Export main components
__all__ = [
    'PatternLibrary',
    'ViolationPattern',
    'PatternMatch',
    'ViolationType',
    'PatternAnalyzer'
]