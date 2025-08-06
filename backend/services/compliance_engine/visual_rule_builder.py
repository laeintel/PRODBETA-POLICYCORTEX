"""
Visual Rule Builder for Compliance Engine
Provides APIs for frontend visual rule creation interface
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field, validator
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from .rule_engine import ComplianceRule, RuleType, RuleOperator, RuleAction

logger = structlog.get_logger(__name__)

class RuleTemplate(BaseModel):
    """Pre-defined rule template"""
    template_id: str
    name: str
    description: str
    category: str
    icon: str
    base_rule: Dict[str, Any]
    configurable_fields: List[str]
    tags: List[str] = Field(default_factory=list)

class VisualRuleComponent(BaseModel):
    """Visual component for rule builder"""
    component_id: str
    type: str  # condition, action, logical_operator
    display_name: str
    icon: str
    properties: Dict[str, Any]
    children: List['VisualRuleComponent'] = Field(default_factory=list)
    position: Dict[str, float] = Field(default_factory=dict)  # x, y coordinates

class RuleBuilderState(BaseModel):
    """State of the visual rule builder"""
    session_id: str
    rule_id: Optional[str] = None
    components: List[VisualRuleComponent] = Field(default_factory=list)
    connections: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_modified: datetime = Field(default_factory=datetime.utcnow)

class VisualRuleBuilder:
    """
    Visual Rule Builder API for creating compliance rules through UI
    """
    
    def __init__(self):
        self.templates = self._load_default_templates()
        self.sessions: Dict[str, RuleBuilderState] = {}
        self.saved_rules: Dict[str, ComplianceRule] = {}
        
    def _load_default_templates(self) -> Dict[str, RuleTemplate]:
        """Load default rule templates"""
        templates = {}
        
        # Security Templates
        templates['encryption_required'] = RuleTemplate(
            template_id='encryption_required',
            name='Encryption Required',
            description='Ensure resources are encrypted',
            category='Security',
            icon='🔒',
            base_rule={
                'rule_type': RuleType.SECURITY,
                'conditions': [{
                    'type': 'custom',
                    'function': 'check_encryption'
                }],
                'actions': [{
                    'type': RuleAction.ALERT,
                    'level': 'high'
                }],
                'severity': 'high'
            },
            configurable_fields=['severity', 'actions']
        )
        
        templates['public_access_blocked'] = RuleTemplate(
            template_id='public_access_blocked',
            name='Block Public Access',
            description='Prevent public access to resources',
            category='Security',
            icon='🚫',
            base_rule={
                'rule_type': RuleType.SECURITY,
                'conditions': [{
                    'type': 'custom',
                    'function': 'check_public_access'
                }],
                'actions': [{
                    'type': RuleAction.BLOCK,
                    'block_type': 'access'
                }],
                'severity': 'critical'
            },
            configurable_fields=['actions']
        )
        
        # Compliance Templates
        templates['required_tags'] = RuleTemplate(
            template_id='required_tags',
            name='Required Tags',
            description='Ensure resources have required tags',
            category='Compliance',
            icon='🏷️',
            base_rule={
                'rule_type': RuleType.TAGGING,
                'conditions': [{
                    'type': 'custom',
                    'function': 'check_tags',
                    'parameters': {
                        'required_tags': ['Environment', 'Owner', 'CostCenter']
                    }
                }],
                'actions': [{
                    'type': RuleAction.TAG,
                    'tags': {
                        'ComplianceStatus': 'NonCompliant'
                    }
                }],
                'severity': 'medium'
            },
            configurable_fields=['conditions.parameters.required_tags', 'actions.tags']
        )
        
        # Cost Management Templates
        templates['cost_threshold'] = RuleTemplate(
            template_id='cost_threshold',
            name='Cost Threshold Alert',
            description='Alert when resource costs exceed threshold',
            category='Cost Management',
            icon='💰',
            base_rule={
                'rule_type': RuleType.COST,
                'conditions': [{
                    'type': 'custom',
                    'function': 'check_cost_threshold',
                    'parameters': {
                        'threshold': 1000
                    }
                }],
                'actions': [{
                    'type': RuleAction.NOTIFY,
                    'channels': ['email'],
                    'recipients': []
                }],
                'severity': 'medium'
            },
            configurable_fields=['conditions.parameters.threshold', 'actions.recipients']
        )
        
        # Configuration Templates
        templates['backup_required'] = RuleTemplate(
            template_id='backup_required',
            name='Backup Configuration',
            description='Ensure backup is configured',
            category='Configuration',
            icon='💾',
            base_rule={
                'rule_type': RuleType.CONFIGURATION,
                'conditions': [{
                    'type': 'custom',
                    'function': 'check_backup'
                }],
                'actions': [{
                    'type': RuleAction.REMEDIATE,
                    'steps': ['Enable backup configuration']
                }],
                'severity': 'high'
            },
            configurable_fields=['actions.steps']
        )
        
        return templates
        
    def create_session(self) -> str:
        """Create a new rule builder session"""
        session_id = str(uuid4())
        self.sessions[session_id] = RuleBuilderState(session_id=session_id)
        logger.info(f"Created rule builder session: {session_id}")
        return session_id
        
    def get_session(self, session_id: str) -> RuleBuilderState:
        """Get session state"""
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        return self.sessions[session_id]
        
    def update_session(self, session_id: str, state: RuleBuilderState) -> None:
        """Update session state"""
        state.last_modified = datetime.utcnow()
        self.sessions[session_id] = state
        
    def get_templates(self, category: Optional[str] = None) -> List[RuleTemplate]:
        """Get available rule templates"""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
            
        return templates
        
    def get_component_library(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get available components for rule building"""
        return {
            'conditions': [
                {
                    'type': 'field_comparison',
                    'display_name': 'Field Comparison',
                    'icon': '⚖️',
                    'operators': [op.value for op in RuleOperator],
                    'description': 'Compare field values'
                },
                {
                    'type': 'custom_function',
                    'display_name': 'Custom Function',
                    'icon': '🔧',
                    'functions': [
                        'check_encryption',
                        'check_public_access',
                        'check_backup',
                        'check_tags',
                        'check_cost_threshold'
                    ],
                    'description': 'Use pre-defined functions'
                },
                {
                    'type': 'regex_match',
                    'display_name': 'Pattern Match',
                    'icon': '🔍',
                    'description': 'Match text patterns'
                },
                {
                    'type': 'date_range',
                    'display_name': 'Date Range',
                    'icon': '📅',
                    'description': 'Check date ranges'
                }
            ],
            'actions': [
                {
                    'type': RuleAction.ALERT,
                    'display_name': 'Alert',
                    'icon': '🚨',
                    'levels': ['low', 'medium', 'high', 'critical'],
                    'description': 'Create an alert'
                },
                {
                    'type': RuleAction.REMEDIATE,
                    'display_name': 'Auto-Remediate',
                    'icon': '🔧',
                    'description': 'Automatically fix issues'
                },
                {
                    'type': RuleAction.TAG,
                    'display_name': 'Tag Resource',
                    'icon': '🏷️',
                    'description': 'Add tags to resources'
                },
                {
                    'type': RuleAction.NOTIFY,
                    'display_name': 'Send Notification',
                    'icon': '📧',
                    'channels': ['email', 'slack', 'teams', 'webhook'],
                    'description': 'Send notifications'
                },
                {
                    'type': RuleAction.BLOCK,
                    'display_name': 'Block Access',
                    'icon': '🚫',
                    'description': 'Block resource access'
                }
            ],
            'logical_operators': [
                {
                    'type': 'AND',
                    'display_name': 'AND',
                    'icon': '∧',
                    'description': 'All conditions must match'
                },
                {
                    'type': 'OR',
                    'display_name': 'OR',
                    'icon': '∨',
                    'description': 'Any condition must match'
                },
                {
                    'type': 'NOT',
                    'display_name': 'NOT',
                    'icon': '¬',
                    'description': 'Negate condition'
                }
            ]
        }
        
    def add_component(self,
                     session_id: str,
                     component: VisualRuleComponent) -> str:
        """Add a component to the rule builder"""
        session = self.get_session(session_id)
        
        component.component_id = str(uuid4())
        session.components.append(component)
        
        self.update_session(session_id, session)
        
        logger.info(f"Added component {component.component_id} to session {session_id}")
        return component.component_id
        
    def remove_component(self, session_id: str, component_id: str) -> bool:
        """Remove a component from the rule builder"""
        session = self.get_session(session_id)
        
        session.components = [
            c for c in session.components 
            if c.component_id != component_id
        ]
        
        # Remove connections involving this component
        session.connections = [
            conn for conn in session.connections
            if conn['source'] != component_id and conn['target'] != component_id
        ]
        
        self.update_session(session_id, session)
        
        logger.info(f"Removed component {component_id} from session {session_id}")
        return True
        
    def connect_components(self,
                          session_id: str,
                          source_id: str,
                          target_id: str,
                          connection_type: str = 'flow') -> bool:
        """Connect two components"""
        session = self.get_session(session_id)
        
        connection = {
            'source': source_id,
            'target': target_id,
            'type': connection_type
        }
        
        session.connections.append(connection)
        self.update_session(session_id, session)
        
        logger.info(f"Connected {source_id} to {target_id} in session {session_id}")
        return True
        
    def validate_rule(self, session_id: str) -> Dict[str, Any]:
        """Validate the rule configuration"""
        session = self.get_session(session_id)
        
        errors = []
        warnings = []
        
        # Check for at least one condition
        conditions = [c for c in session.components if c.type == 'condition']
        if not conditions:
            errors.append("Rule must have at least one condition")
            
        # Check for at least one action
        actions = [c for c in session.components if c.type == 'action']
        if not actions:
            errors.append("Rule must have at least one action")
            
        # Check for orphaned components
        for component in session.components:
            connected = any(
                conn['source'] == component.component_id or 
                conn['target'] == component.component_id
                for conn in session.connections
            )
            
            if not connected and len(session.components) > 1:
                warnings.append(f"Component {component.display_name} is not connected")
                
        # Check for circular dependencies
        if self._has_circular_dependency(session.connections):
            errors.append("Rule has circular dependencies")
            
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
        
    def _has_circular_dependency(self, connections: List[Dict[str, str]]) -> bool:
        """Check for circular dependencies in connections"""
        # Build adjacency list
        graph = {}
        for conn in connections:
            source = conn['source']
            target = conn['target']
            
            if source not in graph:
                graph[source] = []
            graph[source].append(target)
            
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
                    
            rec_stack.remove(node)
            return False
            
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
                    
        return False
        
    def compile_rule(self, session_id: str) -> ComplianceRule:
        """Compile visual components into a compliance rule"""
        session = self.get_session(session_id)
        
        # Validate first
        validation = self.validate_rule(session_id)
        if not validation['valid']:
            raise ValueError(f"Invalid rule: {validation['errors']}")
            
        # Extract conditions
        conditions = []
        for component in session.components:
            if component.type == 'condition':
                conditions.append(component.properties)
                
        # Extract actions
        actions = []
        for component in session.components:
            if component.type == 'action':
                actions.append(component.properties)
                
        # Determine logical operator
        logical_operator = 'AND'
        for component in session.components:
            if component.type == 'logical_operator':
                logical_operator = component.properties.get('operator', 'AND')
                break
                
        # Create rule
        rule = ComplianceRule(
            rule_id=session.rule_id or str(uuid4()),
            name=session.metadata.get('name', 'Untitled Rule'),
            description=session.metadata.get('description', ''),
            rule_type=RuleType(session.metadata.get('rule_type', 'custom')),
            conditions=conditions,
            logical_operator=logical_operator,
            actions=actions,
            severity=session.metadata.get('severity', 'medium'),
            tags=session.metadata.get('tags', [])
        )
        
        return rule
        
    def save_rule(self, session_id: str) -> str:
        """Save the compiled rule"""
        rule = self.compile_rule(session_id)
        
        self.saved_rules[rule.rule_id] = rule
        
        # Update session with rule ID
        session = self.get_session(session_id)
        session.rule_id = rule.rule_id
        self.update_session(session_id, session)
        
        logger.info(f"Saved rule {rule.rule_id} from session {session_id}")
        return rule.rule_id
        
    def load_rule(self, rule_id: str) -> RuleBuilderState:
        """Load a saved rule into visual builder format"""
        if rule_id not in self.saved_rules:
            raise ValueError(f"Rule not found: {rule_id}")
            
        rule = self.saved_rules[rule_id]
        
        # Create new session
        session_id = self.create_session()
        session = self.get_session(session_id)
        
        session.rule_id = rule_id
        session.metadata = {
            'name': rule.name,
            'description': rule.description,
            'rule_type': rule.rule_type.value,
            'severity': rule.severity,
            'tags': rule.tags
        }
        
        # Convert conditions to visual components
        y_position = 100
        for i, condition in enumerate(rule.conditions):
            component = VisualRuleComponent(
                component_id=str(uuid4()),
                type='condition',
                display_name=f"Condition {i+1}",
                icon='⚖️',
                properties=condition,
                position={'x': 100, 'y': y_position}
            )
            session.components.append(component)
            y_position += 100
            
        # Add logical operator
        if len(rule.conditions) > 1:
            operator_component = VisualRuleComponent(
                component_id=str(uuid4()),
                type='logical_operator',
                display_name=rule.logical_operator,
                icon='∧' if rule.logical_operator == 'AND' else '∨',
                properties={'operator': rule.logical_operator},
                position={'x': 300, 'y': 150}
            )
            session.components.append(operator_component)
            
        # Convert actions to visual components
        y_position = 100
        for i, action in enumerate(rule.actions):
            component = VisualRuleComponent(
                component_id=str(uuid4()),
                type='action',
                display_name=f"Action {i+1}",
                icon='⚡',
                properties=action,
                position={'x': 500, 'y': y_position}
            )
            session.components.append(component)
            y_position += 100
            
        self.update_session(session_id, session)
        return session
        
    def export_rule(self, rule_id: str, format: str = 'json') -> str:
        """Export a rule in specified format"""
        if rule_id not in self.saved_rules:
            raise ValueError(f"Rule not found: {rule_id}")
            
        rule = self.saved_rules[rule_id]
        
        if format == 'json':
            return json.dumps(rule.dict(), indent=2, default=str)
        elif format == 'yaml':
            import yaml
            return yaml.dump(rule.dict(), default_flow_style=False)
        elif format == 'python':
            return self._generate_python_code(rule)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _generate_python_code(self, rule: ComplianceRule) -> str:
        """Generate Python code for a rule"""
        code = f'''
# Auto-generated compliance rule: {rule.name}
# Description: {rule.description}

from compliance_engine import ComplianceRule, RuleType

rule = ComplianceRule(
    rule_id="{rule.rule_id}",
    name="{rule.name}",
    description="""{rule.description}""",
    rule_type=RuleType.{rule.rule_type.value.upper()},
    conditions={json.dumps(rule.conditions, indent=8)},
    logical_operator="{rule.logical_operator}",
    actions={json.dumps(rule.actions, indent=8)},
    severity="{rule.severity}",
    tags={rule.tags}
)

# Add to rule engine
engine.add_rule(rule)
'''
        return code
        
    def get_rule_preview(self, session_id: str) -> Dict[str, Any]:
        """Get a preview of the compiled rule"""
        try:
            rule = self.compile_rule(session_id)
            return {
                'valid': True,
                'rule': rule.dict()
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
            
    def apply_template(self, session_id: str, template_id: str) -> bool:
        """Apply a template to the current session"""
        if template_id not in self.templates:
            raise ValueError(f"Template not found: {template_id}")
            
        template = self.templates[template_id]
        session = self.get_session(session_id)
        
        # Clear existing components
        session.components = []
        session.connections = []
        
        # Apply template
        base_rule = template.base_rule
        
        session.metadata = {
            'name': template.name,
            'description': template.description,
            'rule_type': base_rule.get('rule_type', 'custom'),
            'severity': base_rule.get('severity', 'medium')
        }
        
        # Add conditions from template
        for i, condition in enumerate(base_rule.get('conditions', [])):
            component = VisualRuleComponent(
                component_id=str(uuid4()),
                type='condition',
                display_name=f"Condition {i+1}",
                icon='⚖️',
                properties=condition,
                position={'x': 100, 'y': 100 + i * 100}
            )
            session.components.append(component)
            
        # Add actions from template
        for i, action in enumerate(base_rule.get('actions', [])):
            component = VisualRuleComponent(
                component_id=str(uuid4()),
                type='action',
                display_name=f"Action {i+1}",
                icon='⚡',
                properties=action,
                position={'x': 400, 'y': 100 + i * 100}
            )
            session.components.append(component)
            
        self.update_session(session_id, session)
        
        logger.info(f"Applied template {template_id} to session {session_id}")
        return True


# FastAPI Router for Visual Rule Builder
router = APIRouter(prefix="/api/v1/rule-builder", tags=["rule-builder"])

visual_builder = VisualRuleBuilder()

@router.post("/sessions")
async def create_session():
    """Create a new rule builder session"""
    session_id = visual_builder.create_session()
    return {"session_id": session_id}

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session state"""
    try:
        session = visual_builder.get_session(session_id)
        return session.dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/templates")
async def get_templates(category: Optional[str] = Query(None)):
    """Get available rule templates"""
    templates = visual_builder.get_templates(category)
    return [t.dict() for t in templates]

@router.get("/components")
async def get_components():
    """Get component library"""
    return visual_builder.get_component_library()

@router.post("/sessions/{session_id}/components")
async def add_component(session_id: str, component: VisualRuleComponent):
    """Add a component to the rule builder"""
    try:
        component_id = visual_builder.add_component(session_id, component)
        return {"component_id": component_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/sessions/{session_id}/components/{component_id}")
async def remove_component(session_id: str, component_id: str):
    """Remove a component"""
    try:
        success = visual_builder.remove_component(session_id, component_id)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/sessions/{session_id}/connections")
async def connect_components(
    session_id: str,
    source_id: str,
    target_id: str,
    connection_type: str = "flow"
):
    """Connect two components"""
    try:
        success = visual_builder.connect_components(
            session_id, source_id, target_id, connection_type
        )
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/sessions/{session_id}/validate")
async def validate_rule(session_id: str):
    """Validate the rule configuration"""
    try:
        validation = visual_builder.validate_rule(session_id)
        return validation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/sessions/{session_id}/compile")
async def compile_rule(session_id: str):
    """Compile visual components into a rule"""
    try:
        rule = visual_builder.compile_rule(session_id)
        return rule.dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sessions/{session_id}/save")
async def save_rule(session_id: str):
    """Save the compiled rule"""
    try:
        rule_id = visual_builder.save_rule(session_id)
        return {"rule_id": rule_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/rules/{rule_id}")
async def load_rule(rule_id: str):
    """Load a saved rule"""
    try:
        session = visual_builder.load_rule(rule_id)
        return session.dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/rules/{rule_id}/export")
async def export_rule(rule_id: str, format: str = Query("json")):
    """Export a rule in specified format"""
    try:
        exported = visual_builder.export_rule(rule_id, format)
        return JSONResponse(content={"data": exported})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/sessions/{session_id}/templates/{template_id}")
async def apply_template(session_id: str, template_id: str):
    """Apply a template to the session"""
    try:
        success = visual_builder.apply_template(session_id, template_id)
        return {"success": success}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/sessions/{session_id}/preview")
async def preview_rule(session_id: str):
    """Get a preview of the compiled rule"""
    try:
        preview = visual_builder.get_rule_preview(session_id)
        return preview
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))