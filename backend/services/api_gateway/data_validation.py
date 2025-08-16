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
Comprehensive data validation and sanitization for secure input handling
"""

import re
import html
import json
from typing import Dict, Any, List, Optional, Union, Type
from datetime import datetime
from decimal import Decimal
from enum import Enum
import logging

from pydantic import BaseModel, Field, validator, root_validator, EmailStr, HttpUrl, constr, conint, confloat
from pydantic.types import SecretStr
import bleach

logger = logging.getLogger(__name__)

# Validation patterns
PATTERNS = {
    "azure_resource_id": r"^/subscriptions/[a-f0-9-]+/resourceGroups/[a-zA-Z0-9-_]+/providers/[a-zA-Z.]+/[a-zA-Z0-9-_/]+$",
    "uuid": r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
    "tenant_id": r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
    "safe_string": r"^[a-zA-Z0-9-_\s\.]+$",
    "tag_key": r"^[a-zA-Z][a-zA-Z0-9-_]{0,127}$",
    "tag_value": r"^[a-zA-Z0-9-_\s\.]{0,256}$",
    "policy_name": r"^[a-zA-Z][a-zA-Z0-9-_]{2,63}$",
    "resource_name": r"^[a-zA-Z][a-zA-Z0-9-_]{2,63}$",
}

class InputSanitizer:
    """Sanitize user input to prevent XSS and injection attacks"""
    
    @staticmethod
    def sanitize_html(text: str, allowed_tags: List[str] = None) -> str:
        """Sanitize HTML content"""
        allowed_tags = allowed_tags or ['b', 'i', 'u', 'strong', 'em', 'p', 'br']
        allowed_attrs = {}
        
        return bleach.clean(
            text,
            tags=allowed_tags,
            attributes=allowed_attrs,
            strip=True
        )
    
    @staticmethod
    def escape_html(text: str) -> str:
        """Escape HTML special characters"""
        return html.escape(text)
    
    @staticmethod
    def sanitize_json(data: Union[str, Dict]) -> Dict[str, Any]:
        """Sanitize JSON data"""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format")
        
        def sanitize_value(value):
            if isinstance(value, str):
                # Remove control characters
                value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
                # Limit string length
                return value[:10000]
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [sanitize_value(item) for item in value]
            return value
        
        return sanitize_value(data)
    
    @staticmethod
    def sanitize_sql(text: str) -> str:
        """Basic SQL injection prevention (use parameterized queries instead)"""
        # Remove common SQL injection patterns
        dangerous_patterns = [
            r"(;|--|\*|\/\*|\*\/|xp_|sp_|exec|execute|declare|drop|delete|truncate|alter)",
            r"(union|select|insert|update|from|where|having|group by|order by)",
        ]
        
        sanitized = text
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path components
        filename = filename.replace("..", "")
        filename = filename.replace("/", "")
        filename = filename.replace("\\", "")
        
        # Keep only safe characters
        filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
        
        # Limit length
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        if ext:
            return f"{name[:100]}.{ext[:10]}"
        return filename[:100]


# Custom Pydantic types
AzureResourceId = constr(regex=PATTERNS["azure_resource_id"])
TenantId = constr(regex=PATTERNS["tenant_id"])
PolicyName = constr(regex=PATTERNS["policy_name"], min_length=3, max_length=64)
ResourceName = constr(regex=PATTERNS["resource_name"], min_length=3, max_length=64)
SafeString = constr(regex=PATTERNS["safe_string"], max_length=256)


class ResourceValidation(BaseModel):
    """Validation for resource-related requests"""
    
    resource_id: AzureResourceId
    resource_name: ResourceName
    resource_type: str = Field(..., min_length=3, max_length=100)
    provider: str = Field(..., regex="^(azure|aws|gcp)$")
    region: str = Field(..., min_length=2, max_length=50)
    tags: Optional[Dict[str, str]] = Field(default_factory=dict)
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags format"""
        if not v:
            return v
        
        for key, value in v.items():
            if not re.match(PATTERNS["tag_key"], key):
                raise ValueError(f"Invalid tag key: {key}")
            if not re.match(PATTERNS["tag_value"], value):
                raise ValueError(f"Invalid tag value: {value}")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "resource_id": "/subscriptions/12345/resourceGroups/myRG/providers/Microsoft.Compute/virtualMachines/myVM",
                "resource_name": "myVM",
                "resource_type": "Microsoft.Compute/virtualMachines",
                "provider": "azure",
                "region": "eastus",
                "tags": {"Environment": "Production", "Owner": "TeamA"}
            }
        }


class PolicyValidation(BaseModel):
    """Validation for policy-related requests"""
    
    policy_id: Optional[str] = Field(None, regex=PATTERNS["uuid"])
    policy_name: PolicyName
    policy_type: str = Field(..., regex="^(compliance|security|cost|operational)$")
    severity: str = Field(..., regex="^(low|medium|high|critical)$")
    enabled: bool = True
    rules: List[Dict[str, Any]]
    
    @validator('rules')
    def validate_rules(cls, v):
        """Validate policy rules structure"""
        if not v:
            raise ValueError("At least one rule is required")
        
        for rule in v:
            if 'condition' not in rule or 'action' not in rule:
                raise ValueError("Each rule must have 'condition' and 'action'")
        
        return v


class CostValidation(BaseModel):
    """Validation for cost-related data"""
    
    amount: confloat(ge=0, le=1000000000)  # Max 1 billion
    currency: str = Field(..., regex="^[A-Z]{3}$")  # ISO 4217 currency code
    period_start: datetime
    period_end: datetime
    
    @root_validator
    def validate_period(cls, values):
        """Validate period dates"""
        start = values.get('period_start')
        end = values.get('period_end')
        
        if start and end and start >= end:
            raise ValueError("period_start must be before period_end")
        
        return values


class UserInputValidation(BaseModel):
    """Validation for user input"""
    
    email: Optional[EmailStr]
    url: Optional[HttpUrl]
    message: constr(max_length=10000)
    
    @validator('message')
    def sanitize_message(cls, v):
        """Sanitize user message"""
        # Remove control characters
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        # Escape HTML
        v = html.escape(v)
        return v


class QueryValidation(BaseModel):
    """Validation for query parameters"""
    
    page: conint(ge=1, le=10000) = 1
    limit: conint(ge=1, le=100) = 20
    sort_by: Optional[SafeString]
    order: str = Field("asc", regex="^(asc|desc)$")
    filters: Optional[Dict[str, Any]]
    
    @validator('filters')
    def validate_filters(cls, v):
        """Validate filter parameters"""
        if not v:
            return v
        
        # Limit filter depth to prevent DoS
        def check_depth(obj, depth=0):
            if depth > 5:
                raise ValueError("Filter depth exceeds maximum")
            if isinstance(obj, dict):
                for value in obj.values():
                    check_depth(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, depth + 1)
        
        check_depth(v)
        return v


class DataValidator:
    """Main data validation service"""
    
    @staticmethod
    def validate_request(data: Dict[str, Any], model: Type[BaseModel]) -> BaseModel:
        """Validate request data against model"""
        try:
            return model(**data)
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            raise ValueError(f"Invalid input: {str(e)}")
    
    @staticmethod
    def validate_azure_resource_id(resource_id: str) -> bool:
        """Validate Azure resource ID format"""
        return bool(re.match(PATTERNS["azure_resource_id"], resource_id))
    
    @staticmethod
    def validate_uuid(uuid_str: str) -> bool:
        """Validate UUID format"""
        return bool(re.match(PATTERNS["uuid"], uuid_str))
    
    @staticmethod
    def validate_tenant_id(tenant_id: str) -> bool:
        """Validate tenant ID format"""
        return bool(re.match(PATTERNS["tenant_id"], tenant_id))
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        try:
            EmailStr.validate(email)
            return True
        except:
            return False
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        try:
            HttpUrl.validate(url)
            return True
        except:
            return False
    
    @staticmethod
    def validate_json(json_str: str) -> bool:
        """Validate JSON format"""
        try:
            json.loads(json_str)
            return True
        except:
            return False
    
    @staticmethod
    def validate_date_range(start: datetime, end: datetime, max_days: int = 365) -> bool:
        """Validate date range"""
        if start >= end:
            return False
        
        delta = end - start
        if delta.days > max_days:
            return False
        
        return True


class RequestLimiter:
    """Limit request size and complexity"""
    
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_ARRAY_LENGTH = 1000
    MAX_STRING_LENGTH = 100000
    MAX_OBJECT_DEPTH = 10
    
    @classmethod
    def check_request_size(cls, data: Union[str, bytes]) -> bool:
        """Check if request size is within limits"""
        size = len(data) if isinstance(data, (str, bytes)) else len(json.dumps(data))
        return size <= cls.MAX_REQUEST_SIZE
    
    @classmethod
    def check_complexity(cls, obj: Any, depth: int = 0) -> bool:
        """Check object complexity"""
        if depth > cls.MAX_OBJECT_DEPTH:
            return False
        
        if isinstance(obj, str):
            return len(obj) <= cls.MAX_STRING_LENGTH
        
        elif isinstance(obj, list):
            if len(obj) > cls.MAX_ARRAY_LENGTH:
                return False
            return all(cls.check_complexity(item, depth + 1) for item in obj)
        
        elif isinstance(obj, dict):
            if len(obj) > cls.MAX_ARRAY_LENGTH:
                return False
            return all(
                cls.check_complexity(v, depth + 1) 
                for v in obj.values()
            )
        
        return True


class ValidationMiddleware:
    """Middleware for automatic request validation"""
    
    def __init__(self, sanitizer: InputSanitizer = None):
        self.sanitizer = sanitizer or InputSanitizer()
    
    async def validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize request data"""
        # Check request size
        if not RequestLimiter.check_request_size(json.dumps(request_data)):
            raise ValueError("Request size exceeds limit")
        
        # Check complexity
        if not RequestLimiter.check_complexity(request_data):
            raise ValueError("Request complexity exceeds limit")
        
        # Sanitize JSON
        sanitized = self.sanitizer.sanitize_json(request_data)
        
        return sanitized


# Validation decorators
def validate_input(model: Type[BaseModel]):
    """Decorator to validate function input"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Find the data argument
            data = kwargs.get('data') or (args[0] if args else {})
            
            # Validate data
            validated = DataValidator.validate_request(data, model)
            
            # Replace with validated data
            if 'data' in kwargs:
                kwargs['data'] = validated.dict()
            elif args:
                args = (validated.dict(),) + args[1:]
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def sanitize_output(func):
    """Decorator to sanitize function output"""
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        
        # Sanitize output
        if isinstance(result, dict):
            result = InputSanitizer.sanitize_json(result)
        elif isinstance(result, str):
            result = InputSanitizer.escape_html(result)
        
        return result
    
    return wrapper


# Export key components
__all__ = [
    "InputSanitizer",
    "DataValidator",
    "RequestLimiter",
    "ValidationMiddleware",
    "ResourceValidation",
    "PolicyValidation",
    "CostValidation",
    "UserInputValidation",
    "QueryValidation",
    "validate_input",
    "sanitize_output",
    "AzureResourceId",
    "TenantId",
    "PolicyName",
    "ResourceName",
    "SafeString"
]