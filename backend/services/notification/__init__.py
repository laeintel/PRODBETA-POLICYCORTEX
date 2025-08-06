"""
Notification service package for PolicyCortex.
"""

__version__ = "1.0.0"
__author__ = "PolicyCortex Team"
__email__ = "dev@policycortex.com"
__description__ = "Comprehensive notification service for PolicyCortex platform"

from .auth import AuthManager
from .main import app
from .models import *
from .services import *

__all__ = ["app", "AuthManager"]
