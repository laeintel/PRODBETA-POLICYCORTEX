"""
Notification service package for PolicyCortex.
"""

__version__ = "1.0.0"
__author__ = "PolicyCortex Team"
__email__ = "dev@policycortex.com"
__description__ = "Comprehensive notification service for PolicyCortex platform"

from .main import app
from .models import *
from .auth import AuthManager
from .services import *

__all__ = [
    "app",
    "AuthManager"
]