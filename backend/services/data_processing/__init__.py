"""
Data Processing Service for PolicyCortex.
Handles ETL pipelines, stream processing, data transformation, and quality checks.
"""

__version__ = "1.0.0"
__author__ = "PolicyCortex Team"
__email__ = "team@policycortex.com"

from .main import app
from .models import *
from .auth import AuthManager
from .services import *

__all__ = [
    "app",
    "AuthManager"
]