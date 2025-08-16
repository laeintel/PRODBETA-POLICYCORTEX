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
Cloud Providers Module for PolicyCortex
Provides multi-cloud support for Azure, AWS, and GCP
"""

from .multi_cloud_provider import MultiCloudProvider, CloudProvider, multi_cloud_provider
from .aws_provider import AWSProvider
from .gcp_provider import GCPProvider

__all__ = [
    'MultiCloudProvider',
    'CloudProvider',
    'multi_cloud_provider',
    'AWSProvider',
    'GCPProvider'
]