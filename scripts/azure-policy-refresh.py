#!/usr/bin/env python3
"""
Azure Policy Refresh Script
Fetches and updates Azure policy definitions, compliance standards, and SDK data
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from azure.identity import DefaultAzureCredential
from azure.mgmt.policyinsights import PolicyInsightsClient
from azure.mgmt.resource import PolicyClient, ResourceManagementClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzurePolicyRefresh:
    def __init__(self):
        """Initialize Azure clients"""
        self.subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')
        if not self.subscription_id:
            raise ValueError("AZURE_SUBSCRIPTION_ID environment variable is required")
        
        self.credential = DefaultAzureCredential()
        self.policy_client = PolicyClient(self.credential, self.subscription_id)
        self.policy_insights_client = PolicyInsightsClient(self.credential, self.subscription_id)
        self.resource_client = ResourceManagementClient(self.credential, self.subscription_id)
        
        # Create data directories
        self.data_dir = 'data'
        self.policies_dir = os.path.join(self.data_dir, 'policies')
        self.compliance_dir = os.path.join(self.data_dir, 'compliance')
        os.makedirs(self.policies_dir, exist_ok=True)
        os.makedirs(self.compliance_dir, exist_ok=True)
    
    def fetch_policy_definitions(self) -> List[Dict[str, Any]]:
        """Fetch all policy definitions"""
        logger.info("Fetching policy definitions...")
        definitions = []
        
        try:
            # Get built-in policy definitions
            for policy in self.policy_client.policy_definitions.list():
                definitions.append({
                    'id': policy.id,
                    'name': policy.name,
                    'display_name': policy.display_name,
                    'description': policy.description,
                    'policy_type': policy.policy_type,
                    'mode': policy.mode,
                    'metadata': policy.metadata,
                    'parameters': policy.parameters,
                    'policy_rule': policy.policy_rule
                })
            
            logger.info(f"Fetched {len(definitions)} policy definitions")
            return definitions
        except Exception as e:
            logger.error(f"Error fetching policy definitions: {e}")
            return []
    
    def fetch_policy_set_definitions(self) -> List[Dict[str, Any]]:
        """Fetch policy set definitions (initiatives)"""
        logger.info("Fetching policy set definitions...")
        initiatives = []
        
        try:
            for initiative in self.policy_client.policy_set_definitions.list():
                initiatives.append({
                    'id': initiative.id,
                    'name': initiative.name,
                    'display_name': initiative.display_name,
                    'description': initiative.description,
                    'policy_type': initiative.policy_type,
                    'metadata': initiative.metadata,
                    'parameters': initiative.parameters,
                    'policy_definitions': initiative.policy_definitions
                })
            
            logger.info(f"Fetched {len(initiatives)} policy set definitions")
            return initiatives
        except Exception as e:
            logger.error(f"Error fetching policy set definitions: {e}")
            return []
    
    def fetch_policy_assignments(self) -> List[Dict[str, Any]]:
        """Fetch policy assignments"""
        logger.info("Fetching policy assignments...")
        assignments = []
        
        try:
            for assignment in self.policy_client.policy_assignments.list():
                assignments.append({
                    'id': assignment.id,
                    'name': assignment.name,
                    'display_name': assignment.display_name,
                    'description': assignment.description,
                    'enforcement_mode': assignment.enforcement_mode,
                    'policy_definition_id': assignment.policy_definition_id,
                    'parameters': assignment.parameters,
                    'scope': assignment.scope
                })
            
            logger.info(f"Fetched {len(assignments)} policy assignments")
            return assignments
        except Exception as e:
            logger.error(f"Error fetching policy assignments: {e}")
            return []
    
    def fetch_compliance_results(self) -> Dict[str, Any]:
        """Fetch policy compliance results"""
        logger.info("Fetching compliance results...")
        compliance_data = {
            'summary': {},
            'details': []
        }
        
        try:
            # Get policy states
            states = self.policy_insights_client.policy_states.list_query_results_for_subscription(
                policy_states_resource="latest",
                subscription_id=self.subscription_id
            )
            
            for state in states:
                compliance_data['details'].append({
                    'resource_id': state.resource_id,
                    'policy_assignment_id': state.policy_assignment_id,
                    'compliance_state': state.compliance_state,
                    'timestamp': state.timestamp.isoformat() if state.timestamp else None
                })
            
            # Calculate summary
            total = len(compliance_data['details'])
            compliant = sum(1 for s in compliance_data['details'] if s['compliance_state'] == 'Compliant')
            non_compliant = sum(1 for s in compliance_data['details'] if s['compliance_state'] == 'NonCompliant')
            
            compliance_data['summary'] = {
                'total_resources': total,
                'compliant': compliant,
                'non_compliant': non_compliant,
                'compliance_percentage': (compliant / total * 100) if total > 0 else 0
            }
            
            logger.info(f"Fetched compliance data for {total} resources")
            return compliance_data
        except Exception as e:
            logger.error(f"Error fetching compliance results: {e}")
            return compliance_data
    
    def save_data(self, data: Any, filename: str, directory: str):
        """Save data to JSON file"""
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved data to {filepath}")
    
    def create_metadata(self) -> Dict[str, Any]:
        """Create metadata for the refresh"""
        return {
            'last_updated': datetime.utcnow().isoformat(),
            'subscription_id': self.subscription_id,
            'version': '1.0.0'
        }
    
    def run(self):
        """Main execution"""
        logger.info("Starting Azure policy refresh...")
        
        # Fetch all data
        policy_definitions = self.fetch_policy_definitions()
        policy_sets = self.fetch_policy_set_definitions()
        policy_assignments = self.fetch_policy_assignments()
        compliance_results = self.fetch_compliance_results()
        metadata = self.create_metadata()
        
        # Save policy data
        self.save_data(policy_definitions, 'definitions.json', self.policies_dir)
        self.save_data(policy_sets, 'initiatives.json', self.policies_dir)
        self.save_data(policy_assignments, 'assignments.json', self.policies_dir)
        self.save_data(metadata, 'metadata.json', self.policies_dir)
        
        # Save compliance data
        self.save_data(compliance_results, 'compliance.json', self.compliance_dir)
        self.save_data(metadata, 'metadata.json', self.compliance_dir)
        
        # Create summary report
        summary = {
            'metadata': metadata,
            'statistics': {
                'policy_definitions': len(policy_definitions),
                'policy_initiatives': len(policy_sets),
                'policy_assignments': len(policy_assignments),
                'compliance': compliance_results['summary']
            }
        }
        self.save_data(summary, 'summary.json', self.data_dir)
        
        logger.info("Azure policy refresh completed successfully!")
        return summary

if __name__ == "__main__":
    try:
        refresher = AzurePolicyRefresh()
        summary = refresher.run()
        print(json.dumps(summary, indent=2, default=str))
    except Exception as e:
        logger.error(f"Failed to refresh Azure policies: {e}")
        exit(1)