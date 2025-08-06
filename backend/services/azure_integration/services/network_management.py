"""
Azure Network Management service for handling virtual networks and network security.
"""

from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import structlog
from azure.core.exceptions import AzureError
from azure.core.exceptions import ResourceNotFoundError
from azure.mgmt.network import NetworkManagementClient
from shared.config import get_settings

from ..models import NetworkResponse
from ..models import NetworkSecurityAnalysis
from ..models import NetworkSecurityGroupResponse
from .azure_auth import AzureAuthService

settings = get_settings()
logger = structlog.get_logger(__name__)


class NetworkManagementService:
    """Service for managing Azure virtual networks and network security."""

    def __init__(self):
        self.settings = settings
        self.auth_service = AzureAuthService()
        self.network_clients = {}

    async def _get_network_client(self, subscription_id: str) -> NetworkManagementClient:
        """Get or create Network client for subscription."""
        if subscription_id not in self.network_clients:
            credential = await self.auth_service.get_credential(settings.azure.tenant_id)
            self.network_clients[subscription_id] = NetworkManagementClient(
                credential, subscription_id
            )
        return self.network_clients[subscription_id]

    async def list_virtual_networks(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None
    ) -> List[NetworkResponse]:
        """List all virtual networks."""
        try:
            client = await self._get_network_client(subscription_id)
            networks = []

            # List virtual networks
            if resource_group:
                vnet_list = client.virtual_networks.list(resource_group)
            else:
                vnet_list = client.virtual_networks.list_all()

            async for vnet in vnet_list:
                # Get subnets
                subnets = []
                if vnet.subnets:
                    for subnet in vnet.subnets:
                        subnets.append({
                            "id": subnet.id,
                            "name": subnet.name,
                            "address_prefix": subnet.address_prefix,
                            "route_table": subnet.route_table.id if subnet.route_table else None,
                            "network_security_group": subnet.network_security_group.id if subnet.network_security_group else None,
                            "delegations": [
                                {
                                    "name": delegation.name,
                                    "service_name": delegation.service_name
                                }
                                for delegation in subnet.delegations
                            ] if subnet.delegations else []
                        })

                # Get peerings
                peerings = []
                if vnet.virtual_network_peerings:
                    for peering in vnet.virtual_network_peerings:
                        peerings.append({
                            "id": peering.id,
                            "name": peering.name,
                            "remote_virtual_network": peering.remote_virtual_network.id if peering.remote_virtual_network else None,
                            "peering_state": peering.peering_state,
                            "provisioning_state": peering.provisioning_state,
                            "allow_virtual_network_access": peering.allow_virtual_network_access,
                            "allow_forwarded_traffic": peering.allow_forwarded_traffic,
                            "allow_gateway_transit": peering.allow_gateway_transit,
                            "use_remote_gateways": peering.use_remote_gateways
                        })

                networks.append(NetworkResponse(
                    id=vnet.id,
                    name=vnet.name,
                    type=vnet.type,
                    location=vnet.location,
                    resource_group=vnet.id.split('/')[4],  # Extract RG from resource ID
                    address_space=vnet.address_space.address_prefixes if vnet.address_space else [],
                    subnets=subnets,
                    peerings=peerings,
                    dns_servers=vnet.dhcp_options.dns_servers if vnet.dhcp_options else None,
                    tags=vnet.tags
                ))

            logger.info(
                "virtual_networks_listed",
                subscription_id=subscription_id,
                resource_group=resource_group,
                count=len(networks)
            )

            return networks

        except AzureError as e:
            logger.error(
                "list_virtual_networks_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to list virtual networks: {str(e)}")

    async def get_virtual_network(
        self,
        subscription_id: str,
        resource_group: str,
        network_name: str
    ) -> NetworkResponse:
        """Get a specific virtual network."""
        try:
            client = await self._get_network_client(subscription_id)

            # Get virtual network
            vnet = await client.virtual_networks.get(resource_group, network_name)

            # Process subnets
            subnets = []
            if vnet.subnets:
                for subnet in vnet.subnets:
                    subnets.append({
                        "id": subnet.id,
                        "name": subnet.name,
                        "address_prefix": subnet.address_prefix,
                        "route_table": subnet.route_table.id if subnet.route_table else None,
                        "network_security_group": subnet.network_security_group.id if subnet.network_security_group else None
                    })

            # Process peerings
            peerings = []
            if vnet.virtual_network_peerings:
                for peering in vnet.virtual_network_peerings:
                    peerings.append({
                        "id": peering.id,
                        "name": peering.name,
                        "remote_virtual_network": peering.remote_virtual_network.id if peering.remote_virtual_network else None,
                        "peering_state": peering.peering_state,
                        "provisioning_state": peering.provisioning_state
                    })

            logger.info(
                "virtual_network_retrieved",
                subscription_id=subscription_id,
                resource_group=resource_group,
                network_name=network_name
            )

            return NetworkResponse(
                id=vnet.id,
                name=vnet.name,
                type=vnet.type,
                location=vnet.location,
                resource_group=resource_group,
                address_space=vnet.address_space.address_prefixes if vnet.address_space else [],
                subnets=subnets,
                peerings=peerings,
                dns_servers=vnet.dhcp_options.dns_servers if vnet.dhcp_options else None,
                tags=vnet.tags
            )

        except ResourceNotFoundError:
            logger.error(
                "virtual_network_not_found",
                subscription_id=subscription_id,
                resource_group=resource_group,
                network_name=network_name
            )
            raise Exception(f"Virtual network {network_name} not found")
        except AzureError as e:
            logger.error(
                "get_virtual_network_failed",
                error=str(e),
                subscription_id=subscription_id,
                resource_group=resource_group,
                network_name=network_name
            )
            raise Exception(f"Failed to get virtual network: {str(e)}")

    async def list_network_security_groups(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all network security groups."""
        try:
            client = await self._get_network_client(subscription_id)
            nsgs = []

            # List NSGs
            if resource_group:
                nsg_list = client.network_security_groups.list(resource_group)
            else:
                nsg_list = client.network_security_groups.list_all()

            async for nsg in nsg_list:
                # Process security rules
                security_rules = []
                if nsg.security_rules:
                    for rule in nsg.security_rules:
                        security_rules.append({
                            "id": rule.id,
                            "name": rule.name,
                            "description": rule.description,
                            "protocol": rule.protocol,
                            "source_port_range": rule.source_port_range,
                            "destination_port_range": rule.destination_port_range,
                            "source_address_prefix": rule.source_address_prefix,
                            "destination_address_prefix": rule.destination_address_prefix,
                            "access": rule.access,
                            "priority": rule.priority,
                            "direction": rule.direction,
                            "provisioning_state": rule.provisioning_state
                        })

                # Process default security rules
                default_rules = []
                if nsg.default_security_rules:
                    for rule in nsg.default_security_rules:
                        default_rules.append({
                            "id": rule.id,
                            "name": rule.name,
                            "description": rule.description,
                            "protocol": rule.protocol,
                            "source_port_range": rule.source_port_range,
                            "destination_port_range": rule.destination_port_range,
                            "source_address_prefix": rule.source_address_prefix,
                            "destination_address_prefix": rule.destination_address_prefix,
                            "access": rule.access,
                            "priority": rule.priority,
                            "direction": rule.direction
                        })

                # Get associated resources
                network_interfaces = []
                if nsg.network_interfaces:
                    network_interfaces = [ni.id for ni in nsg.network_interfaces]

                subnets = []
                if nsg.subnets:
                    subnets = [subnet.id for subnet in nsg.subnets]

                nsgs.append({
                    "id": nsg.id,
                    "name": nsg.name,
                    "type": nsg.type,
                    "location": nsg.location,
                    "resource_group": nsg.id.split('/')[4],  # Extract RG from resource ID
                    "security_rules": security_rules,
                    "default_security_rules": default_rules,
                    "network_interfaces": network_interfaces,
                    "subnets": subnets,
                    "tags": nsg.tags
                })

            logger.info(
                "network_security_groups_listed",
                subscription_id=subscription_id,
                resource_group=resource_group,
                count=len(nsgs)
            )

            return nsgs

        except AzureError as e:
            logger.error(
                "list_network_security_groups_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to list network security groups: {str(e)}")

    async def analyze_network_security(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze network security configuration."""
        try:
            # Get networks and NSGs
            networks = await self.list_virtual_networks(subscription_id, resource_group)
            nsgs = await self.list_network_security_groups(subscription_id, resource_group)

            # Initialize analysis results
            security_issues = []
            open_ports = []
            overly_permissive_rules = []
            missing_nsgs = []
            recommendations = []

            # Analyze NSG rules
            for nsg in nsgs:
                for rule in nsg.get("security_rules", []):
                    # Check for overly permissive rules
                    if (rule.get("source_address_prefix") == "*" and
                        rule.get("access") == "Allow" and
                        rule.get("direction") == "Inbound"):

                        overly_permissive_rules.append({
                            "nsg_name": nsg["name"],
                            "rule_name": rule["name"],
                            "issue": "Allows inbound traffic from any source",
                            "priority": rule.get("priority"),
                            "protocol": rule.get("protocol"),
                            "destination_port_range": rule.get("destination_port_range")
                        })

                    # Check for open ports to internet
                    if (rule.get("source_address_prefix") in ["*", "0.0.0.0/0", "Internet"] and
                        rule.get("access") == "Allow" and
                        rule.get("direction") == "Inbound"):

                        open_ports.append({
                            "nsg_name": nsg["name"],
                            "rule_name": rule["name"],
                            "port_range": rule.get("destination_port_range"),
                            "protocol": rule.get("protocol"),
                            "priority": rule.get("priority")
                        })

                    # Check for common insecure ports
                    insecure_ports = ["22", "3389", "1433", "1521", "3306", "5432", "27017"]
                    dest_port = rule.get("destination_port_range", "")
                    if (any(port in dest_port for port in insecure_ports) and
                        rule.get("source_address_prefix") == "*" and
                        rule.get("access") == "Allow"):

                        security_issues.append({
                            "type": "insecure_port_exposure",
                            "severity": "high",
                            "nsg_name": nsg["name"],
                            "rule_name": rule["name"],
                            "port": dest_port,
                            "description": f"Insecure port {dest_port} exposed to internet"
                        })

            # Check for subnets without NSGs
            for network in networks:
                for subnet in network.subnets:
                    if not subnet.get("network_security_group"):
                        missing_nsgs.append({
                            "network_name": network.name,
                            "subnet_name": subnet["name"],
                            "address_prefix": subnet["address_prefix"]
                        })

            # Generate recommendations
            if overly_permissive_rules:
                recommendations.append({
                    "type": "security",
                    "priority": "high",
                    "title": "Restrict overly permissive NSG rules",
                    "description": f"Found {len(overly_permissive_rules)} overly permissive rules. Consider restricting source IP ranges.",
                    "affected_count": len(overly_permissive_rules)
                })

            if open_ports:
                recommendations.append({
                    "type": "security",
                    "priority": "high",
                    "title": "Secure open ports",
                    "description": f"Found {len(open_ports)} rules with ports open to internet. Review and
                        restrict access.",
                    "affected_count": len(open_ports)
                })

            if missing_nsgs:
                recommendations.append({
                    "type": "security",
                    "priority": "medium",
                    "title": "Add NSGs to unprotected subnets",
                    "description": f"Found {len(missing_nsgs)} subnets without NSGs. Consider adding network security groups.",
                    "affected_count": len(missing_nsgs)
                })

            # Calculate risk score
            risk_score = 0
            risk_score += len(security_issues) * 20
            risk_score += len(overly_permissive_rules) * 15
            risk_score += len(open_ports) * 10
            risk_score += len(missing_nsgs) * 5
            risk_score = min(risk_score, 100)  # Cap at 100

            logger.info(
                "network_security_analysis_completed",
                subscription_id=subscription_id,
                resource_group=resource_group,
                total_networks=len(networks),
                total_nsgs=len(nsgs),
                security_issues=len(security_issues),
                risk_score=risk_score
            )

            return NetworkSecurityAnalysis(
                total_networks=len(networks),
                total_nsgs=len(nsgs),
                security_issues=security_issues,
                open_ports=open_ports,
                overly_permissive_rules=overly_permissive_rules,
                missing_nsgs=missing_nsgs,
                recommendations=recommendations,
                risk_score=risk_score,
                analysis_timestamp=datetime.utcnow()
            ).dict()

        except Exception as e:
            logger.error(
                "analyze_network_security_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to analyze network security: {str(e)}")

    async def create_network_security_rule(
        self,
        subscription_id: str,
        resource_group: str,
        nsg_name: str,
        rule_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new network security rule."""
        try:
            client = await self._get_network_client(subscription_id)

            # Create security rule parameters
            rule_params = {
                "description": rule_data.get("description"),
                "protocol": rule_data["protocol"],
                "source_port_range": rule_data.get("source_port_range", "*"),
                "destination_port_range": rule_data["destination_port_range"],
                "source_address_prefix": rule_data.get("source_address_prefix", "*"),
                "destination_address_prefix": rule_data.get("destination_address_prefix", "*"),
                "access": rule_data["access"],
                "priority": rule_data["priority"],
                "direction": rule_data["direction"]
            }

            # Create the rule
            rule = await client.security_rules.begin_create_or_update(
                resource_group_name=resource_group,
                network_security_group_name=nsg_name,
                security_rule_name=rule_data["name"],
                security_rule_parameters=rule_params
            )

            logger.info(
                "security_rule_created",
                subscription_id=subscription_id,
                resource_group=resource_group,
                nsg_name=nsg_name,
                rule_name=rule_data["name"]
            )

            return {
                "id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "protocol": rule.protocol,
                "source_port_range": rule.source_port_range,
                "destination_port_range": rule.destination_port_range,
                "source_address_prefix": rule.source_address_prefix,
                "destination_address_prefix": rule.destination_address_prefix,
                "access": rule.access,
                "priority": rule.priority,
                "direction": rule.direction,
                "provisioning_state": rule.provisioning_state
            }

        except AzureError as e:
            logger.error(
                "create_security_rule_failed",
                error=str(e),
                subscription_id=subscription_id,
                resource_group=resource_group,
                nsg_name=nsg_name
            )
            raise Exception(f"Failed to create security rule: {str(e)}")

    async def get_network_topology(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get network topology information."""
        try:
            # Get networks and NSGs
            networks = await self.list_virtual_networks(subscription_id, resource_group)
            nsgs = await self.list_network_security_groups(subscription_id, resource_group)

            # Build topology
            topology = {
                "networks": [],
                "security_groups": [],
                "connections": []
            }

            # Process networks
            for network in networks:
                network_info = {
                    "id": network.id,
                    "name": network.name,
                    "address_space": network.address_space,
                    "subnets": network.subnets,
                    "peerings": network.peerings,
                    "location": network.location
                }
                topology["networks"].append(network_info)

            # Process NSGs
            for nsg in nsgs:
                nsg_info = {
                    "id": nsg["id"],
                    "name": nsg["name"],
                    "location": nsg["location"],
                    "associated_subnets": nsg["subnets"],
                    "security_rules_count": len(nsg.get("security_rules", [])),
                    "default_rules_count": len(nsg.get("default_security_rules", []))
                }
                topology["security_groups"].append(nsg_info)

            # Identify connections (peerings)
            for network in networks:
                for peering in network.peerings:
                    topology["connections"].append({
                        "source_network": network.name,
                        "target_network": peering.get("remote_virtual_network", "").split("/")[-1],
                        "peering_state": peering.get("peering_state"),
                        "connection_type": "peering"
                    })

            logger.info(
                "network_topology_retrieved",
                subscription_id=subscription_id,
                resource_group=resource_group,
                networks_count=len(networks),
                nsgs_count=len(nsgs)
            )

            return topology

        except Exception as e:
            logger.error(
                "get_network_topology_failed",
                error=str(e),
                subscription_id=subscription_id
            )
            raise Exception(f"Failed to get network topology: {str(e)}")
