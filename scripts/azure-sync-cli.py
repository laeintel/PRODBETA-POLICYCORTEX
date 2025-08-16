#!/usr/bin/env python3
"""
PolicyCortex Azure Sync CLI
Command-line tool for managing Azure policy and SDK synchronization
"""

import asyncio
import json
import sys
import argparse
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
import httpx

console = Console()

@dataclass
class PolicySyncStatus:
    scope: str
    last_sync: datetime
    policies_synced: int
    policies_added: int
    policies_updated: int
    policies_deprecated: int

@dataclass
class SDKUpdateStatus:
    package_name: str
    current_version: str
    latest_version: str
    update_available: bool
    security_update: bool
    update_priority: str

class AzureSyncCLI:
    """Main CLI interface for Azure sync management"""
    
    def __init__(self, api_base_url: str = "http://localhost:8085"):
        self.api_base_url = api_base_url
        
    async def initialize(self):
        """Initialize connections"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base_url}/health")
                if response.status_code != 200:
                    console.print("[red]❌ Cannot connect to Azure Sync Service[/red]")
                    sys.exit(1)
            
            console.print("[green]✅ Connected to Azure Sync Service[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Connection failed: {e}[/red]")
            sys.exit(1)

@click.group()
@click.option('--api-url', default='http://localhost:8085', help='Azure Sync Service URL')
@click.option('--format', type=click.Choice(['json', 'yaml', 'table']), default='table', help='Output format')
@click.pass_context
def cli(ctx, api_url, format):
    """PolicyCortex Azure Sync CLI"""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url
    ctx.obj['format'] = format
    ctx.obj['cli'] = AzureSyncCLI(api_url)

@cli.group()
def policy():
    """Azure policy management commands"""
    pass

@cli.group()
def sdk():
    """SDK version management commands"""
    pass

@policy.command()
@click.option('--scope', default='subscription', type=click.Choice(['subscription', 'management_group']), help='Policy scope')
@click.option('--scope-id', help='Scope ID (subscription ID or management group ID)')
@click.option('--include-builtin/--no-builtin', default=True, help='Include built-in policies')
@click.option('--include-custom/--no-custom', default=True, help='Include custom policies')
@click.option('--force', is_flag=True, help='Force refresh all policies')
@click.pass_context
async def sync(ctx, scope, scope_id, include_builtin, include_custom, force):
    """Sync Azure policy definitions"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Syncing Azure policies...", total=None)
        
        try:
            sync_request = {
                "scope": scope,
                "scope_id": scope_id,
                "include_builtin": include_builtin,
                "include_custom": include_custom,
                "force_refresh": force
            }
            
            result = await trigger_policy_sync(cli_instance.api_base_url, sync_request)
            progress.update(task, description="✅ Policy sync started")
            
            console.print(f"[green]Policy sync initiated for scope: {scope}[/green]")
            if scope_id:
                console.print(f"[blue]Scope ID: {scope_id}[/blue]")
            
            # Wait for sync to complete and show status
            await monitor_sync_progress(cli_instance.api_base_url, scope, scope_id)
            
        except Exception as e:
            progress.update(task, description="❌ Policy sync failed")
            console.print(f"[red]Policy sync failed: {e}[/red]")

@policy.command()
@click.option('--scope', default='subscription', help='Policy scope')
@click.option('--scope-id', help='Scope ID')
@click.pass_context
async def status(ctx, scope, scope_id):
    """Check policy sync status"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    status_data = await get_policy_sync_status(cli_instance.api_base_url, scope, scope_id)
    display_policy_sync_status(status_data, ctx.obj['format'])

@policy.command()
@click.option('--scope', default='subscription', help='Policy scope')
@click.option('--scope-id', help='Scope ID')
@click.option('--limit', default=50, help='Limit number of policies to show')
@click.option('--search', help='Search policies by name or description')
@click.option('--policy-type', type=click.Choice(['BuiltIn', 'Custom']), help='Filter by policy type')
@click.pass_context
async def list(ctx, scope, scope_id, limit, search, policy_type):
    """List synchronized policies"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    policies = await get_policies(cli_instance.api_base_url, scope, scope_id, limit, search, policy_type)
    display_policies(policies, ctx.obj['format'])

@policy.command()
@click.argument('policy_id')
@click.pass_context
async def show(ctx, policy_id):
    """Show detailed policy information"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    policy = await get_policy_details(cli_instance.api_base_url, policy_id)
    display_policy_details(policy, ctx.obj['format'])

@sdk.command()
@click.option('--package', help='Check specific package')
@click.option('--include-preview', is_flag=True, help='Include preview versions')
@click.pass_context
async def check(ctx, package, include_preview):
    """Check for SDK updates"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking SDK versions...", total=None)
        
        try:
            check_request = {
                "package_name": package,
                "include_preview": include_preview
            }
            
            updates = await check_sdk_updates(cli_instance.api_base_url, check_request)
            progress.update(task, description="✅ SDK check complete")
            
            display_sdk_updates(updates, ctx.obj['format'])
            
        except Exception as e:
            progress.update(task, description="❌ SDK check failed")
            console.print(f"[red]SDK check failed: {e}[/red]")

@sdk.command()
@click.argument('package_name')
@click.argument('version')
@click.option('--auto-deploy', is_flag=True, help='Automatically deploy after update')
@click.option('--force', is_flag=True, help='Force update without confirmation')
@click.pass_context
async def update(ctx, package_name, version, auto_deploy, force):
    """Update SDK package"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    if not force:
        if not Confirm.ask(f"Update {package_name} to {version}?"):
            return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Updating {package_name}...", total=None)
        
        try:
            result = await update_sdk_package(cli_instance.api_base_url, package_name, version, auto_deploy)
            progress.update(task, description="✅ Update complete")
            
            display_update_result(result, ctx.obj['format'])
            
        except Exception as e:
            progress.update(task, description="❌ Update failed")
            console.print(f"[red]SDK update failed: {e}[/red]")

@sdk.command()
@click.pass_context
async def versions(ctx):
    """Show current SDK versions"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    versions = await get_sdk_versions(cli_instance.api_base_url)
    display_sdk_versions(versions, ctx.obj['format'])

@cli.command()
@click.option('--watch', is_flag=True, help='Watch for changes')
@click.option('--interval', default=30, help='Watch interval in seconds')
@click.pass_context
async def dashboard(ctx, watch, interval):
    """Show sync dashboard"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    if watch:
        while True:
            console.clear()
            await show_dashboard(cli_instance.api_base_url)
            await asyncio.sleep(interval)
    else:
        await show_dashboard(cli_instance.api_base_url)

# Helper functions

async def trigger_policy_sync(api_url: str, sync_request: Dict) -> Dict:
    """Trigger policy synchronization"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api_url}/sync/policies", json=sync_request)
        response.raise_for_status()
        return response.json()

async def get_policy_sync_status(api_url: str, scope: str, scope_id: Optional[str] = None) -> Dict:
    """Get policy sync status"""
    params = {}
    if scope_id:
        params['scope_id'] = scope_id
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/sync/policies/status/{scope}", params=params)
        response.raise_for_status()
        return response.json()

async def get_policies(api_url: str, scope: str, scope_id: Optional[str], limit: int, 
                      search: Optional[str], policy_type: Optional[str]) -> List[Dict]:
    """Get synchronized policies"""
    params = {
        'scope': scope,
        'limit': limit
    }
    if scope_id:
        params['scope_id'] = scope_id
    if search:
        params['search'] = search
    if policy_type:
        params['policy_type'] = policy_type
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/policies", params=params)
        response.raise_for_status()
        return response.json()

async def get_policy_details(api_url: str, policy_id: str) -> Dict:
    """Get detailed policy information"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/policies/{policy_id}")
        response.raise_for_status()
        return response.json()

async def check_sdk_updates(api_url: str, check_request: Dict) -> List[Dict]:
    """Check for SDK updates"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api_url}/sdk/check", json=check_request)
        response.raise_for_status()
        return response.json()

async def update_sdk_package(api_url: str, package_name: str, version: str, auto_deploy: bool) -> Dict:
    """Update SDK package"""
    params = {
        'target_version': version,
        'auto_deploy': auto_deploy
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(f"{api_url}/sdk/update/{package_name}", params=params)
        response.raise_for_status()
        return response.json()

async def get_sdk_versions(api_url: str) -> List[Dict]:
    """Get current SDK versions"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/sdk/versions")
        response.raise_for_status()
        return response.json()

async def monitor_sync_progress(api_url: str, scope: str, scope_id: Optional[str]):
    """Monitor sync progress"""
    for i in range(30):  # Check for up to 5 minutes
        await asyncio.sleep(10)
        try:
            status = await get_policy_sync_status(api_url, scope, scope_id)
            if status.get('status') != 'no_recent_sync':
                console.print("[green]✅ Sync completed[/green]")
                return
        except:
            pass
    
    console.print("[yellow]⏱️  Sync is taking longer than expected[/yellow]")

def display_policy_sync_status(status_data: Dict, format: str):
    """Display policy sync status"""
    if format == 'json':
        console.print(json.dumps(status_data, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(status_data, default_flow_style=False))
    else:
        if status_data.get('status') == 'no_recent_sync':
            console.print("[yellow]No recent sync data available[/yellow]")
            return
        
        table = Table(title="Policy Sync Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        results = status_data.get('results', [])
        
        status_counts = {}
        for result in results:
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        table.add_row("Last Sync", status_data.get('timestamp', 'Unknown'))
        table.add_row("Total Policies", str(len(results)))
        
        for status, count in status_counts.items():
            table.add_row(f"Policies {status.title()}", str(count))
        
        console.print(table)

def display_policies(policies: List[Dict], format: str):
    """Display policy list"""
    if format == 'json':
        console.print(json.dumps(policies, indent=2, default=str))
    elif format == 'yaml':
        console.print(yaml.dump(policies, default_flow_style=False))
    else:
        table = Table(title="Azure Policies")
        table.add_column("Name", style="cyan")
        table.add_column("Display Name", style="green")
        table.add_column("Type", style="blue")
        table.add_column("Updated", style="yellow")
        
        for policy in policies:
            table.add_row(
                policy.get('name', 'Unknown'),
                policy.get('display_name', 'Unknown'),
                policy.get('policy_type', 'Unknown'),
                policy.get('updated_at', 'Unknown')
            )
        
        console.print(table)

def display_policy_details(policy: Dict, format: str):
    """Display detailed policy information"""
    if format == 'json':
        console.print(json.dumps(policy, indent=2, default=str))
    elif format == 'yaml':
        console.print(yaml.dump(policy, default_flow_style=False))
    else:
        console.print(Panel.fit(f"📋 {policy.get('display_name', 'Unknown Policy')}", title="Policy Details"))
        
        details_table = Table()
        details_table.add_column("Property", style="cyan")
        details_table.add_column("Value", style="green")
        
        details_table.add_row("ID", policy.get('policy_id', 'Unknown'))
        details_table.add_row("Name", policy.get('name', 'Unknown'))
        details_table.add_row("Type", policy.get('policy_type', 'Unknown'))
        details_table.add_row("Mode", policy.get('mode', 'Unknown'))
        details_table.add_row("Version", policy.get('version', 'Unknown'))
        details_table.add_row("Description", policy.get('description', 'No description'))
        details_table.add_row("Updated", policy.get('updated_at', 'Unknown'))
        
        console.print(details_table)

def display_sdk_updates(updates: List[Dict], format: str):
    """Display SDK update information"""
    if format == 'json':
        console.print(json.dumps(updates, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(updates, default_flow_style=False))
    else:
        if not updates:
            console.print("[green]✅ All packages are up to date[/green]")
            return
        
        table = Table(title="SDK Update Status")
        table.add_column("Package", style="cyan")
        table.add_column("Current", style="blue")
        table.add_column("Latest", style="green")
        table.add_column("Priority", style="yellow")
        table.add_column("Security", style="red")
        
        for update in updates:
            priority = update.get('update_priority', 'low')
            priority_color = {
                'critical': 'red',
                'high': 'orange',
                'medium': 'yellow',
                'low': 'green'
            }.get(priority, 'white')
            
            security_icon = "🔒" if update.get('security_update') else ""
            
            table.add_row(
                update.get('package_name', 'Unknown'),
                update.get('current_version', 'Unknown'),
                update.get('latest_version', 'Unknown'),
                f"[{priority_color}]{priority}[/{priority_color}]",
                security_icon
            )
        
        console.print(table)

def display_sdk_versions(versions: List[Dict], format: str):
    """Display current SDK versions"""
    if format == 'json':
        console.print(json.dumps(versions, indent=2, default=str))
    elif format == 'yaml':
        console.print(yaml.dump(versions, default_flow_style=False))
    else:
        table = Table(title="Current SDK Versions")
        table.add_column("Package", style="cyan")
        table.add_column("Current", style="blue")
        table.add_column("Latest", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Last Checked", style="gray50")
        
        for version in versions:
            if version.get('update_available'):
                status = "[yellow]Update Available[/yellow]"
                if version.get('security_update'):
                    status = "[red]Security Update[/red]"
            else:
                status = "[green]Up to Date[/green]"
            
            table.add_row(
                version.get('package_name', 'Unknown'),
                version.get('current_version', 'Unknown'),
                version.get('latest_version', 'Unknown'),
                status,
                version.get('last_checked', 'Unknown')
            )
        
        console.print(table)

def display_update_result(result: Dict, format: str):
    """Display SDK update result"""
    if format == 'json':
        console.print(json.dumps(result, indent=2, default=str))
    elif format == 'yaml':
        console.print(yaml.dump(result, default_flow_style=False))
    else:
        status = result.get('status', 'unknown')
        package = result.get('package_name', 'unknown')
        from_version = result.get('from_version', 'unknown')
        to_version = result.get('to_version', 'unknown')
        
        if status == 'completed':
            console.print(f"[green]✅ Successfully updated {package} from {from_version} to {to_version}[/green]")
        else:
            console.print(f"[red]❌ Failed to update {package}: {result.get('error_message', 'Unknown error')}[/red]")
        
        if result.get('steps'):
            console.print("\n[bold]Update Steps:[/bold]")
            for step in result['steps']:
                console.print(f"  ✓ {step.replace('_', ' ').title()}")

async def show_dashboard(api_url: str):
    """Show comprehensive sync dashboard"""
    console.print(Panel.fit("🔄 Azure Sync Dashboard", title="PolicyCortex"))
    
    # Get policy sync status
    try:
        policy_status = await get_policy_sync_status(api_url, 'subscription')
        
        policy_tree = Tree("📋 Policy Sync Status")
        if policy_status.get('status') != 'no_recent_sync':
            results = policy_status.get('results', [])
            status_counts = {}
            for result in results:
                status = result.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            policy_tree.add(f"Last Sync: {policy_status.get('timestamp', 'Unknown')}")
            policy_tree.add(f"Total Policies: {len(results)}")
            for status, count in status_counts.items():
                policy_tree.add(f"{status.title()}: {count}")
        else:
            policy_tree.add("No recent sync data")
        
        console.print(policy_tree)
        
    except Exception as e:
        console.print(f"[red]Failed to get policy status: {e}[/red]")
    
    console.print()
    
    # Get SDK versions
    try:
        sdk_versions = await get_sdk_versions(api_url)
        
        sdk_tree = Tree("📦 SDK Status")
        update_counts = {'up_to_date': 0, 'updates_available': 0, 'security_updates': 0}
        
        for version in sdk_versions:
            if version.get('security_update'):
                update_counts['security_updates'] += 1
            elif version.get('update_available'):
                update_counts['updates_available'] += 1
            else:
                update_counts['up_to_date'] += 1
        
        sdk_tree.add(f"Total Packages: {len(sdk_versions)}")
        sdk_tree.add(f"[green]Up to Date: {update_counts['up_to_date']}[/green]")
        sdk_tree.add(f"[yellow]Updates Available: {update_counts['updates_available']}[/yellow]")
        sdk_tree.add(f"[red]Security Updates: {update_counts['security_updates']}[/red]")
        
        console.print(sdk_tree)
        
    except Exception as e:
        console.print(f"[red]Failed to get SDK status: {e}[/red]")

if __name__ == '__main__':
    # Handle async CLI
    import sys
    import inspect
    
    # Patch click to handle async commands
    original_invoke = click.Context.invoke
    
    def patched_invoke(self, callback, *args, **kwargs):
        if inspect.iscoroutinefunction(callback):
            return asyncio.run(callback(*args, **kwargs))
        return original_invoke(self, callback, *args, **kwargs)
    
    click.Context.invoke = patched_invoke
    
    cli()