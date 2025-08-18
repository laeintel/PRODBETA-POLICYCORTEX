#!/usr/bin/env python3
"""
PolicyCortex What-If Simulation CLI
Interactive command-line tool for governance scenario analysis
"""

import asyncio
import json
import sys
import argparse
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
import httpx
import neo4j

console = Console()

@dataclass
class ScenarioChange:
    """Represents a single change in a what-if scenario"""
    target_type: str  # 'resource', 'policy', 'configuration'
    target_id: str
    property_path: str
    old_value: Any
    new_value: Any
    change_type: str  # 'update', 'add', 'remove'

@dataclass
class WhatIfScenario:
    """Complete what-if scenario definition"""
    scenario_id: str
    name: str
    description: str
    changes: List[ScenarioChange]
    constraints: List[Dict[str, Any]]
    duration_hours: float = 24.0
    tenant_id: Optional[str] = None

@dataclass
class SimulationResult:
    """Results from a what-if simulation"""
    scenario_id: str
    impact_score: float
    risk_score: float
    compliance_impact: float
    cost_impact: float
    affected_resources: List[str]
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    execution_time: float

class WhatIfCLI:
    """Main CLI interface for what-if analysis"""
    
    def __init__(self, api_base_url: str = "http://localhost:8080"):
        self.api_base_url = api_base_url
        self.graph_driver = None
        self.current_scenario = None
        
    async def initialize(self):
        """Initialize connections"""
        try:
            # Test API connection
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base_url}/health")
                if response.status_code != 200:
                    console.print("[red]❌ Cannot connect to PolicyCortex API[/red]")
                    sys.exit(1)
            
            console.print("[green]✅ Connected to PolicyCortex API[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Connection failed: {e}[/red]")
            sys.exit(1)

@click.group()
@click.option('--api-url', default='http://localhost:8080', help='PolicyCortex API URL')
@click.option('--format', type=click.Choice(['json', 'yaml', 'table']), default='table', help='Output format')
@click.pass_context
def cli(ctx, api_url, format):
    """PolicyCortex What-If Simulation CLI"""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url
    ctx.obj['format'] = format
    ctx.obj['cli'] = WhatIfCLI(api_url)

@cli.command()
@click.option('--scenario-file', type=click.Path(exists=True), help='Load scenario from file')
@click.option('--interactive', is_flag=True, help='Interactive scenario builder')
@click.pass_context
async def create(ctx, scenario_file, interactive):
    """Create a new what-if scenario"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    if scenario_file:
        scenario = await load_scenario_from_file(scenario_file)
    elif interactive:
        scenario = await interactive_scenario_builder()
    else:
        console.print("[yellow]Please specify --scenario-file or --interactive[/yellow]")
        return
    
    # Save scenario
    await save_scenario(cli_instance.api_base_url, scenario)
    console.print(f"[green]✅ Scenario '{scenario.name}' created successfully[/green]")

@cli.command()
@click.argument('scenario_id')
@click.option('--save-results', type=click.Path(), help='Save results to file')
@click.option('--detailed', is_flag=True, help='Show detailed results')
@click.pass_context
async def simulate(ctx, scenario_id, save_results, detailed):
    """Run a what-if simulation"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running simulation...", total=None)
        
        try:
            result = await run_simulation(cli_instance.api_base_url, scenario_id)
            progress.update(task, description="✅ Simulation complete")
            
            # Display results
            display_simulation_results(result, detailed, ctx.obj['format'])
            
            # Save results if requested
            if save_results:
                await save_results_to_file(result, save_results)
                console.print(f"[green]Results saved to {save_results}[/green]")
                
        except Exception as e:
            progress.update(task, description="❌ Simulation failed")
            console.print(f"[red]Simulation failed: {e}[/red]")

@cli.command()
@click.option('--tenant-id', help='Filter by tenant')
@click.option('--status', type=click.Choice(['active', 'completed', 'failed']), help='Filter by status')
@click.pass_context
async def list_scenarios(ctx, tenant_id, status):
    """List all what-if scenarios"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    scenarios = await get_scenarios(cli_instance.api_base_url, tenant_id, status)
    
    if ctx.obj['format'] == 'table':
        display_scenarios_table(scenarios)
    else:
        output = scenarios if ctx.obj['format'] == 'json' else yaml.dump(scenarios)
        console.print(output)

@cli.command()
@click.argument('scenario_ids', nargs=-1, required=True)
@click.option('--criteria', type=click.Choice(['impact', 'risk', 'cost', 'compliance']), default='impact')
@click.pass_context
async def compare(ctx, scenario_ids, criteria):
    """Compare multiple scenarios"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    results = []
    for scenario_id in scenario_ids:
        result = await get_simulation_result(cli_instance.api_base_url, scenario_id)
        results.append(result)
    
    comparison = compare_scenarios(results, criteria)
    display_comparison_results(comparison, ctx.obj['format'])

@cli.command()
@click.argument('scenario_id')
@click.option('--force', is_flag=True, help='Force delete without confirmation')
@click.pass_context
async def delete(ctx, scenario_id, force):
    """Delete a scenario"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    if not force:
        if not Confirm.ask(f"Delete scenario {scenario_id}?"):
            return
    
    await delete_scenario(cli_instance.api_base_url, scenario_id)
    console.print(f"[green]✅ Scenario {scenario_id} deleted[/green]")

@cli.command()
@click.argument('scenario_id')
@click.pass_context
async def rollback(ctx, scenario_id):
    """Rollback a simulation"""
    cli_instance = ctx.obj['cli']
    await cli_instance.initialize()
    
    await rollback_simulation(cli_instance.api_base_url, scenario_id)
    console.print(f"[green]✅ Simulation {scenario_id} rolled back[/green]")

@cli.command()
@click.option('--output-file', type=click.Path(), help='Export to file')
@click.pass_context
async def export_template(ctx, output_file):
    """Export scenario template"""
    template = {
        "scenario_id": "example-scenario",
        "name": "Example What-If Scenario",
        "description": "Sample scenario for testing policy changes",
        "changes": [
            {
                "target_type": "policy",
                "target_id": "policy-001",
                "property_path": "parameters.allowedLocations",
                "old_value": ["eastus", "westus"],
                "new_value": ["eastus", "westus", "centralus"],
                "change_type": "update"
            }
        ],
        "constraints": [
            {
                "type": "compliance_threshold",
                "threshold": 0.95,
                "operator": "gte"
            }
        ],
        "duration_hours": 24.0
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            if output_file.endswith('.yaml') or output_file.endswith('.yml'):
                yaml.dump(template, f, default_flow_style=False)
            else:
                json.dump(template, f, indent=2)
        console.print(f"[green]Template exported to {output_file}[/green]")
    else:
        console.print(json.dumps(template, indent=2))

# Helper functions

async def load_scenario_from_file(file_path: str) -> WhatIfScenario:
    """Load scenario from YAML or JSON file"""
    with open(file_path, 'r') as f:
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    changes = [ScenarioChange(**change) for change in data['changes']]
    return WhatIfScenario(
        scenario_id=data['scenario_id'],
        name=data['name'],
        description=data['description'],
        changes=changes,
        constraints=data.get('constraints', []),
        duration_hours=data.get('duration_hours', 24.0),
        tenant_id=data.get('tenant_id')
    )

async def interactive_scenario_builder() -> WhatIfScenario:
    """Interactive scenario builder"""
    console.print(Panel.fit("🔮 What-If Scenario Builder", title="PolicyCortex"))
    
    scenario_id = Prompt.ask("Scenario ID")
    name = Prompt.ask("Scenario name")
    description = Prompt.ask("Description")
    
    changes = []
    while True:
        console.print("\n[bold]Add a change:[/bold]")
        target_type = Prompt.ask("Target type", choices=['resource', 'policy', 'configuration'])
        target_id = Prompt.ask("Target ID")
        property_path = Prompt.ask("Property path")
        old_value = Prompt.ask("Current value")
        new_value = Prompt.ask("New value")
        change_type = Prompt.ask("Change type", choices=['update', 'add', 'remove'], default='update')
        
        changes.append(ScenarioChange(
            target_type=target_type,
            target_id=target_id,
            property_path=property_path,
            old_value=old_value,
            new_value=new_value,
            change_type=change_type
        ))
        
        if not Confirm.ask("Add another change?"):
            break
    
    return WhatIfScenario(
        scenario_id=scenario_id,
        name=name,
        description=description,
        changes=changes,
        constraints=[]
    )

async def save_scenario(api_url: str, scenario: WhatIfScenario):
    """Save scenario to API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{api_url}/api/v1/what-if/scenarios",
            json=asdict(scenario)
        )
        response.raise_for_status()

async def run_simulation(api_url: str, scenario_id: str) -> SimulationResult:
    """Run what-if simulation"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(f"{api_url}/api/v1/what-if/scenarios/{scenario_id}/simulate")
        response.raise_for_status()
        data = response.json()
        return SimulationResult(**data)

async def get_scenarios(api_url: str, tenant_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict]:
    """Get list of scenarios"""
    params = {}
    if tenant_id:
        params['tenant_id'] = tenant_id
    if status:
        params['status'] = status
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/api/v1/what-if/scenarios", params=params)
        response.raise_for_status()
        return response.json()

async def get_simulation_result(api_url: str, scenario_id: str) -> SimulationResult:
    """Get simulation result"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{api_url}/api/v1/what-if/scenarios/{scenario_id}/result")
        response.raise_for_status()
        data = response.json()
        return SimulationResult(**data)

async def delete_scenario(api_url: str, scenario_id: str):
    """Delete scenario"""
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{api_url}/api/v1/what-if/scenarios/{scenario_id}")
        response.raise_for_status()

async def rollback_simulation(api_url: str, scenario_id: str):
    """Rollback simulation"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{api_url}/api/v1/what-if/scenarios/{scenario_id}/rollback")
        response.raise_for_status()

def display_simulation_results(result: SimulationResult, detailed: bool, format: str):
    """Display simulation results"""
    if format == 'json':
        console.print(json.dumps(asdict(result), indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(asdict(result), default_flow_style=False))
    else:
        # Table format
        table = Table(title=f"Simulation Results: {result.scenario_id}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Impact Score", f"{result.impact_score:.2f}")
        table.add_row("Risk Score", f"{result.risk_score:.2f}")
        table.add_row("Compliance Impact", f"{result.compliance_impact:.2f}")
        table.add_row("Cost Impact", f"${result.cost_impact:,.2f}")
        table.add_row("Affected Resources", str(len(result.affected_resources)))
        table.add_row("Violations", str(len(result.violations)))
        table.add_row("Execution Time", f"{result.execution_time:.2f}s")
        
        console.print(table)
        
        if detailed:
            if result.violations:
                console.print("\n[bold red]Violations:[/bold red]")
                for violation in result.violations:
                    console.print(f"  • {violation.get('message', 'Unknown violation')}")
            
            if result.recommendations:
                console.print("\n[bold blue]Recommendations:[/bold blue]")
                for rec in result.recommendations:
                    console.print(f"  • {rec}")

def display_scenarios_table(scenarios: List[Dict]):
    """Display scenarios in table format"""
    table = Table(title="What-If Scenarios")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Created", style="blue")
    
    for scenario in scenarios:
        table.add_row(
            scenario['scenario_id'],
            scenario['name'],
            scenario.get('status', 'unknown'),
            scenario.get('created_at', 'unknown')
        )
    
    console.print(table)

def compare_scenarios(results: List[SimulationResult], criteria: str) -> Dict:
    """Compare multiple scenarios"""
    comparison = {
        'criteria': criteria,
        'scenarios': [],
        'best_scenario': None,
        'ranking': []
    }
    
    for result in results:
        score = getattr(result, f'{criteria}_score', 0) if hasattr(result, f'{criteria}_score') else 0
        comparison['scenarios'].append({
            'scenario_id': result.scenario_id,
            'score': score,
            'impact_score': result.impact_score,
            'risk_score': result.risk_score,
            'compliance_impact': result.compliance_impact,
            'cost_impact': result.cost_impact
        })
    
    # Sort by criteria (lower is better for risk and cost, higher for others)
    reverse = criteria not in ['risk', 'cost']
    sorted_scenarios = sorted(comparison['scenarios'], 
                            key=lambda x: x['score'], reverse=reverse)
    
    comparison['ranking'] = sorted_scenarios
    comparison['best_scenario'] = sorted_scenarios[0]['scenario_id']
    
    return comparison

def display_comparison_results(comparison: Dict, format: str):
    """Display scenario comparison results"""
    if format == 'json':
        console.print(json.dumps(comparison, indent=2))
    elif format == 'yaml':
        console.print(yaml.dump(comparison, default_flow_style=False))
    else:
        table = Table(title=f"Scenario Comparison ({comparison['criteria']})")
        table.add_column("Rank", style="cyan")
        table.add_column("Scenario ID", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Impact", style="blue")
        table.add_column("Risk", style="red")
        
        for i, scenario in enumerate(comparison['ranking'], 1):
            rank_style = "bold green" if i == 1 else ""
            table.add_row(
                f"#{i}",
                scenario['scenario_id'],
                f"{scenario['score']:.2f}",
                f"{scenario['impact_score']:.2f}",
                f"{scenario['risk_score']:.2f}",
                style=rank_style
            )
        
        console.print(table)
        console.print(f"\n[bold green]Best scenario: {comparison['best_scenario']}[/bold green]")

async def save_results_to_file(result: SimulationResult, file_path: str):
    """Save results to file"""
    data = asdict(result)
    with open(file_path, 'w') as f:
        if file_path.endswith('.yaml') or file_path.endswith('.yml'):
            yaml.dump(data, f, default_flow_style=False)
        else:
            json.dump(data, f, indent=2)

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