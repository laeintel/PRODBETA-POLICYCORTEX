/**
 * Test Data Factory
 * Generates consistent test data for all testing scenarios
 */

import { faker } from '@faker-js/faker';
import { v4 as uuidv4 } from 'uuid';

// Types
interface Resource {
  id: string;
  name: string;
  type: string;
  location: string;
  tags: Record<string, string>;
  status: 'running' | 'stopped' | 'deleted';
  createdAt: Date;
  updatedAt: Date;
}

interface Policy {
  id: string;
  name: string;
  description: string;
  type: 'preventive' | 'detective' | 'corrective';
  rules: PolicyRule[];
  effect: 'allow' | 'deny' | 'audit';
  scope: string[];
  enabled: boolean;
}

interface PolicyRule {
  field: string;
  operator: 'equals' | 'notEquals' | 'contains' | 'in' | 'notIn';
  value: any;
}

interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'readonly';
  tenantId: string;
  permissions: string[];
  mfaEnabled: boolean;
}

interface Incident {
  id: string;
  title: string;
  description: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  status: 'open' | 'investigating' | 'resolved' | 'closed';
  assignee?: string;
  createdAt: Date;
  resolvedAt?: Date;
}

interface Metric {
  name: string;
  value: number;
  unit: string;
  timestamp: Date;
  tags: Record<string, string>;
  aggregation: 'sum' | 'avg' | 'min' | 'max' | 'count';
}

interface ComplianceReport {
  framework: string;
  score: number;
  totalControls: number;
  passedControls: number;
  failedControls: number;
  findings: ComplianceFinding[];
}

interface ComplianceFinding {
  controlId: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  resourceId: string;
  description: string;
  recommendation: string;
}

// Base Factory Class
class BaseFactory<T> {
  private defaults: Partial<T> = {};
  
  withDefaults(defaults: Partial<T>): this {
    this.defaults = { ...this.defaults, ...defaults };
    return this;
  }
  
  build(overrides: Partial<T> = {}): T {
    return { ...this.generate(), ...this.defaults, ...overrides } as T;
  }
  
  buildMany(count: number, overrides: Partial<T> = {}): T[] {
    return Array.from({ length: count }, () => this.build(overrides));
  }
  
  protected generate(): T {
    throw new Error('generate() must be implemented by subclass');
  }
}

// Resource Factory
export class ResourceFactory extends BaseFactory<Resource> {
  private static resourceTypes = [
    'Microsoft.Compute/virtualMachines',
    'Microsoft.Storage/storageAccounts',
    'Microsoft.Network/virtualNetworks',
    'Microsoft.Sql/servers',
    'Microsoft.Web/sites',
    'Microsoft.ContainerService/managedClusters'
  ];
  
  private static locations = ['eastus', 'westus', 'northeurope', 'westeurope', 'southeastasia'];
  
  protected generate(): Resource {
    const type = faker.helpers.arrayElement(ResourceFactory.resourceTypes);
    
    return {
      id: uuidv4(),
      name: `${faker.word.adjective()}-${faker.word.noun()}-${faker.number.int({ min: 100, max: 999 })}`,
      type,
      location: faker.helpers.arrayElement(ResourceFactory.locations),
      tags: this.generateTags(),
      status: faker.helpers.arrayElement(['running', 'stopped', 'deleted']),
      createdAt: faker.date.past(),
      updatedAt: faker.date.recent()
    };
  }
  
  private generateTags(): Record<string, string> {
    const tags: Record<string, string> = {};
    const tagCount = faker.number.int({ min: 1, max: 5 });
    
    for (let i = 0; i < tagCount; i++) {
      const key = faker.helpers.arrayElement(['environment', 'owner', 'project', 'costCenter', 'department']);
      const value = faker.helpers.arrayElement(['production', 'staging', 'dev', 'test', 'qa']);
      tags[key] = value;
    }
    
    return tags;
  }
  
  // Specialized builders
  vm(overrides?: Partial<Resource>): Resource {
    return this.build({
      type: 'Microsoft.Compute/virtualMachines',
      ...overrides
    });
  }
  
  storage(overrides?: Partial<Resource>): Resource {
    return this.build({
      type: 'Microsoft.Storage/storageAccounts',
      ...overrides
    });
  }
  
  compliant(overrides?: Partial<Resource>): Resource {
    return this.build({
      tags: {
        environment: 'production',
        owner: 'team-a',
        costCenter: 'cc-123',
        ...overrides?.tags
      },
      status: 'running',
      ...overrides
    });
  }
  
  nonCompliant(overrides?: Partial<Resource>): Resource {
    return this.build({
      tags: {}, // Missing required tags
      ...overrides
    });
  }
}

// Policy Factory
export class PolicyFactory extends BaseFactory<Policy> {
  protected generate(): Policy {
    return {
      id: uuidv4(),
      name: `${faker.word.adjective()}-${faker.word.noun()}-policy`,
      description: faker.lorem.sentence(),
      type: faker.helpers.arrayElement(['preventive', 'detective', 'corrective']),
      rules: this.generateRules(),
      effect: faker.helpers.arrayElement(['allow', 'deny', 'audit']),
      scope: this.generateScope(),
      enabled: faker.datatype.boolean()
    };
  }
  
  private generateRules(): PolicyRule[] {
    const ruleCount = faker.number.int({ min: 1, max: 5 });
    const rules: PolicyRule[] = [];
    
    for (let i = 0; i < ruleCount; i++) {
      rules.push({
        field: faker.helpers.arrayElement(['tags.environment', 'location', 'type', 'sku']),
        operator: faker.helpers.arrayElement(['equals', 'notEquals', 'contains', 'in', 'notIn']),
        value: faker.helpers.arrayElement(['production', 'eastus', 'Standard_D2s_v3'])
      });
    }
    
    return rules;
  }
  
  private generateScope(): string[] {
    return [
      `/subscriptions/${uuidv4()}`,
      `/subscriptions/${uuidv4()}/resourceGroups/${faker.word.noun()}`
    ];
  }
  
  // Specialized builders
  tagPolicy(overrides?: Partial<Policy>): Policy {
    return this.build({
      name: 'require-tags-policy',
      type: 'preventive',
      rules: [
        { field: 'tags.environment', operator: 'notEquals', value: null },
        { field: 'tags.owner', operator: 'notEquals', value: null }
      ],
      effect: 'deny',
      ...overrides
    });
  }
  
  locationPolicy(overrides?: Partial<Policy>): Policy {
    return this.build({
      name: 'allowed-locations-policy',
      type: 'preventive',
      rules: [
        { field: 'location', operator: 'in', value: ['eastus', 'westus'] }
      ],
      effect: 'deny',
      ...overrides
    });
  }
}

// User Factory
export class UserFactory extends BaseFactory<User> {
  protected generate(): User {
    const firstName = faker.person.firstName();
    const lastName = faker.person.lastName();
    const role = faker.helpers.arrayElement(['admin', 'user', 'readonly']);
    
    return {
      id: uuidv4(),
      email: faker.internet.email({ firstName, lastName }).toLowerCase(),
      name: `${firstName} ${lastName}`,
      role,
      tenantId: uuidv4(),
      permissions: this.getPermissionsForRole(role),
      mfaEnabled: faker.datatype.boolean()
    };
  }
  
  private getPermissionsForRole(role: string): string[] {
    switch (role) {
      case 'admin':
        return ['read', 'write', 'delete', 'admin'];
      case 'user':
        return ['read', 'write'];
      case 'readonly':
        return ['read'];
      default:
        return [];
    }
  }
  
  // Specialized builders
  admin(overrides?: Partial<User>): User {
    return this.build({
      role: 'admin',
      permissions: ['read', 'write', 'delete', 'admin'],
      mfaEnabled: true,
      ...overrides
    });
  }
  
  readonlyUser(overrides?: Partial<User>): User {
    return this.build({
      role: 'readonly',
      permissions: ['read'],
      ...overrides
    });
  }
}

// Incident Factory
export class IncidentFactory extends BaseFactory<Incident> {
  protected generate(): Incident {
    const createdAt = faker.date.past();
    const isResolved = faker.datatype.boolean();
    
    return {
      id: uuidv4(),
      title: faker.lorem.sentence(),
      description: faker.lorem.paragraph(),
      priority: faker.helpers.arrayElement(['critical', 'high', 'medium', 'low']),
      status: faker.helpers.arrayElement(['open', 'investigating', 'resolved', 'closed']),
      assignee: faker.datatype.boolean() ? uuidv4() : undefined,
      createdAt,
      resolvedAt: isResolved ? faker.date.between({ from: createdAt, to: new Date() }) : undefined
    };
  }
  
  // Specialized builders
  critical(overrides?: Partial<Incident>): Incident {
    return this.build({
      priority: 'critical',
      status: 'open',
      ...overrides
    });
  }
  
  resolved(overrides?: Partial<Incident>): Incident {
    const createdAt = faker.date.past();
    
    return this.build({
      status: 'resolved',
      resolvedAt: faker.date.between({ from: createdAt, to: new Date() }),
      ...overrides
    });
  }
}

// Metric Factory
export class MetricFactory extends BaseFactory<Metric> {
  protected generate(): Metric {
    return {
      name: faker.helpers.arrayElement([
        'cpu_utilization',
        'memory_usage',
        'disk_io',
        'network_throughput',
        'request_count',
        'error_rate'
      ]),
      value: faker.number.float({ min: 0, max: 100, fractionDigits: 2 }),
      unit: faker.helpers.arrayElement(['percent', 'bytes', 'count', 'ms']),
      timestamp: faker.date.recent(),
      tags: {
        resource: uuidv4(),
        region: faker.helpers.arrayElement(['us-east-1', 'us-west-2', 'eu-west-1'])
      },
      aggregation: faker.helpers.arrayElement(['sum', 'avg', 'min', 'max', 'count'])
    };
  }
  
  // Specialized builders
  cpuMetric(overrides?: Partial<Metric>): Metric {
    return this.build({
      name: 'cpu_utilization',
      unit: 'percent',
      value: faker.number.float({ min: 0, max: 100, fractionDigits: 2 }),
      ...overrides
    });
  }
  
  errorMetric(overrides?: Partial<Metric>): Metric {
    return this.build({
      name: 'error_rate',
      unit: 'count',
      value: faker.number.int({ min: 0, max: 100 }),
      ...overrides
    });
  }
}

// Compliance Factory
export class ComplianceFactory extends BaseFactory<ComplianceReport> {
  protected generate(): ComplianceReport {
    const totalControls = faker.number.int({ min: 50, max: 200 });
    const passedControls = faker.number.int({ min: 0, max: totalControls });
    const failedControls = totalControls - passedControls;
    
    return {
      framework: faker.helpers.arrayElement(['SOC2', 'ISO27001', 'HIPAA', 'PCI-DSS', 'GDPR']),
      score: (passedControls / totalControls) * 100,
      totalControls,
      passedControls,
      failedControls,
      findings: this.generateFindings(failedControls)
    };
  }
  
  private generateFindings(count: number): ComplianceFinding[] {
    const findings: ComplianceFinding[] = [];
    const maxFindings = Math.min(count, 10); // Limit to 10 findings for performance
    
    for (let i = 0; i < maxFindings; i++) {
      findings.push({
        controlId: `CTRL-${faker.number.int({ min: 100, max: 999 })}`,
        severity: faker.helpers.arrayElement(['critical', 'high', 'medium', 'low']),
        resourceId: uuidv4(),
        description: faker.lorem.sentence(),
        recommendation: faker.lorem.sentence()
      });
    }
    
    return findings;
  }
  
  // Specialized builders
  compliantReport(overrides?: Partial<ComplianceReport>): ComplianceReport {
    const totalControls = 100;
    const passedControls = faker.number.int({ min: 90, max: 100 });
    
    return this.build({
      score: (passedControls / totalControls) * 100,
      totalControls,
      passedControls,
      failedControls: totalControls - passedControls,
      findings: [],
      ...overrides
    });
  }
  
  nonCompliantReport(overrides?: Partial<ComplianceReport>): ComplianceReport {
    const totalControls = 100;
    const passedControls = faker.number.int({ min: 0, max: 50 });
    
    return this.build({
      score: (passedControls / totalControls) * 100,
      totalControls,
      passedControls,
      failedControls: totalControls - passedControls,
      ...overrides
    });
  }
}

// Main Factory Export
export class TestDataFactory {
  static resource = new ResourceFactory();
  static policy = new PolicyFactory();
  static user = new UserFactory();
  static incident = new IncidentFactory();
  static metric = new MetricFactory();
  static compliance = new ComplianceFactory();
  
  // Utility method to seed database with test data
  static async seedDatabase(options: {
    resources?: number;
    policies?: number;
    users?: number;
    incidents?: number;
    metrics?: number;
  } = {}) {
    const data = {
      resources: this.resource.buildMany(options.resources || 10),
      policies: this.policy.buildMany(options.policies || 5),
      users: this.user.buildMany(options.users || 3),
      incidents: this.incident.buildMany(options.incidents || 5),
      metrics: this.metric.buildMany(options.metrics || 100)
    };
    
    // In a real implementation, this would save to database
    console.log(`Seeded database with:
      - ${data.resources.length} resources
      - ${data.policies.length} policies
      - ${data.users.length} users
      - ${data.incidents.length} incidents
      - ${data.metrics.length} metrics
    `);
    
    return data;
  }
  
  // Generate realistic scenarios
  static generateScenarios() {
    return {
      // Compliance violation scenario
      complianceViolation: {
        resource: this.resource.nonCompliant(),
        policy: this.policy.tagPolicy(),
        incident: this.incident.critical({
          title: 'Resource missing required tags',
          description: 'Production resource detected without required tags'
        })
      },
      
      // High CPU usage scenario
      highCpuUsage: {
        resource: this.resource.vm(),
        metrics: Array.from({ length: 10 }, () => 
          this.metric.cpuMetric({ value: faker.number.float({ min: 85, max: 100 }) })
        ),
        incident: this.incident.build({
          title: 'High CPU utilization detected',
          priority: 'high'
        })
      },
      
      // Security breach scenario
      securityBreach: {
        user: this.user.build({ mfaEnabled: false }),
        incident: this.incident.critical({
          title: 'Suspicious login activity detected',
          description: 'Multiple failed login attempts from unknown location'
        }),
        metrics: [
          this.metric.errorMetric({ name: 'failed_login_attempts', value: 25 })
        ]
      }
    };
  }
}

// Export individual factories for convenience
export const resourceFactory = TestDataFactory.resource;
export const policyFactory = TestDataFactory.policy;
export const userFactory = TestDataFactory.user;
export const incidentFactory = TestDataFactory.incident;
export const metricFactory = TestDataFactory.metric;
export const complianceFactory = TestDataFactory.compliance;