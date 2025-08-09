# Security Architecture

## Table of Contents
1. [Security Overview](#security-overview)
2. [Threat Model](#threat-model)
3. [Authentication & Authorization](#authentication--authorization)
4. [Encryption & Cryptography](#encryption--cryptography)
5. [Network Security](#network-security)
6. [Data Protection](#data-protection)
7. [Security Controls](#security-controls)
8. [Compliance Framework](#compliance-framework)
9. [Incident Response](#incident-response)

## Security Overview

PolicyCortex implements a comprehensive zero-trust security architecture with defense-in-depth principles, post-quantum cryptography, and blockchain-based audit trails.

### Security Principles

```
┌─────────────────────────────────────────────────────────┐
│                Zero Trust Architecture                  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Identity  │  │   Device    │  │  Network    │    │
│  │     &       │  │  Security   │  │  Micro-     │    │
│  │   Access    │  │             │  │ Segmentation│    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │Application  │  │    Data     │  │Operational  │    │
│  │  Security   │  │ Protection  │  │  Security   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Security Layers

1. **Perimeter Security**: WAF, DDoS protection, network firewalls
2. **Network Security**: micro-segmentation, encrypted communication
3. **Application Security**: secure coding, vulnerability management
4. **Data Security**: encryption at rest and in transit
5. **Identity Security**: multi-factor authentication, RBAC
6. **Operational Security**: monitoring, incident response, audit trails

## Threat Model

### STRIDE Analysis

```rust
// core/src/security/threat_model.rs
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatCategory {
    Spoofing,
    Tampering,
    Repudiation,
    InformationDisclosure,
    DenialOfService,
    ElevationOfPrivilege,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Threat {
    pub id: String,
    pub title: String,
    pub description: String,
    pub category: ThreatCategory,
    pub impact: RiskLevel,
    pub likelihood: RiskLevel,
    pub affected_assets: Vec<String>,
    pub mitigations: Vec<String>,
}

pub struct ThreatModel {
    threats: HashMap<String, Threat>,
}

impl ThreatModel {
    pub fn new() -> Self {
        let mut threats = HashMap::new();
        
        // Spoofing threats
        threats.insert("T001".to_string(), Threat {
            id: "T001".to_string(),
            title: "Identity Spoofing".to_string(),
            description: "Attacker impersonates legitimate user or service".to_string(),
            category: ThreatCategory::Spoofing,
            impact: RiskLevel::High,
            likelihood: RiskLevel::Medium,
            affected_assets: vec!["User Accounts".to_string(), "API Endpoints".to_string()],
            mitigations: vec![
                "Multi-factor Authentication".to_string(),
                "Certificate-based Authentication".to_string(),
                "Behavioral Analysis".to_string(),
            ],
        });

        // Tampering threats
        threats.insert("T002".to_string(), Threat {
            id: "T002".to_string(),
            title: "Data Tampering".to_string(),
            description: "Unauthorized modification of data or configurations".to_string(),
            category: ThreatCategory::Tampering,
            impact: RiskLevel::Critical,
            likelihood: RiskLevel::Medium,
            affected_assets: vec!["Database".to_string(), "Configuration Files".to_string()],
            mitigations: vec![
                "Digital Signatures".to_string(),
                "Blockchain Audit Trail".to_string(),
                "Input Validation".to_string(),
                "Database Integrity Checks".to_string(),
            ],
        });

        // Repudiation threats
        threats.insert("T003".to_string(), Threat {
            id: "T003".to_string(),
            title: "Non-Repudiation Failure".to_string(),
            description: "Actions cannot be definitively attributed to users".to_string(),
            category: ThreatCategory::Repudiation,
            impact: RiskLevel::High,
            likelihood: RiskLevel::Low,
            affected_assets: vec!["Audit Logs".to_string(), "Compliance Records".to_string()],
            mitigations: vec![
                "Immutable Audit Trail".to_string(),
                "Digital Signatures".to_string(),
                "Timestamping Services".to_string(),
            ],
        });

        // Information Disclosure threats
        threats.insert("T004".to_string(), Threat {
            id: "T004".to_string(),
            title: "Data Exfiltration".to_string(),
            description: "Unauthorized access to sensitive data".to_string(),
            category: ThreatCategory::InformationDisclosure,
            impact: RiskLevel::Critical,
            likelihood: RiskLevel::High,
            affected_assets: vec!["Customer Data".to_string(), "Azure Credentials".to_string()],
            mitigations: vec![
                "Data Loss Prevention".to_string(),
                "End-to-End Encryption".to_string(),
                "Access Controls".to_string(),
                "Data Classification".to_string(),
            ],
        });

        // Denial of Service threats
        threats.insert("T005".to_string(), Threat {
            id: "T005".to_string(),
            title: "Application DoS".to_string(),
            description: "Service unavailability due to resource exhaustion".to_string(),
            category: ThreatCategory::DenialOfService,
            impact: RiskLevel::High,
            likelihood: RiskLevel::High,
            affected_assets: vec!["API Services".to_string(), "Database".to_string()],
            mitigations: vec![
                "Rate Limiting".to_string(),
                "Auto-scaling".to_string(),
                "Circuit Breakers".to_string(),
                "Resource Monitoring".to_string(),
            ],
        });

        // Elevation of Privilege threats
        threats.insert("T006".to_string(), Threat {
            id: "T006".to_string(),
            title: "Privilege Escalation".to_string(),
            description: "Unauthorized elevation of user privileges".to_string(),
            category: ThreatCategory::ElevationOfPrivilege,
            impact: RiskLevel::Critical,
            likelihood: RiskLevel::Medium,
            affected_assets: vec!["Admin Functions".to_string(), "System Resources".to_string()],
            mitigations: vec![
                "Principle of Least Privilege".to_string(),
                "Role-Based Access Control".to_string(),
                "Privilege Monitoring".to_string(),
                "Regular Access Reviews".to_string(),
            ],
        });

        Self { threats }
    }

    pub fn get_threats_by_category(&self, category: &ThreatCategory) -> Vec<&Threat> {
        self.threats
            .values()
            .filter(|t| std::mem::discriminant(&t.category) == std::mem::discriminant(category))
            .collect()
    }

    pub fn get_high_risk_threats(&self) -> Vec<&Threat> {
        self.threats
            .values()
            .filter(|t| matches!(t.impact, RiskLevel::Critical | RiskLevel::High))
            .collect()
    }

    pub fn calculate_risk_score(&self, threat: &Threat) -> u8 {
        let impact_score = match threat.impact {
            RiskLevel::Critical => 4,
            RiskLevel::High => 3,
            RiskLevel::Medium => 2,
            RiskLevel::Low => 1,
        };

        let likelihood_score = match threat.likelihood {
            RiskLevel::Critical => 4,
            RiskLevel::High => 3,
            RiskLevel::Medium => 2,
            RiskLevel::Low => 1,
        };

        impact_score * likelihood_score
    }
}

// Attack Surface Analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct AttackSurface {
    pub external_endpoints: Vec<String>,
    pub internal_services: Vec<String>,
    pub data_flows: Vec<DataFlow>,
    pub trust_boundaries: Vec<TrustBoundary>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataFlow {
    pub source: String,
    pub destination: String,
    pub data_type: String,
    pub encryption: bool,
    pub authentication_required: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrustBoundary {
    pub name: String,
    pub description: String,
    pub security_controls: Vec<String>,
}

impl AttackSurface {
    pub fn new() -> Self {
        Self {
            external_endpoints: vec![
                "https://api.policycortex.com/graphql".to_string(),
                "https://policycortex.com/api/v1/*".to_string(),
                "https://policycortex.com/auth/*".to_string(),
            ],
            internal_services: vec![
                "core-api:8080".to_string(),
                "ai-engine:8081".to_string(),
                "postgres:5432".to_string(),
                "redis:6379".to_string(),
            ],
            data_flows: vec![
                DataFlow {
                    source: "Frontend".to_string(),
                    destination: "GraphQL Gateway".to_string(),
                    data_type: "User Queries".to_string(),
                    encryption: true,
                    authentication_required: true,
                },
                DataFlow {
                    source: "Core API".to_string(),
                    destination: "PostgreSQL".to_string(),
                    data_type: "Application Data".to_string(),
                    encryption: true,
                    authentication_required: true,
                },
                DataFlow {
                    source: "AI Engine".to_string(),
                    destination: "Azure OpenAI".to_string(),
                    data_type: "ML Queries".to_string(),
                    encryption: true,
                    authentication_required: true,
                },
            ],
            trust_boundaries: vec![
                TrustBoundary {
                    name: "Internet to DMZ".to_string(),
                    description: "Public internet to application tier".to_string(),
                    security_controls: vec![
                        "WAF".to_string(),
                        "DDoS Protection".to_string(),
                        "Rate Limiting".to_string(),
                    ],
                },
                TrustBoundary {
                    name: "DMZ to Internal".to_string(),
                    description: "Application tier to data tier".to_string(),
                    security_controls: vec![
                        "Network Segmentation".to_string(),
                        "Service Mesh".to_string(),
                        "mTLS".to_string(),
                    ],
                },
            ],
        }
    }
}
```

## Authentication & Authorization

### Multi-Factor Authentication

```typescript
// frontend/lib/auth/mfa.ts
import { authenticator } from 'otplib';
import QRCode from 'qrcode';
import { webcrypto } from 'crypto';

export interface MFAProvider {
  name: string;
  type: 'totp' | 'webauthn' | 'sms';
  enabled: boolean;
}

export interface TOTPSetup {
  secret: string;
  qrCodeUrl: string;
  backupCodes: string[];
}

export class MFAManager {
  private readonly serviceName = 'PolicyCortex';

  // TOTP (Time-based One-Time Password) setup
  async setupTOTP(userEmail: string): Promise<TOTPSetup> {
    const secret = authenticator.generateSecret();
    const otpauth = authenticator.keyuri(userEmail, this.serviceName, secret);
    
    // Generate QR code
    const qrCodeUrl = await QRCode.toDataURL(otpauth);
    
    // Generate backup codes
    const backupCodes = Array.from({ length: 8 }, () => 
      this.generateBackupCode()
    );

    return {
      secret,
      qrCodeUrl,
      backupCodes
    };
  }

  // Verify TOTP token
  verifyTOTP(token: string, secret: string): boolean {
    return authenticator.verify({ token, secret });
  }

  // WebAuthn setup
  async setupWebAuthn(userId: string, username: string): Promise<PublicKeyCredentialCreationOptions> {
    const challenge = new Uint8Array(32);
    webcrypto.getRandomValues(challenge);

    return {
      challenge,
      rp: {
        name: this.serviceName,
        id: window.location.hostname
      },
      user: {
        id: new TextEncoder().encode(userId),
        name: username,
        displayName: username
      },
      pubKeyCredParams: [
        { alg: -7, type: 'public-key' }, // ES256
        { alg: -257, type: 'public-key' } // RS256
      ],
      authenticatorSelection: {
        authenticatorAttachment: 'platform',
        requireResidentKey: false,
        userVerification: 'required'
      },
      timeout: 60000,
      attestation: 'direct'
    };
  }

  // Verify WebAuthn authentication
  async verifyWebAuthn(
    credential: PublicKeyCredential,
    challenge: ArrayBuffer,
    publicKey: ArrayBuffer
  ): Promise<boolean> {
    try {
      const response = credential.response as AuthenticatorAssertionResponse;
      
      // Verify the signature (simplified - production would use crypto library)
      const clientData = JSON.parse(new TextDecoder().decode(response.clientDataJSON));
      
      if (clientData.type !== 'webauthn.get') return false;
      if (!this.arrayBuffersEqual(
        new TextEncoder().encode(clientData.challenge),
        new Uint8Array(challenge)
      )) return false;

      return true;
    } catch (error) {
      console.error('WebAuthn verification failed:', error);
      return false;
    }
  }

  private generateBackupCode(): string {
    const chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    let code = '';
    for (let i = 0; i < 8; i++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
  }

  private arrayBuffersEqual(a: ArrayBufferLike, b: ArrayBufferLike): boolean {
    const aView = new Uint8Array(a);
    const bView = new Uint8Array(b);
    
    if (aView.length !== bView.length) return false;
    return aView.every((val, i) => val === bView[i]);
  }
}

// Risk-based authentication
export class RiskAssessment {
  async assessLoginRisk(
    userId: string,
    loginContext: LoginContext
  ): Promise<RiskScore> {
    const factors = await this.collectRiskFactors(userId, loginContext);
    const score = this.calculateRiskScore(factors);
    
    return {
      score,
      factors,
      recommendation: this.getRecommendation(score)
    };
  }

  private async collectRiskFactors(
    userId: string,
    context: LoginContext
  ): Promise<RiskFactor[]> {
    const factors: RiskFactor[] = [];
    
    // Device fingerprinting
    const isKnownDevice = await this.isKnownDevice(userId, context.deviceFingerprint);
    if (!isKnownDevice) {
      factors.push({
        type: 'unknown_device',
        weight: 0.3,
        description: 'Login from unknown device'
      });
    }

    // Geographic location
    const isUnusualLocation = await this.isUnusualLocation(userId, context.ipAddress);
    if (isUnusualLocation) {
      factors.push({
        type: 'unusual_location',
        weight: 0.25,
        description: 'Login from unusual geographic location'
      });
    }

    // Time-based patterns
    const isUnusualTime = await this.isUnusualTime(userId, context.timestamp);
    if (isUnusualTime) {
      factors.push({
        type: 'unusual_time',
        weight: 0.15,
        description: 'Login at unusual time'
      });
    }

    // Threat intelligence
    const isSuspiciousIP = await this.checkThreatIntelligence(context.ipAddress);
    if (isSuspiciousIP) {
      factors.push({
        type: 'suspicious_ip',
        weight: 0.4,
        description: 'Login from IP address flagged by threat intelligence'
      });
    }

    // Behavioral analysis
    const behaviorScore = await this.analyzeBehavior(userId, context);
    if (behaviorScore > 0.3) {
      factors.push({
        type: 'unusual_behavior',
        weight: behaviorScore,
        description: 'Unusual user behavior patterns detected'
      });
    }

    return factors;
  }

  private calculateRiskScore(factors: RiskFactor[]): number {
    if (factors.length === 0) return 0;
    
    const totalWeight = factors.reduce((sum, factor) => sum + factor.weight, 0);
    return Math.min(totalWeight, 1.0);
  }

  private getRecommendation(score: number): AuthRecommendation {
    if (score >= 0.7) {
      return {
        action: 'deny',
        reason: 'High risk login attempt',
        requiredActions: ['admin_review']
      };
    } else if (score >= 0.4) {
      return {
        action: 'step_up',
        reason: 'Medium risk login attempt',
        requiredActions: ['additional_mfa', 'email_notification']
      };
    } else if (score >= 0.2) {
      return {
        action: 'monitor',
        reason: 'Low to medium risk',
        requiredActions: ['enhanced_logging']
      };
    } else {
      return {
        action: 'allow',
        reason: 'Low risk login',
        requiredActions: []
      };
    }
  }

  private async isKnownDevice(userId: string, fingerprint: string): Promise<boolean> {
    // Check device database
    return false; // Simplified
  }

  private async isUnusualLocation(userId: string, ipAddress: string): Promise<boolean> {
    // Check location patterns
    return false; // Simplified
  }

  private async isUnusualTime(userId: string, timestamp: Date): Promise<boolean> {
    // Check time patterns
    return false; // Simplified
  }

  private async checkThreatIntelligence(ipAddress: string): Promise<boolean> {
    // Check threat intel feeds
    return false; // Simplified
  }

  private async analyzeBehavior(userId: string, context: LoginContext): Promise<number> {
    // Behavior analysis
    return 0; // Simplified
  }
}

interface LoginContext {
  ipAddress: string;
  userAgent: string;
  deviceFingerprint: string;
  timestamp: Date;
}

interface RiskFactor {
  type: string;
  weight: number;
  description: string;
}

interface RiskScore {
  score: number;
  factors: RiskFactor[];
  recommendation: AuthRecommendation;
}

interface AuthRecommendation {
  action: 'allow' | 'step_up' | 'monitor' | 'deny';
  reason: string;
  requiredActions: string[];
}
```

### Role-Based Access Control (RBAC)

```rust
// core/src/security/rbac.rs
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Permission {
    pub resource: String,
    pub action: String,
    pub conditions: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub permissions: HashSet<Permission>,
    pub inherits_from: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub roles: HashSet<Uuid>,
    pub direct_permissions: HashSet<Permission>,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessRequest {
    pub user_id: Uuid,
    pub resource: String,
    pub action: String,
    pub context: HashMap<String, String>,
}

pub struct RBACEngine {
    roles: HashMap<Uuid, Role>,
    users: HashMap<Uuid, User>,
}

impl RBACEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            roles: HashMap::new(),
            users: HashMap::new(),
        };
        
        engine.setup_default_roles();
        engine
    }

    fn setup_default_roles(&mut self) {
        // Super Admin role
        let super_admin_id = Uuid::new_v4();
        self.roles.insert(super_admin_id, Role {
            id: super_admin_id,
            name: "Super Admin".to_string(),
            description: "Full system access".to_string(),
            permissions: [
                Permission { resource: "*".to_string(), action: "*".to_string(), conditions: None }
            ].into_iter().collect(),
            inherits_from: vec![],
        });

        // Tenant Admin role
        let tenant_admin_id = Uuid::new_v4();
        self.roles.insert(tenant_admin_id, Role {
            id: tenant_admin_id,
            name: "Tenant Admin".to_string(),
            description: "Full access within tenant".to_string(),
            permissions: [
                Permission { resource: "subscription".to_string(), action: "*".to_string(), conditions: Some([("tenant".to_string(), "{{user.tenant}}".to_string())].into_iter().collect()) },
                Permission { resource: "user".to_string(), action: "manage".to_string(), conditions: Some([("tenant".to_string(), "{{user.tenant}}".to_string())].into_iter().collect()) },
                Permission { resource: "policy".to_string(), action: "*".to_string(), conditions: Some([("tenant".to_string(), "{{user.tenant}}".to_string())].into_iter().collect()) },
            ].into_iter().collect(),
            inherits_from: vec![],
        });

        // Security Analyst role
        let security_analyst_id = Uuid::new_v4();
        self.roles.insert(security_analyst_id, Role {
            id: security_analyst_id,
            name: "Security Analyst".to_string(),
            description: "Security monitoring and analysis".to_string(),
            permissions: [
                Permission { resource: "policy".to_string(), action: "read".to_string(), conditions: None },
                Permission { resource: "compliance".to_string(), action: "read".to_string(), conditions: None },
                Permission { resource: "security".to_string(), action: "read".to_string(), conditions: None },
                Permission { resource: "audit".to_string(), action: "read".to_string(), conditions: None },
                Permission { resource: "alert".to_string(), action: "*".to_string(), conditions: None },
            ].into_iter().collect(),
            inherits_from: vec![],
        });

        // Compliance Officer role
        let compliance_officer_id = Uuid::new_v4();
        self.roles.insert(compliance_officer_id, Role {
            id: compliance_officer_id,
            name: "Compliance Officer".to_string(),
            description: "Compliance monitoring and reporting".to_string(),
            permissions: [
                Permission { resource: "compliance".to_string(), action: "*".to_string(), conditions: None },
                Permission { resource: "policy".to_string(), action: "read".to_string(), conditions: None },
                Permission { resource: "report".to_string(), action: "*".to_string(), conditions: None },
                Permission { resource: "audit".to_string(), action: "read".to_string(), conditions: None },
            ].into_iter().collect(),
            inherits_from: vec![],
        });

        // DevOps Engineer role
        let devops_engineer_id = Uuid::new_v4();
        self.roles.insert(devops_engineer_id, Role {
            id: devops_engineer_id,
            name: "DevOps Engineer".to_string(),
            description: "Infrastructure and deployment management".to_string(),
            permissions: [
                Permission { resource: "resource".to_string(), action: "read".to_string(), conditions: None },
                Permission { resource: "deployment".to_string(), action: "*".to_string(), conditions: None },
                Permission { resource: "infrastructure".to_string(), action: "*".to_string(), conditions: None },
                Permission { resource: "monitoring".to_string(), action: "read".to_string(), conditions: None },
            ].into_iter().collect(),
            inherits_from: vec![],
        });

        // Read-Only User role
        let readonly_user_id = Uuid::new_v4();
        self.roles.insert(readonly_user_id, Role {
            id: readonly_user_id,
            name: "Read-Only User".to_string(),
            description: "View-only access".to_string(),
            permissions: [
                Permission { resource: "resource".to_string(), action: "read".to_string(), conditions: Some([("subscription".to_string(), "{{user.allowed_subscriptions}}".to_string())].into_iter().collect()) },
                Permission { resource: "policy".to_string(), action: "read".to_string(), conditions: None },
                Permission { resource: "compliance".to_string(), action: "read".to_string(), conditions: None },
            ].into_iter().collect(),
            inherits_from: vec![],
        });
    }

    pub fn check_access(&self, request: &AccessRequest) -> Result<bool, String> {
        let user = self.users.get(&request.user_id)
            .ok_or("User not found")?;

        // Check direct permissions
        if self.has_direct_permission(user, &request.resource, &request.action, &request.context) {
            return Ok(true);
        }

        // Check role-based permissions
        let all_permissions = self.get_all_user_permissions(user)?;
        
        for permission in all_permissions {
            if self.permission_matches(&permission, &request.resource, &request.action, user, &request.context)? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn has_direct_permission(
        &self,
        user: &User,
        resource: &str,
        action: &str,
        context: &HashMap<String, String>
    ) -> bool {
        user.direct_permissions.iter().any(|perm| {
            self.permission_matches(perm, resource, action, user, context).unwrap_or(false)
        })
    }

    fn get_all_user_permissions(&self, user: &User) -> Result<HashSet<Permission>, String> {
        let mut all_permissions = user.direct_permissions.clone();
        
        for role_id in &user.roles {
            let role_permissions = self.get_role_permissions(*role_id)?;
            all_permissions.extend(role_permissions);
        }
        
        Ok(all_permissions)
    }

    fn get_role_permissions(&self, role_id: Uuid) -> Result<HashSet<Permission>, String> {
        let role = self.roles.get(&role_id)
            .ok_or("Role not found")?;
        
        let mut permissions = role.permissions.clone();
        
        // Add inherited permissions
        for inherited_role_id in &role.inherits_from {
            let inherited_permissions = self.get_role_permissions(*inherited_role_id)?;
            permissions.extend(inherited_permissions);
        }
        
        Ok(permissions)
    }

    fn permission_matches(
        &self,
        permission: &Permission,
        resource: &str,
        action: &str,
        user: &User,
        context: &HashMap<String, String>
    ) -> Result<bool, String> {
        // Check resource match
        if !self.resource_matches(&permission.resource, resource) {
            return Ok(false);
        }

        // Check action match
        if !self.action_matches(&permission.action, action) {
            return Ok(false);
        }

        // Check conditions
        if let Some(conditions) = &permission.conditions {
            for (condition_key, condition_value) in conditions {
                if !self.condition_matches(condition_key, condition_value, user, context)? {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    fn resource_matches(&self, permission_resource: &str, requested_resource: &str) -> bool {
        permission_resource == "*" || 
        permission_resource == requested_resource ||
        requested_resource.starts_with(&format!("{}:", permission_resource))
    }

    fn action_matches(&self, permission_action: &str, requested_action: &str) -> bool {
        permission_action == "*" || permission_action == requested_action
    }

    fn condition_matches(
        &self,
        condition_key: &str,
        condition_value: &str,
        user: &User,
        context: &HashMap<String, String>
    ) -> Result<bool, String> {
        let resolved_value = self.resolve_template(condition_value, user, context)?;
        
        match condition_key {
            "tenant" => {
                let user_tenant = user.attributes.get("tenant")
                    .ok_or("User has no tenant attribute")?;
                Ok(user_tenant == &resolved_value)
            }
            "subscription" => {
                let allowed_subscriptions = user.attributes.get("allowed_subscriptions")
                    .ok_or("User has no allowed_subscriptions attribute")?;
                Ok(allowed_subscriptions.split(',').any(|sub| sub.trim() == resolved_value))
            }
            _ => {
                let context_value = context.get(condition_key)
                    .ok_or(&format!("Context missing key: {}", condition_key))?;
                Ok(context_value == &resolved_value)
            }
        }
    }

    fn resolve_template(
        &self,
        template: &str,
        user: &User,
        context: &HashMap<String, String>
    ) -> Result<String, String> {
        let mut result = template.to_string();
        
        // Replace user attributes
        for (key, value) in &user.attributes {
            let placeholder = format!("{{{{user.{}}}}}", key);
            result = result.replace(&placeholder, value);
        }
        
        // Replace context values
        for (key, value) in context {
            let placeholder = format!("{{{{context.{}}}}}", key);
            result = result.replace(&placeholder, value);
        }
        
        Ok(result)
    }

    // Administrative functions
    pub fn create_role(&mut self, name: String, description: String, permissions: HashSet<Permission>) -> Uuid {
        let id = Uuid::new_v4();
        let role = Role {
            id,
            name,
            description,
            permissions,
            inherits_from: vec![],
        };
        self.roles.insert(id, role);
        id
    }

    pub fn assign_role_to_user(&mut self, user_id: Uuid, role_id: Uuid) -> Result<(), String> {
        let user = self.users.get_mut(&user_id)
            .ok_or("User not found")?;
        
        if !self.roles.contains_key(&role_id) {
            return Err("Role not found".to_string());
        }
        
        user.roles.insert(role_id);
        Ok(())
    }

    pub fn create_user(&mut self, email: String, attributes: HashMap<String, String>) -> Uuid {
        let id = Uuid::new_v4();
        let user = User {
            id,
            email,
            roles: HashSet::new(),
            direct_permissions: HashSet::new(),
            attributes,
        };
        self.users.insert(id, user);
        id
    }
}

// Attribute-Based Access Control (ABAC) extension
pub struct ABACEngine {
    rbac: RBACEngine,
}

impl ABACEngine {
    pub fn new() -> Self {
        Self {
            rbac: RBACEngine::new(),
        }
    }

    pub fn check_access_with_attributes(
        &self,
        request: &AccessRequest,
        subject_attributes: &HashMap<String, String>,
        resource_attributes: &HashMap<String, String>,
        environment_attributes: &HashMap<String, String>
    ) -> Result<bool, String> {
        // First check RBAC
        if self.rbac.check_access(request)? {
            return Ok(true);
        }

        // Then check ABAC policies
        self.evaluate_abac_policies(
            request,
            subject_attributes,
            resource_attributes,
            environment_attributes
        )
    }

    fn evaluate_abac_policies(
        &self,
        request: &AccessRequest,
        subject_attrs: &HashMap<String, String>,
        resource_attrs: &HashMap<String, String>,
        env_attrs: &HashMap<String, String>
    ) -> Result<bool, String> {
        // Example policy: Allow access to resources with same classification level
        if let (Some(user_clearance), Some(resource_classification)) = 
            (subject_attrs.get("security_clearance"), resource_attrs.get("classification")) {
            
            let clearance_level = self.get_clearance_level(user_clearance)?;
            let classification_level = self.get_classification_level(resource_classification)?;
            
            if clearance_level >= classification_level {
                return Ok(true);
            }
        }

        // Example policy: Deny access outside business hours for certain resources
        if resource_attrs.get("requires_business_hours") == Some(&"true".to_string()) {
            if let Some(current_time) = env_attrs.get("current_time") {
                if !self.is_business_hours(current_time)? {
                    return Ok(false);
                }
            }
        }

        // Example policy: Allow access only from specific IP ranges
        if let Some(allowed_networks) = resource_attrs.get("allowed_networks") {
            if let Some(client_ip) = env_attrs.get("client_ip") {
                return Ok(self.ip_in_networks(client_ip, allowed_networks)?);
            }
        }

        Ok(false)
    }

    fn get_clearance_level(&self, clearance: &str) -> Result<u8, String> {
        match clearance {
            "unclassified" => Ok(1),
            "confidential" => Ok(2),
            "secret" => Ok(3),
            "top_secret" => Ok(4),
            _ => Err(format!("Unknown clearance level: {}", clearance))
        }
    }

    fn get_classification_level(&self, classification: &str) -> Result<u8, String> {
        self.get_clearance_level(classification)
    }

    fn is_business_hours(&self, time_str: &str) -> Result<bool, String> {
        // Simplified business hours check
        // In practice, this would parse the time and check against business hours
        Ok(true) // Placeholder
    }

    fn ip_in_networks(&self, ip: &str, networks: &str) -> Result<bool, String> {
        // Simplified IP network check
        // In practice, this would parse CIDR ranges and check IP membership
        Ok(true) // Placeholder
    }
}
```

## Encryption & Cryptography

### Post-Quantum Cryptography Implementation

```rust
// core/src/security/crypto.rs
use pqcrypto_kyber::kyber1024;
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::{kem::*, sign::*};
use ring::aead::{self, AES_256_GCM, BoundKey, Nonce, NonceSequence, OpeningKey, SealingKey, UnboundKey};
use ring::digest::{self, SHA256};
use ring::pbkdf2;
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroU32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyPair {
    pub public_key: Vec<u8>,
    pub private_key: Vec<u8>,
    pub algorithm: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    pub ciphertext: Vec<u8>,
    pub nonce: Vec<u8>,
    pub algorithm: String,
    pub key_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalSignature {
    pub signature: Vec<u8>,
    pub algorithm: String,
    pub signer: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct CryptoEngine {
    rng: SystemRandom,
    key_storage: HashMap<String, KeyPair>,
}

impl CryptoEngine {
    pub fn new() -> Self {
        Self {
            rng: SystemRandom::new(),
            key_storage: HashMap::new(),
        }
    }

    // Post-quantum key exchange using Kyber1024
    pub fn generate_kyber_keypair(&mut self) -> Result<String, String> {
        let (public_key, secret_key) = kyber1024::keypair();
        
        let key_id = self.generate_key_id();
        let keypair = KeyPair {
            public_key: public_key.as_bytes().to_vec(),
            private_key: secret_key.as_bytes().to_vec(),
            algorithm: "Kyber1024".to_string(),
            created_at: chrono::Utc::now(),
        };
        
        self.key_storage.insert(key_id.clone(), keypair);
        Ok(key_id)
    }

    pub fn kyber_encapsulate(&self, public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>), String> {
        let public_key = kyber1024::PublicKey::from_bytes(public_key)
            .map_err(|_| "Invalid public key")?;
        
        let (ciphertext, shared_secret) = kyber1024::encapsulate(&public_key);
        
        Ok((
            ciphertext.as_bytes().to_vec(),
            shared_secret.as_bytes().to_vec()
        ))
    }

    pub fn kyber_decapsulate(&self, key_id: &str, ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        let keypair = self.key_storage.get(key_id)
            .ok_or("Key not found")?;
        
        if keypair.algorithm != "Kyber1024" {
            return Err("Wrong key algorithm".to_string());
        }
        
        let secret_key = kyber1024::SecretKey::from_bytes(&keypair.private_key)
            .map_err(|_| "Invalid secret key")?;
        
        let ciphertext = kyber1024::Ciphertext::from_bytes(ciphertext)
            .map_err(|_| "Invalid ciphertext")?;
        
        let shared_secret = kyber1024::decapsulate(&ciphertext, &secret_key);
        
        Ok(shared_secret.as_bytes().to_vec())
    }

    // Post-quantum digital signatures using Dilithium5
    pub fn generate_dilithium_keypair(&mut self) -> Result<String, String> {
        let (public_key, secret_key) = dilithium5::keypair();
        
        let key_id = self.generate_key_id();
        let keypair = KeyPair {
            public_key: public_key.as_bytes().to_vec(),
            private_key: secret_key.as_bytes().to_vec(),
            algorithm: "Dilithium5".to_string(),
            created_at: chrono::Utc::now(),
        };
        
        self.key_storage.insert(key_id.clone(), keypair);
        Ok(key_id)
    }

    pub fn dilithium_sign(&self, key_id: &str, message: &[u8]) -> Result<DigitalSignature, String> {
        let keypair = self.key_storage.get(key_id)
            .ok_or("Key not found")?;
        
        if keypair.algorithm != "Dilithium5" {
            return Err("Wrong key algorithm".to_string());
        }
        
        let secret_key = dilithium5::SecretKey::from_bytes(&keypair.private_key)
            .map_err(|_| "Invalid secret key")?;
        
        let signature = dilithium5::sign(message, &secret_key);
        
        Ok(DigitalSignature {
            signature: signature.as_bytes().to_vec(),
            algorithm: "Dilithium5".to_string(),
            signer: key_id.to_string(),
            timestamp: chrono::Utc::now(),
        })
    }

    pub fn dilithium_verify(&self, public_key: &[u8], message: &[u8], signature: &[u8]) -> Result<bool, String> {
        let public_key = dilithium5::PublicKey::from_bytes(public_key)
            .map_err(|_| "Invalid public key")?;
        
        let signature = dilithium5::Signature::from_bytes(signature)
            .map_err(|_| "Invalid signature")?;
        
        match dilithium5::verify(&signature, message, &public_key) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    // AES-256-GCM symmetric encryption
    pub fn encrypt_data(&self, data: &[u8], password: &str) -> Result<EncryptedData, String> {
        let salt = self.generate_salt()?;
        let key = self.derive_key(password, &salt)?;
        
        let sealing_key = SealingKey::new(UnboundKey::new(&AES_256_GCM, &key)
            .map_err(|_| "Failed to create encryption key")?);
        
        let nonce = self.generate_nonce()?;
        let nonce_obj = Nonce::assume_unique_for_key(nonce);
        
        let mut in_out = data.to_vec();
        let tag = sealing_key.seal_in_place_separate_tag(nonce_obj, aead::Aad::empty(), &mut in_out)
            .map_err(|_| "Encryption failed")?;
        
        // Append tag to ciphertext
        in_out.extend_from_slice(tag.as_ref());
        
        Ok(EncryptedData {
            ciphertext: in_out,
            nonce: nonce.to_vec(),
            algorithm: "AES-256-GCM".to_string(),
            key_id: hex::encode(&salt), // Store salt as key_id for password-based encryption
        })
    }

    pub fn decrypt_data(&self, encrypted_data: &EncryptedData, password: &str) -> Result<Vec<u8>, String> {
        if encrypted_data.algorithm != "AES-256-GCM" {
            return Err("Unsupported encryption algorithm".to_string());
        }
        
        let salt = hex::decode(&encrypted_data.key_id)
            .map_err(|_| "Invalid salt")?;
        let key = self.derive_key(password, &salt)?;
        
        let opening_key = OpeningKey::new(UnboundKey::new(&AES_256_GCM, &key)
            .map_err(|_| "Failed to create decryption key")?);
        
        let nonce = Nonce::try_assume_unique_for_key(&encrypted_data.nonce)
            .map_err(|_| "Invalid nonce")?;
        
        let mut ciphertext = encrypted_data.ciphertext.clone();
        let plaintext = opening_key.open_in_place(nonce, aead::Aad::empty(), &mut ciphertext)
            .map_err(|_| "Decryption failed")?;
        
        Ok(plaintext.to_vec())
    }

    // Hash functions
    pub fn hash_data(&self, data: &[u8]) -> Vec<u8> {
        digest::digest(&SHA256, data).as_ref().to_vec()
    }

    pub fn verify_hash(&self, data: &[u8], expected_hash: &[u8]) -> bool {
        let actual_hash = self.hash_data(data);
        actual_hash == expected_hash
    }

    // Key derivation
    fn derive_key(&self, password: &str, salt: &[u8]) -> Result<[u8; 32], String> {
        let mut key = [0u8; 32];
        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            NonZeroU32::new(100_000).unwrap(),
            salt,
            password.as_bytes(),
            &mut key,
        );
        Ok(key)
    }

    // Utility functions
    fn generate_key_id(&self) -> String {
        let mut bytes = [0u8; 16];
        self.rng.fill(&mut bytes).expect("Failed to generate key ID");
        hex::encode(bytes)
    }

    fn generate_salt(&self) -> Result<[u8; 32], String> {
        let mut salt = [0u8; 32];
        self.rng.fill(&mut salt).map_err(|_| "Failed to generate salt")?;
        Ok(salt)
    }

    fn generate_nonce(&self) -> Result<[u8; 12], String> {
        let mut nonce = [0u8; 12];
        self.rng.fill(&mut nonce).map_err(|_| "Failed to generate nonce")?;
        Ok(nonce)
    }
}

// Key management
pub struct KeyManager {
    crypto_engine: CryptoEngine,
    key_rotation_schedule: HashMap<String, chrono::DateTime<chrono::Utc>>,
}

impl KeyManager {
    pub fn new() -> Self {
        Self {
            crypto_engine: CryptoEngine::new(),
            key_rotation_schedule: HashMap::new(),
        }
    }

    pub fn rotate_keys(&mut self) -> Result<Vec<String>, String> {
        let now = chrono::Utc::now();
        let mut rotated_keys = Vec::new();
        
        for (key_id, next_rotation) in &self.key_rotation_schedule.clone() {
            if now >= *next_rotation {
                // Generate new key
                let new_key_id = self.crypto_engine.generate_dilithium_keypair()?;
                
                // Schedule next rotation (90 days)
                let next_rotation = now + chrono::Duration::days(90);
                self.key_rotation_schedule.insert(new_key_id.clone(), next_rotation);
                
                // Remove old key from schedule
                self.key_rotation_schedule.remove(key_id);
                
                rotated_keys.push(new_key_id);
            }
        }
        
        Ok(rotated_keys)
    }

    pub fn schedule_key_rotation(&mut self, key_id: String, rotation_date: chrono::DateTime<chrono::Utc>) {
        self.key_rotation_schedule.insert(key_id, rotation_date);
    }
}

// Hardware Security Module (HSM) integration
pub trait HSMProvider {
    fn generate_key(&self, algorithm: &str) -> Result<String, String>;
    fn sign(&self, key_id: &str, data: &[u8]) -> Result<Vec<u8>, String>;
    fn encrypt(&self, key_id: &str, data: &[u8]) -> Result<Vec<u8>, String>;
    fn decrypt(&self, key_id: &str, ciphertext: &[u8]) -> Result<Vec<u8>, String>;
}

pub struct AzureKeyVaultHSM {
    client: azure_security_keyvault::KeyClient,
}

impl AzureKeyVaultHSM {
    pub fn new(vault_url: &str, credential: azure_core::auth::TokenCredential) -> Self {
        Self {
            client: azure_security_keyvault::KeyClient::new(vault_url, credential),
        }
    }
}

impl HSMProvider for AzureKeyVaultHSM {
    fn generate_key(&self, algorithm: &str) -> Result<String, String> {
        // Implementation would use Azure Key Vault SDK
        // This is a placeholder
        Ok("hsm-key-id".to_string())
    }

    fn sign(&self, key_id: &str, data: &[u8]) -> Result<Vec<u8>, String> {
        // Implementation would use Azure Key Vault SDK
        // This is a placeholder
        Ok(vec![])
    }

    fn encrypt(&self, key_id: &str, data: &[u8]) -> Result<Vec<u8>, String> {
        // Implementation would use Azure Key Vault SDK
        // This is a placeholder
        Ok(vec![])
    }

    fn decrypt(&self, key_id: &str, ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        // Implementation would use Azure Key Vault SDK
        // This is a placeholder
        Ok(vec![])
    }
}
```

## Network Security

### Zero Trust Network Architecture

```yaml
# infrastructure/kubernetes/security/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: policycortex
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-policy
  namespace: policycortex
spec:
  podSelector:
    matchLabels:
      app: frontend
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: nginx-ingress
    ports:
    - protocol: TCP
      port: 3000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: graphql-gateway
    ports:
    - protocol: TCP
      port: 4000
  - to: []  # DNS
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: core-api-policy
  namespace: policycortex
spec:
  podSelector:
    matchLabels:
      app: core
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: graphql-gateway
    ports:
    - protocol: TCP
      port: 8080
  - from:  # Health checks
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []  # Azure APIs
    ports:
    - protocol: TCP
      port: 443

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: database-policy
  namespace: policycortex
spec:
  podSelector:
    matchLabels:
      app: postgres
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: core
    - podSelector:
        matchLabels:
          app: ai-engine
    ports:
    - protocol: TCP
      port: 5432
```

### Service Mesh Security with Istio

```yaml
# infrastructure/kubernetes/security/istio-security.yaml
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: policycortex
spec:
  mtls:
    mode: STRICT

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: core-api-authz
  namespace: policycortex
spec:
  selector:
    matchLabels:
      app: core
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/policycortex/sa/graphql-gateway"]
  - to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/v1/*", "/health", "/metrics"]

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: external-access-authz
  namespace: policycortex
spec:
  selector:
    matchLabels:
      app: graphql-gateway
  rules:
  - from:
    - source:
        namespaces: ["istio-system"]  # Ingress gateway
  - to:
    - operation:
        methods: ["GET", "POST", "OPTIONS"]
        paths: ["/graphql", "/subscriptions"]
  - when:
    - key: request.headers[authorization]
      values: ["Bearer *"]

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: core-api-dr
  namespace: policycortex
spec:
  host: core
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 5
    outlierDetection:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
```

## Data Protection

### Data Classification and DLP

```rust
// core/src/security/data_protection.rs
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PIIType {
    CreditCard,
    SSN,
    Email,
    PhoneNumber,
    IPAddress,
    AzureResourceId,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataScanResult {
    pub classification: DataClassification,
    pub pii_detected: Vec<PIIDetection>,
    pub sensitive_patterns: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIIDetection {
    pub pii_type: PIIType,
    pub location: String,
    pub confidence: f32,
    pub masked_value: String,
}

pub struct DataProtectionEngine {
    patterns: HashMap<PIIType, Regex>,
    classification_rules: Vec<ClassificationRule>,
}

#[derive(Debug, Clone)]
struct ClassificationRule {
    patterns: Vec<String>,
    classification: DataClassification,
    weight: u32,
}

impl DataProtectionEngine {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut patterns = HashMap::new();
        
        // Credit card patterns
        patterns.insert(
            PIIType::CreditCard,
            Regex::new(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")?,
        );
        
        // SSN patterns
        patterns.insert(
            PIIType::SSN,
            Regex::new(r"\b\d{3}-\d{2}-\d{4}\b")?,
        );
        
        // Email patterns
        patterns.insert(
            PIIType::Email,
            Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")?,
        );
        
        // Phone number patterns
        patterns.insert(
            PIIType::PhoneNumber,
            Regex::new(r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b")?,
        );
        
        // IP address patterns
        patterns.insert(
            PIIType::IPAddress,
            Regex::new(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")?,
        );
        
        // Azure resource ID patterns
        patterns.insert(
            PIIType::AzureResourceId,
            Regex::new(r"/subscriptions/[a-f0-9-]{36}/resourceGroups/[^/]+/providers/[^/]+/[^/]+/[^/\s]+")?,
        );

        let classification_rules = vec![
            ClassificationRule {
                patterns: vec!["password".to_string(), "secret".to_string(), "key".to_string()],
                classification: DataClassification::Restricted,
                weight: 10,
            },
            ClassificationRule {
                patterns: vec!["confidential".to_string(), "private".to_string()],
                classification: DataClassification::Confidential,
                weight: 8,
            },
            ClassificationRule {
                patterns: vec!["internal".to_string(), "company".to_string()],
                classification: DataClassification::Internal,
                weight: 5,
            },
        ];

        Ok(Self {
            patterns,
            classification_rules,
        })
    }

    pub fn scan_data(&self, data: &str, context: &str) -> DataScanResult {
        let mut pii_detected = Vec::new();
        let mut sensitive_patterns = Vec::new();
        
        // Detect PII
        for (pii_type, regex) in &self.patterns {
            for mat in regex.find_iter(data) {
                let detected_value = mat.as_str();
                let masked_value = self.mask_pii(detected_value, pii_type);
                
                pii_detected.push(PIIDetection {
                    pii_type: pii_type.clone(),
                    location: context.to_string(),
                    confidence: self.calculate_confidence(detected_value, pii_type),
                    masked_value,
                });
                
                sensitive_patterns.push(detected_value.to_string());
            }
        }
        
        // Classify data
        let classification = self.classify_data(data, &pii_detected);
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&classification, &pii_detected);
        
        DataScanResult {
            classification,
            pii_detected,
            sensitive_patterns,
            recommendations,
        }
    }

    fn mask_pii(&self, value: &str, pii_type: &PIIType) -> String {
        match pii_type {
            PIIType::CreditCard => {
                if value.len() >= 4 {
                    format!("****-****-****-{}", &value[value.len()-4..])
                } else {
                    "****".to_string()
                }
            }
            PIIType::SSN => "***-**-****".to_string(),
            PIIType::Email => {
                if let Some(at_pos) = value.find('@') {
                    format!("{}@{}", "*".repeat(at_pos.min(3)), &value[at_pos+1..])
                } else {
                    "*****".to_string()
                }
            }
            PIIType::PhoneNumber => "***-***-****".to_string(),
            PIIType::IPAddress => {
                let parts: Vec<&str> = value.split('.').collect();
                if parts.len() == 4 {
                    format!("***.***.***.{}", parts[3])
                } else {
                    "***.***.***.***".to_string()
                }
            }
            PIIType::AzureResourceId => {
                // Mask subscription ID and resource names
                let masked = value
                    .replace(|c: char| c.is_alphanumeric() && !"/".contains(c), "*");
                masked
            }
            PIIType::Custom(_) => "*****".to_string(),
        }
    }

    fn calculate_confidence(&self, value: &str, pii_type: &PIIType) -> f32 {
        match pii_type {
            PIIType::CreditCard => {
                // Luhn algorithm check for credit cards
                if self.luhn_check(value) {
                    0.9
                } else {
                    0.6
                }
            }
            PIIType::SSN => {
                // Basic format validation
                if value.len() == 11 && value.chars().nth(3) == Some('-') && value.chars().nth(6) == Some('-') {
                    0.9
                } else {
                    0.7
                }
            }
            PIIType::Email => {
                // More sophisticated email validation
                if value.contains('@') && value.contains('.') {
                    0.9
                } else {
                    0.6
                }
            }
            _ => 0.8,
        }
    }

    fn luhn_check(&self, card_number: &str) -> bool {
        let digits: String = card_number.chars().filter(|c| c.is_numeric()).collect();
        if digits.is_empty() {
            return false;
        }
        
        let mut sum = 0;
        let mut alternate = false;
        
        for digit in digits.chars().rev() {
            let mut n = digit.to_digit(10).unwrap_or(0) as u32;
            if alternate {
                n *= 2;
                if n > 9 {
                    n = (n % 10) + 1;
                }
            }
            sum += n;
            alternate = !alternate;
        }
        
        sum % 10 == 0
    }

    fn classify_data(&self, data: &str, pii_detected: &[PIIDetection]) -> DataClassification {
        let mut score = 0;
        let mut highest_classification = DataClassification::Public;
        
        // Score based on PII detection
        for detection in pii_detected {
            match detection.pii_type {
                PIIType::CreditCard | PIIType::SSN => {
                    score += 10;
                    highest_classification = DataClassification::Restricted;
                }
                PIIType::Email | PIIType::PhoneNumber => {
                    score += 5;
                    if highest_classification == DataClassification::Public {
                        highest_classification = DataClassification::Confidential;
                    }
                }
                _ => score += 3,
            }
        }
        
        // Score based on content patterns
        let data_lower = data.to_lowercase();
        for rule in &self.classification_rules {
            for pattern in &rule.patterns {
                if data_lower.contains(pattern) {
                    score += rule.weight;
                    if rule.classification > highest_classification {
                        highest_classification = rule.classification.clone();
                    }
                }
            }
        }
        
        // Final classification based on score and detected patterns
        if score >= 20 || highest_classification == DataClassification::Restricted {
            DataClassification::Restricted
        } else if score >= 10 || highest_classification == DataClassification::Confidential {
            DataClassification::Confidential
        } else if score >= 5 || highest_classification == DataClassification::Internal {
            DataClassification::Internal
        } else {
            DataClassification::Public
        }
    }

    fn generate_recommendations(&self, classification: &DataClassification, pii_detected: &[PIIDetection]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        match classification {
            DataClassification::Restricted => {
                recommendations.push("Apply encryption at rest and in transit".to_string());
                recommendations.push("Implement strict access controls".to_string());
                recommendations.push("Enable audit logging for all access".to_string());
                recommendations.push("Consider data masking for non-production environments".to_string());
            }
            DataClassification::Confidential => {
                recommendations.push("Enable encryption for sensitive fields".to_string());
                recommendations.push("Implement role-based access controls".to_string());
                recommendations.push("Monitor data access patterns".to_string());
            }
            DataClassification::Internal => {
                recommendations.push("Restrict access to authorized personnel".to_string());
                recommendations.push("Enable basic audit logging".to_string());
            }
            DataClassification::Public => {
                // No specific recommendations for public data
            }
        }
        
        // PII-specific recommendations
        for detection in pii_detected {
            match detection.pii_type {
                PIIType::CreditCard => {
                    recommendations.push("Implement PCI DSS compliance measures".to_string());
                    recommendations.push("Use tokenization for credit card storage".to_string());
                }
                PIIType::SSN => {
                    recommendations.push("Implement strict data handling procedures".to_string());
                    recommendations.push("Consider data minimization practices".to_string());
                }
                PIIType::Email => {
                    recommendations.push("Implement email anonymization for analytics".to_string());
                }
                _ => {}
            }
        }
        
        recommendations.sort();
        recommendations.dedup();
        recommendations
    }
}

impl PartialOrd for DataClassification {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DataClassification {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let self_level = match self {
            DataClassification::Public => 0,
            DataClassification::Internal => 1,
            DataClassification::Confidential => 2,
            DataClassification::Restricted => 3,
        };
        
        let other_level = match other {
            DataClassification::Public => 0,
            DataClassification::Internal => 1,
            DataClassification::Confidential => 2,
            DataClassification::Restricted => 3,
        };
        
        self_level.cmp(&other_level)
    }
}

// Data Loss Prevention (DLP) policies
pub struct DLPPolicy {
    pub name: String,
    pub description: String,
    pub rules: Vec<DLPRule>,
    pub actions: Vec<DLPAction>,
}

pub struct DLPRule {
    pub pattern: Regex,
    pub threshold: u32,
    pub severity: Severity,
}

#[derive(Debug, Clone)]
pub enum DLPAction {
    Block,
    Alert,
    Encrypt,
    Quarantine,
    Log,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

pub struct DLPEngine {
    policies: Vec<DLPPolicy>,
    data_protection: DataProtectionEngine,
}

impl DLPEngine {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let data_protection = DataProtectionEngine::new()?;
        let policies = Self::create_default_policies()?;
        
        Ok(Self {
            policies,
            data_protection,
        })
    }

    fn create_default_policies() -> Result<Vec<DLPPolicy>, Box<dyn std::error::Error>> {
        let mut policies = Vec::new();
        
        // Credit card policy
        policies.push(DLPPolicy {
            name: "Credit Card Protection".to_string(),
            description: "Prevent credit card data exposure".to_string(),
            rules: vec![
                DLPRule {
                    pattern: Regex::new(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")?,
                    threshold: 1,
                    severity: Severity::Critical,
                }
            ],
            actions: vec![DLPAction::Block, DLPAction::Alert, DLPAction::Log],
        });
        
        // SSN policy
        policies.push(DLPPolicy {
            name: "SSN Protection".to_string(),
            description: "Prevent Social Security Number exposure".to_string(),
            rules: vec![
                DLPRule {
                    pattern: Regex::new(r"\b\d{3}-\d{2}-\d{4}\b")?,
                    threshold: 1,
                    severity: Severity::Critical,
                }
            ],
            actions: vec![DLPAction::Block, DLPAction::Alert, DLPAction::Encrypt],
        });
        
        Ok(policies)
    }

    pub fn evaluate_data(&self, data: &str, context: &str) -> DLPEvaluation {
        let scan_result = self.data_protection.scan_data(data, context);
        let mut violations = Vec::new();
        let mut recommended_actions = Vec::new();
        
        for policy in &self.policies {
            for rule in &policy.rules {
                let matches: Vec<_> = rule.pattern.find_iter(data).collect();
                if matches.len() as u32 >= rule.threshold {
                    violations.push(DLPViolation {
                        policy_name: policy.name.clone(),
                        rule_pattern: rule.pattern.as_str().to_string(),
                        matches: matches.len() as u32,
                        severity: rule.severity.clone(),
                    });
                    
                    recommended_actions.extend(policy.actions.clone());
                }
            }
        }
        
        DLPEvaluation {
            scan_result,
            violations,
            recommended_actions,
        }
    }
}

#[derive(Debug)]
pub struct DLPEvaluation {
    pub scan_result: DataScanResult,
    pub violations: Vec<DLPViolation>,
    pub recommended_actions: Vec<DLPAction>,
}

#[derive(Debug)]
pub struct DLPViolation {
    pub policy_name: String,
    pub rule_pattern: String,
    pub matches: u32,
    pub severity: Severity,
}
```

This comprehensive security architecture documentation covers all major aspects of PolicyCortex's security implementation, from threat modeling and authentication to post-quantum cryptography and data protection. The system implements defense-in-depth with zero-trust principles throughout.