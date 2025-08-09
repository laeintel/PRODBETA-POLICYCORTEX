# 6. AI & Knowledge Base

## 6.1 Components
- Domain Expert (Python) – deep rules/heuristics + pattern knowledge
- GPT‑5 Integration – NL analysis, generation, predictions (fallback to expert if unavailable)
- Multi‑Cloud Knowledge Base – service mappings, patterns, controls, equivalence

## 6.2 Training & Context
- Use existing training specs and data; add retrieval into tenants’ facts for grounded responses
- Context windows constructed from resources, policies, costs, violations, actions

## 6.3 Capabilities
- Policy generation (multi‑cloud equivalence)
- Compliance prediction (drift, conflicts, effective policies)
- FinOps optimization strategies (commitments, rightsizing)
- Security exposure analysis (attack paths)

## 6.4 Policy Equivalence (Examples)
| Control | Azure Policy | AWS | GCP |
|---|---|---|---|
| Encryption at rest | `Microsoft.Storage/storageAccounts` encryption enabled | S3/Bucket Server-Side Encryption | CMEK for GCS Buckets |
| Allowed locations | Azure location allowlist | SCP `aws:RequestedRegion` allowlist | Org Policy `constraints/gcp.resourceLocations` |
| Required tags/labels | Require tag policy | Config Rule + Tagging enforcement | Label enforcement via Org Policy |

## 6.5 Generation Examples
Input: “Enforce encryption at rest for storage across production.”
- Azure: Policy with `then.effect: Deny`; scope `prod` management group
- AWS: SCP denying `s3:PutObject` without SSE; Config rule for EBS encryption
- GCP: Org Policy for CMEK + enforce required key usage

## 6.6 Guardrails
- Deterministic rules backstop; never run actions from LLMs without orchestrator safeguards
- Evidence requirement; confidence thresholds; prompt red teaming

## 6.7 APIs
- POST `DEEP_API_BASE/api/v1/policies/generate`
- POST `DEEP_API_BASE/api/v1/analyze` (prompt‑engineered with evidence requests)

## 6.8 Sample Generated Policies

### Azure Policy (JSON)
```json
{
  "properties": {
    "displayName": "Enforce Storage Encryption",
    "policyRule": {
      "if": {
        "allOf": [
          {"field": "type", "equals": "Microsoft.Storage/storageAccounts"},
          {"field": "Microsoft.Storage/storageAccounts/encryption.enabled", "equals": false}
        ]
      },
      "then": {"effect": "Deny"}
    },
    "parameters": {}
  }
}
```

### AWS SCP (JSON)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Deny",
      "Action": ["s3:PutObject"],
      "Resource": "*",
      "Condition": {"StringNotEquals": {"s3:x-amz-server-side-encryption": ["AES256","aws:kms"]}}
    }
  ]
}
```

### GCP Org Policy (YAML)
```yaml
name: constraints/gcp.restrictCmekCryptoKeyProjects
spec:
  rules:
    - values:
        allowedValues:
          - projects/my-kms-project
```
