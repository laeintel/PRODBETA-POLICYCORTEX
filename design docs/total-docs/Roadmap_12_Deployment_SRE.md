# 12. Deployment, SRE & Operations

## 12.1 Environments
- Dev, Staging, Prod; per‑tenant isolation options; secrets via KeyVault/Secrets Manager

## 12.2 CI/CD
- Lint/test/build; contract tests; integration tests against mock cloud; blue/green deploys

## 12.3 Runtime
- K8s (AKS/EKS/GKE) + HPA; Postgres HA; Redis cluster; observability stack

## 12.4 Backups & DR
- Nightly DB backups; artifact store versioning; RPO < 1h, RTO < 2h

## 12.5 Security Ops
- Image signing, SBOM, vulnerability scans; least privilege IAM; periodic pen‑tests

## 12.6 IaC Outline
- Terraform modules: Core API, AI services, Postgres, Redis, NATS, ClickHouse/Timescale, Object store
- Per‑env workspaces; secrets via Key Vault/SM; pipelines enforce plan/apply approvals

## 12.7 Runbooks
- API outage; DB failover; queue backlog; evidence generation failures; high error rate; budget overrun
- Each includes detection, diagnostics (dashboards/queries), remediation steps, and rollback
