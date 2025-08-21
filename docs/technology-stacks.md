# PolicyCortex – Technology Stack Snapshot (2025-08)

_This file is generated from static manifest inspection (`package.json`, `Cargo.toml`, `requirements*.txt`, Dockerfiles, Terraform providers) and will drift as dependencies change. Update during each release cut._

## Front-End
| Layer | Tech | Version Source |
|-------|------|----------------|
| Framework | Next.js | "next": **14.x** (`frontend/package.json`) |
| UI Library | React | "react": **18.x** |
| Language | TypeScript | "typescript": **5.x** |
| Styling | Tailwind CSS | "tailwindcss": **3.x** |
| State | React Context + SWR | "swr": **2.x** |
| Testing | Playwright | "@playwright/test": **1.x** |
| E2E runner | Jest (legacy) | "jest": **29.x** |

## Back-End (Rust)
| Component | Crate | Version |
|-----------|-------|---------|
| Web Framework | axum | 0.6 |
| Async runtime | tokio | 1.x |
| Serialization | serde / serde_json | 1.x / 1.x |
| Config | figment | 0.10 |
| Database ORM | sqlx | 0.7 (postgres feature) |
| Graph Client | reqwest + serde | 0.11 / 1.x |
| Logging | tracing / tracing-subscriber | 0.1 / 0.3 |
| Feature flags | cargo features `simulated`, `live-data` |

Rust toolchain: `rustc 1.89.0`, `cargo 1.89.0` (stable-2025-08-04).

## AI/ML Services (Python sidecars)
| Service | Library | Version |
|---------|---------|---------|
| RLHF tooling | transformers | 4.41.x |
| Drift monitor | river | 0.23 |
| Explainability | shap / captum | 0.43 / 0.6 |
| Data | pandas | 2.x |
| Serving | FastAPI | 0.111 |
| Deployment | Azure Container Apps Job |
| Python runtime | 3.10 slim (Dockerfile) |

## GraphQL Gateway (Node)
| Package | Version |
|---------|---------|
| `@apollo/gateway` | 2.x |
| `graphql` | 16.x |
| Node runtime | 18 LTS (Docker ARG) |

## Infrastructure-as-Code
| Tool | Version |
|------|---------|
| Terraform | 1.6.x |
| AzureRM provider | 3.85 |
| Helm provider | 2.12 |
| Kubernetes provider | 2.30 |

## Cloud Platform
- Azure Container Registry (Standard) 2025-08 SKU
- Azure Container Apps (revision-mode: multiple; min=0, max=20)
- Log Analytics Workspace (2020-08-01 API)
- Key Vault (Premium) per environment
- Optional AKS (1.28) + Istio 1.20 (beta branch)

## Observability
| Component | Tech / Package |
|-----------|----------------|
| Metrics & traces | OpenTelemetry SDK (Rust `opentelemetry 0.19`); OTEL collector 0.91 Docker |
| Dashboards | Grafana 10 (container) |
| Log shipper | builtin ACA -> Log Analytics |

## CI / CD
- GitHub Actions runner (`ubuntu-22.04`), `actions/setup-node@v4`, `actions-rs/toolchain@v1`.
- Docker buildx QEMU 0.11.
- fs caching with `actions/cache@v4`.

## Container base images
| Image | Tag |
|-------|-----|
| rust | 1.89-slim-bullseye |
| node | 18-alpine |
| python | 3.10-slim |
| otel/opentelemetry-collector-contrib | 0.91 |

## Dev Toolchain
- Node 18 LTS (npm 9)
- yarn 4 (optional)
- Rust `cargo-audit`, `cargo-watch`
- npm scripts: `dev`, `type-check`, `lint`, `e2e`
- Pre-commit hooks: `eslint`, `prettier`, `clippy`, `terraform fmt`.

---
_Last regenerated: 2025-08-18_
