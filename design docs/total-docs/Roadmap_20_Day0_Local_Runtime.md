# 20. Day‑0 Local Runtime (Developer Guide)

## 20.1 Prerequisites
- Node 18+, Docker, Docker Compose
- Python 3.10+ (for FastAPI deep service)
- (Optional) Rust toolchain (for Core build) or use container

## 20.2 Services
- Postgres, Redis
- Core API (port 8080)
- Python Deep API (port 8090)
- Frontend Next.js (port 3000)

## 20.3 Compose (Sketch)
```yaml
version: '3.9'
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: postgres
    ports: ["5432:5432"]
  redis:
    image: redis:7
    ports: ["6379:6379"]
  deep:
    build: ./backend/services/api_gateway
    environment:
      - PYTHONUNBUFFERED=1
    ports: ["8090:8080"]
  core:
    build: ./core
    environment:
      - DEEP_API_BASE=http://deep:8080
    ports: ["8080:8080"]
    depends_on: [db, redis, deep]
  frontend:
    build: ./frontend
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8080
    ports: ["3000:3000"]
    depends_on: [core]
```

## 20.4 Make Targets
- `make up` / `make down` / `make logs`
- `make db-migrate` / `make db-seed`

## 20.5 Quick Test
- Open `http://localhost:3000` → dashboard populated
- Policies deep view → calls Core `/policies/deep` → proxied to Python
- Trigger an action and watch SSE stream

## 20.6 Troubleshooting
- If Rust not installed: run core via Docker build
- If SSE blocked: check reverse proxy and CORS
- Deep API down: Core returns fallback mock data; check `DEEP_API_BASE`
