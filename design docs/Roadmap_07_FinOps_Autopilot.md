# 7. FinOps Autopilot

## 7.1 Scope
- Idle detection and cleanup
- Rightsizing (CPU/mem/storage/IOPS)
- Commitment planning (RIs/SPs/CUDs)
- Budget guardrails and anomaly defense

## 7.2 Inputs
- Cost timeseries (daily), utilization metrics, pricing catalogs, tags, schedules

## 7.3 Algorithms (Sketch)
- Idle Detection: `avg(cpu) < 5%` OR `no network io` for N days → schedule deallocate
- Rightsizing: `P95(cpu) < 30%` and `P95(mem) < 50%` → propose next lower SKU; forecast headroom
- Commitment Planner: forecast next 12 months compute hours by family; optimize mix of 1y/3y, zonal/region
- Anomaly Defense: STL decomposition + z‑scores; alert/auto‑halt if > 3σ with approval path

## 7.4 Outputs & Actions
- Deallocation schedules; SKU resize plan; purchase recommendations; lifecycle rules; guardrail policies

## 7.5 Examples
- VM `D4s_v3` → `D2s_v3`; save $78/month; risk low (P95 cpu 15%)
- 200k hours/month on M family → 60% 1y SP, 20% 3y SP, 20% on‑demand; projected savings 32%

## 7.6 KPIs & Dashboards
- Net savings/month; realized vs potential; ROI per action; anomaly MTTR; budget adherence

## 7.7 Commitment Planner (Worked Example)
Inputs:
- Compute hours/month by family M: 200,000
- On‑demand rate: $0.10/h; 1y SP: $0.07/h; 3y SP: $0.05/h
- Forecast: flat ±10%

Heuristic mix:
- Baseline (guaranteed): 60% 1y SP
- Stable workload: 20% 3y SP
- Variable: 20% on‑demand/spare

Savings:
- 60% * 200k * (0.10-0.07) = $3,600/mo
- 20% * 200k * (0.10-0.05) = $2,000/mo
- Total ≈ $5,600/mo (~28%)

Guardrails:
- Budget ceiling; cancellation penalties; diversification across SKUs/regions; rebalancing monthly
