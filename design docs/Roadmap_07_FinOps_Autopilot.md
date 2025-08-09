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
