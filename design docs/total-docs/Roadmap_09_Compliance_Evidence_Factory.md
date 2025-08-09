# 9. Compliance Evidence Factory

## 9.1 Purpose
Automate continuous control testing and produce auditor‑ready evidence packs per framework and period.

## 9.2 Control Mapping Model
- controls(id, framework, name, description)
- control_tests(control_id, signal_source, query, success_criteria)
- control_artifacts(control_id, artifact_type, generator)

## 9.3 Signals & Artifacts
- Signals: policy evaluations, config snapshots, logs, cost data, identity graphs
- Artifacts:
  - JSON exports (policy definitions, exceptions)
  - CSV/Parquet (control test results)
  - PDFs (screenshots or rendered reports)
  - Signed manifests with checksums

## 9.4 Schedules
- Continuous for critical controls; daily/weekly for others
- Ad‑hoc on action completion or drift detection

## 9.5 Pack Assembly
- Gather latest passing control evidence for the period
- Include diffs, exceptions, remediation proof, and approvals
- Output ZIP with manifest and index HTML

## 9.6 KPIs
- Evidence freshness; audit cycle time; control pass rate; exception count
