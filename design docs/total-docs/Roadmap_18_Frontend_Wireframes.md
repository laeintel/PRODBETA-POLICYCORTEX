# 18. Frontend Wireframes (Low‑Fi)

## 18.1 Dashboard
```
+---------------------------------------------------------------+
| Unified Cloud Governance              [AI Learning 87% ▓▓▓ ]  |
|---------------------------------------------------------------|
| [Policies] [RBAC] [Costs] [Network] [Resources] [AI Expert]   |
|                                                               |
| Proactive AI Recommendations                                   |
|  - VM Right-Sizing (Auto‑Remediate)                           |
|  - Unencrypted Storage (Auto‑Remediate)                       |
|                                                               |
| Real‑time Activity: Policies Automated | Users Managed ...     |
+---------------------------------------------------------------+
```

## 18.2 List → Detail → Action Drawer
```
Policies (list)                          Filters [Category][Status]
---------------------------------------------------------------
| Require Encryption at Rest         |  Deny  | 93.8% | Active |
| Require Tags                       |  Audit | 84.7% | Active |

Detail (right panel)
---------------------------------------------------------------
| Name, Description, Type/Effect/Scope, Compliance bar           |
| Non‑compliant resources (table with Remediate/Exception)      |
| [View in Portal] [Export] [Edit] [Open Action Drawer]         |

Action Drawer (bottom/right)
---------------------------------------------------------------
| Summary | Preflight Diff | Blast‑Radius | Approvals | Progress|
| - Change set preview (JSON diff)                               |
| - Requires Approval: Yes (Operator + Approver)                 |
| - [Approve & Run] [Reject] [Close]                             |
| Progress (SSE): queued → in_progress → verifying → completed   |
+---------------------------------------------------------------+
```

## 18.3 Evidence Pack Download
```
Evidence
- Plan (JSON), Human Summary (MD)
- Control Tests (CSV), Policy Export (JSON)
- Screenshots/Artifacts
[Download ZIP]
```

## 18.4 Deep‑Links
- Resource: `/resources/[id]`
- Policy: `/policies/[id]`
- Action: `/actions/[id]` (read‑only with SSE replay)

Notes: color‑contrast accessible, keyboard shortcuts for approvals, skeleton loaders.
