
    = note: you can suppress this lint if you plan to use the trait only in your own code, or do not care about auto traits like `Send` on the `Future`
help: you can alternatively desugar to a normal `fn` that returns `impl Future` and add any desired bounds such as `Send`, but these cannot be relaxed without a breaking API change
    |
362 -     async fn get_status(&self, request_id: Uuid) -> Result<RemediationStatus, String>;
362 +     fn get_status(&self, request_id: Uuid) -> impl std::future::Future<Output = Result<RemediationStatus, String>> + Send;
    |

warning: `policycortex-core` (lib) generated 353 warnings (run `cargo fix --lib -p policycortex-core` to apply 42 suggestions)
error[E0412]: cannot find type `AzureClient` in this scope
  --> src/azure_integration.rs:15:17
   |
15 |     client: Arc<AzureClient>,
   |                 ^^^^^^^^^^^ not found in this scope
   |
help: consider importing one of these items
   |
4  + use crate::AzureClient;
   |
4  + use crate::remediation::arm_executor::AzureClient;
   |
4  + use policycortex_core::azure::AzureClient;
   |
4  + use policycortex_core::azure_client::AzureClient;
   |
     and 1 other candidate

error[E0412]: cannot find type `MonitorService` in this scope
  --> src/azure_integration.rs:16:18
   |
16 |     monitor: Arc<MonitorService>,
   |                  ^^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::MonitorService;
   |

error[E0412]: cannot find type `GovernanceService` in this scope
  --> src/azure_integration.rs:17:21
   |
17 |     governance: Arc<GovernanceService>,
   |                     ^^^^^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::GovernanceService;
   |

error[E0412]: cannot find type `SecurityService` in this scope
  --> src/azure_integration.rs:18:19
   |
18 |     security: Arc<SecurityService>,
   |                   ^^^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::SecurityService;
   |

error[E0412]: cannot find type `OperationsService` in this scope
  --> src/azure_integration.rs:19:21
   |
19 |     operations: Arc<OperationsService>,
   |                     ^^^^^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::OperationsService;
   |

error[E0412]: cannot find type `DevOpsService` in this scope
  --> src/azure_integration.rs:20:17
   |
20 |     devops: Arc<DevOpsService>,
   |                 ^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::DevOpsService;
   |

error[E0412]: cannot find type `CostService` in this scope
  --> src/azure_integration.rs:21:15
   |
21 |     cost: Arc<CostService>,
   |               ^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::CostService;
   |

error[E0412]: cannot find type `ActivityService` in this scope
  --> src/azure_integration.rs:22:19
   |
22 |     activity: Arc<ActivityService>,
   |                   ^^^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::ActivityService;
   |

error[E0412]: cannot find type `ResourceGraphService` in this scope
  --> src/azure_integration.rs:23:25
   |
23 |     resource_graph: Arc<ResourceGraphService>,
   |                         ^^^^^^^^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::ResourceGraphService;
   |

error[E0425]: cannot find function `create_shared_client` in this scope
  --> src/azure_integration.rs:32:22
   |
32 |         let client = create_shared_client().await?;
   |                      ^^^^^^^^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this function
   |
4  + use policycortex_core::azure::create_shared_client;
   |

error[E0433]: failed to resolve: use of undeclared type `MonitorService`
  --> src/azure_integration.rs:35:31
   |
35 |             monitor: Arc::new(MonitorService::new((*client).clone())),
   |                               ^^^^^^^^^^^^^^ use of undeclared type `MonitorService`
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::MonitorService;
   |

error[E0433]: failed to resolve: use of undeclared type `GovernanceService`
  --> src/azure_integration.rs:36:34
   |
36 |             governance: Arc::new(GovernanceService::new((*client).clone())),
   |                                  ^^^^^^^^^^^^^^^^^ use of undeclared type `GovernanceService`
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::GovernanceService;
   |

error[E0433]: failed to resolve: use of undeclared type `SecurityService`
  --> src/azure_integration.rs:37:32
   |
37 |             security: Arc::new(SecurityService::new((*client).clone())),
   |                                ^^^^^^^^^^^^^^^ use of undeclared type `SecurityService`
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::SecurityService;
   |

error[E0433]: failed to resolve: use of undeclared type `OperationsService`
  --> src/azure_integration.rs:38:34
   |
38 |             operations: Arc::new(OperationsService::new((*client).clone())),
   |                                  ^^^^^^^^^^^^^^^^^ use of undeclared type `OperationsService`
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::OperationsService;
   |

error[E0433]: failed to resolve: use of undeclared type `DevOpsService`
  --> src/azure_integration.rs:39:30
   |
39 |             devops: Arc::new(DevOpsService::new((*client).clone())),
   |                              ^^^^^^^^^^^^^ use of undeclared type `DevOpsService`
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::DevOpsService;
   |

error[E0433]: failed to resolve: use of undeclared type `CostService`
  --> src/azure_integration.rs:40:28
   |
40 |             cost: Arc::new(CostService::new((*client).clone())),
   |                            ^^^^^^^^^^^ use of undeclared type `CostService`
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::CostService;
   |

error[E0433]: failed to resolve: use of undeclared type `ActivityService`
  --> src/azure_integration.rs:41:32
   |
41 |             activity: Arc::new(ActivityService::new((*client).clone())),
   |                                ^^^^^^^^^^^^^^^ use of undeclared type `ActivityService`
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::ActivityService;
   |

error[E0433]: failed to resolve: use of undeclared type `ResourceGraphService`
  --> src/azure_integration.rs:42:38
   |
42 |             resource_graph: Arc::new(ResourceGraphService::new((*client).clone())),
   |                                      ^^^^^^^^^^^^^^^^^^^^ use of undeclared type `ResourceGraphService`
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::ResourceGraphService;
   |

error[E0412]: cannot find type `MonitorService` in this scope
  --> src/azure_integration.rs:93:31
   |
93 |     pub fn monitor(&self) -> &MonitorService {
   |                               ^^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::MonitorService;
   |

error[E0412]: cannot find type `GovernanceService` in this scope
  --> src/azure_integration.rs:98:34
   |
98 |     pub fn governance(&self) -> &GovernanceService {
   |                                  ^^^^^^^^^^^^^^^^^ not found in this scope
   |
help: consider importing this struct
   |
4  + use policycortex_core::azure::GovernanceService;
   |

error[E0412]: cannot find type `SecurityService` in this scope
   --> src/azure_integration.rs:103:32
    |
103 |     pub fn security(&self) -> &SecurityService {
    |                                ^^^^^^^^^^^^^^^ not found in this scope
    |
help: consider importing this struct
    |
4   + use policycortex_core::azure::SecurityService;
    |

error[E0412]: cannot find type `OperationsService` in this scope
   --> src/azure_integration.rs:108:34
    |
108 |     pub fn operations(&self) -> &OperationsService {
    |                                  ^^^^^^^^^^^^^^^^^ not found in this scope
    |
help: consider importing this struct
    |
4   + use policycortex_core::azure::OperationsService;
    |

error[E0412]: cannot find type `DevOpsService` in this scope
   --> src/azure_integration.rs:113:30
    |
113 |     pub fn devops(&self) -> &DevOpsService {
    |                              ^^^^^^^^^^^^^ not found in this scope
    |
help: consider importing this struct
    |
4   + use policycortex_core::azure::DevOpsService;
    |

error[E0412]: cannot find type `CostService` in this scope
   --> src/azure_integration.rs:118:28
    |
118 |     pub fn cost(&self) -> &CostService {
    |                            ^^^^^^^^^^^ not found in this scope
    |
help: consider importing this struct
    |
4   + use policycortex_core::azure::CostService;
    |

error[E0412]: cannot find type `ActivityService` in this scope
   --> src/azure_integration.rs:123:32
    |
123 |     pub fn activity(&self) -> &ActivityService {
    |                                ^^^^^^^^^^^^^^^ not found in this scope
    |
help: consider importing this struct
    |
4   + use policycortex_core::azure::ActivityService;
    |

error[E0412]: cannot find type `ResourceGraphService` in this scope
   --> src/azure_integration.rs:128:38
    |
128 |     pub fn resource_graph(&self) -> &ResourceGraphService {
    |                                      ^^^^^^^^^^^^^^^^^^^^ not found in this scope
    |
help: consider importing this struct
    |
4   + use policycortex_core::azure::ResourceGraphService;
    |

warning: unused doc comment
   --> src/azure_integration.rs:151:1
    |
151 | /// Create a global Azure integration service instance
    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ rustdoc does not generate documentation for macro invocations
    |
    = help: to document an item produced by a macro, the macro must produce the documentation as part of its expansion
    = note: `#[warn(unused_doc_comments)]` on by default

warning: unused variable: `query`
   --> src/audit_chain.rs:367:13
    |
367 |         let query = r#"
    |             ^^^^^ help: if this is intentional, prefix it with an underscore: `_query`

warning: unused variable: `control`
   --> src/compliance/mod.rs:439:9
    |
439 |         control: &ComplianceControl,
    |         ^^^^^^^ help: if this is intentional, prefix it with an underscore: `_control`

warning: unused variable: `key_id`
   --> src/evidence_pipeline.rs:450:9
    |
450 |         key_id: &str,
    |         ^^^^^^ help: if this is intentional, prefix it with an underscore: `_key_id`

warning: unused variable: `rule`
   --> src/approvals.rs:292:21
    |
292 |         if let Some(rule) = policy.sod_rules.first() {
    |                     ^^^^ help: if this is intentional, prefix it with an underscore: `_rule`

warning: unused variable: `requester_id`
   --> src/approvals.rs:288:9
    |
288 |         requester_id: &str,
    |         ^^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_requester_id`

warning: unused variable: `approver_id`
   --> src/approvals.rs:289:9
    |
289 |         approver_id: &str,
    |         ^^^^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_approver_id`

warning: unused variable: `result`
   --> src/change_management.rs:445:17
    |
445 |             let result: serde_json::Value = response
    |                 ^^^^^^ help: if this is intentional, prefix it with an underscore: `_result`

Some errors have detailed explanations: E0412, E0425, E0433.
For more information about an error, try `rustc --explain E0412`.
warning: `policycortex-core` (bin "policycortex-core") generated 203 warnings (195 duplicates)
error: could not compile `policycortex-core` (bin "policycortex-core") due to 26 previous errors; 203 warnings emitted
Error: Process completed with exit code 101.