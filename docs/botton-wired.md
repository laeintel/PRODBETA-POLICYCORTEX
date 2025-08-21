# Buttons wired with onClick or navigation

Generated from clickable-inventory.json

## app/ai/chat/page.tsx
- app/ai/chat/page.tsx:407 :: onClick={() => router.push(action.action)}
- app/ai/chat/page.tsx:510 :: onClick={() => setShowHistory(false)}
- app/ai/chat/page.tsx:546 :: onClick={() => setShowHistory(true)}
- app/ai/chat/page.tsx:559 :: <button type="button" className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg" onClick={() => toast({ title: 'Download', description: 'Exporting conversation...' })}>
- app/ai/chat/page.tsx:562 :: <button type="button" className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg" onClick={() => toast({ title: 'Share', description: 'Generating share link...' })}>
- app/ai/chat/page.tsx:565 :: <button type="button" className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg" onClick={() => toast({ title: 'Settings', description: 'Opening chat settings' })}>

## app/ai/correlations/page.tsx
- app/ai/correlations/page.tsx:125 :: onClick={() => toast({ title: 'Refreshed', description: 'Correlation data refreshed' })}
- app/ai/correlations/page.tsx:133 :: onClick={() => toast({ title: 'Export', description: 'Exporting correlation report...' })}
- app/ai/correlations/page.tsx:251 :: onClick={() => toast({ title: 'Expanded', description: 'Opening full-screen view' })}

## app/ai/page.tsx
- app/ai/page.tsx:513 :: onClick={() => toast({ title: 'Deploy', description: 'Deploy model flow coming soon' })}
- app/ai/page.tsx:875 :: onClick={() => toast({ title: 'Train model', description: 'Training job queued' })}
- app/ai/page.tsx:883 :: onClick={() => toast({ title: 'Experiment', description: 'Experiment started' })}
- app/ai/page.tsx:891 :: onClick={() => toast({ title: 'Pipeline', description: 'Pipeline deployment queued' })}
- app/ai/page.tsx:899 :: onClick={() => toast({ title: 'Report', description: 'Generating performance report...' })}
- app/ai/page.tsx:956 :: <button type="button" className="p-2 hover:bg-gray-800 rounded" onClick={() => toast({ title: 'Filters', description: 'Filter dialog coming soon' })}>
- app/ai/page.tsx:959 :: <button type="button" className="p-2 hover:bg-gray-800 rounded" onClick={() => toast({ title: 'Export', description: 'Exporting dashboard data...' })}>
- app/ai/page.tsx:1109 :: <button type="button" className="p-1 hover:bg-gray-700 rounded" onClick={() => toast({ title: 'Monitor', description: 'Opening monitoring view (coming soon)' })}>
- app/ai/page.tsx:1112 :: <button type="button" className="p-1 hover:bg-gray-700 rounded" onClick={() => toast({ title: 'More', description: 'More actions menu (coming soon)' })}>

## app/ai/predictive/page.tsx
- app/ai/predictive/page.tsx:265 :: onClick={() => toast({ title: 'Action', description: 'Applying recommended action...' })}
- app/ai/predictive/page.tsx:332 :: onClick={() => toast({ title: 'Applied', description: 'Recommendation applied' })}
- app/ai/predictive/page.tsx:339 :: onClick={() => toast({ title: 'Noted', description: 'Marked as false positive' })}
- app/ai/predictive/page.tsx:346 :: onClick={() => toast({ title: 'Retrain', description: 'Model retraining started' })}

## app/ai/unified/page.tsx
- app/ai/unified/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/ai/unified/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/dashboard/page.tsx
- app/dashboard/page.tsx:157 :: onClick={() => router.push('/operations/alerts')}
- app/dashboard/page.tsx:383 :: onClick={() => toast({ title: 'Export started', description: `Generating ${timeRange} report...` })}
- app/dashboard/page.tsx:505 :: onClick={() => router.push('/operations/alerts')}
- app/dashboard/page.tsx:616 :: onClick={() => router.push('/ai/chat')}
- app/dashboard/page.tsx:629 :: onClick={() => router.push('/governance/policies')}
- app/dashboard/page.tsx:642 :: onClick={() => router.push('/operations/remediation')}
- app/dashboard/page.tsx:655 :: onClick={() => router.push('/governance/cost')}
- app/dashboard/page.tsx:679 :: onClick={() => router.push(item.link)}

## app/devops/artifacts/page.tsx
- app/devops/artifacts/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/devops/artifacts/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/devops/releases/page.tsx
- app/devops/releases/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/devops/releases/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/devops/repos/page.tsx
- app/devops/repos/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/devops/repos/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/governance/cost/page.tsx
- app/governance/cost/page.tsx:341 :: onClick={() => router.push('/governance/cost/optimize')}

## app/operations/automation/page.tsx
- app/operations/automation/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/operations/automation/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/operations/monitoring/page.tsx
- app/operations/monitoring/page.tsx:372 :: onClick={() => toast({ title: 'Dashboard', description: 'Create dashboard flow coming soon' })}
- app/operations/monitoring/page.tsx:516 :: onClick={() => toast({ title: 'Filter', description: 'Opening metric filters' })}
- app/operations/monitoring/page.tsx:571 :: onClick={() => toast({ title: 'Alert', description: 'Creating new alert rule...' })}

## app/operations/notifications/page.tsx
- app/operations/notifications/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/operations/notifications/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/operations/page.tsx
- app/operations/page.tsx:649 :: onClick={() => toast({ title: 'Resource', description: `Viewing ${name}` })}
- app/operations/page.tsx:656 :: onClick={() => toast({ title: 'Manage', description: `Managing ${name}` })}
- app/operations/page.tsx:678 :: onClick={() => toast({ title: 'Optimize', description: `Optimizing ${resource}` })}

## app/operations/resources/page.tsx
- app/operations/resources/page.tsx:688 :: <button type="button" className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors" onClick={() => console.log('Configure resource', selectedResource?.id)}>
- app/operations/resources/page.tsx:692 :: <button type="button" className="bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors" onClick={() => console.log('Edit tags for', selectedResource?.id)}>
- app/operations/resources/page.tsx:696 :: <button type="button" className="bg-red-600/20 hover:bg-red-600/30 text-red-400 px-4 py-2 rounded-lg flex items-center gap-2 transition-colors" onClick={() => console.log('Delete resource', selectedResource?.id)}>

## app/page.tsx
- app/page.tsx:99 :: <button type="button" onClick={() => router.push('/dashboard')} className="py-2 text-xs bg-gray-800 hover:bg-gray-700 rounded-lg text-white flex items-center justify-center gap-2"><Cpu className="w-4 h-4" /> Guest</button>

## app/security/access-reviews/page.tsx
- app/security/access-reviews/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/security/access-reviews/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/security/conditional-access/page.tsx
- app/security/conditional-access/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/security/conditional-access/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/security/entitlements/page.tsx
- app/security/entitlements/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/security/entitlements/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/security/iam/page.tsx
- app/security/iam/page.tsx:368 :: onClick={() => router.push('/security/iam/users/new')}

## app/security/rbac/page.tsx
- app/security/rbac/page.tsx:336 :: onClick={() => toast({ title: 'Export', description: 'Exporting permissions report...' })}
- app/security/rbac/page.tsx:343 :: onClick={() => router.push('/security/rbac/review')}
- app/security/rbac/page.tsx:476 :: <button type="button" className="p-4 bg-gray-900/50 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors" onClick={() => toast({ title: 'Privileged Accounts', description: 'Opening review (coming soon)' })}>
- app/security/rbac/page.tsx:480 :: <button type="button" className="p-4 bg-gray-900/50 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors" onClick={() => toast({ title: 'JIT Requests', description: 'Viewing pending requests (coming soon)' })}>
- app/security/rbac/page.tsx:484 :: <button type="button" className="p-4 bg-gray-900/50 border border-gray-800 rounded-lg hover:bg-gray-800/50 transition-colors" onClick={() => toast({ title: 'SoD Conflicts', description: 'Navigating to conflicts (coming soon)' })}>
- app/security/rbac/page.tsx:509 :: onClick={() => toast({ title: 'Filter', description: 'Filter dialog coming soon' })}
- app/security/rbac/page.tsx:582 :: <button type="button" className="p-1 hover:bg-gray-700 rounded" onClick={() => toast({ title: 'User', description: 'Viewing user details' })}>
- app/security/rbac/page.tsx:585 :: <button type="button" className="p-1 hover:bg-gray-700 rounded" onClick={() => toast({ title: 'User', description: 'Opening user settings' })}>

## app/security/zero-trust/page.tsx
- app/security/zero-trust/page.tsx:90 :: <ActionButton label="Export Report" onClick={() => console.log('Export')} />
- app/security/zero-trust/page.tsx:91 :: <ActionButton label="Configure" onClick={() => console.log('Configure')} />

## app/settings/page.tsx
- app/settings/page.tsx:137 :: onClick={() => toast({ title: 'Change avatar', description: 'Avatar change coming soon' })}
- app/settings/page.tsx:184 :: onClick={() => toast({ title: 'Cancelled', description: 'No changes were saved' })}
- app/settings/page.tsx:191 :: onClick={() => toast({ title: 'Saved', description: 'Profile updated successfully' })}
- app/settings/page.tsx:338 :: onClick={() => toast({ title: 'API key', description: 'Generating new API key...' })}
- app/settings/page.tsx:366 :: onClick={() => toast({ title: 'Rotated', description: 'API key rotated' })}
- app/settings/page.tsx:373 :: onClick={() => toast({ title: 'Deleted', description: 'API key revoked' })}
- app/settings/page.tsx:572 :: onClick={() => toast({ title: 'Password', description: 'Password change flow coming soon' })}
- app/settings/page.tsx:616 :: onClick={() => toast({ title: 'Integration', description: `${integration.name}: ${integration.status === 'connected' ? 'Disconnecting' : 'Connecting'}...` })}
- app/settings/page.tsx:694 :: onClick={() => toast({ title: 'Export', description: 'Exporting settings...' })}
- app/settings/page.tsx:708 :: onClick={() => toast({ title: 'Export', description: 'Exporting audit logs...' })}
- app/settings/page.tsx:722 :: onClick={() => toast({ title: 'Export', description: 'Exporting full backup...' })}
- app/settings/page.tsx:744 :: onClick={() => toast({ title: 'Import', description: 'Opening file picker...' })}
- app/settings/page.tsx:829 :: onClick={() => toast({ title: 'Diagnostics', description: 'Downloading diagnostic report...' })}

## app/tactical/page.tsx
- app/tactical/page.tsx:153 :: onClick={() => toast({ title: 'Alert settings', description: 'Opening alert settings (coming soon)' })}
- app/tactical/page.tsx:327 :: onClick={() => toast({ title: 'Playbook', description: 'Executing emergency playbook...' })}
- app/tactical/page.tsx:363 :: onClick={() => toast({ title: 'Broadcast', description: 'Broadcasting alert to channels...' })}
- app/tactical/page.tsx:449 :: onClick={() => toast({ title: 'Acknowledged', description: `${selectedAlert.title}` })}
- app/tactical/page.tsx:456 :: onClick={() => toast({ title: 'Resolved', description: `${selectedAlert.title}` })}
- app/tactical/page.tsx:463 :: onClick={() => toast({ title: 'Escalated', description: `${selectedAlert.title}` })}
- app/tactical/page.tsx:470 :: onClick={() => toast({ title: 'Note added', description: 'Added note to alert' })}

## components/compliance/ComplianceDeepDrill.tsx
- components/compliance/ComplianceDeepDrill.tsx:253 :: onClick={() => toast({ title: 'Auto-remediation', description: 'Queued automated remediation' })}
- components/compliance/ComplianceDeepDrill.tsx:262 :: onClick={() => toast({ title: 'Export', description: 'Exporting violation report...' })}
- components/compliance/ComplianceDeepDrill.tsx:433 :: onClick={() => toast({ title: 'Remediation', description: 'Executing automated steps...' })}
- components/compliance/ComplianceDeepDrill.tsx:530 :: onClick={() => toast({ title: 'Step', description: 'Marked as complete' })}
- components/compliance/ComplianceDeepDrill.tsx:675 :: onClick={() => toast({ title: 'Azure', description: 'Opening Azure portal link' })}
- components/compliance/ComplianceDeepDrill.tsx:682 :: onClick={() => toast({ title: 'Dependencies', description: 'Showing dependencies' })}

## components/cost/CostAnomalyDeepDrill.tsx
- components/cost/CostAnomalyDeepDrill.tsx:253 :: onClick={() => toast({ title: 'Export', description: 'Exporting analysis...' })}
- components/cost/CostAnomalyDeepDrill.tsx:496 :: onClick={() => toast({ title: 'Filter', description: 'Opening resource filters' })}
- components/cost/CostAnomalyDeepDrill.tsx:719 :: onClick={() => toast({ title: 'Automation', description: 'Auto-implementation queued' })}
- components/cost/CostAnomalyDeepDrill.tsx:727 :: onClick={() => toast({ title: 'Details', description: 'Opening recommendation details' })}
- components/cost/CostAnomalyDeepDrill.tsx:823 :: onClick={() => toast({ title: 'Optimize', description: 'Optimization workflow coming soon' })}
- components/cost/CostAnomalyDeepDrill.tsx:830 :: onClick={() => toast({ title: 'Azure', description: 'Opening Azure portal link' })}

## components/rbac/DeepDrillDashboard.tsx
- components/rbac/DeepDrillDashboard.tsx:400 :: <button type="button" className="px-3 py-1 text-sm bg-blue-50 text-blue-600 rounded-md hover:bg-blue-100" onClick={() => console.log('Refresh permissions') }>
- components/rbac/DeepDrillDashboard.tsx:404 :: <button type="button" className="px-3 py-1 text-sm bg-gray-50 text-gray-600 rounded-md hover:bg-gray-100" onClick={() => console.log('Export permissions analysis') }>
- components/rbac/DeepDrillDashboard.tsx:536 :: <button type="button" className="px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700" onClick={() => console.log('Request removal for', selectedPermission?.permissionId)}>
- components/rbac/DeepDrillDashboard.tsx:539 :: <button type="button" className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700" onClick={() => console.log('Convert to JIT for', selectedPermission?.permissionId)}>