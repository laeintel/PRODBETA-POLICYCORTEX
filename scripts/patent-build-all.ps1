$roots = @(
  'NON-PROVISIONAL/Patent1_CrossDomainCorrelation',
  'NON-PROVISIONAL/Patent2_ConversationalGovernance',
  'NON-PROVISIONAL/Patent3_UnifiedAIGovernancePlatform',
  'NON-PROVISIONAL/Patent4_PredictivePolicyCompliance'
)

foreach ($p in $roots) {
  Write-Host "Building $p"
  scripts/patent-build.ps1 -PatentFolder $p
}
