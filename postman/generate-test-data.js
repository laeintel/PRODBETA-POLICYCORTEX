/**
 * Test Data Generator for PolicyCortex API
 * Generates realistic test data for API testing
 */

const fs = require('fs');
const path = require('path');

// Generate test policies
const generatePolicies = (count = 10) => {
    const policyTypes = ['BuiltIn', 'Custom', 'Initiative'];
    const categories = ['Security', 'Compliance', 'Cost Management', 'Performance', 'Governance'];
    const effects = ['Audit', 'Deny', 'Deploy', 'AuditIfNotExists', 'DeployIfNotExists'];
    
    return Array.from({ length: count }, (_, i) => ({
        id: `/providers/Microsoft.Authorization/policyDefinitions/test-policy-${i + 1}`,
        displayName: `Test Policy ${i + 1}`,
        description: `Test policy for ${categories[i % categories.length]} compliance`,
        policyType: policyTypes[i % policyTypes.length],
        category: categories[i % categories.length],
        effect: effects[i % effects.length],
        compliancePercentage: Math.floor(Math.random() * 40) + 60, // 60-100%
        resourcesScanned: Math.floor(Math.random() * 100) + 50,
        nonCompliantResources: Math.floor(Math.random() * 20),
        metadata: {
            version: '1.0.0',
            createdOn: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString(),
            updatedOn: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString()
        }
    }));
};

// Generate test resources
const generateResources = (count = 50) => {
    const resourceTypes = [
        'Microsoft.Compute/virtualMachines',
        'Microsoft.Storage/storageAccounts',
        'Microsoft.Network/virtualNetworks',
        'Microsoft.Sql/servers',
        'Microsoft.Web/sites',
        'Microsoft.KeyVault/vaults'
    ];
    
    const complianceStates = ['Compliant', 'NonCompliant', 'Unknown'];
    const locations = ['eastus', 'westus2', 'centralus', 'northeurope', 'westeurope'];
    
    return Array.from({ length: count }, (_, i) => ({
        id: `/subscriptions/test-sub/resourceGroups/test-rg/providers/${resourceTypes[i % resourceTypes.length]}/test-resource-${i + 1}`,
        name: `test-resource-${i + 1}`,
        type: resourceTypes[i % resourceTypes.length],
        location: locations[i % locations.length],
        resourceGroup: `test-rg-${Math.floor(i / 10) + 1}`,
        complianceState: complianceStates[Math.random() < 0.7 ? 0 : Math.floor(Math.random() * 3)],
        lastEvaluated: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString(),
        tags: {
            Environment: ['Dev', 'Test', 'Prod'][i % 3],
            Owner: `team-${Math.floor(i / 5) + 1}`,
            CostCenter: `CC${1000 + i}`
        }
    }));
};

// Generate test conversations
const generateConversations = (count = 5) => {
    const topics = [
        'Compliance improvement strategies',
        'Cost optimization recommendations',
        'Security best practices',
        'Policy violation analysis',
        'Resource governance planning'
    ];
    
    return Array.from({ length: count }, (_, i) => ({
        id: `conv-${Date.now()}-${i}`,
        title: topics[i % topics.length],
        created_at: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
        updated_at: new Date(Date.now() - i * 60 * 60 * 1000).toISOString(),
        messages: [
            {
                id: `msg-${i}-1`,
                role: 'user',
                content: `What are the main ${topics[i % topics.length].toLowerCase()} for my Azure environment?`,
                timestamp: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString()
            },
            {
                id: `msg-${i}-2`,
                role: 'assistant',
                content: `Based on your current Azure configuration, here are the key recommendations for ${topics[i % topics.length].toLowerCase()}...`,
                timestamp: new Date(Date.now() - i * 24 * 60 * 60 * 1000 + 5000).toISOString()
            }
        ]
    }));
};

// Generate test compliance data
const generateComplianceData = () => {
    const days = 30;
    const data = [];
    
    for (let i = 0; i < days; i++) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        
        data.push({
            date: date.toISOString().split('T')[0],
            overallCompliance: Math.floor(Math.random() * 20) + 75, // 75-95%
            compliantResources: Math.floor(Math.random() * 50) + 150,
            nonCompliantResources: Math.floor(Math.random() * 30) + 20,
            totalResources: Math.floor(Math.random() * 50) + 200,
            byCategory: {
                Security: Math.floor(Math.random() * 15) + 80,
                Compliance: Math.floor(Math.random() * 15) + 85,
                'Cost Management': Math.floor(Math.random() * 20) + 70,
                Performance: Math.floor(Math.random() * 10) + 85,
                Governance: Math.floor(Math.random() * 15) + 80
            }
        });
    }
    
    return data;
};

// Write test data to files
const testData = {
    policies: generatePolicies(20),
    resources: generateResources(100),
    conversations: generateConversations(10),
    complianceTrends: generateComplianceData(),
    timestamp: new Date().toISOString()
};

// Create test-data directory if it doesn't exist
const testDataDir = path.join(__dirname, 'test-data');
if (!fs.existsSync(testDataDir)) {
    fs.mkdirSync(testDataDir);
}

// Write individual data files
fs.writeFileSync(
    path.join(testDataDir, 'policies.json'),
    JSON.stringify(testData.policies, null, 2)
);

fs.writeFileSync(
    path.join(testDataDir, 'resources.json'),
    JSON.stringify(testData.resources, null, 2)
);

fs.writeFileSync(
    path.join(testDataDir, 'conversations.json'),
    JSON.stringify(testData.conversations, null, 2)
);

fs.writeFileSync(
    path.join(testDataDir, 'compliance-trends.json'),
    JSON.stringify(testData.complianceTrends, null, 2)
);

// Write complete test data
fs.writeFileSync(
    path.join(testDataDir, 'complete-test-data.json'),
    JSON.stringify(testData, null, 2)
);

console.log('✅ Test data generated successfully!');
console.log(`📁 Files created in: ${testDataDir}`);
console.log(`- policies.json (${testData.policies.length} policies)`);
console.log(`- resources.json (${testData.resources.length} resources)`);
console.log(`- conversations.json (${testData.conversations.length} conversations)`);
console.log(`- compliance-trends.json (${testData.complianceTrends.length} days)`);
console.log(`- complete-test-data.json (all data combined)`);