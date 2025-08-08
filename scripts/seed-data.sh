#!/bin/bash

# PolicyCortex v2 - Seed Data Script

echo "🌱 Seeding PolicyCortex v2 Development Data"
echo "=========================================="

# Wait for services to be ready
echo "⏳ Waiting for services..."
sleep 5

# Check if PostgreSQL is ready
until docker exec policycortex-v2-postgres-1 pg_isready -U postgres; do
  echo "Waiting for PostgreSQL..."
  sleep 2
done

# Check if EventStore is ready
until curl -s http://localhost:2113/stats > /dev/null; do
  echo "Waiting for EventStore..."
  sleep 2
done

echo "✅ Services are ready"
echo ""

# Seed PostgreSQL
echo "📊 Seeding PostgreSQL..."
docker exec -i policycortex-v2-postgres-1 psql -U postgres -d policycortex << EOF
-- Organizations
INSERT INTO organizations (id, name, tier, created_at) VALUES
  ('org-1', 'Contoso Corporation', 'enterprise', NOW()),
  ('org-2', 'Fabrikam Industries', 'professional', NOW()),
  ('org-3', 'Adventure Works', 'starter', NOW());

-- Users
INSERT INTO users (id, organization_id, email, name, role, created_at) VALUES
  ('user-1', 'org-1', 'admin@contoso.com', 'Admin User', 'admin', NOW()),
  ('user-2', 'org-1', 'analyst@contoso.com', 'Policy Analyst', 'analyst', NOW()),
  ('user-3', 'org-2', 'admin@fabrikam.com', 'Fabrikam Admin', 'admin', NOW());

-- Azure Subscriptions
INSERT INTO azure_subscriptions (id, organization_id, subscription_id, name, created_at) VALUES
  ('sub-1', 'org-1', 'sub-contoso-prod', 'Contoso Production', NOW()),
  ('sub-2', 'org-1', 'sub-contoso-dev', 'Contoso Development', NOW()),
  ('sub-3', 'org-2', 'sub-fabrikam-prod', 'Fabrikam Production', NOW());

-- Policies
INSERT INTO policies (id, organization_id, name, category, severity, status, created_at) VALUES
  ('pol-1', 'org-1', 'Require HTTPS for Storage Accounts', 'Security', 'high', 'active', NOW()),
  ('pol-2', 'org-1', 'Enforce Tagging Standards', 'Governance', 'medium', 'active', NOW()),
  ('pol-3', 'org-1', 'Restrict VM SKUs', 'Cost', 'medium', 'active', NOW()),
  ('pol-4', 'org-2', 'Require MFA for Admin Accounts', 'Security', 'critical', 'active', NOW()),
  ('pol-5', 'org-2', 'Backup Policy for Databases', 'Resilience', 'high', 'draft', NOW());

-- Resources
INSERT INTO resources (id, subscription_id, resource_id, name, type, location, tags, created_at) VALUES
  ('res-1', 'sub-1', '/subscriptions/sub-1/resourceGroups/rg-prod/providers/Microsoft.Storage/storageAccounts/stcontoso01', 'stcontoso01', 'Microsoft.Storage/storageAccounts', 'eastus', '{"env": "prod", "dept": "finance"}', NOW()),
  ('res-2', 'sub-1', '/subscriptions/sub-1/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/vm-web-01', 'vm-web-01', 'Microsoft.Compute/virtualMachines', 'eastus', '{"env": "prod", "dept": "it"}', NOW()),
  ('res-3', 'sub-2', '/subscriptions/sub-2/resourceGroups/rg-dev/providers/Microsoft.Sql/servers/sql-dev-01', 'sql-dev-01', 'Microsoft.Sql/servers', 'westus', '{"env": "dev"}', NOW());

-- Compliance Results
INSERT INTO compliance_results (id, policy_id, resource_id, status, reason, checked_at) VALUES
  ('comp-1', 'pol-1', 'res-1', 'compliant', 'HTTPS is enforced', NOW()),
  ('comp-2', 'pol-2', 'res-1', 'compliant', 'All required tags present', NOW()),
  ('comp-3', 'pol-2', 'res-2', 'non_compliant', 'Missing required tag: cost-center', NOW()),
  ('comp-4', 'pol-3', 'res-2', 'compliant', 'VM SKU is in allowed list', NOW());

-- Achievements (Gamification)
INSERT INTO achievements (id, name, description, points, icon, criteria) VALUES
  ('ach-1', 'First Policy', 'Create your first policy', 10, '🎯', '{"type": "policy_count", "value": 1}'),
  ('ach-2', 'Compliance Champion', 'Achieve 95% compliance', 50, '🏆', '{"type": "compliance_rate", "value": 95}'),
  ('ach-3', 'Cost Saver', 'Save $1000 through optimization', 100, '💰', '{"type": "cost_saved", "value": 1000}'),
  ('ach-4', 'Security Expert', 'Fix 10 security issues', 25, '🛡️', '{"type": "security_fixes", "value": 10}');

-- User Achievements
INSERT INTO user_achievements (user_id, achievement_id, earned_at) VALUES
  ('user-1', 'ach-1', NOW() - INTERVAL '7 days'),
  ('user-1', 'ach-4', NOW() - INTERVAL '3 days'),
  ('user-2', 'ach-1', NOW() - INTERVAL '5 days');

-- Policy Templates (Marketplace)
INSERT INTO policy_templates (id, name, description, category, author, downloads, rating, price, template) VALUES
  ('tpl-1', 'CIS Azure Foundations Benchmark', 'Complete CIS benchmark implementation', 'Security', 'PolicyCortex', 1250, 4.8, 0, '{}'),
  ('tpl-2', 'FinOps Cost Optimization Pack', 'Comprehensive cost optimization policies', 'Cost', 'Community', 890, 4.6, 49.99, '{}'),
  ('tpl-3', 'HIPAA Compliance Suite', 'Healthcare compliance policies', 'Compliance', 'HealthTech Corp', 450, 4.9, 199.99, '{}');

COMMIT;
EOF

echo "✅ PostgreSQL seeded"
echo ""

# Seed EventStore with sample events
echo "📝 Seeding EventStore..."
curl -X POST http://localhost:2113/streams/policy-events \
  -H "Content-Type: application/vnd.eventstore.events+json" \
  -d '[
    {
      "eventId": "'"$(uuidgen)"'",
      "eventType": "PolicyCreated",
      "data": {
        "policyId": "pol-1",
        "name": "Require HTTPS for Storage Accounts",
        "createdBy": "user-1",
        "timestamp": "2024-01-15T10:00:00Z"
      }
    },
    {
      "eventId": "'"$(uuidgen)"'",
      "eventType": "PolicyActivated",
      "data": {
        "policyId": "pol-1",
        "activatedBy": "user-1",
        "timestamp": "2024-01-15T10:05:00Z"
      }
    },
    {
      "eventId": "'"$(uuidgen)"'",
      "eventType": "ComplianceChecked",
      "data": {
        "policyId": "pol-1",
        "resourceCount": 25,
        "compliantCount": 23,
        "timestamp": "2024-01-15T11:00:00Z"
      }
    }
  ]' > /dev/null 2>&1

echo "✅ EventStore seeded"
echo ""

# Seed Redis/DragonflyDB cache
echo "💾 Seeding cache..."
docker exec policycortex-v2-dragonfly-1 redis-cli << EOF
SET session:user-1 '{"userId":"user-1","org":"org-1","role":"admin"}' EX 3600
SET cache:policies:org-1 '[{"id":"pol-1","name":"Require HTTPS"},{"id":"pol-2","name":"Enforce Tagging"}]' EX 300
SET rate_limit:user-1 "10" EX 60
ZADD leaderboard 150 "user-1" 85 "user-2" 45 "user-3"
HSET feature_flags "quantum_optimization" "true" "blockchain_audit" "true" "ai_recommendations" "true"
QUIT
EOF

echo "✅ Cache seeded"
echo ""

echo "🎉 Seed data loaded successfully!"
echo ""
echo "📊 Summary:"
echo "  - 3 Organizations"
echo "  - 3 Users"
echo "  - 3 Azure Subscriptions"
echo "  - 5 Policies"
echo "  - 3 Resources"
echo "  - 4 Compliance Results"
echo "  - 4 Achievements"
echo "  - 3 Policy Templates"
echo "  - Sample events in EventStore"
echo "  - Cache data in DragonflyDB"
echo ""
echo "🚀 You can now access the application at:"
echo "   http://localhost:3000"
echo ""
echo "📧 Test credentials:"
echo "   admin@contoso.com (Admin)"
echo "   analyst@contoso.com (Analyst)"
echo "   admin@fabrikam.com (Admin)"