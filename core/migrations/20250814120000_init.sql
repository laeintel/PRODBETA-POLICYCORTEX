-- Create exceptions table used by exceptions APIs
CREATE TABLE IF NOT EXISTS exceptions (
  id UUID PRIMARY KEY,
  tenant_id TEXT NOT NULL,
  resource_id TEXT NOT NULL,
  policy_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  status TEXT NOT NULL,
  created_by TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  recertify_at TIMESTAMPTZ NULL,
  evidence JSONB NULL,
  metadata JSONB NULL
);

CREATE INDEX IF NOT EXISTS idx_exceptions_tenant ON exceptions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_exceptions_expires ON exceptions(expires_at);
