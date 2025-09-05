import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { createRemoteJWKSet, jwtVerify, JWTPayload } from 'jose';

// ---- Env
const PORT = Number(process.env.GATEWAY_PORT || 8000);
const CORE_URL = process.env.CORE_URL || 'http://localhost:8081';
const AZURE_AGENTS_URL = process.env.AZURE_AGENTS_URL || 'http://localhost:8084';
const ISSUER = process.env.JWT_ISSUER!;
const AUDIENCE = process.env.JWT_AUDIENCE!;
const JWKS_URL = process.env.JWT_JWKS_URL!;
const ADMIN_GROUP_IDS = (process.env.ADMIN_GROUP_IDS || '').split(',').map(s => s.trim()).filter(Boolean);
const AUDITOR_GROUP_IDS = (process.env.AUDITOR_GROUP_IDS || '').split(',').map(s => s.trim()).filter(Boolean);
const OPERATOR_GROUP_IDS = (process.env.OPERATOR_GROUP_IDS || '').split(',').map(s => s.trim()).filter(Boolean);

if (!ISSUER || !AUDIENCE || !JWKS_URL) {
  console.warn('[gateway] WARNING: JWT_ISSUER, JWT_AUDIENCE, JWT_JWKS_URL are not fully set.');
}

// ---- App
const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// ---- Auth helpers
const JWKS = createRemoteJWKSet(new URL(JWKS_URL));
type Role = 'admin' | 'auditor' | 'operator' | 'unknown';

function mapRoleFromPayload(p: JWTPayload): Role {
  const groups: string[] = Array.isArray((p as any).groups) ? (p as any).groups : [];
  const inAny = (ids: string[]) => groups.some(g => ids.includes(g));
  if (inAny(ADMIN_GROUP_IDS)) return 'admin';
  if (inAny(AUDITOR_GROUP_IDS)) return 'auditor';
  if (inAny(OPERATOR_GROUP_IDS)) return 'operator';
  return 'unknown';
}

async function verifyBearer(authHeader?: string): Promise<{ role: Role; payload: JWTPayload } | null> {
  if (!authHeader || !authHeader.startsWith('Bearer ')) return null;
  const token = authHeader.slice(7);
  const { payload } = await jwtVerify(token, JWKS, { issuer: ISSUER, audience: AUDIENCE });
  const role = mapRoleFromPayload(payload);
  return { role, payload };
}

function requireRole(allowed: Role[]) {
  return async (req: express.Request, res: express.Response, next: express.NextFunction) => {
    try {
      const v = await verifyBearer(req.get('authorization'));
      if (!v) return res.status(401).json({ error: 'missing_or_invalid_token' });
      if (!allowed.includes(v.role)) return res.status(403).json({ error: 'forbidden', role: v.role });
      // attach user to request
      (req as any).user = { role: v.role, sub: v.payload.sub, tid: (v.payload as any).tid, oid: (v.payload as any).oid };
      return next();
    } catch (e: any) {
      return res.status(401).json({ error: 'token_verify_failed', detail: String(e?.message || e) });
    }
  };
}

// ---- Public health
app.get('/health', (_req, res) => res.json({ ok: true, time: new Date().toISOString() }));

// ---- RBAC policy (default deny)
// Core (events/evidence): auditors & admins can read; operators/admins can write events
app.use(
  '/api/v1/events',
  requireRole(['operator', 'admin']),
  createProxyMiddleware({
    target: CORE_URL,
    changeOrigin: true,
    xfwd: true
  })
);
app.get(
  '/api/v1/events/replay',
  requireRole(['auditor', 'admin', 'operator']),
  createProxyMiddleware({ target: CORE_URL, changeOrigin: true, xfwd: true })
);
app.get(
  '/api/v1/verify/*',
  requireRole(['auditor', 'admin', 'operator']),
  createProxyMiddleware({ target: CORE_URL, changeOrigin: true, xfwd: true })
);

// Azure Agents (predictions + P&L): all roles can read
app.get(
  '/api/v1/predictions',
  requireRole(['auditor', 'admin', 'operator']),
  createProxyMiddleware({ target: AZURE_AGENTS_URL, changeOrigin: true, xfwd: true })
);
app.get(
  '/api/v1/costs/pnl',
  requireRole(['auditor', 'admin', 'operator']),
  createProxyMiddleware({ target: AZURE_AGENTS_URL, changeOrigin: true, xfwd: true })
);

// ---- Fallback
app.use((_req, res) => res.status(404).json({ error: 'not_found' }));

app.listen(PORT, () => {
  console.log(`[gateway] listening on http://0.0.0.0:${PORT}`);
  console.log(`[gateway] → CORE_URL=${CORE_URL}  AZURE_AGENTS_URL=${AZURE_AGENTS_URL}`);
});