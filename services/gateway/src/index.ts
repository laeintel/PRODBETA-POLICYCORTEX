import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { createRemoteJWKSet, jwtVerify, JWTPayload } from 'jose';
import { createClient } from 'redis';
import { Registry, collectDefaultMetrics, Counter, Histogram } from 'prom-client';

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
const WINDOW_SEC = Number(process.env.RATELIMIT_WINDOW_SEC || '60');
const USER_LIMIT = Number(process.env.RATE_LIMIT_USER_PER_MIN || '120');
const TENANT_LIMIT = Number(process.env.RATE_LIMIT_TENANT_PER_MIN || '1200');
const SAMPLE = Number(process.env.TRACE_SAMPLE || '1') === 1;

if (!ISSUER || !AUDIENCE || !JWKS_URL) {
  console.warn('[gateway] WARNING: JWT_ISSUER, JWT_AUDIENCE, JWT_JWKS_URL are not fully set.');
}

// Redis client (optional - works without it)
let redis: ReturnType<typeof createClient> | null = null;
(async () => {
  try {
    redis = createClient({ url: process.env.REDIS_URL });
    await redis.connect();
    console.log('[gateway] Redis connected for rate limiting');
  } catch (e) {
    console.warn('[gateway] Redis not available, rate limiting disabled');
  }
})();

// ---- Metrics (Prometheus)
const reg = new Registry();
if ((process.env.PROM_ENABLE_DEFAULT_METRICS || 'true') === 'true') {
  collectDefaultMetrics({ register: reg });
}
const httpReqs = new Counter({
  name: 'pcx_http_requests_total',
  help: 'Requests count',
  registers: [reg],
  labelNames: ['route', 'method', 'status'] as const
});
const httpDur = new Histogram({
  name: 'pcx_http_request_duration_seconds',
  help: 'Request duration (s)',
  registers: [reg],
  labelNames: ['route', 'method'] as const,
  buckets: [0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
});
const rateLimited = new Counter({
  name: 'pcx_rate_limited_total',
  help: 'Requests blocked by rate limiter',
  registers: [reg]
});

// ---- App
const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// ---- W3C traceparent utilities
function randHex(bytes: number) {
  const crypto = require('crypto');
  return crypto.randomBytes(bytes).toString('hex');
}

function ensureTraceparent(req: express.Request, res: express.Response) {
  let tp = req.get('traceparent');
  if (!tp || !/^00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]$/i.test(tp)) {
    if (!SAMPLE) return; // sampling off—don't add
    const traceId = randHex(16);
    const spanId = randHex(8);
    tp = `00-${traceId}-${spanId}-01`;
    req.headers['traceparent'] = tp;
  }
  if (tp) res.setHeader('traceparent', tp);
  return tp;
}

// per-request timing + traceparent
app.use((req, res, next) => {
  const start = process.hrtime.bigint();
  const routeTag = req.path.split('?')[0];
  const tp = ensureTraceparent(req, res);
  res.on('finish', () => {
    const end = process.hrtime.bigint();
    const sec = Number(end - start) / 1e9;
    httpDur.labels(routeTag, req.method).observe(sec);
    httpReqs.labels(routeTag, req.method, String(res.statusCode)).inc(1);
    if (res.statusCode === 429) rateLimited.inc();
  });
  next();
});

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

async function slidingWindowAllow(key: string, limit: number, nowMs: number): Promise<boolean> {
  if (!redis) return true; // If Redis is not available, allow all requests
  const windowStart = nowMs - WINDOW_SEC * 1000;
  // prune + count + add atomically
  const multi = redis.multi();
  multi.zRemRangeByScore(key, 0, windowStart);
  multi.zCard(key);
  multi.zAdd(key, { score: nowMs, value: `${nowMs}:${Math.random()}` });
  multi.expire(key, WINDOW_SEC);
  const results = await multi.exec();
  const cardResp = results[1] as number;
  return cardResp < limit;
}

const rateLimit = async (req: express.Request, res: express.Response, next: express.NextFunction) => {
  if (!redis) return next(); // Skip rate limiting if Redis is not available
  const u = (req as any).user as { sub?: string; tid?: string; role?: Role } | undefined;
  const sub = u?.sub || 'anon';
  const tid = u?.tid || 'unknown';
  const now = Date.now();
  if (!(await slidingWindowAllow(`rl:u:${sub}`, USER_LIMIT, now))) return res.status(429).json({ error: 'rate_limited', scope: 'user' });
  if (!(await slidingWindowAllow(`rl:t:${tid}`, TENANT_LIMIT, now))) return res.status(429).json({ error: 'rate_limited', scope: 'tenant' });
  return next();
};

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

// ---- Public health & metrics
app.get('/health', (_req, res) => res.json({ ok: true, time: new Date().toISOString() }));
app.get('/metrics', async (_req, res) => res.type('text/plain').send(await reg.metrics()));

// ---- RBAC policy (default deny)
// Core (events/evidence): auditors & admins can read; operators/admins can write events
app.use(
  '/api/v1/events',
  requireRole(['operator', 'admin']), rateLimit,
  createProxyMiddleware({
    target: CORE_URL,
    changeOrigin: true,
    xfwd: true,
    onProxyReq: (proxyReq, req) => {
      const tp = req.get('traceparent');
      if (tp) proxyReq.setHeader('traceparent', tp);
    }
  })
);
app.get(
  '/api/v1/events/replay',
  requireRole(['auditor', 'admin', 'operator']), rateLimit,
  createProxyMiddleware({ target: CORE_URL, changeOrigin: true, xfwd: true,
    onProxyReq: (proxyReq, req) => {
      const tp = req.get('traceparent');
      if (tp) proxyReq.setHeader('traceparent', tp);
    }
  })
);
app.get(
  '/api/v1/verify/*',
  requireRole(['auditor', 'admin', 'operator']), rateLimit,
  createProxyMiddleware({ target: CORE_URL, changeOrigin: true, xfwd: true,
    onProxyReq: (proxyReq, req) => {
      const tp = req.get('traceparent');
      if (tp) proxyReq.setHeader('traceparent', tp);
    }
  })
);

// Azure Agents (predictions + P&L): all roles can read
app.get(
  '/api/v1/predictions',
  requireRole(['auditor', 'admin', 'operator']), rateLimit,
  createProxyMiddleware({ target: AZURE_AGENTS_URL, changeOrigin: true, xfwd: true,
    onProxyReq: (proxyReq, req) => {
      const tp = req.get('traceparent');
      if (tp) proxyReq.setHeader('traceparent', tp);
    }
  })
);
app.get(
  '/api/v1/costs/pnl',
  requireRole(['auditor', 'admin', 'operator']), rateLimit,
  createProxyMiddleware({ target: AZURE_AGENTS_URL, changeOrigin: true, xfwd: true,
    onProxyReq: (proxyReq, req) => {
      const tp = req.get('traceparent');
      if (tp) proxyReq.setHeader('traceparent', tp);
    }
  })
);

// ---- Fallback
app.use((_req, res) => res.status(404).json({ error: 'not_found' }));

app.listen(PORT, () => {
  console.log(`[gateway] listening on http://0.0.0.0:${PORT}`);
  console.log(`[gateway] → CORE_URL=${CORE_URL}  AZURE_AGENTS_URL=${AZURE_AGENTS_URL}`);
});