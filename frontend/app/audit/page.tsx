'use client';
// Simple Audit verifier UI: enter a contentHash, call Gateway /verify/{hash}, show Merkle proof.
import { useState } from 'react';
import { real } from '@/lib/real';

const API_BASE = process.env.NEXT_PUBLIC_REAL_API_BASE || '';
const TEST_BEARER = process.env.NEXT_PUBLIC_TEST_BEARER;

type VerifyResp = {
  verified: boolean;
  merkleRoot: string;
  proof: string[];
  day: string;
};

export default function AuditPage() {
  const [hash, setHash] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<VerifyResp | null>(null);

  async function exportEvidence() {
    try {
      if (!res?.verified) throw new Error('Verify a hash first');
      const url = `${API_BASE}/api/v1/evidence/export?hash=${encodeURIComponent(hash)}`;
      const headers: Record<string, string> = {};
      if (TEST_BEARER) headers['Authorization'] = `Bearer ${TEST_BEARER}`;
      const r = await fetch(url, { headers });
      if (!r.ok) throw new Error(`Export failed: ${r.status}`);
      const blob = await r.blob();
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `pcx-evidence-${hash}.json`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 0);
    } catch (e: any) {
      setErr(String(e?.message || e));
    }
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true); setErr(null); setRes(null);
    try {
      if (!/^[a-f0-9]{64}$/i.test(hash)) {
        throw new Error('Enter a 64-char SHA-256 hex hash');
      }
      const data = await real<VerifyResp>(`/api/v1/verify/${hash}`);
      setRes(data);
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl p-6">
      <h1 className="text-2xl font-semibold">Audit Verification</h1>
      <p className="text-sm text-zinc-500">Verify an evidence export hash (tamper-evident Merkle proof).</p>

      <form onSubmit={onSubmit} className="mt-4 flex gap-2 items-center">
        <input
          value={hash}
          onChange={(e)=>setHash(e.target.value.trim())}
          placeholder="aaaaaaaa... (64-hex)"
          className="w-full rounded-xl border px-3 py-2 text-sm"
          aria-label="Evidence hash"
        />
        <button
          disabled={loading}
          className="rounded-xl border px-4 py-2 text-sm font-medium hover:bg-zinc-50 disabled:opacity-60"
        >
          {loading ? 'Verifying…' : 'Verify'}
        </button>
      </form>

      {err && <div className="mt-3 text-sm text-red-600">Error: {err}</div>}

      {res && (
        <div className="mt-6 rounded-2xl border p-4">
          <div className="flex items-center justify-between">
            <div className="text-base font-medium">
              {res.verified ? '✔ Verified' : '✖ Not Found'}
            </div>
            <div className="text-xs text-zinc-500">Day: {res.day || '—'}</div>
          </div>
          {res.verified && (
            <>
              <div className="mt-2 text-sm">
                <div><span className="font-mono text-xs">merkleRoot</span>: <span className="font-mono">{res.merkleRoot}</span></div>
                <div className="text-zinc-500 text-xs">Proof length: {res.proof.length} hop(s)</div>
              </div>
              <details className="mt-3">
                <summary className="cursor-pointer text-sm">View Merkle proof</summary>
                <pre className="mt-2 text-xs overflow-auto rounded-xl bg-zinc-50 p-3">{JSON.stringify(res.proof, null, 2)}</pre>
              </details>
              <div className="mt-4">
                <button className="rounded-xl border px-4 py-2 text-sm font-medium hover:bg-zinc-50" onClick={exportEvidence}>Export Evidence</button>
              </div>
            </>
          )}
        </div>
      )}

      <div className="mt-8 text-xs text-zinc-500">
        Tip: In the demo stack, a sample evidence hash is seeded:
        <span className="font-mono"> aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</span>
      </div>
    </div>
  );
}