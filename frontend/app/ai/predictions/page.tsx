// T06: Predictions page with explanations + Create Fix PR link
import { real } from '@/lib/real';

type Item = {
  ruleId: string;
  etaDays: number;
  confidence: number;
  repo: string;
  fixBranch: string;
  explanations?: { top?: [string, number][] };
};

export default async function PredictionsPage() {
  const data = await real<{ items: Item[] }>('/api/v1/predictions');
  const items = (data.items || []).sort((a, b) => b.confidence - a.confidence);
  return (
    <div className="mx-auto max-w-5xl p-6">
      <h1 className="text-2xl font-semibold">AI Predictions</h1>
      <p className="text-sm text-zinc-500">Top risks with ETA, confidence, explanations, and one-click PR.</p>
      <div className="mt-6 space-y-4">
        {items.map((p) => (
          <div key={p.ruleId} className="rounded-2xl border p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-base font-medium">Rule: {p.ruleId}</div>
                <div className="text-sm text-zinc-500">
                  ETA: {p.etaDays} day(s) • Confidence: {(p.confidence * 100).toFixed(0)}%
                </div>
              </div>
              <a
                className="rounded-xl px-3 py-2 text-sm font-medium border hover:bg-zinc-50"
                href={`https://github.com/${p.repo}/compare/${p.fixBranch}?quick_pull=1&title=Auto-fix:${encodeURIComponent(
                  p.ruleId
                )}`}
                target="_blank"
                rel="noreferrer"
              >
                Create Fix PR
              </a>
            </div>
            {p.explanations?.top?.length ? (
              <div className="mt-3">
                <div className="text-xs font-semibold uppercase text-zinc-500">Top Factors</div>
                <ul className="mt-1 text-sm">
                  {p.explanations.top.slice(0, 5).map(([f, w]) => (
                    <li key={f}>
                      {f}: {w.toFixed(2)}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ))}
        {!items.length && <div className="text-sm text-zinc-500">No predictions available.</div>}
      </div>
    </div>
  );
}