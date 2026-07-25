'use client';

import { Job } from '@prisma/client';
import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import { Finding, FindingLevel } from '@/utils/preflight';
import { RunSummary } from '@/utils/runAnalysis';
import { LuTriangleAlert, LuCircleAlert, LuInfo, LuCircleCheck, LuLoader, LuStar, LuThumbsUp, LuThumbsDown } from 'react-icons/lu';

interface Props {
  job: Job;
}

const levelMeta: Record<FindingLevel, { icon: React.ReactNode; ring: string; text: string; label: string }> = {
  error: { icon: <LuCircleAlert />, ring: 'border-rose-500/40 bg-rose-500/5', text: 'text-rose-400', label: 'Problem' },
  warning: { icon: <LuTriangleAlert />, ring: 'border-amber-500/40 bg-amber-500/5', text: 'text-amber-400', label: 'Warning' },
  info: { icon: <LuInfo />, ring: 'border-sky-500/40 bg-sky-500/5', text: 'text-sky-400', label: 'Note' },
  ok: { icon: <LuCircleCheck />, ring: 'border-emerald-500/40 bg-emerald-500/5', text: 'text-emerald-400', label: 'OK' },
};

interface RunRating {
  score?: number | null;
  likeness?: 'good' | 'bad' | null;
  notes?: string;
  rated_at?: string;
}

function fmtDuration(sec: number | null): string {
  if (sec == null) return '—';
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  return `${s}s`;
}

export default function JobAnalysis({ job }: Props) {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [rating, setRating] = useState<RunRating>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([
      apiClient.get(`/api/jobs/${job.id}/analysis`).then(r => r.data).catch(() => null),
      apiClient.get(`/api/jobs/${job.id}/rating`).then(r => r.data).catch(() => null),
    ]).then(([analysis, ratingResp]) => {
      if (cancelled) return;
      if (analysis) {
        setSummary(analysis.summary ?? null);
        setFindings(analysis.findings ?? []);
      }
      if (ratingResp?.rating) setRating(ratingResp.rating);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [job.id]);

  const saveRating = async (next: RunRating) => {
    setRating(next);
    setSaving(true);
    setSaved(false);
    try {
      const resp = await apiClient.post(`/api/jobs/${job.id}/rating`, next);
      if (resp.data?.rating) setRating(resp.data.rating);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      // best-effort; leave optimistic state
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center gap-2 text-gray-400 py-16">
        <LuLoader className="animate-spin" /> Analyzing this run…
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 space-y-6">
      {/* Summary tiles */}
      <div>
        <h2 className="text-gray-100 text-lg font-medium mb-3">Run summary</h2>
        {summary && summary.sampleCount > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Stat label="Speed" value={summary.secPerStep != null ? `${summary.secPerStep.toFixed(1)} s/step` : '—'} />
            <Stat label="Elapsed" value={fmtDuration(summary.elapsedSec)} sub={summary.stepsCovered != null ? `${summary.stepsCovered} steps` : undefined} />
            <Stat
              label="Peak VRAM"
              value={summary.vramPeakPct != null ? `${summary.vramPeakPct.toFixed(0)}%` : '—'}
              sub={summary.vramPeakGB != null && summary.vramTotalGB != null ? `${summary.vramPeakGB.toFixed(1)}/${summary.vramTotalGB.toFixed(0)} GB` : undefined}
            />
            <Stat label="GPU util (mean)" value={summary.gpuMeanPct != null ? `${summary.gpuMeanPct.toFixed(0)}%` : '—'} sub={summary.gpuP90Pct != null ? `p90 ${summary.gpuP90Pct.toFixed(0)}%` : undefined} />
            <Stat label="Peak RAM" value={summary.ramPeakPct != null ? `${summary.ramPeakPct.toFixed(0)}%` : '—'} sub={summary.procRamPeakGB != null ? `proc ${summary.procRamPeakGB.toFixed(1)} GB` : undefined} />
            <Stat label="CPU (mean)" value={summary.cpuMeanPct != null ? `${summary.cpuMeanPct.toFixed(0)}%` : '—'} />
            <Stat label="Loss (mean)" value={summary.lossFirst != null && summary.lossLast != null ? `${summary.lossFirst.toFixed(3)} → ${summary.lossLast.toFixed(3)}` : '—'} sub={summary.lossDeltaPct != null ? `${summary.lossDeltaPct >= 0 ? '+' : ''}${summary.lossDeltaPct.toFixed(1)}%` : undefined} />
            <Stat label="Samples" value={String(summary.sampleCount)} sub="telemetry rows" />
          </div>
        ) : (
          <div className="text-sm text-gray-400">No telemetry recorded yet for this run.</div>
        )}
      </div>

      {/* Findings */}
      {findings.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-gray-100 text-lg font-medium">What the logs show</h2>
          {findings.map(f => {
            const meta = levelMeta[f.level];
            return (
              <div key={f.id} className={`rounded-lg border p-3 ${meta.ring}`}>
                <div className="flex items-start gap-2">
                  <span className={`mt-0.5 ${meta.text}`}>{meta.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs uppercase tracking-wide ${meta.text}`}>{meta.label}</span>
                      <span className="text-gray-100 text-sm font-medium">{f.title}</span>
                    </div>
                    <p className="text-sm text-gray-300 mt-1 leading-relaxed">{f.detail}</p>
                    {(f.setting || f.current || f.recommended) && (
                      <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-xs">
                        {f.setting && <span className="text-gray-500">Setting: <span className="text-gray-300 font-mono">{f.setting}</span></span>}
                        {f.current != null && <span className="text-gray-500">This run: <span className="text-amber-300 font-mono">{f.current}</span></span>}
                        {f.recommended != null && <span className="text-gray-500">Try: <span className="text-emerald-300 font-mono">{f.recommended}</span></span>}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Rating */}
      <div className="rounded-lg border border-gray-700 bg-gray-800/40 p-4">
        <h2 className="text-gray-100 text-lg font-medium">Rate this run</h2>
        <p className="text-sm text-gray-400 mt-1 mb-3">
          Loss and utilization can&apos;t tell whether the output is actually good — only you can. Your rating is the missing signal that lets
          future runs be compared and better settings learned over time.
        </p>

        <div className="flex flex-wrap items-center gap-6">
          {/* Stars */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 uppercase tracking-wide w-16">Quality</span>
            <div className="flex items-center gap-1">
              {[1, 2, 3, 4, 5].map(n => (
                <button
                  key={n}
                  type="button"
                  onClick={() => saveRating({ ...rating, score: n })}
                  className="p-0.5"
                  title={`${n} star${n > 1 ? 's' : ''}`}
                >
                  <LuStar className={`w-5 h-5 ${rating.score != null && n <= rating.score ? 'fill-amber-400 text-amber-400' : 'text-gray-600 hover:text-gray-400'}`} />
                </button>
              ))}
            </div>
          </div>

          {/* Likeness */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500 uppercase tracking-wide w-16">Likeness</span>
            <button
              type="button"
              onClick={() => saveRating({ ...rating, likeness: rating.likeness === 'good' ? null : 'good' })}
              className={`flex items-center gap-1 px-2 py-1 rounded-md text-sm border ${rating.likeness === 'good' ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300' : 'border-gray-700 text-gray-400 hover:text-gray-200'}`}
            >
              <LuThumbsUp className="w-4 h-4" /> Good
            </button>
            <button
              type="button"
              onClick={() => saveRating({ ...rating, likeness: rating.likeness === 'bad' ? null : 'bad' })}
              className={`flex items-center gap-1 px-2 py-1 rounded-md text-sm border ${rating.likeness === 'bad' ? 'border-rose-500/50 bg-rose-500/10 text-rose-300' : 'border-gray-700 text-gray-400 hover:text-gray-200'}`}
            >
              <LuThumbsDown className="w-4 h-4" /> Bad
            </button>
          </div>

          <div className="text-xs text-gray-500 ml-auto">
            {saving ? 'Saving…' : saved ? 'Saved ✓' : rating.rated_at ? `Rated ${new Date(rating.rated_at).toLocaleString()}` : 'Not rated yet'}
          </div>
        </div>

        {/* Notes */}
        <textarea
          value={rating.notes ?? ''}
          onChange={e => setRating({ ...rating, notes: e.target.value })}
          onBlur={() => saveRating(rating)}
          placeholder="Optional notes — what worked, what didn't, anything to remember for next time."
          className="mt-3 w-full h-20 rounded-md bg-gray-900 border border-gray-700 text-sm text-gray-200 p-2 resize-y focus:outline-none focus:border-gray-500"
        />
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/40 p-3">
      <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-gray-100 text-lg font-medium mt-0.5">{value}</div>
      {sub && <div className="text-[11px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}
