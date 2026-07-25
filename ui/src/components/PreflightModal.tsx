'use client';

import { Dialog, DialogPanel, DialogTitle } from '@headlessui/react';
import { useEffect, useState } from 'react';
import { JobConfig } from '@/types';
import { apiClient } from '@/utils/api';
import { analyzePreflight, Finding, FindingLevel, PreflightHardware } from '@/utils/preflight';
import { reviewTrainingConfig } from '@/utils/configReview';
import { LuTriangleAlert, LuCircleAlert, LuInfo, LuCircleCheck, LuLoader, LuCpu, LuMemoryStick, LuHardDrive } from 'react-icons/lu';

interface Props {
  open: boolean;
  jobConfig: JobConfig | null;
  onConfirm: () => void;
  onCancel: () => void;
}

const levelMeta: Record<FindingLevel, { icon: React.ReactNode; ring: string; text: string; label: string }> = {
  error: { icon: <LuCircleAlert />, ring: 'border-rose-500/40 bg-rose-500/5', text: 'text-rose-400', label: 'Problem' },
  warning: { icon: <LuTriangleAlert />, ring: 'border-amber-500/40 bg-amber-500/5', text: 'text-amber-400', label: 'Warning' },
  info: { icon: <LuInfo />, ring: 'border-sky-500/40 bg-sky-500/5', text: 'text-sky-400', label: 'Note' },
  ok: { icon: <LuCircleCheck />, ring: 'border-emerald-500/40 bg-emerald-500/5', text: 'text-emerald-400', label: 'OK' },
};

export default function PreflightModal({ open, jobConfig, onConfirm, onCancel }: Props) {
  const [loading, setLoading] = useState(false);
  const [hw, setHw] = useState<PreflightHardware | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !jobConfig) return;
    let cancelled = false;
    setLoading(true);
    setErr(null);

    Promise.all([
      apiClient.get('/api/gpu').then(r => r.data).catch(() => null),
      apiClient.get('/api/cpu').then(r => r.data).catch(() => null),
      apiClient.get('/api/disk').then(r => r.data).catch(() => null),
      apiClient.get('/api/datasets/stats').then(r => r.data).catch(() => null),
    ])
      .then(([gpu, cpu, disk, dsStats]) => {
        if (cancelled) return;
        const hardware: PreflightHardware = {
          gpus: (gpu?.gpus ?? []).map((g: any) => ({
            name: g.name,
            memTotalMB: g.memory?.total ?? 0,
            memFreeMB: g.memory?.free ?? 0,
          })),
          cpuCores: cpu?.cores ?? 0,
          ramTotalMB: cpu?.totalMemory ?? 0,
          ramFreeMB: cpu?.availableMemory ?? cpu?.freeMemory ?? 0,
          diskFreeGB: disk?.training ? disk.training.freeBytes / 1024 ** 3 : null,
          diskTotalGB: disk?.training ? disk.training.totalBytes / 1024 ** 3 : null,
        };
        setHw(hardware);

        // Sum real image counts for the job's datasets by matching folder
        // basenames against /api/datasets/stats. null if stats unavailable, so
        // the steps-per-image check is skipped rather than guessed.
        let imageCount: number | null = null;
        const statList: { name: string; image_count: number }[] = dsStats?.datasets ?? [];
        if (statList.length > 0) {
          const byName = new Map(statList.map(s => [s.name, s.image_count]));
          const jobDatasets = jobConfig.config?.process?.[0]?.datasets ?? [];
          let sum = 0;
          let matched = 0;
          for (const d of jobDatasets) {
            const base = (d.folder_path || '').replace(/[/\\]+$/, '').split(/[/\\]/).pop() || '';
            if (byName.has(base)) {
              sum += byName.get(base) ?? 0;
              matched += 1;
            }
          }
          if (matched > 0) imageCount = sum;
        }

        const order: Record<FindingLevel, number> = { error: 0, warning: 1, info: 2, ok: 3 };
        const merged = [...analyzePreflight(jobConfig, hardware), ...reviewTrainingConfig(jobConfig, imageCount, hardware)];
        merged.sort((a, b) => order[a.level] - order[b.level]);
        setFindings(merged);
      })
      .catch(e => {
        if (!cancelled) setErr(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, jobConfig]);

  const errorCount = findings.filter(f => f.level === 'error').length;
  const warnCount = findings.filter(f => f.level === 'warning').length;
  const problemCount = errorCount + warnCount;

  const proceedLabel = errorCount > 0 ? 'Create anyway' : warnCount > 0 ? 'Create anyway' : 'Looks good — Create';

  return (
    <Dialog open={open} onClose={onCancel} className="relative z-50">
      <div className="fixed inset-0 bg-black/60" aria-hidden="true" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="w-full max-w-3xl max-h-[85vh] flex flex-col rounded-xl bg-gray-900 border border-gray-700 shadow-2xl">
          <div className="px-5 py-4 border-b border-gray-800 shrink-0">
            <DialogTitle className="text-gray-100 text-lg font-medium">Pre-flight check</DialogTitle>
            <p className="text-sm text-gray-400 mt-0.5">
              We checked your settings against this machine and reviewed the training config. Review any advice below, then confirm.
            </p>
          </div>

          {/* Hardware summary */}
          {hw && (
            <div className="px-5 py-3 border-b border-gray-800 shrink-0 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <HwStat icon={<LuMemoryStick />} label="GPU / VRAM" value={hw.gpus[0] ? `${hw.gpus[0].name.replace(/NVIDIA\s*/i, '')} · ${(hw.gpus[0].memTotalMB / 1024).toFixed(0)} GB` : 'none'} />
              <HwStat icon={<LuMemoryStick />} label="System RAM" value={`${(hw.ramTotalMB / 1024).toFixed(0)} GB`} />
              <HwStat icon={<LuCpu />} label="CPU cores" value={`${hw.cpuCores || '?'}`} />
              <HwStat icon={<LuHardDrive />} label="Disk free" value={hw.diskFreeGB != null ? `${hw.diskFreeGB.toFixed(0)} GB` : '?'} />
            </div>
          )}

          {/* Findings */}
          <div className="flex-1 overflow-y-auto px-5 py-4 space-y-3 min-h-[120px]">
            {loading ? (
              <div className="flex items-center justify-center gap-2 text-gray-400 py-10">
                <LuLoader className="animate-spin" /> Inspecting hardware and settings…
              </div>
            ) : err ? (
              <div className="text-rose-400 text-sm">Could not run the check: {err}. You can still create the job.</div>
            ) : problemCount === 0 ? (
              <div className="flex flex-col items-center justify-center text-center py-8 gap-2">
                <LuCircleCheck className="w-8 h-8 text-emerald-400" />
                <div className="text-gray-100 font-medium">No problems found</div>
                <div className="text-sm text-gray-400">Your settings look compatible with this machine.</div>
                {findings.length > 0 && <div className="text-xs text-gray-500">{findings.length} note(s) below.</div>}
              </div>
            ) : null}

            {!loading &&
              findings.map(f => {
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
                            {f.setting && (
                              <span className="text-gray-500">
                                Setting: <span className="text-gray-300 font-mono">{f.setting}</span>
                              </span>
                            )}
                            {f.current != null && (
                              <span className="text-gray-500">
                                Yours: <span className="text-amber-300 font-mono">{f.current}</span>
                              </span>
                            )}
                            {f.recommended != null && (
                              <span className="text-gray-500">
                                Recommended: <span className="text-emerald-300 font-mono">{f.recommended}</span>
                              </span>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>

          {/* Actions */}
          <div className="px-5 py-4 border-t border-gray-800 shrink-0 flex items-center justify-between gap-3">
            <div className="text-xs text-gray-500">
              {problemCount > 0 ? (
                <>
                  {errorCount > 0 && <span className="text-rose-400">{errorCount} problem(s)</span>}
                  {errorCount > 0 && warnCount > 0 && ' · '}
                  {warnCount > 0 && <span className="text-amber-400">{warnCount} warning(s)</span>}
                </>
              ) : (
                'Estimates are approximate — they flag likely issues, not certainties.'
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onCancel}
                className="px-4 py-1.5 rounded-md text-sm bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700"
              >
                Back to settings
              </button>
              <button
                type="button"
                onClick={onConfirm}
                disabled={loading}
                className={`px-4 py-1.5 rounded-md text-sm text-white disabled:opacity-50 ${
                  errorCount > 0 ? 'bg-rose-600 hover:bg-rose-700' : warnCount > 0 ? 'bg-amber-600 hover:bg-amber-700' : 'bg-green-600 hover:bg-green-700'
                }`}
              >
                {proceedLabel}
              </button>
            </div>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}

function HwStat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-gray-500">{icon}</span>
      <div className="min-w-0">
        <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
        <div className="text-gray-200 truncate" title={value}>
          {value}
        </div>
      </div>
    </div>
  );
}
