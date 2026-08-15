'use client';

import { Dialog, DialogPanel, DialogTitle } from '@headlessui/react';
import { useEffect, useState } from 'react';
import { JobConfig } from '@/types';
import { apiClient } from '@/utils/api';
import { analyzePreflight, Finding, FindingFix, FindingLevel, FindingProfile, PreflightHardware } from '@/utils/preflight';
import { reviewTrainingConfig } from '@/utils/configReview';
import { LuTriangleAlert, LuCircleAlert, LuInfo, LuCircleCheck, LuLoader, LuCpu, LuMemoryStick, LuHardDrive, LuWandSparkles, LuZap, LuSparkles, LuShield } from 'react-icons/lu';

interface Props {
  open: boolean;
  jobConfig: JobConfig | null;
  onConfirm: () => void;
  onCancel: () => void;
  // Apply selected suggestions back to the job form. Given a flat list of
  // path/value changes to write. Optional — without it, findings are advisory only.
  onApplyFixes?: (fixes: FindingFix[]) => void;
  // Verb for the proceed button. 'Create' for the new-job flow (default),
  // 'Start' for the restart/resume flow. Only the wording changes.
  confirmVerb?: string;
}

const levelMeta: Record<FindingLevel, { icon: React.ReactNode; ring: string; text: string; label: string }> = {
  error: { icon: <LuCircleAlert />, ring: 'border-rose-500/40 bg-rose-500/5', text: 'text-rose-400', label: 'Problem' },
  warning: { icon: <LuTriangleAlert />, ring: 'border-amber-500/40 bg-amber-500/5', text: 'text-amber-400', label: 'Warning' },
  info: { icon: <LuInfo />, ring: 'border-sky-500/40 bg-sky-500/5', text: 'text-sky-400', label: 'Note' },
  ok: { icon: <LuCircleCheck />, ring: 'border-emerald-500/40 bg-emerald-500/5', text: 'text-emerald-400', label: 'OK' },
};

// Intent chips for the configuration-goal tabs. Colour-coded so the user can
// pick by goal (speed / quality / fail-safe) at a glance.
const profileMeta: Record<FindingProfile, { icon: React.ReactNode; badge: string; ring: string; activeTab: string; text: string; label: string }> = {
  speed: { icon: <LuZap />, badge: 'bg-amber-500/10 text-amber-300 border-amber-500/30', ring: 'border-amber-500/50', activeTab: 'border-amber-500 text-amber-300', text: 'text-amber-300', label: 'Speed' },
  quality: { icon: <LuSparkles />, badge: 'bg-violet-500/10 text-violet-300 border-violet-500/30', ring: 'border-violet-500/50', activeTab: 'border-violet-500 text-violet-300', text: 'text-violet-300', label: 'Quality' },
  safe: { icon: <LuShield />, badge: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30', ring: 'border-emerald-500/50', activeTab: 'border-emerald-500 text-emerald-300', text: 'text-emerald-300', label: 'Fail-safe' },
};

// Human-readable names for the config paths a strategy profile touches, so the
// per-tab diff reads in plain language instead of raw dotted paths.
const settingLabels: Record<string, string> = {
  'model.quantize': 'Transformer quantization',
  'model.quantize_te': 'Text-encoder quantization',
  'model.layer_offloading': 'Layer offloading',
  'model.low_vram': 'Low VRAM mode',
  'train.cache_text_embeddings': 'Cache text embeddings',
};

// Resolve a value at a setNestedValue-style path (e.g. config.process[0].model.quantize).
function getAtPath(obj: unknown, path: string): unknown {
  const re = /([^[.\]]+)|\[(\d+)\]/g;
  let m: RegExpExecArray | null;
  let cur: any = obj;
  while ((m = re.exec(path)) !== null) {
    if (cur == null) return undefined;
    cur = cur[m[1] !== undefined ? m[1] : Number(m[2])];
  }
  return cur;
}

// Compact display for a config value in the diff.
function fmtVal(v: unknown): string {
  if (v === true) return 'on';
  if (v === false) return 'off';
  if (v === null || v === undefined) return 'none';
  return String(v);
}

// Friendly label for a fix path: the mapped name, else the last path segment.
function labelForPath(path: string): string {
  const tail = path.replace(/^config\.process\[0\]\./, '');
  return settingLabels[tail] ?? tail.split('.').pop() ?? tail;
}

export default function PreflightModal({ open, jobConfig, onConfirm, onCancel, onApplyFixes, confirmVerb = 'Create' }: Props) {
  const [loading, setLoading] = useState(false);
  const [hw, setHw] = useState<PreflightHardware | null>(null);
  const [imageCount, setImageCount] = useState<number | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [err, setErr] = useState<string | null>(null);
  // ids of fixable findings the user has ticked to apply
  const [selected, setSelected] = useState<Set<string>>(new Set());
  // for findings that offer mutually-exclusive options: findingId -> chosen optionId.
  // Nothing is pre-picked, so an option is applied only when the user chooses it.
  const [optionChoice, setOptionChoice] = useState<Map<string, string>>(new Map());

  // Fetch hardware + dataset image counts once when the modal opens. Kept
  // separate from analysis so applying a fix (which changes jobConfig) re-runs
  // the cheap analysis without re-hitting the hardware/stats endpoints.
  useEffect(() => {
    if (!open || !jobConfig) return;
    let cancelled = false;
    setLoading(true);
    setErr(null);
    setSelected(new Set());
    setOptionChoice(new Map());

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
        let count: number | null = null;
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
          if (matched > 0) count = sum;
        }
        setImageCount(count);
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

  // Recompute findings whenever the config, hardware, or image count changes.
  // After an apply, jobConfig updates and this re-runs — resolved findings drop
  // off automatically, so the list reflects the new state.
  useEffect(() => {
    if (!open || !jobConfig || !hw) return;
    const order: Record<FindingLevel, number> = { error: 0, warning: 1, info: 2, ok: 3 };
    const merged = [...analyzePreflight(jobConfig, hw), ...reviewTrainingConfig(jobConfig, imageCount, hw)];
    merged.sort((a, b) => order[a.level] - order[b.level]);
    setFindings(merged);
    // prune selections whose finding no longer exists
    setSelected(prev => {
      const ids = new Set(merged.map(f => f.id));
      const next = new Set([...prev].filter(id => ids.has(id)));
      return next.size === prev.size ? prev : next;
    });
    // Default each options-finding's active tab to its recommended profile
    // (view only — applying is done per-tab, so this never auto-writes anything).
    setOptionChoice(prev => {
      const optionFinds = merged.filter(f => f.options && f.options.length > 0);
      const ids = new Set(optionFinds.map(f => f.id));
      const next = new Map([...prev].filter(([fid]) => ids.has(fid)));
      for (const f of optionFinds) {
        if (!next.has(f.id)) {
          const rec = f.options!.find(o => o.recommended) ?? f.options![0];
          next.set(f.id, rec.id);
        }
      }
      return next;
    });
  }, [open, jobConfig, hw, imageCount]);

  const fixable = findings.filter(f => f.fix && f.fix.length > 0);
  const toggle = (id: string) =>
    setSelected(prev => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  // Which strategy tab is shown for an options-finding (view state only).
  const selectTab = (fid: string, oid: string) => setOptionChoice(prev => new Map(prev).set(fid, oid));
  // Apply one strategy profile's whole bundle immediately.
  const applyProfile = (fixes: FindingFix[]) => {
    if (onApplyFixes && fixes.length > 0) onApplyFixes(fixes);
  };
  const applySelected = () => {
    if (!onApplyFixes) return;
    const fixes = fixable.filter(f => selected.has(f.id)).flatMap(f => f.fix ?? []);
    if (fixes.length > 0) onApplyFixes(fixes);
    // selection is pruned by the recompute effect once jobConfig updates
  };

  const errorCount = findings.filter(f => f.level === 'error').length;
  const warnCount = findings.filter(f => f.level === 'warning').length;
  const problemCount = errorCount + warnCount;

  const proceedLabel = errorCount > 0 ? `${confirmVerb} anyway` : warnCount > 0 ? `${confirmVerb} anyway` : `Looks good — ${confirmVerb}`;

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
                const canFix = !!onApplyFixes && !!f.fix && f.fix.length > 0;
                const isSel = selected.has(f.id);
                return (
                  <div key={f.id} className={`rounded-lg border p-3 ${meta.ring} ${canFix && isSel ? 'ring-1 ring-emerald-500/50' : ''}`}>
                    <div className="flex items-start gap-2">
                      {canFix ? (
                        <input
                          type="checkbox"
                          checked={isSel}
                          onChange={() => toggle(f.id)}
                          className="mt-1 accent-emerald-500 cursor-pointer"
                          title="Select this suggestion to apply"
                        />
                      ) : (
                        <span className={`mt-0.5 ${meta.text}`}>{meta.icon}</span>
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`text-xs uppercase tracking-wide ${meta.text}`}>{meta.label}</span>
                          <span className="text-gray-100 text-sm font-medium">{f.title}</span>
                          {canFix && <span className="text-[10px] uppercase tracking-wide text-emerald-400/80 border border-emerald-500/30 rounded px-1">applyable</span>}
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
                        {f.options && f.options.length > 0 && (() => {
                          const active = f.options.find(o => o.id === optionChoice.get(f.id)) ?? f.options[0];
                          const apm = profileMeta[active.profile];
                          // Per-setting diff for the active tab: current -> target, mismatches highlighted.
                          const rows = active.fix.map(fx => {
                            const current = getAtPath(jobConfig, fx.path);
                            return { path: fx.path, current, target: fx.value, changed: current !== fx.value };
                          });
                          const changedCount = rows.filter(r => r.changed).length;
                          return (
                            <div className="mt-3">
                              {/* Goal tabs */}
                              <div className="flex gap-1 border-b border-gray-800">
                                {f.options!.map(o => {
                                  const pm = profileMeta[o.profile];
                                  const isActive = o.id === active.id;
                                  return (
                                    <button
                                      key={o.id}
                                      type="button"
                                      onClick={() => selectTab(f.id, o.id)}
                                      className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border-b-2 -mb-px transition-colors ${
                                        isActive ? apm.activeTab : 'border-transparent text-gray-400 hover:text-gray-200'
                                      }`}
                                    >
                                      <span className={isActive ? pm.text : ''}>{pm.icon}</span>
                                      {pm.label}
                                      {o.recommended && <span className="text-[9px] uppercase tracking-wide text-emerald-400/80">rec</span>}
                                    </button>
                                  );
                                })}
                              </div>
                              {/* Active goal panel */}
                              <div className="pt-3">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <span className={`inline-flex items-center gap-1 text-[10px] uppercase tracking-wide border rounded px-1 py-0.5 ${apm.badge}`}>
                                    {apm.icon}
                                    {apm.label}
                                  </span>
                                  <span className="text-gray-100 text-sm font-medium">{active.label}</span>
                                  {active.recommended && (
                                    <span className="text-[10px] uppercase tracking-wide text-emerald-400/80 border border-emerald-500/30 rounded px-1">
                                      Recommended
                                    </span>
                                  )}
                                </div>
                                <p className="text-xs text-gray-400 mt-1 leading-relaxed">{active.detail}</p>
                                {/* Diff table */}
                                <div className="mt-2 rounded-md border border-gray-800 divide-y divide-gray-800">
                                  {rows.map(r => (
                                    <div key={r.path} className={`flex items-center justify-between gap-3 px-2.5 py-1.5 text-xs ${r.changed ? 'bg-amber-500/5' : ''}`}>
                                      <span className="text-gray-300">{labelForPath(r.path)}</span>
                                      <span className="flex items-center gap-1.5 font-mono shrink-0">
                                        {r.changed ? (
                                          <>
                                            <span className="text-amber-300">{fmtVal(r.current)}</span>
                                            <span className="text-gray-600">→</span>
                                            <span className={apm.text}>{fmtVal(r.target)}</span>
                                          </>
                                        ) : (
                                          <span className="text-gray-500">{fmtVal(r.current)} <span className="text-emerald-500/70">✓</span></span>
                                        )}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                                {onApplyFixes && (
                                  <div className="mt-2 flex items-center justify-between gap-2">
                                    <span className="text-[11px] text-gray-500">
                                      {changedCount === 0 ? 'Your config already matches this goal.' : `${changedCount} setting${changedCount > 1 ? 's' : ''} would change.`}
                                    </span>
                                    <button
                                      type="button"
                                      onClick={() => applyProfile(active.fix)}
                                      disabled={changedCount === 0}
                                      className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs bg-emerald-600 hover:bg-emerald-700 text-white disabled:opacity-40 disabled:cursor-not-allowed"
                                    >
                                      <LuWandSparkles /> Apply {apm.label}
                                    </button>
                                  </div>
                                )}
                              </div>
                            </div>
                          );
                        })()}
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>

          {/* Apply-suggestions bar (single-fix findings; strategy tabs apply via their own button) */}
          {!loading && onApplyFixes && fixable.length > 0 && (
            <div className="px-5 py-2.5 border-t border-gray-800 shrink-0 flex items-center justify-between gap-3 bg-gray-900/60">
              <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer select-none">
                <input
                  type="checkbox"
                  className="accent-emerald-500 cursor-pointer"
                  checked={selected.size === fixable.length && fixable.length > 0}
                  ref={el => {
                    if (el) el.indeterminate = selected.size > 0 && selected.size < fixable.length;
                  }}
                  onChange={() =>
                    setSelected(prev => (prev.size === fixable.length ? new Set() : new Set(fixable.map(f => f.id))))
                  }
                />
                Select all applyable ({fixable.length})
              </label>
              <button
                type="button"
                onClick={applySelected}
                disabled={selected.size === 0}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm bg-emerald-600 hover:bg-emerald-700 text-white disabled:opacity-40 disabled:cursor-not-allowed"
              >
                <LuWandSparkles /> Apply {selected.size > 0 ? selected.size : ''} selected
              </button>
            </div>
          )}

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
