'use client';

import { Job } from '@prisma/client';
import useJobSystemStats, { SystemStatsPoint } from '@/hooks/useJobSystemStats';
import { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

interface Props {
  job: Job;
}

// Which scale (y-axis group) a metric belongs to. Metrics that share a unit and
// magnitude share an axis so the chart stays readable; disk (often 100s of GB)
// gets its own axis so it doesn't flatten VRAM/RAM.
type ScaleId = 'mem' | 'pct' | 'disk';

interface MetricDef {
  key: string; // key in the JSONL record
  label: string;
  unit: string;
  scale: ScaleId;
  color: string;
  defaultOn: boolean;
}

const METRICS: MetricDef[] = [
  { key: 'vram_used_mb', label: 'VRAM Used', unit: 'MB', scale: 'mem', color: 'rgba(96,165,250,1)', defaultOn: true }, // blue
  { key: 'ram_used_mb', label: 'RAM Used', unit: 'MB', scale: 'mem', color: 'rgba(52,211,153,1)', defaultOn: true }, // emerald
  { key: 'proc_ram_mb', label: 'Process RAM', unit: 'MB', scale: 'mem', color: 'rgba(45,212,191,1)', defaultOn: false }, // teal
  { key: 'gpu_percent', label: 'GPU Util', unit: '%', scale: 'pct', color: 'rgba(251,191,36,1)', defaultOn: true }, // amber
  { key: 'cpu_percent', label: 'CPU Util', unit: '%', scale: 'pct', color: 'rgba(244,114,182,1)', defaultOn: true }, // pink
  { key: 'vram_percent', label: 'VRAM %', unit: '%', scale: 'pct', color: 'rgba(129,140,248,1)', defaultOn: false }, // indigo
  { key: 'ram_percent', label: 'RAM %', unit: '%', scale: 'pct', color: 'rgba(34,211,238,1)', defaultOn: false }, // cyan
  { key: 'disk_used_mb', label: 'Disk Used', unit: 'MB', scale: 'disk', color: 'rgba(167,139,250,1)', defaultOn: false }, // purple
  { key: 'disk_free_mb', label: 'Disk Free', unit: 'MB', scale: 'disk', color: 'rgba(248,113,113,1)', defaultOn: false }, // red
  { key: 'disk_percent', label: 'Disk %', unit: '%', scale: 'pct', color: 'rgba(251,146,60,1)', defaultOn: false }, // orange
];

const METRIC_BY_KEY: Record<string, MetricDef> = Object.fromEntries(METRICS.map(m => [m.key, m]));

const FALLBACK_CANVAS_HEIGHT = 360;
const MIN_CANVAS_HEIGHT = 160;

function computeCanvasSize(host: HTMLElement): { width: number; height: number } | null {
  const { width, height } = host.getBoundingClientRect();
  if (width <= 0 || height <= 0) return null;
  const legend = host.querySelector('.u-legend') as HTMLElement | null;
  const legendH = legend?.getBoundingClientRect().height ?? 0;
  return { width, height: Math.max(MIN_CANVAS_HEIGHT, height - legendH) };
}

function formatMB(v: number): string {
  if (!Number.isFinite(v)) return '';
  if (v >= 1024) return `${(v / 1024).toFixed(1)} GB`;
  return `${v.toFixed(0)} MB`;
}

function formatPct(v: number): string {
  if (!Number.isFinite(v)) return '';
  return `${v.toFixed(0)}%`;
}

function formatValueForScale(scale: ScaleId, v: number): string {
  return scale === 'pct' ? formatPct(v) : formatMB(v);
}

// elapsed seconds -> compact h:mm:ss / m:ss
function formatElapsed(sec: number): string {
  if (!Number.isFinite(sec)) return '';
  const s = Math.max(0, Math.floor(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
  return `${m}:${String(ss).padStart(2, '0')}`;
}

// Selectable trailing-window lengths for the x-axis. `null` = show everything.
const TIME_SCALES: { label: string; seconds: number | null }[] = [
  { label: '1m', seconds: 60 },
  { label: '5m', seconds: 300 },
  { label: '15m', seconds: 900 },
  { label: '1h', seconds: 3600 },
  { label: 'All', seconds: null },
];

export default function JobSystemStats({ job }: Props) {
  const { points, keys, status, refresh } = useJobSystemStats(job.id, 3000);

  // which metrics are enabled; only metrics actually present in the data are shown
  const [enabled, setEnabled] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(METRICS.map(m => [m.key, m.defaultOn])),
  );

  // Horizontal window state: `windowSec` is the visible span (null = all).
  // `follow` keeps the right edge pinned to the latest sample; scrolling back
  // turns it off and reveals a "Live" button to resume.
  const [windowSec, setWindowSec] = useState<number | null>(300);
  const [follow, setFollow] = useState(true);
  // Right-edge elapsed (seconds) when not following. State drives the scrollbar;
  // the ref gives event handlers a synchronous read.
  const [viewEnd, setViewEndState] = useState<number | null>(null);
  const viewEndRef = useRef<number | null>(null);
  const windowSecRef = useRef<number | null>(windowSec);
  const followRef = useRef(true);
  useEffect(() => {
    windowSecRef.current = windowSec;
  }, [windowSec]);
  useEffect(() => {
    followRef.current = follow;
  }, [follow]);
  useEffect(() => {
    viewEndRef.current = viewEnd;
  }, [viewEnd]);
  // Set both the ref (immediate) and state (UI) together.
  const setViewEnd = useCallback((v: number | null) => {
    viewEndRef.current = v;
    setViewEndState(v);
  }, []);

  // metrics that have appeared in the log at least once, in canonical order
  const availableMetrics = useMemo(
    () => METRICS.filter(m => keys.includes(m.key)),
    [keys],
  );

  const activeMetrics = useMemo(
    () => availableMetrics.filter(m => enabled[m.key] !== false),
    [availableMetrics, enabled],
  );

  // Build uPlot aligned data (x = elapsed seconds) + per-scale series/axes.
  const built = useMemo(() => {
    const xs = points.map(p => (typeof p.elapsed === 'number' ? p.elapsed : 0));

    const data: (number | null)[][] = [xs];
    const seriesConfigs: uPlot.Series[] = [{}];

    const scales: uPlot.Scales = { x: { time: false } };
    const axes: uPlot.Axis[] = [
      {
        stroke: 'rgba(255,255,255,0.55)',
        grid: { stroke: 'rgba(255,255,255,0.06)' },
        ticks: { stroke: 'rgba(255,255,255,0.15)' },
        values: (_u, ticks) => ticks.map(t => formatElapsed(t)),
      },
    ];

    // one axis per scale that is actually in use
    const usedScales: ScaleId[] = [];
    for (const m of activeMetrics) if (!usedScales.includes(m.scale)) usedScales.push(m.scale);

    usedScales.forEach((scaleId, i) => {
      scales[scaleId] = { auto: true };
      axes.push({
        scale: scaleId,
        side: i % 2 === 0 ? 3 : 1, // alternate left / right
        stroke: 'rgba(255,255,255,0.6)',
        label: scaleId === 'pct' ? 'Percent' : scaleId === 'disk' ? 'Disk' : 'Memory',
        labelSize: 14,
        grid: { show: i === 0, stroke: 'rgba(255,255,255,0.06)' },
        ticks: { stroke: 'rgba(255,255,255,0.15)' },
        size: 62,
        values: (_u, ticks) => ticks.map(t => formatValueForScale(scaleId, t)),
      });
    });

    for (const m of activeMetrics) {
      const col = points.map(p => {
        const v = p[m.key];
        return typeof v === 'number' && Number.isFinite(v) ? v : null;
      });
      data.push(col);
      seriesConfigs.push({
        label: `${m.label} (${m.unit})`,
        scale: m.scale,
        stroke: m.color,
        width: 1.75,
        spanGaps: true,
        points: { show: false },
        value: (_u, v) => (v == null ? '' : formatValueForScale(m.scale, v)),
      });
    }

    return { data: data as uPlot.AlignedData, seriesConfigs, scales, axes };
  }, [points, activeMetrics]);

  const chartHostRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const uplotRef = useRef<uPlot | null>(null);

  const hasData = points.length > 0 && activeMetrics.length > 0;

  // Apply the current horizontal window (time scale + follow/scroll position) to
  // the x-axis. Called after every data push and whenever the window changes.
  const applyXWindow = useCallback((u: uPlot) => {
    const xs = u.data[0] as number[];
    if (!xs || !xs.length) return;
    const dataMin = xs[0];
    const dataMax = xs[xs.length - 1];
    const win = windowSecRef.current;
    if (win == null) {
      u.setScale('x', { min: dataMin, max: dataMax });
      return;
    }
    let end = followRef.current ? dataMax : Math.min(viewEndRef.current ?? dataMax, dataMax);
    let start = end - win;
    if (start < dataMin) {
      start = dataMin;
      end = Math.min(dataMax, start + win);
    }
    u.setScale('x', { min: start, max: end });
  }, []);

  // Recreate uPlot only when the series shape changes; data updates go via setData.
  const structuralKey = useMemo(
    () => `${activeMetrics.map(m => m.key).join('|')}|has=${hasData}`,
    [activeMetrics, hasData],
  );

  useEffect(() => {
    if (uplotRef.current) {
      uplotRef.current.destroy();
      uplotRef.current = null;
    }
    if (!containerRef.current || !chartHostRef.current) return;
    if (!hasData) return;

    const host = chartHostRef.current;
    const rect = host.getBoundingClientRect();
    const initialHeight = rect.height > 0 ? Math.max(MIN_CANVAS_HEIGHT, rect.height - 40) : FALLBACK_CANVAS_HEIGHT;
    const opts: uPlot.Options = {
      width: rect.width || 800,
      height: initialHeight,
      padding: [12, 16, 0, 4],
      series: built.seriesConfigs,
      scales: built.scales,
      axes: built.axes,
      cursor: {
        drag: { x: false, y: false },
        points: { size: 6 },
      },
      legend: { show: true },
    };

    const u = new uPlot(opts, built.data, containerRef.current);
    uplotRef.current = u;
    applyXWindow(u);

    const raf = requestAnimationFrame(() => {
      const uu = uplotRef.current;
      if (!uu) return;
      const fitted = computeCanvasSize(host);
      if (fitted) uu.setSize(fitted);
    });

    return () => {
      cancelAnimationFrame(raf);
      uplotRef.current?.destroy();
      uplotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structuralKey]);

  // Push new data without recreating, then re-apply the window (keeps y auto-ranged).
  useEffect(() => {
    const u = uplotRef.current;
    if (!u) return;
    u.setData(built.data, true);
    applyXWindow(u);
  }, [built, applyXWindow]);

  // Re-apply the window when the time scale, follow state, or scroll position changes.
  useEffect(() => {
    const u = uplotRef.current;
    if (u) applyXWindow(u);
  }, [windowSec, follow, viewEnd, applyXWindow]);

  useEffect(() => {
    const el = chartHostRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const u = uplotRef.current;
      if (!u) return;
      const fitted = computeCanvasSize(el);
      if (fitted) u.setSize(fitted);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasData]);

  // Wheel over the chart scrolls the window through time (when a finite window is
  // active). Scrolling to the newest sample re-enables follow. Bound natively
  // with { passive: false } so preventDefault() actually stops the page scroll
  // (React's synthetic onWheel can be passive).
  useEffect(() => {
    const el = chartHostRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const win = windowSecRef.current;
      if (win == null) return; // "All" — nothing to scroll
      const u = uplotRef.current;
      if (!u) return;
      const xs = u.data[0] as number[];
      if (!xs || !xs.length) return;
      e.preventDefault();
      const dataMin = xs[0];
      const dataMax = xs[xs.length - 1];
      const delta = e.deltaY !== 0 ? e.deltaY : e.deltaX;
      const shift = (delta > 0 ? 1 : -1) * win * 0.15;
      const curEnd = followRef.current ? dataMax : (viewEndRef.current ?? dataMax);
      let newEnd = curEnd + shift;
      if (newEnd >= dataMax) {
        followRef.current = true;
        setFollow(true);
        setViewEnd(null);
      } else {
        if (newEnd < dataMin + win) newEnd = Math.min(dataMax, dataMin + win);
        followRef.current = false;
        setFollow(false);
        setViewEnd(newEnd);
      }
      applyXWindow(u);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [hasData, applyXWindow, setViewEnd]);

  const goLive = useCallback(() => {
    followRef.current = true;
    setViewEnd(null);
    setFollow(true);
    const u = uplotRef.current;
    if (u) applyXWindow(u);
  }, [applyXWindow, setViewEnd]);

  // Scrubbing the scrollbar sets the window's right edge; dragging to the end
  // resumes live-follow.
  const onScrub = useCallback(
    (val: number, dataMin: number, dataMax: number) => {
      if (val >= dataMax - 0.001) {
        followRef.current = true;
        setFollow(true);
        setViewEnd(null);
      } else {
        followRef.current = false;
        setFollow(false);
        setViewEnd(val);
      }
      const u = uplotRef.current;
      if (u) applyXWindow(u);
    },
    [applyXWindow, setViewEnd],
  );

  const latest: SystemStatsPoint | null = points.length ? points[points.length - 1] : null;

  // First/last elapsed times, for the scrollbar range.
  const timeBounds = useMemo(() => {
    if (!points.length) return null;
    const first = typeof points[0].elapsed === 'number' ? points[0].elapsed : 0;
    const last = typeof points[points.length - 1].elapsed === 'number' ? (points[points.length - 1].elapsed as number) : 0;
    return { first, last };
  }, [points]);

  // Whether there's more data than the current window (so scrolling is meaningful).
  const canScroll = windowSec != null && timeBounds != null && timeBounds.last - timeBounds.first > windowSec;
  const scrubValue =
    timeBounds == null ? 0 : follow ? timeBounds.last : Math.min(viewEnd ?? timeBounds.last, timeBounds.last);

  // peak of each active metric across the run, for the summary row
  const peaks = useMemo(() => {
    const out: Record<string, number> = {};
    for (const m of activeMetrics) {
      let peak = -Infinity;
      for (const p of points) {
        const v = p[m.key];
        if (typeof v === 'number' && Number.isFinite(v) && v > peak) peak = v;
      }
      if (peak !== -Infinity) out[m.key] = peak;
    }
    return out;
  }, [points, activeMetrics]);

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col h-full">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-emerald-400" />
          <h2 className="text-gray-100 text-sm font-medium">System resources</h2>
          <span className="text-xs text-gray-400">
            {status === 'loading' && 'Loading...'}
            {status === 'refreshing' && 'Refreshing...'}
            {status === 'error' && 'Error'}
            {status === 'success' && hasData && `${points.length.toLocaleString()} samples`}
            {status === 'success' && !hasData && 'No data yet'}
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {/* Time scale selector */}
          <div className="flex items-center rounded-md border border-gray-700 overflow-hidden">
            {TIME_SCALES.map(ts => {
              const active = windowSec === ts.seconds;
              return (
                <button
                  key={ts.label}
                  type="button"
                  onClick={() => {
                    setWindowSec(ts.seconds);
                    if (ts.seconds == null) {
                      setFollow(true);
                      viewEndRef.current = null;
                    }
                  }}
                  className={`px-2 py-1 text-xs ${
                    active ? 'bg-emerald-600/80 text-white' : 'bg-gray-800/60 text-gray-300 hover:bg-gray-700'
                  }`}
                  title={ts.seconds == null ? 'Show the entire run' : `Show the last ${ts.label}`}
                >
                  {ts.label}
                </button>
              );
            })}
          </div>
          <button
            type="button"
            onClick={refresh}
            className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Current / peak summary */}
      {hasData && latest && (
        <div className="px-4 pt-3 shrink-0">
          <div className="flex flex-wrap gap-2">
            {activeMetrics.map(m => {
              const cur = latest[m.key];
              const peak = peaks[m.key];
              return (
                <div
                  key={m.key}
                  className="bg-gray-950 border border-gray-800 rounded-lg px-3 py-1.5 flex items-center gap-2"
                >
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: m.color }} />
                  <span className="text-[11px] text-gray-400">{m.label}</span>
                  <span className="text-xs text-gray-100 tabular-nums">
                    {typeof cur === 'number' ? formatValueForScale(m.scale, cur) : '—'}
                  </span>
                  {typeof peak === 'number' && (
                    <span className="text-[10px] text-gray-500 tabular-nums">
                      peak {formatValueForScale(m.scale, peak)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="px-4 pt-3 pb-3 flex-1 min-h-0 flex flex-col">
        <div
          className="bg-gray-950 rounded-lg border border-gray-800 relative select-none flex-1 min-h-0"
          style={{ minHeight: 240 }}
        >
          {!hasData ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-400 text-center px-4">
              {status === 'error'
                ? 'Failed to load system stats.'
                : availableMetrics.length === 0
                  ? 'Waiting for the first samples — stats appear once training starts.'
                  : 'No metrics selected. Enable one below.'}
            </div>
          ) : (
            <>
              {windowSec != null && !follow && (
                <button
                  type="button"
                  onClick={goLive}
                  className="absolute top-2 right-2 z-10 px-2 py-1 rounded text-xs bg-emerald-600/80 hover:bg-emerald-600 text-white border border-emerald-500/50 flex items-center gap-1.5"
                  title="Jump back to the latest samples and follow live"
                >
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-white" />
                  Live
                </button>
              )}
              {windowSec != null && follow && (
                <span className="absolute top-2 right-2 z-10 px-2 py-1 rounded text-[11px] bg-gray-900/70 text-emerald-400 border border-gray-700 flex items-center gap-1.5">
                  <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  Live
                </span>
              )}
              <div ref={chartHostRef} className="absolute top-0 left-0 right-0 bottom-2 overflow-hidden">
                <div ref={containerRef} />
              </div>
            </>
          )}
        </div>

        {/* Horizontal time scrollbar */}
        {hasData && windowSec != null && timeBounds && (
          <div className="mt-2 flex items-center gap-2 shrink-0">
            <span className="text-[10px] text-gray-500 tabular-nums w-10 text-right">
              {formatElapsed(follow ? Math.max(timeBounds.first, scrubValue - windowSec) : scrubValue - windowSec)}
            </span>
            <input
              type="range"
              className="flex-1 accent-emerald-500 h-1.5"
              min={timeBounds.first}
              max={timeBounds.last}
              step="any"
              value={scrubValue}
              disabled={!canScroll}
              onChange={e => onScrub(Number(e.target.value), timeBounds.first, timeBounds.last)}
              title={canScroll ? 'Scroll through the run' : 'Not enough data to scroll yet'}
            />
            <span className="text-[10px] text-gray-500 tabular-nums w-10">{formatElapsed(scrubValue)}</span>
          </div>
        )}
      </div>

      {/* Resource filter */}
      <div className="px-4 pb-3 shrink-0">
        <div className="bg-gray-950 border border-gray-800 rounded-lg p-3">
          <label className="block text-xs text-gray-400 mb-2">Filter resources</label>
          <div className="flex flex-wrap gap-2">
            {(availableMetrics.length ? availableMetrics : METRICS.filter(m => m.defaultOn)).map(m => {
              const on = enabled[m.key] !== false;
              return (
                <button
                  key={m.key}
                  type="button"
                  onClick={() => setEnabled(prev => ({ ...prev, [m.key]: !(prev[m.key] !== false) }))}
                  className="px-3 py-1 rounded-md text-xs border transition-colors bg-gray-900 border-gray-800 hover:bg-gray-800/60"
                  style={{ color: on ? '#e5e7eb' : '#6b7280' }}
                  aria-pressed={on}
                  title={`${m.label} (${m.unit})`}
                >
                  <span
                    className="inline-block h-2 w-2 rounded-full mr-2"
                    style={{ background: on ? m.color : 'rgba(120,120,120,1)' }}
                  />
                  {m.label}
                </button>
              );
            })}
          </div>
          <div className="mt-2 text-[11px] text-gray-500">
            Pick a time scale above; drag the scrollbar or scroll over the chart to move through the run (return to the
            end to go live). Samples are logged to <code className="text-gray-400">system_stats.jsonl</code> in the job folder.
          </div>
        </div>
      </div>

      <style jsx global>{`
        .uplot,
        .uplot * {
          font-family: inherit;
        }
        .uplot .u-legend {
          color: rgba(255, 255, 255, 0.85);
          font-size: 12px;
          margin-top: 4px;
        }
        .uplot .u-legend th,
        .uplot .u-legend td {
          color: rgba(255, 255, 255, 0.85);
        }
        .uplot .u-legend .u-marker {
          border-radius: 2px;
        }
        .uplot .u-select {
          background: rgba(16, 185, 129, 0.15);
          border: 1px solid rgba(16, 185, 129, 0.4);
        }
      `}</style>
    </div>
  );
}
