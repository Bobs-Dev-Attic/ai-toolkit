'use client';

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import useJobLog from '@/hooks/useJobLog';
import {
  LuCopy,
  LuCheck,
  LuLayoutDashboard,
  LuAlignLeft,
  LuChevronRight,
  LuChevronDown,
  LuClock,
  LuDownload,
  LuBox,
  LuImage,
  LuMessageSquare,
  LuTriangleAlert,
} from 'react-icons/lu';

interface Props {
  jobID: string;
}

type ViewMode = 'raw' | 'dashboard';
type Category = 'loading' | 'sampling' | 'message';
type Level = 'info' | 'warning' | 'error';

interface LogRow {
  idx: number; // 1-based line number
  text: string;
  level: Level;
  category: Category;
  ts: number | null;
}

const SETTINGS_KEY = 'jobLogPanel:v2';
const PANEL_HEIGHT = 560; // fixed panel height (px)

interface PersistedSettings {
  viewMode: ViewMode;
  verbose: boolean;
  wrap: boolean;
}

function detectLevel(line: string): Level {
  const l = line.toLowerCase();
  if (
    l.includes('error') ||
    l.includes('traceback') ||
    l.includes('exception') ||
    l.includes('failed') ||
    l.includes('out of memory') ||
    l.includes('must match')
  ) {
    return 'error';
  }
  if (l.includes('warn') || l.includes('skipping') || l.includes('deprecat')) return 'warning';
  return 'info';
}

function categorize(line: string): Category {
  const l = line.toLowerCase();
  if (/generating samples|generating baseline|generating images|\bsample/.test(l)) return 'sampling';
  if (
    /loading|downloading|fetching|download complete|checkpoint shards|quantiz|accuracy recovery|making pipe|preparing model|\bvae\b|text encoder|umt5|qwen|transformer|moving |unloading|create lora|enable lora|attaching quantization|bucket|caching (latents|text)|found \d+ images/.test(
      l,
    )
  ) {
    return 'loading';
  }
  return 'message';
}

function splitLogLines(log: string): string[] {
  let splits: string[] = log.split(/\n|\r\n/);
  splits = splits.map(line => line.split(/\r/).pop() as string);
  return splits;
}

function fmtClock(ts: number | null): string {
  if (ts == null) return '';
  const d = new Date(ts);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map(n => String(n).padStart(2, '0')).join(':');
}

const levelColor: Record<Level, string> = {
  info: 'text-gray-300',
  warning: 'text-amber-400',
  error: 'text-rose-400',
};

const CATEGORY_META: Record<Category, { label: string; icon: React.ReactNode; accent: string }> = {
  loading: { label: 'Loading & setup', icon: <LuBox />, accent: 'text-sky-400' },
  sampling: { label: 'Generating samples', icon: <LuImage />, accent: 'text-purple-400' },
  message: { label: 'Messages', icon: <LuMessageSquare />, accent: 'text-emerald-400' },
};

export default function JobLogPanel({ jobID }: Props) {
  const { log, status } = useJobLog(jobID, 2000);

  const [viewMode, setViewMode] = useState<ViewMode>('raw');
  const [verbose, setVerbose] = useState(false);
  const [wrap, setWrap] = useState(true);
  const [copied, setCopied] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  // which dashboard sections are collapsed
  const [collapsed, setCollapsed] = useState<Record<Category, boolean>>({
    loading: true,
    sampling: false,
    message: false,
  });

  const logRef = useRef<HTMLDivElement>(null);
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);

  const stampsRef = useRef<(number | null)[]>([]);
  const prevLinesRef = useRef<string[]>([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (raw) {
        const s = JSON.parse(raw) as Partial<PersistedSettings>;
        if (s.viewMode === 'raw' || s.viewMode === 'dashboard') setViewMode(s.viewMode);
        if (typeof s.verbose === 'boolean') setVerbose(s.verbose);
        if (typeof s.wrap === 'boolean') setWrap(s.wrap);
      }
    } catch {
      /* ignore */
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({ viewMode, verbose, wrap } as PersistedSettings));
    } catch {
      /* ignore */
    }
  }, [hydrated, viewMode, verbose, wrap]);

  const rawLines = useMemo(() => splitLogLines(log), [log]);

  // per-line client receive timestamps (verbose only; append-aware)
  const stamps = useMemo(() => {
    const now = Date.now();
    const prev = prevLinesRef.current;
    const prevStamps = stampsRef.current;
    let isPrefix = prev.length <= rawLines.length;
    if (isPrefix) {
      for (let i = 0; i < prev.length; i++) {
        if (prev[i] !== rawLines[i]) {
          isPrefix = false;
          break;
        }
      }
    }
    let next: (number | null)[];
    if (isPrefix) {
      next = prevStamps.slice(0, rawLines.length);
      const firstEver = prevStamps.length === 0;
      for (let i = prev.length; i < rawLines.length; i++) next[i] = firstEver ? null : now;
    } else {
      next = rawLines.map(() => null);
    }
    prevLinesRef.current = rawLines;
    stampsRef.current = next;
    return next;
  }, [rawLines]);

  const rows: LogRow[] = useMemo(
    () =>
      rawLines.map((text, i) => ({
        idx: i + 1,
        text,
        level: detectLevel(text),
        category: categorize(text),
        ts: stamps[i] ?? null,
      })),
    [rawLines, stamps],
  );

  const rawViewRows = useMemo(() => {
    const maxLines = verbose ? Infinity : 1000;
    if (rows.length > maxLines) return rows.slice(rows.length - maxLines);
    return rows;
  }, [rows, verbose]);

  // group non-empty rows by category for the dashboard
  const grouped = useMemo(() => {
    const g: Record<Category, LogRow[]> = { loading: [], sampling: [], message: [] };
    for (const r of rows) {
      if (!r.text.trim()) continue;
      g[r.category].push(r);
    }
    return g;
  }, [rows]);

  const alerts = useMemo(() => rows.filter(r => r.text.trim() && r.level !== 'info').slice(-6), [rows]);

  const handleScroll = () => {
    if (logRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = logRef.current;
      setIsScrolledToBottom(scrollHeight - scrollTop - clientHeight < 10);
    }
  };
  useEffect(() => {
    if (viewMode === 'raw' && logRef.current && isScrolledToBottom) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log, isScrolledToBottom, viewMode]);

  const handleCopy = useCallback(async () => {
    try {
      const text = rawViewRows.map(r => (verbose && r.ts != null ? `[${fmtClock(r.ts)}] ${r.text}` : r.text)).join('\n');
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (err) {
      console.error('Copy failed:', err);
    }
  }, [rawViewRows, verbose]);

  const handleExport = useCallback(() => {
    const esc = (s: string) => `"${s.replace(/"/g, '""')}"`;
    const header = ['line', 'time', 'category', 'level', 'message'].join(',');
    const body = rows
      .filter(r => r.text.trim())
      .map(r => [r.idx, fmtClock(r.ts), r.category, r.level, r.text].map(v => esc(String(v))).join(','))
      .join('\n');
    const blob = new Blob([`${header}\n${body}`], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `job-log-${jobID}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }, [rows, jobID]);

  const btn = 'px-2 py-1 rounded-md text-xs border transition-colors flex items-center gap-1.5';
  const btnIdle = 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800/60';
  const btnActive = 'bg-blue-500/10 text-blue-300 border-blue-500/30 hover:bg-blue-500/15';

  const totalNonEmpty = grouped.loading.length + grouped.sampling.length + grouped.message.length;

  return (
    <div
      className="bg-gray-950 rounded-lg relative flex flex-col border border-gray-800 overflow-hidden"
      style={{ height: PANEL_HEIGHT }}
    >
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-2 px-3 py-2 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-1">
          <button className={`${btn} ${viewMode === 'raw' ? btnActive : btnIdle}`} onClick={() => setViewMode('raw')} title="Raw log">
            <LuAlignLeft /> Raw
          </button>
          <button
            className={`${btn} ${viewMode === 'dashboard' ? btnActive : btnIdle}`}
            onClick={() => setViewMode('dashboard')}
            title="Parsed dashboard grouped into loading / samples / messages"
          >
            <LuLayoutDashboard /> Dashboard
          </button>
        </div>

        <div className="w-px h-5 bg-gray-800" />

        <button
          className={`${btn} ${verbose ? btnActive : btnIdle}`}
          onClick={() => setVerbose(v => !v)}
          title="Verbose: line numbers, timestamps for new lines, and the full (untruncated) log"
        >
          <LuClock /> Verbose
        </button>
        {viewMode === 'raw' && (
          <button className={`${btn} ${wrap ? btnActive : btnIdle}`} onClick={() => setWrap(w => !w)} title="Wrap long lines">
            Wrap
          </button>
        )}

        <div className="flex-1" />

        <span className="hidden sm:inline text-[11px] text-gray-400">{totalNonEmpty.toLocaleString()} lines</span>
        <button className={`${btn} ${btnIdle}`} onClick={handleCopy} title="Copy log to clipboard">
          {copied ? <LuCheck className="text-emerald-400" /> : <LuCopy />}
          {copied ? 'Copied' : 'Copy'}
        </button>
        <button className={`${btn} ${btnIdle}`} onClick={handleExport} title="Export the full log as CSV">
          <LuDownload /> Export
        </button>
      </div>

      {/* Body */}
      {status === 'loading' && rows.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-sm text-gray-400">Loading log...</div>
      ) : status === 'error' ? (
        <div className="flex-1 flex items-center justify-center text-sm text-rose-400">Error loading log</div>
      ) : viewMode === 'raw' ? (
        <div ref={logRef} onScroll={handleScroll} className="text-xs text-gray-300 flex-1 overflow-auto p-3 font-mono">
          {rawViewRows.map(r => (
            <div key={r.idx} className="flex">
              {verbose && (
                <>
                  <span className="select-none text-gray-600 pr-2 text-right shrink-0" style={{ minWidth: '3.5em' }}>
                    {r.idx}
                  </span>
                  <span className="select-none text-gray-600 pr-2 shrink-0" style={{ minWidth: '5.5em' }}>
                    {fmtClock(r.ts)}
                  </span>
                </>
              )}
              <pre className={`${wrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre'} ${levelColor[r.level]} flex-1 m-0`}>
                {r.text}
              </pre>
            </div>
          ))}
        </div>
      ) : (
        // Dashboard
        <div className="flex-1 overflow-auto p-3 space-y-3">
          {alerts.length > 0 && (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-2">
              <div className="flex items-center gap-1.5 text-xs text-amber-400 mb-1">
                <LuTriangleAlert /> Recent warnings & errors
              </div>
              <div className="space-y-0.5 font-mono text-[11px]">
                {alerts.map(r => (
                  <div key={r.idx} className={`${levelColor[r.level]} truncate`} title={r.text}>
                    {r.text}
                  </div>
                ))}
              </div>
            </div>
          )}

          {(['loading', 'sampling', 'message'] as Category[]).map(cat => {
            const meta = CATEGORY_META[cat];
            const list = grouped[cat];
            const errs = list.filter(r => r.level === 'error').length;
            const warns = list.filter(r => r.level === 'warning').length;
            const last = list[list.length - 1];
            const isCollapsed = collapsed[cat];
            return (
              <div key={cat} className="rounded-lg border border-gray-800 bg-gray-900/40">
                <button
                  type="button"
                  onClick={() => setCollapsed(prev => ({ ...prev, [cat]: !prev[cat] }))}
                  className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-gray-800/40"
                >
                  <span className="text-gray-500">{isCollapsed ? <LuChevronRight /> : <LuChevronDown />}</span>
                  <span className={meta.accent}>{meta.icon}</span>
                  <span className="text-sm text-gray-100">{meta.label}</span>
                  <span className="text-xs text-gray-500">{list.length}</span>
                  {errs > 0 && <span className="text-xs text-rose-400">{errs} err</span>}
                  {warns > 0 && <span className="text-xs text-amber-400">{warns} warn</span>}
                  {isCollapsed && last && (
                    <span className="ml-auto text-[11px] text-gray-500 font-mono truncate max-w-[45%]" title={last.text}>
                      {last.text}
                    </span>
                  )}
                </button>
                {!isCollapsed && (
                  <div className="border-t border-gray-800 max-h-64 overflow-auto px-3 py-2 font-mono text-[11px] space-y-0.5">
                    {list.length === 0 ? (
                      <div className="text-gray-600">No lines yet.</div>
                    ) : (
                      list.map(r => (
                        <div key={r.idx} className="flex gap-2">
                          {verbose && <span className="text-gray-600 shrink-0">{fmtClock(r.ts)}</span>}
                          <span className={`${levelColor[r.level]} whitespace-pre-wrap break-words`}>{r.text}</span>
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
