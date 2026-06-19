'use client';

import { ReactNode, useCallback, useEffect, useRef, useState } from 'react';
import classNames from 'classnames';
import { GripVertical } from 'lucide-react';

export interface ChangeEntry {
  id: number;
  path: string;
  value: any;
  ts: number;
  origin: 'simple' | 'advanced';
}

interface Props {
  leftLabel: string;
  rightLabel: string;
  leftPane: ReactNode;
  rightPane: ReactNode;
  changes: ChangeEntry[];
  topOffsetPx?: number;
}

const STORAGE_KEY = 'AITK_SPLIT_FRACTION';
const HIGHLIGHT_MS = 30_000;

function shortPath(p: string): string {
  // Trim noisy "config.process[0]." prefix.
  return p.replace(/^config\.process\[0\]\./, '');
}

function fmtValue(v: any): string {
  if (v === null) return 'null';
  if (v === undefined) return 'undefined';
  if (typeof v === 'object') {
    try {
      const s = JSON.stringify(v);
      return s.length > 40 ? s.slice(0, 37) + '…' : s;
    } catch {
      return '[object]';
    }
  }
  const s = String(v);
  return s.length > 40 ? s.slice(0, 37) + '…' : s;
}

export default function SplitWorkspace({
  leftLabel,
  rightLabel,
  leftPane,
  rightPane,
  changes,
  topOffsetPx = 48,
}: Props) {
  const [fraction, setFraction] = useState(0.5);
  const [dragging, setDragging] = useState(false);
  const [now, setNow] = useState(Date.now());
  const containerRef = useRef<HTMLDivElement>(null);

  // Restore persisted fraction on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      const f = raw ? parseFloat(raw) : NaN;
      if (Number.isFinite(f) && f > 0.15 && f < 0.85) setFraction(f);
    } catch {}
  }, []);

  // Persist on change (debounced trivially via the dragging flag)
  useEffect(() => {
    if (dragging) return;
    try {
      localStorage.setItem(STORAGE_KEY, fraction.toFixed(3));
    } catch {}
  }, [fraction, dragging]);

  // Drag handlers
  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => {
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      const f = (e.clientX - rect.left) / rect.width;
      const clamped = Math.min(0.85, Math.max(0.15, f));
      setFraction(clamped);
    };
    const onUp = () => setDragging(false);
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [dragging]);

  // Re-render every 500ms so fade-out animates smoothly.
  useEffect(() => {
    const active = changes.some(c => now - c.ts < HIGHLIGHT_MS);
    if (!active) return;
    const t = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(t);
  }, [changes, now]);

  const recent = changes.filter(c => now - c.ts < HIGHLIGHT_MS);
  const opacityForPane = (origin: 'simple' | 'advanced') => {
    // Highlight intensity = strongest among recent changes from the OTHER pane.
    const other = origin === 'simple' ? 'advanced' : 'simple';
    let best = 0;
    for (const c of recent) {
      if (c.origin !== other) continue;
      const age = now - c.ts;
      const a = Math.max(0, 1 - age / HIGHLIGHT_MS);
      if (a > best) best = a;
    }
    return best;
  };

  const recentBy = (origin: 'simple' | 'advanced') => recent.filter(c => c.origin === origin);

  const leftOpacity = opacityForPane('simple');
  const rightOpacity = opacityForPane('advanced');

  const reset = () => setFraction(0.5);

  return (
    <div
      ref={containerRef}
      className="absolute left-0 w-full flex select-text"
      style={{ top: topOffsetPx, height: `calc(100% - ${topOffsetPx}px)` }}
    >
      <Pane
        label={leftLabel}
        widthPct={fraction * 100}
        highlightOpacity={leftOpacity}
        changes={recentBy('advanced')}
        now={now}
        receivingFrom="Advanced"
      >
        {leftPane}
      </Pane>

      <div
        onMouseDown={() => setDragging(true)}
        onDoubleClick={reset}
        title="Drag to resize. Double-click to reset to 50/50."
        className={classNames(
          'relative flex-shrink-0 w-1.5 cursor-col-resize transition-colors',
          dragging ? 'bg-blue-500' : 'bg-gray-800 hover:bg-blue-600',
        )}
      >
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-gray-500 pointer-events-none">
          <GripVertical className="w-4 h-4" />
        </div>
      </div>

      <Pane
        label={rightLabel}
        widthPct={(1 - fraction) * 100}
        highlightOpacity={rightOpacity}
        changes={recentBy('simple')}
        now={now}
        receivingFrom="Simple"
      >
        {rightPane}
      </Pane>
    </div>
  );
}

interface PaneProps {
  label: string;
  widthPct: number;
  highlightOpacity: number;
  changes: ChangeEntry[];
  now: number;
  receivingFrom: string;
  children: ReactNode;
}

function Pane({ label, widthPct, highlightOpacity, changes, now, receivingFrom, children }: PaneProps) {
  return (
    <div
      className="relative h-full overflow-auto"
      style={{ width: `${widthPct}%` }}
    >
      {/* Fading highlight ring layered above the pane */}
      {highlightOpacity > 0 && (
        <div
          className="pointer-events-none sticky top-0 left-0 w-full h-0 z-30"
          style={{
            boxShadow: `inset 0 0 0 3px rgba(59, 130, 246, ${highlightOpacity.toFixed(3)}), inset 0 0 24px rgba(59, 130, 246, ${(highlightOpacity * 0.35).toFixed(3)})`,
          }}
        />
      )}

      <div className="sticky top-0 z-20 bg-gray-900/90 backdrop-blur border-b border-gray-800 px-4 py-1 text-xs uppercase tracking-wide text-gray-400 flex items-center gap-2">
        <span>{label}</span>
        {changes.length > 0 && (
          <span
            className="text-[10px] normal-case tracking-normal text-blue-300 bg-blue-900/40 border border-blue-700/50 px-1.5 py-0.5 rounded"
            title={`Recent edits from the ${receivingFrom} pane`}
          >
            {changes.length} from {receivingFrom}
          </span>
        )}
      </div>

      {children}

      {/* Floating recent-changes tray pinned to the bottom-right of this pane */}
      {changes.length > 0 && (
        <div className="sticky bottom-2 ml-auto mr-2 w-72 max-w-full z-30 pointer-events-none">
          <div className="space-y-1 text-[11px] font-mono">
            {changes.slice(-6).map(c => {
              const age = now - c.ts;
              const op = Math.max(0, 1 - age / HIGHLIGHT_MS);
              return (
                <div
                  key={c.id}
                  className="bg-blue-950/80 border border-blue-700/50 rounded px-2 py-1 text-blue-100 shadow"
                  style={{ opacity: op.toFixed(3) }}
                >
                  <div className="truncate text-blue-300">{shortPath(c.path)}</div>
                  <div className="truncate text-gray-100">{fmtValue(c.value)}</div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
