'use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import { CheckCircle2, Loader2, AlertTriangle, X } from 'lucide-react';

type Phase = 'confirm' | 'sending' | 'done';

interface LogEntry {
  ts: number;
  text: string;
}

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function RestartModal({ open, onClose }: Props) {
  const [phase, setPhase] = useState<Phase>('confirm');
  const [log, setLog] = useState<LogEntry[]>([]);
  const logEndRef = useRef<HTMLDivElement>(null);

  const append = (text: string) => {
    setLog(prev => [...prev, { ts: Date.now(), text }]);
  };

  useEffect(() => {
    if (!open) {
      setPhase('confirm');
      setLog([]);
    }
  }, [open]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [log]);

  const stop = async () => {
    setPhase('sending');
    append('Sending stop request to /api/restart…');
    try {
      await apiClient.post('/api/restart', {});
      append('Server acknowledged. Process is exiting now.');
    } catch {
      append('Connection dropped — this is expected as the server exits.');
    }
    append('Server stopped. Relaunch with Start-AI-Toolkit.bat when ready.');
    setPhase('done');
  };

  if (!open) return null;

  const phaseIcon = (() => {
    switch (phase) {
      case 'confirm':
        return <AlertTriangle className="w-5 h-5 text-amber-400" />;
      case 'sending':
        return <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />;
      case 'done':
        return <CheckCircle2 className="w-5 h-5 text-green-400" />;
    }
  })();

  const phaseTitle = (() => {
    switch (phase) {
      case 'confirm':
        return 'Stop server?';
      case 'sending':
        return 'Stopping…';
      case 'done':
        return 'Server stopped';
    }
  })();

  const canClose = phase === 'confirm' || phase === 'done';

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4"
      onClick={canClose ? onClose : undefined}
    >
      <div
        className="bg-gray-900 border border-gray-700 rounded-lg shadow-2xl w-full max-w-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center gap-3 px-5 py-3 border-b border-gray-800">
          {phaseIcon}
          <div className="flex-1 font-medium">{phaseTitle}</div>
          {canClose && (
            <button onClick={onClose} className="text-gray-500 hover:text-gray-300">
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        <div className="px-5 py-4 space-y-3">
          {phase === 'confirm' && (
            <>
              <p className="text-sm text-gray-300">
                This will gracefully exit the Next.js server. The command window will close on its own once the
                launcher's <code className="bg-gray-800 px-1 rounded">Start-AI-Toolkit.bat</code> script finishes.
              </p>
              <p className="text-sm text-amber-400">
                The server will <strong>not</strong> auto-restart. When you're ready, double-click{' '}
                <code className="bg-gray-800 px-1 rounded">Start-AI-Toolkit.bat</code> to bring it back.
              </p>
            </>
          )}

          {phase !== 'confirm' && (
            <div className="bg-black/40 border border-gray-800 rounded p-3 max-h-64 overflow-y-auto font-mono text-xs">
              {log.map((entry, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-gray-500 shrink-0">
                    {new Date(entry.ts).toLocaleTimeString([], { hour12: false })}
                  </span>
                  <span className="text-gray-200">{entry.text}</span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          )}

          {phase === 'done' && (
            <p className="text-sm text-gray-300">
              You can close this dialog and the browser tab. Run{' '}
              <code className="bg-gray-800 px-1 rounded">Start-AI-Toolkit.bat</code> when you want to bring the server
              back up.
            </p>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-gray-800">
          {phase === 'confirm' && (
            <>
              <button
                onClick={onClose}
                className="px-3 py-2 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={stop}
                className="px-4 py-2 text-sm text-white bg-red-700 hover:bg-red-600 rounded-lg"
              >
                Stop Server
              </button>
            </>
          )}
          {phase === 'done' && (
            <button
              onClick={onClose}
              className="px-3 py-2 text-sm text-gray-300 bg-gray-800 hover:bg-gray-700 rounded-lg"
            >
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
