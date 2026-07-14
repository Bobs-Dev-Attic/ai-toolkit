'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { apiClient } from '@/utils/api';

export interface SystemStatsPoint {
  t: number;
  elapsed: number;
  step?: number | null;
  [key: string]: number | null | undefined;
}

export default function useJobSystemStats(jobID: string, reloadInterval: null | number = null) {
  const [points, setPoints] = useState<SystemStatsPoint[]>([]);
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);
  // last wall-clock timestamp seen, so we can poll for only newer samples
  const lastTRef = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    if (!jobID) return;
    if (inFlightRef.current) return;
    inFlightRef.current = true;

    setStatus(didInitialLoadRef.current ? 'refreshing' : 'loading');

    try {
      const params: Record<string, any> = {};
      if (reloadInterval && lastTRef.current != null) {
        params.since = lastTRef.current;
      }

      const data = await apiClient
        .get(`/api/jobs/${jobID}/system-stats`, { params })
        .then(res => res.data as { points?: SystemStatsPoint[]; keys?: string[] });

      const newPoints = (data.points ?? []).filter(p => typeof p.t === 'number');

      if (data.keys && data.keys.length) {
        setKeys(prev => {
          const merged = new Set(prev);
          for (const k of data.keys as string[]) merged.add(k);
          return Array.from(merged);
        });
      }

      if (!didInitialLoadRef.current) {
        setPoints(newPoints);
      } else if (newPoints.length) {
        setPoints(prev => {
          const prevLast = prev.length ? prev[prev.length - 1].t : null;
          const fresh = prevLast == null ? newPoints : newPoints.filter(p => p.t > prevLast);
          return fresh.length ? [...prev, ...fresh] : prev;
        });
      }

      if (newPoints.length) {
        lastTRef.current = newPoints[newPoints.length - 1].t;
      }

      setStatus('success');
      didInitialLoadRef.current = true;
    } catch (err) {
      console.error('Error fetching system stats:', err);
      setStatus('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [jobID, reloadInterval]);

  useEffect(() => {
    // reset when the job changes
    didInitialLoadRef.current = false;
    lastTRef.current = null;
    setPoints([]);
    setKeys([]);
    setStatus('idle');

    refresh();

    if (reloadInterval) {
      const interval = setInterval(() => {
        refresh();
      }, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID, reloadInterval, refresh]);

  return { points, keys, status, refresh };
}
