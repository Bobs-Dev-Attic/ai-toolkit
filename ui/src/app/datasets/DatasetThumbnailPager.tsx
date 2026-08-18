'use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import { ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';
import ImageLightbox from '@/components/ImageLightbox';

interface Props {
  datasetName: string;
  initialThumbs: string[];
  statsLoading: boolean;
  pageSize?: number;
  // Dataset folder mtime from stats. When it changes (files added/removed/
  // renamed on disk), the cached thumb paths are stale and must be dropped.
  modifiedAt?: number;
}

const THUMB_PX = 56;
const THUMB_GAP_PX = 4;
// Approx reserved width for the two chevrons + the counter on the right.
const CHROME_PX = 80 + 70;

interface CacheEntry {
  loading: boolean;
  paths: string[];
  loaded: boolean;
  error?: string;
  // The dataset's modified_at this cache was built from; used to detect that the
  // folder changed on disk (rename/add/delete) so the stale paths are dropped.
  version?: number;
}

const cache: Record<string, CacheEntry> = {};

export default function DatasetThumbnailPager({
  datasetName,
  initialThumbs,
  statsLoading,
  pageSize,
  modifiedAt,
}: Props) {
  const [page, setPage] = useState(0);
  const [, setTick] = useState(0);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);

  useEffect(() => {
    if (!wrapperRef.current) return;
    const ro = new ResizeObserver(entries => {
      for (const e of entries) setContainerWidth(e.contentRect.width);
    });
    ro.observe(wrapperRef.current);
    return () => ro.disconnect();
  }, []);

  const dynamicSize = (() => {
    if (pageSize) return pageSize;
    const usable = Math.max(0, containerWidth - CHROME_PX);
    const fit = Math.floor((usable + THUMB_GAP_PX) / (THUMB_PX + THUMB_GAP_PX));
    return Math.max(3, Math.min(40, fit));
  })();

  const entry: CacheEntry = cache[datasetName] || {
    loading: false,
    paths: initialThumbs,
    loaded: false,
    version: modifiedAt,
  };
  // Keep the cache hydrated from props for the very first render.
  if (!cache[datasetName]) cache[datasetName] = entry;

  // If the dataset changed on disk since this entry was cached (rename/add/
  // delete bumps the folder mtime), the cached paths point at files that no
  // longer exist. Drop them and re-seed from the fresh stats thumbs so a later
  // "next" re-fetches the full list.
  if (modifiedAt !== undefined && entry.version !== undefined && entry.version !== modifiedAt) {
    entry.paths = initialThumbs;
    entry.loaded = false;
    entry.loading = false;
    entry.error = undefined;
    entry.version = modifiedAt;
  }
  if (entry.version === undefined) entry.version = modifiedAt;

  // Keep cache.paths in sync with initialThumbs until a full fetch happens.
  if (!entry.loaded && entry.paths.length < initialThumbs.length) {
    entry.paths = initialThumbs;
  }

  const fetchAll = async () => {
    if (entry.loading || entry.loaded) return;
    entry.loading = true;
    setTick(t => t + 1);
    try {
      const res = await apiClient.post('/api/datasets/listImages', { datasetName });
      const imgs: { img_path: string }[] = res.data?.images || [];
      // Filter to still-image extensions for previews; videos/audio not previewable here.
      const stillExts = ['.png', '.jpg', '.jpeg', '.webp'];
      const paths = imgs
        .map(i => i.img_path)
        .filter(p => stillExts.some(ext => p.toLowerCase().endsWith(ext)));
      entry.paths = paths.length > 0 ? paths : entry.paths;
      entry.loaded = true;
    } catch (err: any) {
      entry.error = err?.response?.data?.error || 'Failed to load images';
    } finally {
      entry.loading = false;
      setTick(t => t + 1);
    }
  };

  const totalKnown = entry.paths.length;
  const effectiveSize = dynamicSize;
  const totalPages = Math.max(1, Math.ceil(totalKnown / effectiveSize));
  const startIdx = Math.min(page * effectiveSize, Math.max(0, totalKnown - effectiveSize));
  const visible = entry.paths.slice(startIdx, startIdx + effectiveSize);

  const goLeft = () => {
    if (page > 0) {
      setPage(p => p - 1);
    }
  };

  const goRight = async () => {
    // If we're near the end of what we have, fetch the full set so
    // "next" can keep going past the stats-provided thumbs.
    if (!entry.loaded && page + 1 >= totalPages - 1) {
      await fetchAll();
    }
    setPage(p => {
      const newTotal = Math.max(1, Math.ceil(entry.paths.length / effectiveSize));
      return Math.min(p + 1, newTotal - 1);
    });
  };

  if (statsLoading && totalKnown === 0) {
    return <span className="text-gray-500 text-xs">…</span>;
  }
  if (totalKnown === 0) {
    return <span className="text-gray-500 text-xs italic">no images</span>;
  }

  const canGoLeft = page > 0;
  // We can always try to go right unless we know we're on the last page of the full list.
  const canGoRight = entry.loaded ? page < totalPages - 1 : true;
  const hasMore = !entry.loaded;

  return (
    <div ref={wrapperRef} className="flex items-center gap-2 py-1 w-full">
      <button
        type="button"
        onClick={goLeft}
        disabled={!canGoLeft}
        className="text-gray-400 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed flex-shrink-0"
        title="Previous page"
      >
        <ChevronLeft className="w-5 h-5" />
      </button>

      <div
        className="flex flex-wrap gap-1 flex-1 min-w-0"
        style={{ gap: `${THUMB_GAP_PX}px` }}
      >
        {visible.map(p => {
          const url = `/api/img/${encodeURIComponent(p)}`;
          return (
            <button
              key={p}
              type="button"
              onClick={() => setLightboxSrc(url)}
              style={{ height: THUMB_PX, width: THUMB_PX }}
              className="rounded border border-gray-800 flex-shrink-0 overflow-hidden cursor-zoom-in hover:border-gray-500 transition-colors p-0"
              title="Click to expand"
            >
              <img
                src={url}
                alt=""
                loading="lazy"
                className="h-full w-full object-cover"
              />
            </button>
          );
        })}
        {entry.loading && (
          <span
            style={{ height: THUMB_PX, width: THUMB_PX }}
            className="flex items-center justify-center text-gray-400 flex-shrink-0"
          >
            <Loader2 className="w-5 h-5 animate-spin" />
          </span>
        )}
      </div>
      <ImageLightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />

      <button
        type="button"
        onClick={goRight}
        disabled={!canGoRight || entry.loading}
        className="text-gray-400 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed"
        title={hasMore ? 'Next (loads full list)' : 'Next page'}
      >
        <ChevronRight className="w-5 h-5" />
      </button>

      <div className="text-[11px] text-gray-500 tabular-nums whitespace-nowrap min-w-[64px] text-right flex-shrink-0">
        {entry.loading ? (
          'loading…'
        ) : (
          <>
            {startIdx + 1}–{Math.min(startIdx + effectiveSize, totalKnown)} / {totalKnown}
            {hasMore && '+'}
          </>
        )}
      </div>
    </div>
  );
}
