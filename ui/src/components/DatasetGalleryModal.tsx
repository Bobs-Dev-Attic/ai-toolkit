'use client';

import { Dialog, DialogPanel, DialogTitle } from '@headlessui/react';
import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import { LuLoader, LuImage } from 'react-icons/lu';
import { X } from 'lucide-react';

interface GalleryImage {
  img_path: string;
  size?: number;
  width?: number;
  height?: number;
  caption?: string;
}

interface Props {
  open: boolean;
  datasetName: string | null;
  onClose: () => void;
}

const VIDEO_EXT = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'];

function isVideo(p: string): boolean {
  const lower = p.toLowerCase();
  return VIDEO_EXT.some(e => lower.endsWith(e));
}

function fmtSize(bytes?: number): string {
  if (bytes == null) return '';
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

export default function DatasetGalleryModal({ open, datasetName, onClose }: Props) {
  const [loading, setLoading] = useState(false);
  const [images, setImages] = useState<GalleryImage[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !datasetName) return;
    let cancelled = false;
    setLoading(true);
    setErr(null);
    setImages([]);
    apiClient
      .post('/api/datasets/listImages', { datasetName })
      .then(r => {
        if (cancelled) return;
        setImages(r.data?.images ?? []);
      })
      .catch(e => {
        if (!cancelled) setErr(e?.response?.data?.error ?? String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, datasetName]);

  const captioned = images.filter(i => i.caption && i.caption.length > 0).length;

  return (
    <Dialog open={open} onClose={onClose} className="relative z-50">
      <div className="fixed inset-0 bg-black/60" aria-hidden="true" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="w-full max-w-5xl max-h-[88vh] flex flex-col rounded-xl bg-gray-900 border border-gray-700 shadow-2xl">
          {/* Header */}
          <div className="px-5 py-4 border-b border-gray-800 shrink-0 flex items-center justify-between gap-3">
            <div className="min-w-0">
              <DialogTitle className="text-gray-100 text-lg font-medium truncate flex items-center gap-2">
                <LuImage className="shrink-0" /> {datasetName || 'Dataset'}
              </DialogTitle>
              <p className="text-sm text-gray-400 mt-0.5">
                {loading
                  ? 'Loading…'
                  : `${images.length} image${images.length === 1 ? '' : 's'}` +
                    (images.length > 0 ? ` · ${captioned} captioned` : '')}
              </p>
            </div>
            <button type="button" onClick={onClose} className="p-1 text-gray-400 hover:text-gray-200 rounded hover:bg-gray-800 shrink-0">
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Body */}
          <div className="flex-1 overflow-y-auto px-5 py-4 min-h-[200px]">
            {loading ? (
              <div className="flex items-center justify-center gap-2 text-gray-400 py-16">
                <LuLoader className="animate-spin" /> Loading images…
              </div>
            ) : err ? (
              <div className="text-rose-400 text-sm py-8 text-center">{err}</div>
            ) : images.length === 0 ? (
              <div className="text-gray-400 text-sm py-16 text-center">No images found in this dataset.</div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
                {images.map(img => {
                  const url = `/api/img/${encodeURIComponent(img.img_path)}`;
                  const dims = img.width && img.height ? `${img.width}×${img.height}` : '';
                  const meta = [dims, fmtSize(img.size)].filter(Boolean).join(' · ');
                  return (
                    <div
                      key={img.img_path}
                      className="group relative aspect-square rounded-lg overflow-hidden border border-gray-700 bg-gray-800"
                    >
                      {isVideo(img.img_path) ? (
                        <video src={url} muted loop className="w-full h-full object-cover" onMouseEnter={e => e.currentTarget.play().catch(() => {})} onMouseLeave={e => e.currentTarget.pause()} />
                      ) : (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={url} alt="" loading="lazy" className="w-full h-full object-cover" />
                      )}

                      {/* Hover overlay: prompt + metadata */}
                      <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 transition-opacity duration-150 flex flex-col p-2 text-left">
                        <div className="text-[10px] text-gray-400 shrink-0">{meta || 'metadata unavailable'}</div>
                        <div className="mt-1 flex-1 overflow-y-auto text-xs text-gray-100 leading-snug whitespace-pre-wrap">
                          {img.caption && img.caption.length > 0 ? (
                            img.caption
                          ) : (
                            <span className="text-gray-500 italic">no caption</span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
