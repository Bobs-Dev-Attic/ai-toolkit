'use client';

import { useEffect } from 'react';
import { X } from 'lucide-react';

interface Props {
  src: string | null;
  alt?: string;
  onClose: () => void;
}

/**
 * Fullscreen click-to-dismiss image viewer. Mounts a fixed overlay that
 * fits the image to ~95% of the viewport while preserving aspect ratio.
 * Click outside the image or press Escape to close.
 */
export default function ImageLightbox({ src, alt, onClose }: Props) {
  useEffect(() => {
    if (!src) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    // Lock body scroll while open
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [src, onClose]);

  if (!src) return null;

  return (
    <div
      className="fixed inset-0 z-[100] bg-black/85 flex items-center justify-center p-4 cursor-zoom-out"
      onClick={onClose}
    >
      <button
        type="button"
        onClick={onClose}
        className="absolute top-4 right-4 text-gray-200 hover:text-white bg-gray-900/60 hover:bg-gray-800 rounded-full p-2"
        title="Close (Esc)"
        aria-label="Close"
      >
        <X className="w-6 h-6" />
      </button>
      <img
        src={src}
        alt={alt || ''}
        onClick={e => e.stopPropagation()}
        className="max-w-[95vw] max-h-[95vh] object-contain rounded shadow-2xl cursor-default"
      />
    </div>
  );
}
