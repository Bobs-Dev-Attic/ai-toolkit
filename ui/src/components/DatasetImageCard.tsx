import React, { useRef, useEffect, useState, ReactNode, KeyboardEvent } from 'react';
import { FaCheck } from 'react-icons/fa';
import { Search } from 'lucide-react';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import AudioPlayer from './AudioPlayer';
import { isVideo, isAudio } from '@/utils/basic';
import ImageLightbox from './ImageLightbox';

interface DatasetImageCardProps {
  imageUrl: string;
  alt: string;
  children?: ReactNode;
  className?: string;
  selected?: boolean;
  onToggleSelect?: () => void;
  /** Bump this number to force the caption to be re-fetched from disk. */
  reloadSignal?: number;
  /** When true, overlay filename / dimensions / size on the image. */
  showMetadata?: boolean;
  width?: number;
  height?: number;
  size?: number;
}

const formatSize = (bytes?: number): string => {
  if (!bytes && bytes !== 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const filenameFromPath = (p: string): string => p.replace(/^.*[\\/]/, '');

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  children,
  className = '',
  selected = false,
  onToggleSelect = () => {},
  reloadSignal = 0,
  showMetadata = false,
  width,
  height,
  size,
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState<boolean>(false);
  const [inViewport, setInViewport] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);
  const [lightboxOpen, setLightboxOpen] = useState<boolean>(false);
  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchCaption = async (force = false) => {
    if (isCaptionLoaded && !force) return;
    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    apiClient
      .post(`/api/caption/get`, { imgPath: imageUrl }, { signal: controller.signal })
      .then(res => res.data)
      .then(data => {
        if (data) {
          data = `${data}`;
        }
        setCaption(data || '');
        setSavedCaption(data || '');
        setIsCaptionLoaded(true);
      })
      .catch(error => {
        if (controller.signal.aborted) return;
        console.error('Error fetching caption:', error);
      })
      .finally(() => {
        if (abortControllerRef.current === controller) {
          abortControllerRef.current = null;
        }
      });
  };

  const saveCaption = () => {
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) return;
    apiClient
      .post('/api/img/caption', { imgPath: imageUrl, caption: trimmedCaption })
      .then(() => {
        setSavedCaption(trimmedCaption);
      })
      .catch(error => {
        console.error('Error saving caption:', error);
      });
  };

  useEffect(() => {
    if (inViewport && isVisible) {
      fetchCaption();
    }
  }, [inViewport, isVisible]);

  // When the parent signals captions changed (e.g. after generation), drop the
  // cached caption so it is re-read from disk.
  useEffect(() => {
    if (reloadSignal === 0) return;
    setIsCaptionLoaded(false);
    if (inViewport && isVisible) {
      fetchCaption(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reloadSignal]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          setInViewport(true);
          if (!isVisible) {
            setIsVisible(true);
          }
        } else {
          setInViewport(false);
          abortControllerRef.current?.abort();
        }
      },
      { threshold: 0.1 },
    );

    if (cardRef.current) {
      observer.observe(cardRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, []);

  const handleLoad = (): void => {
    setLoaded(true);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  const isCaptionCurrent = caption.trim() === savedCaption;

  const isItAVideo = isVideo(imageUrl);
  const isItAudio = isAudio(imageUrl);
  const isItImage = !isItAVideo && !isItAudio;

  const canSelect = isItImage;
  const filename = filenameFromPath(imageUrl);
  // Cache-buster: re-fetch when the file size changes (file was rewritten by an
  // op) or when the parent bumps reloadSignal (manual Refresh).
  const cacheKey = `${size ?? 0}-${reloadSignal}`;
  const mediaSrc = `/api/img/${encodeURIComponent(imageUrl)}?v=${cacheKey}`;

  return (
    <div
      className={classNames('flex flex-col rounded-lg', className, {
        'ring-2 ring-cyan-400': canSelect && selected,
        'ring-1 ring-transparent': !(canSelect && selected),
      })}
    >
      {/* Square image container */}
      <div
        ref={cardRef}
        className={classNames('relative w-full', { 'cursor-pointer': canSelect })}
        style={{ paddingBottom: '100%' }}
        onClick={canSelect ? () => onToggleSelect() : undefined}
      >
        <div className="absolute inset-0 rounded-t-lg shadow-md overflow-hidden">
          {inViewport && isVisible && (
            <>
              {isItAVideo && (
                <video
                  src={mediaSrc}
                  className={`w-full h-full object-contain`}
                  autoPlay={false}
                  loop
                  muted
                  controls
                />
              )}
              {isItAudio && (
                <AudioPlayer src={mediaSrc} title={imageUrl.replace(/^.*[\\/]/, '')} />
              )}
              {isItImage && (
                <img
                  src={mediaSrc}
                  alt={alt}
                  onLoad={handleLoad}
                  className={`w-full h-full object-contain transition-opacity duration-300 ${
                    loaded ? 'opacity-100' : 'opacity-0'
                  }`}
                />
              )}
            </>
          )}
          {!isVisible && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800 bg-opacity-75 rounded-t-lg">
              <span className="text-white text-lg"></span>
            </div>
          )}
          {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
          {canSelect && (
            <div className="absolute top-1 left-1 z-10 pointer-events-none">
              <span
                className={classNames('flex h-6 w-6 items-center justify-center rounded-md border-2 shadow', {
                  'border-cyan-400 bg-cyan-500 text-white': selected,
                  'border-gray-300 bg-gray-900/70 text-transparent': !selected,
                })}
              >
                <FaCheck className="h-3 w-3" />
              </span>
            </div>
          )}
          {isItImage && (
            <button
              type="button"
              onClick={e => {
                // Don't let the click bubble to the card's select-toggle handler.
                e.stopPropagation();
                setLightboxOpen(true);
              }}
              title="Expand to full screen"
              aria-label="Expand image"
              className="absolute top-1 right-1 z-10 flex h-7 w-7 items-center justify-center rounded-md bg-gray-900/70 text-gray-200 hover:bg-gray-800 hover:text-white shadow border border-gray-700/60 transition-colors"
            >
              <Search className="h-4 w-4" />
            </button>
          )}
          {showMetadata && isItImage && (
            <div className="pointer-events-none absolute inset-x-0 bottom-0 z-10 bg-gradient-to-t from-black/85 via-black/60 to-transparent px-2 py-1.5 text-xs leading-tight text-white">
              <div className="truncate font-medium" title={filename}>
                {filename}
              </div>
              <div className="flex justify-between text-[11px] text-gray-200">
                <span>{width && height ? `${width}×${height}` : ''}</span>
                <span>{formatSize(size)}</span>
              </div>
            </div>
          )}
        </div>
      </div>
      <div
        className={classNames('w-full p-2 bg-gray-800 text-white text-sm rounded-b-lg h-[75px]', {
          'border-blue-500 border-2': !isCaptionCurrent,
          'border-transparent border-2': isCaptionCurrent,
        })}
      >
        {inViewport && isVisible && isCaptionLoaded && (
          <form
            onSubmit={e => {
              e.preventDefault();
              saveCaption();
            }}
            onBlur={saveCaption}
          >
            <textarea
              className="w-full bg-transparent resize-none outline-none focus:ring-0 focus:outline-none"
              value={caption}
              rows={3}
              onChange={e => setCaption(e.target.value)}
              onKeyDown={handleKeyDown}
            />
          </form>
        )}
        {(!inViewport || !isVisible) && isCaptionLoaded && (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            {isVisible ? 'Scroll into view to edit caption' : 'Show content to edit caption'}
          </div>
        )}
        {!isCaptionLoaded && (
          <div className="w-full h-full flex items-center justify-center text-gray-400">Loading caption...</div>
        )}
      </div>
      {lightboxOpen && <ImageLightbox src={mediaSrc} alt={alt} onClose={() => setLightboxOpen(false)} />}
    </div>
  );
};

export default DatasetImageCard;
