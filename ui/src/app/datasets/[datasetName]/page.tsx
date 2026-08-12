'use client';

import { useEffect, useState, use, useMemo, useCallback } from 'react';
import {
  LuImageOff,
  LuLoader,
  LuBan,
  LuSparkles,
  LuDownload,
  LuTrash2,
  LuInfo,
  LuScaling,
  LuImageUpscale,
  LuScissors,
  LuEraser,
  LuChevronDown,
  LuChevronUp,
  LuRefreshCw,
  LuLayoutGrid,
  LuCrop,
} from 'react-icons/lu';
import { FaChevronLeft } from 'react-icons/fa';
import DatasetImageCard from '@/components/DatasetImageCard';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal, useOpenImagesModalOnDrag } from '@/components/AddImagesModal';
import { Modal } from '@/components/Modal';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'];

interface ImageItem {
  img_path: string;
  size?: number;
  width?: number;
  height?: number;
}

const isCaptionableExt = (imgPath: string) =>
  IMAGE_EXTENSIONS.includes(imgPath.slice(imgPath.lastIndexOf('.')).toLowerCase());

const formatMB = (bytes: number) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

type OpPhase = 'idle' | 'starting' | 'downloading' | 'loading' | 'processing' | 'done' | 'error';

interface OpProgress {
  phase: OpPhase;
  label: string;
  downloadedBytes: number;
  totalBytes: number;
  current: number;
  total: number;
}

const INITIAL_PROGRESS: OpProgress = {
  phase: 'idle',
  label: '',
  downloadedBytes: 0,
  totalBytes: 0,
  current: 0,
  total: 0,
};

type CaptionStyle = 'short' | 'standard' | 'detailed';

interface CaptionOptions {
  triggerWord: string;
  overwrite: boolean;
  style: CaptionStyle;
  prompt: string;
  repetitionPenalty: number;
}

interface ResizeOptions {
  mode: 'longest_side' | 'exact';
  width: number;
  height: number;
  fit: 'fit' | 'cover' | 'pad' | 'stretch';
  padColor: string;
  format: 'keep' | 'jpg' | 'png' | 'webp';
  quality: number;
}

interface RemoveBgOptions {
  model: 'u2net' | 'u2netp' | 'isnet-general-use' | 'birefnet-general' | 'birefnet-general-lite';
  bg: string;
  outputMode: 'replace' | 'sidecar';
}

interface RemoveWatermarkOptions {
  model: 'grounding-dino-tiny' | 'grounding-dino-base';
  text: boolean;
  watermark: boolean;
  logo: boolean;
  prompt: string;
  threshold: number;
  dilate: number;
  outputMode: 'replace' | 'sidecar';
}

interface UpscaleOptions {
  model: 'x2' | 'x4' | 'x4plus';
  maxSide: number;
  outputMode: 'replace' | 'sidecar';
}

interface AutoCropOptions {
  target: 'face' | 'upper_body' | 'torso' | 'person';
  padding: number;
  outputSizeMode: 'none' | 'square' | 'exact';
  squareSize: number;
  exactWidth: number;
  exactHeight: number;
  fit: 'fit' | 'cover' | 'pad' | 'stretch';
  padColor: string;
  outputMode: 'replace' | 'sidecar';
  minConfidence: number;
  torsoTightness: 'tight' | 'medium' | 'wide';
}

const MODEL_DOWNLOAD_LABELS: Record<string, string> = {
  caption: 'Downloading BLIP captioning model (one-time)',
  removeBackground: 'Downloading background-removal model (one-time)',
  removeWatermark: 'Downloading detection + inpainting models (one-time)',
  upscale: 'Downloading Real-ESRGAN upscaler (one-time)',
  autoCrop: 'Downloading detection model (one-time)',
  resize: '',
};

const OPERATION_LABELS: Record<string, string> = {
  caption: 'Generating captions',
  resize: 'Resizing',
  removeBackground: 'Removing background',
  removeWatermark: 'Removing text / watermarks',
  upscale: 'Upscaling',
  autoCrop: 'Auto-cropping',
};

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName;

  const [imgList, setImgList] = useState<ImageItem[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [reloadSignal, setReloadSignal] = useState(0);
  const [showMetadata, setShowMetadata] = useState(false);
  const [gridCols, setGridCols] = useState<number>(() => {
    if (typeof window === 'undefined') return 4;
    const raw = window.localStorage.getItem('AI_TOOLKIT_DATASET_COLS');
    const parsed = raw ? parseInt(raw, 10) : NaN;
    return Number.isFinite(parsed) && parsed >= 1 && parsed <= 6 ? parsed : 4;
  });

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem('AI_TOOLKIT_DATASET_COLS', String(gridCols));
    }
  }, [gridCols]);

  // Single active op slot - we serialize ops so the progress UI is unambiguous.
  const [activeOp, setActiveOp] = useState<
    null | 'caption' | 'resize' | 'removeBackground' | 'removeWatermark' | 'upscale' | 'autoCrop'
  >(null);
  const [progress, setProgress] = useState<OpProgress>(INITIAL_PROGRESS);
  const [opMessage, setOpMessage] = useState('');
  const [opMessageKind, setOpMessageKind] = useState<'info' | 'error'>('info');

  // Modal visibility
  const [resizeOpen, setResizeOpen] = useState(false);
  const [removeBgOpen, setRemoveBgOpen] = useState(false);
  const [removeWatermarkOpen, setRemoveWatermarkOpen] = useState(false);
  const [upscaleOpen, setUpscaleOpen] = useState(false);
  const [autoCropOpen, setAutoCropOpen] = useState(false);
  const [captionAdvancedOpen, setCaptionAdvancedOpen] = useState(false);

  const [captionOpts, setCaptionOpts] = useState<CaptionOptions>({
    triggerWord: '',
    overwrite: false,
    style: 'standard',
    prompt: '',
    repetitionPenalty: 1.2,
  });
  const [resizeOpts, setResizeOpts] = useState<ResizeOptions>({
    mode: 'longest_side',
    width: 1024,
    height: 1024,
    fit: 'fit',
    padColor: 'black',
    format: 'keep',
    quality: 92,
  });
  const [removeBgOpts, setRemoveBgOpts] = useState<RemoveBgOptions>({
    model: 'u2net',
    bg: 'transparent',
    outputMode: 'sidecar',
  });
  const [removeWatermarkOpts, setRemoveWatermarkOpts] = useState<RemoveWatermarkOptions>({
    model: 'grounding-dino-tiny',
    text: true,
    watermark: true,
    logo: true,
    prompt: '',
    threshold: 0.3,
    dilate: 6,
    outputMode: 'sidecar',
  });
  const [upscaleOpts, setUpscaleOpts] = useState<UpscaleOptions>({
    model: 'x4',
    maxSide: 2048,
    outputMode: 'sidecar',
  });
  const [autoCropOpts, setAutoCropOpts] = useState<AutoCropOptions>({
    target: 'face',
    padding: 0.2,
    outputSizeMode: 'square',
    squareSize: 512,
    exactWidth: 512,
    exactHeight: 512,
    fit: 'cover',
    padColor: 'black',
    outputMode: 'sidecar',
    minConfidence: 0.4,
    torsoTightness: 'medium',
  });

  const refreshImageList = useCallback(
    (dbName: string) => {
      setStatus('loading');
      apiClient
        .post('/api/datasets/listImages', { datasetName: dbName })
        .then((res: any) => {
          const data = res.data;
          data.images.sort((a: ImageItem, b: ImageItem) => a.img_path.localeCompare(b.img_path));
          setImgList(data.images);
          setStatus('success');
        })
        .catch(error => {
          console.error('Error fetching images:', error);
          setStatus('error');
        });
    },
    [],
  );

  useOpenImagesModalOnDrag(datasetName, () => refreshImageList(datasetName));

  useEffect(() => {
    if (datasetName) {
      refreshImageList(datasetName);
    }
  }, [datasetName, refreshImageList]);

  // Drop selections that no longer exist after a refresh.
  useEffect(() => {
    setSelected(prev => {
      if (prev.size === 0) return prev;
      const stillPresent = new Set(imgList.map(i => i.img_path));
      const next = new Set<string>();
      for (const p of prev) if (stillPresent.has(p)) next.add(p);
      return next.size === prev.size ? prev : next;
    });
  }, [imgList]);

  const captionableImages = useMemo(
    () => imgList.map(img => img.img_path).filter(isCaptionableExt),
    [imgList],
  );

  const selectedArray = useMemo(() => Array.from(selected), [selected]);
  const isBusy = activeOp !== null;
  const hasSelection = selectedArray.length > 0;

  const toggleSelect = (imgPath: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(imgPath)) next.delete(imgPath);
      else next.add(imgPath);
      return next;
    });
  };

  const selectAll = () => setSelected(new Set(captionableImages));
  const clearSelection = () => setSelected(new Set());

  /**
   * Stream a Server-Sent-Events response from an op endpoint.
   * The op label determines what's shown in the progress bar.
   */
  const runStreamingOp = async (
    op: 'caption' | 'resize' | 'removeBackground' | 'removeWatermark' | 'upscale' | 'autoCrop',
    endpoint: string,
    payload: object,
  ) => {
    setActiveOp(op);
    setOpMessage('');
    setOpMessageKind('info');
    setProgress({ ...INITIAL_PROGRESS, phase: 'starting' });

    let sawError = false;
    let processedCount = 0;

    const handleEvent = (evt: any) => {
      switch (evt.type) {
        case 'model_download_start':
          setProgress(p => ({
            ...p,
            phase: 'downloading',
            label: MODEL_DOWNLOAD_LABELS[op] || 'Downloading model (one-time)',
            downloadedBytes: 0,
            totalBytes: evt.totalBytes || 0,
          }));
          break;
        case 'model_download':
          setProgress(p => ({
            ...p,
            phase: 'downloading',
            label: MODEL_DOWNLOAD_LABELS[op] || 'Downloading model (one-time)',
            downloadedBytes: evt.downloadedBytes || 0,
            totalBytes: evt.totalBytes || 0,
          }));
          break;
        case 'model_loading':
          setProgress(p => ({ ...p, phase: 'loading', label: 'Loading model into memory…' }));
          break;
        case 'progress':
          setProgress(p => ({
            ...p,
            phase: 'processing',
            label: OPERATION_LABELS[op] || 'Processing',
            current: evt.current || 0,
            total: evt.total || 0,
          }));
          break;
        case 'done':
          processedCount = evt.processed ?? 0;
          break;
        case 'warning':
          // surface per-image errors but keep going
          console.warn(`[${op}] ${evt.image}: ${evt.message}`);
          break;
        case 'error':
          sawError = true;
          setOpMessage(evt.message || 'Operation failed.');
          setOpMessageKind('error');
          break;
        default:
          break;
      }
    };

    try {
      const token = typeof window !== 'undefined' ? localStorage.getItem('AI_TOOLKIT_AUTH') : null;
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        let message = 'Operation failed.';
        try {
          const json = await res.json();
          message = json.error || message;
        } catch {
          // non-JSON
        }
        throw new Error(message);
      }
      if (!res.body) throw new Error('No response stream from server.');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let sepIndex: number;
        while ((sepIndex = buffer.indexOf('\n\n')) >= 0) {
          const rawEvent = buffer.slice(0, sepIndex);
          buffer = buffer.slice(sepIndex + 2);
          const dataLine = rawEvent.split('\n').find(line => line.startsWith('data:'));
          if (!dataLine) continue;
          try {
            handleEvent(JSON.parse(dataLine.slice(5).trim()));
          } catch {
            // ignore
          }
        }
      }
    } catch (error: any) {
      sawError = true;
      setOpMessage(error?.message || 'Operation failed.');
      setOpMessageKind('error');
    }

    if (!sawError) {
      setOpMessage(`${OPERATION_LABELS[op]}: ${processedCount} processed.`);
      setOpMessageKind('info');
    }
    setProgress(p => ({ ...p, phase: sawError ? 'error' : 'done' }));
    setActiveOp(null);

    refreshImageList(datasetName);
    setReloadSignal(s => s + 1);
    clearSelection();
  };

  const runCaption = async (useSelection: boolean) => {
    const images = useSelection ? selectedArray : undefined;
    await runStreamingOp('caption', '/api/datasets/caption', {
      datasetName,
      images,
      triggerWord: captionOpts.triggerWord.trim(),
      overwrite: captionOpts.overwrite,
      style: captionOpts.style,
      prompt: captionOpts.prompt,
      repetitionPenalty: captionOpts.repetitionPenalty,
    });
  };

  const runResize = async () => {
    setResizeOpen(false);
    await runStreamingOp('resize', '/api/datasets/resize', {
      datasetName,
      images: selectedArray,
      ...resizeOpts,
    });
  };

  const runRemoveBackground = async () => {
    setRemoveBgOpen(false);
    await runStreamingOp('removeBackground', '/api/datasets/removeBackground', {
      datasetName,
      images: selectedArray,
      ...removeBgOpts,
    });
  };

  const runRemoveWatermark = async () => {
    setRemoveWatermarkOpen(false);
    const targets: string[] = [];
    if (removeWatermarkOpts.text) targets.push('text');
    if (removeWatermarkOpts.watermark) targets.push('watermark');
    if (removeWatermarkOpts.logo) targets.push('logo');
    await runStreamingOp('removeWatermark', '/api/datasets/removeWatermark', {
      datasetName,
      images: selectedArray,
      model: removeWatermarkOpts.model,
      targets: targets.length ? targets : ['text', 'watermark', 'logo'],
      prompt: removeWatermarkOpts.prompt,
      threshold: removeWatermarkOpts.threshold,
      dilate: removeWatermarkOpts.dilate,
      outputMode: removeWatermarkOpts.outputMode,
    });
  };

  const runUpscale = async () => {
    setUpscaleOpen(false);
    await runStreamingOp('upscale', '/api/datasets/upscale', {
      datasetName,
      images: selectedArray,
      ...upscaleOpts,
    });
  };

  const runAutoCrop = async () => {
    setAutoCropOpen(false);
    await runStreamingOp('autoCrop', '/api/datasets/autoCrop', {
      datasetName,
      images: selectedArray,
      ...autoCropOpts,
    });
  };

  const runBulkDelete = () => {
    if (selectedArray.length === 0) return;
    openConfirm({
      title: `Delete ${selectedArray.length} image${selectedArray.length === 1 ? '' : 's'}`,
      message: `Are you sure you want to delete ${selectedArray.length} image${selectedArray.length === 1 ? '' : 's'} (and their captions)? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/img/bulkDelete', { imgPaths: selectedArray })
          .then((res: any) => {
            const deleted = res.data?.deleted ?? 0;
            setOpMessage(`Deleted ${deleted} image${deleted === 1 ? '' : 's'}.`);
            setOpMessageKind('info');
            refreshImageList(datasetName);
            clearSelection();
          })
          .catch(err => {
            console.error('bulk delete failed', err);
            setOpMessage('Failed to delete images.');
            setOpMessageKind('error');
          });
      },
    });
  };

  const PageInfoContent = useMemo(() => {
    if (status === 'loading') {
      return (
        <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
          <LuLoader className="animate-spin w-8 h-8 text-gray-400 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Loading Images</h3>
          <p className="text-sm opacity-75 leading-relaxed">Please wait while we fetch your dataset images...</p>
        </div>
      );
    }
    if (status === 'error') {
      return (
        <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-red-600/20 text-red-100 mx-auto max-w-md text-center">
          <LuBan className="w-8 h-8 text-red-400 mb-4" />
          <h3 className="text-lg font-semibold mb-2">Error Loading Images</h3>
          <p className="text-sm opacity-75 leading-relaxed">There was a problem fetching the images. Please refresh.</p>
        </div>
      );
    }
    if (status === 'success' && imgList.length === 0) {
      return (
        <div className="mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed bg-gray-800/50 text-gray-100 mx-auto max-w-md text-center">
          <LuImageOff className="w-8 h-8 text-gray-400 mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Images Found</h3>
          <p className="text-sm opacity-75 leading-relaxed">This dataset is empty. Click &quot;Add Images&quot; to get started.</p>
        </div>
      );
    }
    return null;
  }, [status, imgList.length]);

  const FloatingProgress = useMemo(() => {
    if (!isBusy) return null;
    const { phase, label, downloadedBytes, totalBytes, current, total } = progress;
    let title = label || 'Working…';
    let subtitle = '';
    let percent: number | null = null;
    let indeterminate = false;

    if (phase === 'downloading') {
      const hasTotal = totalBytes > 0;
      if (hasTotal) {
        percent = Math.min(100, Math.round((downloadedBytes / totalBytes) * 100));
        subtitle = `${formatMB(downloadedBytes)} / ${formatMB(totalBytes)}`;
      } else {
        subtitle = formatMB(downloadedBytes);
        indeterminate = true;
      }
    } else if (phase === 'processing') {
      if (total > 0) {
        percent = Math.min(100, Math.round((current / total) * 100));
        subtitle = `Image ${current} of ${total}`;
      } else {
        subtitle = `Image ${current}`;
        indeterminate = true;
      }
    } else if (phase === 'loading' || phase === 'starting') {
      indeterminate = true;
    }

    return (
      <div className="fixed bottom-4 right-4 z-40 w-80 rounded-lg border border-gray-700 bg-gray-900/95 px-4 py-3 text-sm text-gray-100 shadow-2xl backdrop-blur">
        <div className="flex items-center gap-2">
          {phase === 'downloading' ? (
            <LuDownload className="h-4 w-4 shrink-0 text-cyan-300" />
          ) : (
            <LuLoader className="h-4 w-4 shrink-0 animate-spin text-cyan-300" />
          )}
          <div className="flex min-w-0 flex-1 flex-col">
            <span className="truncate font-medium">{title}</span>
            {(subtitle || percent !== null) && (
              <span className="flex items-center justify-between text-xs text-gray-400">
                <span className="truncate">{subtitle}</span>
                {percent !== null && <span className="ml-2 shrink-0">{percent}%</span>}
              </span>
            )}
          </div>
        </div>
        <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-gray-700">
          <div
            className={`h-full bg-cyan-400 transition-all duration-300 ${indeterminate ? 'w-1/3 animate-pulse' : ''}`}
            style={!indeterminate && percent !== null ? { width: `${percent}%` } : undefined}
          />
        </div>
      </div>
    );
  }, [isBusy, progress]);

  const selectionLabel = hasSelection
    ? `${selectedArray.length} selected`
    : `${captionableImages.length} image${captionableImages.length === 1 ? '' : 's'}`;

  return (
    <>
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => history.back()}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">Dataset: {datasetName}</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-white bg-slate-600 px-3 py-1 rounded-md"
            onClick={() => openImagesModal(datasetName, () => refreshImageList(datasetName))}
          >
            Add Images
          </Button>
        </div>
      </TopBar>
      <MainContent>
        {status === 'success' && imgList.length > 0 && (
          <div className="mb-3 flex flex-col gap-3 rounded-lg border border-gray-700 bg-gray-900/60 p-3 text-gray-100">
            {/* Top row: caption controls */}
            <div className="flex flex-col gap-2 md:flex-row md:items-center">
              <div className="flex min-w-0 flex-1 items-center gap-3">
                <LuSparkles className="h-5 w-5 shrink-0 text-cyan-300" />
                <input
                  className="min-w-0 flex-1 rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-sm outline-none focus:border-cyan-400"
                  value={captionOpts.triggerWord}
                  onChange={e => setCaptionOpts(o => ({ ...o, triggerWord: e.target.value }))}
                  placeholder="Optional trigger word"
                  disabled={isBusy}
                />
                <label className="flex shrink-0 items-center gap-2 text-sm text-gray-300">
                  <input
                    type="checkbox"
                    checked={captionOpts.overwrite}
                    onChange={e => setCaptionOpts(o => ({ ...o, overwrite: e.target.checked }))}
                    disabled={isBusy || hasSelection}
                  />
                  Overwrite
                </label>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Button
                  className="flex items-center justify-center gap-2 rounded-md bg-cyan-700 px-3 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-60"
                  onClick={() => runCaption(hasSelection)}
                  disabled={isBusy}
                >
                  {isBusy ? <LuLoader className="h-4 w-4 animate-spin" /> : <LuSparkles className="h-4 w-4" />}
                  {hasSelection ? `Caption ${selectedArray.length} selected` : 'Generate Captions'}
                </Button>
                <Button
                  className="flex items-center justify-center gap-1 rounded-md border border-gray-600 px-2 py-2 text-xs text-gray-200 hover:bg-gray-800"
                  onClick={() => setCaptionAdvancedOpen(v => !v)}
                  disabled={isBusy}
                  title="Advanced caption options"
                >
                  {captionAdvancedOpen ? <LuChevronUp className="h-3 w-3" /> : <LuChevronDown className="h-3 w-3" />}
                  Advanced
                </Button>
              </div>
            </div>

            {captionAdvancedOpen && (
              <div className="grid grid-cols-1 gap-3 rounded-md border border-gray-700 bg-gray-950/40 p-3 text-sm md:grid-cols-3">
                <label className="flex flex-col gap-1">
                  <span className="text-gray-300">Style</span>
                  <select
                    className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                    value={captionOpts.style}
                    onChange={e => setCaptionOpts(o => ({ ...o, style: e.target.value as CaptionStyle }))}
                    disabled={isBusy}
                  >
                    <option value="short">Short (~10 words)</option>
                    <option value="standard">Standard</option>
                    <option value="detailed">Detailed</option>
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-gray-300">Conditional prompt (optional)</span>
                  <input
                    className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                    value={captionOpts.prompt}
                    onChange={e => setCaptionOpts(o => ({ ...o, prompt: e.target.value }))}
                    placeholder="e.g. a photograph of"
                    disabled={isBusy}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-gray-300">Repetition penalty ({captionOpts.repetitionPenalty.toFixed(2)})</span>
                  <input
                    type="range"
                    min={1.0}
                    max={2.0}
                    step={0.05}
                    value={captionOpts.repetitionPenalty}
                    onChange={e => setCaptionOpts(o => ({ ...o, repetitionPenalty: parseFloat(e.target.value) }))}
                    disabled={isBusy}
                  />
                </label>
              </div>
            )}

            {/* Bulk-action bar */}
            <div className="flex flex-wrap items-center gap-2 border-t border-gray-700 pt-3 text-sm">
              <span className="mr-1 text-gray-400">{selectionLabel}</span>
              <Button
                className="rounded-md border border-gray-600 px-2 py-1.5 text-xs text-gray-200 hover:bg-gray-800 disabled:opacity-60"
                onClick={selectAll}
                disabled={isBusy || captionableImages.length === 0}
              >
                Select all
              </Button>
              <Button
                className="rounded-md border border-gray-600 px-2 py-1.5 text-xs text-gray-200 hover:bg-gray-800 disabled:opacity-60"
                onClick={clearSelection}
                disabled={isBusy || !hasSelection}
              >
                Clear
              </Button>
              <span className="mx-1 h-5 w-px bg-gray-700" />
              <Button
                className="flex items-center gap-1 rounded-md border border-red-700/60 bg-red-900/30 px-2 py-1.5 text-xs text-red-100 hover:bg-red-900/50 disabled:opacity-60"
                onClick={runBulkDelete}
                disabled={isBusy || !hasSelection}
              >
                <LuTrash2 className="h-3.5 w-3.5" />
                Delete {hasSelection ? selectedArray.length : ''}
              </Button>
              <Button
                className="flex items-center gap-1 rounded-md border border-gray-600 px-2 py-1.5 text-xs text-gray-200 hover:bg-gray-800 disabled:opacity-60"
                onClick={() => setResizeOpen(true)}
                disabled={isBusy || !hasSelection}
              >
                <LuScaling className="h-3.5 w-3.5" />
                Resize…
              </Button>
              <Button
                className="flex items-center gap-1 rounded-md border border-gray-600 px-2 py-1.5 text-xs text-gray-200 hover:bg-gray-800 disabled:opacity-60"
                onClick={() => setRemoveBgOpen(true)}
                disabled={isBusy || !hasSelection}
              >
                <LuScissors className="h-3.5 w-3.5" />
                Remove background…
              </Button>
              <Button
                className="flex items-center gap-1 rounded-md border border-gray-600 px-2 py-1.5 text-xs text-gray-200 hover:bg-gray-800 disabled:opacity-60"
                onClick={() => setRemoveWatermarkOpen(true)}
                disabled={isBusy || !hasSelection}
              >
                <LuEraser className="h-3.5 w-3.5" />
                Remove text/watermark…
              </Button>
              <Button
                className="flex items-center gap-1 rounded-md border border-gray-600 px-2 py-1.5 text-xs text-gray-200 hover:bg-gray-800 disabled:opacity-60"
                onClick={() => setUpscaleOpen(true)}
                disabled={isBusy || !hasSelection}
              >
                <LuImageUpscale className="h-3.5 w-3.5" />
                Upscale…
              </Button>
              <Button
                className="flex items-center gap-1 rounded-md border border-gray-600 px-2 py-1.5 text-xs text-gray-200 hover:bg-gray-800 disabled:opacity-60"
                onClick={() => setAutoCropOpen(true)}
                disabled={isBusy || !hasSelection}
              >
                <LuCrop className="h-3.5 w-3.5" />
                Auto-Crop…
              </Button>
              <span className="ml-auto" />
              <div className="flex items-center gap-1.5 rounded-md border border-gray-600 px-2 py-1 text-xs text-gray-200">
                <LuLayoutGrid className="h-3.5 w-3.5" />
                <input
                  type="range"
                  min={1}
                  max={6}
                  step={1}
                  value={gridCols}
                  onChange={e => setGridCols(parseInt(e.target.value, 10))}
                  className="h-1 w-20 accent-cyan-400"
                  aria-label="Columns per row"
                />
                <span className="tabular-nums">{gridCols}/row</span>
              </div>
              <Button
                className={`flex items-center gap-1 rounded-md border px-2 py-1.5 text-xs hover:bg-gray-800 disabled:opacity-60 ${
                  showMetadata ? 'border-cyan-500 bg-cyan-900/30 text-cyan-100' : 'border-gray-600 text-gray-200'
                }`}
                onClick={() => setShowMetadata(v => !v)}
                disabled={isBusy}
              >
                <LuInfo className="h-3.5 w-3.5" />
                Show Metadata
              </Button>
              <Button
                className="flex items-center gap-1 rounded-md border border-gray-600 px-2 py-1.5 text-xs text-gray-200 hover:bg-gray-800 disabled:opacity-60"
                onClick={() => {
                  refreshImageList(datasetName);
                  setReloadSignal(s => s + 1);
                }}
                disabled={isBusy}
                title="Reload images and captions from disk"
              >
                <LuRefreshCw className="h-3.5 w-3.5" />
                Refresh
              </Button>
            </div>
          </div>
        )}

        {opMessage && !isBusy && (
          <div
            className={`mb-4 rounded-md border px-3 py-2 text-sm ${
              opMessageKind === 'error'
                ? 'border-red-700 bg-red-950/50 text-red-100'
                : 'border-gray-700 bg-gray-900/60 text-gray-200'
            }`}
          >
            {opMessage}
          </div>
        )}
        {PageInfoContent}
        {status === 'success' && imgList.length > 0 && (
          <div
            className="grid gap-4"
            style={{ gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))` }}
          >
            {imgList.map(img => (
              <DatasetImageCard
                key={img.img_path}
                alt="image"
                imageUrl={img.img_path}
                selected={selected.has(img.img_path)}
                onToggleSelect={() => toggleSelect(img.img_path)}
                reloadSignal={reloadSignal}
                showMetadata={showMetadata}
                width={img.width}
                height={img.height}
                size={img.size}
              />
            ))}
          </div>
        )}
      </MainContent>
      <AddImagesModal />
      {FloatingProgress}

      {/* Resize modal */}
      <Modal isOpen={resizeOpen} onClose={() => setResizeOpen(false)} title="Resize selected images" size="md">
        <div className="flex flex-col gap-3 text-sm text-gray-200">
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Mode</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={resizeOpts.mode}
              onChange={e => setResizeOpts(o => ({ ...o, mode: e.target.value as ResizeOptions['mode'] }))}
            >
              <option value="longest_side">Longest side</option>
              <option value="exact">Exact W × H</option>
            </select>
          </label>
          {resizeOpts.mode === 'longest_side' ? (
            <label className="flex flex-col gap-1">
              <span className="text-gray-300">Longest side (px)</span>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                  value={resizeOpts.width}
                  onChange={e => setResizeOpts(o => ({ ...o, width: parseInt(e.target.value) || 0 }))}
                />
                <div className="flex shrink-0 gap-1">
                  {[512, 768, 1024, 1280, 1536].map(p => (
                    <Button
                      key={p}
                      className="rounded border border-gray-600 px-2 py-1 text-xs hover:bg-gray-800"
                      onClick={() => setResizeOpts(o => ({ ...o, width: p }))}
                    >
                      {p}
                    </Button>
                  ))}
                </div>
              </div>
            </label>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3">
                <label className="flex flex-col gap-1">
                  <span className="text-gray-300">Width</span>
                  <input
                    type="number"
                    className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                    value={resizeOpts.width}
                    onChange={e => setResizeOpts(o => ({ ...o, width: parseInt(e.target.value) || 0 }))}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-gray-300">Height</span>
                  <input
                    type="number"
                    className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                    value={resizeOpts.height}
                    onChange={e => setResizeOpts(o => ({ ...o, height: parseInt(e.target.value) || 0 }))}
                  />
                </label>
              </div>
              <label className="flex flex-col gap-1">
                <span className="text-gray-300">Fit</span>
                <select
                  className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                  value={resizeOpts.fit}
                  onChange={e => setResizeOpts(o => ({ ...o, fit: e.target.value as ResizeOptions['fit'] }))}
                >
                  <option value="fit">Fit (preserve aspect)</option>
                  <option value="cover">Cover (crop center)</option>
                  <option value="pad">Pad (preserve aspect, fill with color)</option>
                  <option value="stretch">Stretch (distort)</option>
                </select>
              </label>
              {resizeOpts.fit === 'pad' && (
                <label className="flex flex-col gap-1">
                  <span className="text-gray-300">Pad color</span>
                  <input
                    className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                    value={resizeOpts.padColor}
                    onChange={e => setResizeOpts(o => ({ ...o, padColor: e.target.value }))}
                    placeholder="black, white, or #RRGGBB"
                  />
                </label>
              )}
            </>
          )}
          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-gray-300">Format</span>
              <select
                className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                value={resizeOpts.format}
                onChange={e => setResizeOpts(o => ({ ...o, format: e.target.value as ResizeOptions['format'] }))}
              >
                <option value="keep">Keep original</option>
                <option value="jpg">JPG</option>
                <option value="png">PNG</option>
                <option value="webp">WebP</option>
              </select>
            </label>
            {resizeOpts.format !== 'png' && (
              <label className="flex flex-col gap-1">
                <span className="text-gray-300">Quality ({resizeOpts.quality})</span>
                <input
                  type="range"
                  min={50}
                  max={100}
                  value={resizeOpts.quality}
                  onChange={e => setResizeOpts(o => ({ ...o, quality: parseInt(e.target.value) }))}
                />
              </label>
            )}
          </div>
          <p className="rounded-md bg-amber-950/40 px-2 py-1.5 text-xs text-amber-100">
            This overwrites the {selectedArray.length} selected image{selectedArray.length === 1 ? '' : 's'} in place.
          </p>
          <div className="mt-2 flex justify-end gap-2">
            <Button
              className="rounded-md border border-gray-600 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-800"
              onClick={() => setResizeOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="rounded-md bg-cyan-700 px-3 py-1.5 text-sm font-medium text-white"
              onClick={runResize}
            >
              Resize {selectedArray.length}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Remove BG modal */}
      <Modal isOpen={removeBgOpen} onClose={() => setRemoveBgOpen(false)} title="Remove background" size="md">
        <div className="flex flex-col gap-3 text-sm text-gray-200">
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Model</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={removeBgOpts.model}
              onChange={e => setRemoveBgOpts(o => ({ ...o, model: e.target.value as RemoveBgOptions['model'] }))}
            >
              <option value="u2net">u2net (170 MB, balanced)</option>
              <option value="u2netp">u2netp (5 MB, fast, lower quality)</option>
              <option value="isnet-general-use">isnet-general-use (170 MB)</option>
              <option value="birefnet-general-lite">birefnet-general-lite (220 MB)</option>
              <option value="birefnet-general">birefnet-general (885 MB, best quality)</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Output background</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={removeBgOpts.bg}
              onChange={e => setRemoveBgOpts(o => ({ ...o, bg: e.target.value }))}
            >
              <option value="transparent">Transparent (PNG)</option>
              <option value="white">White</option>
              <option value="black">Black</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Save as</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={removeBgOpts.outputMode}
              onChange={e =>
                setRemoveBgOpts(o => ({ ...o, outputMode: e.target.value as RemoveBgOptions['outputMode'] }))
              }
            >
              <option value="sidecar">Sidecar (&lt;name&gt;.nobg.png next to original)</option>
              <option value="replace">Replace original (forces PNG extension)</option>
            </select>
          </label>
          <div className="mt-2 flex justify-end gap-2">
            <Button
              className="rounded-md border border-gray-600 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-800"
              onClick={() => setRemoveBgOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="rounded-md bg-cyan-700 px-3 py-1.5 text-sm font-medium text-white"
              onClick={runRemoveBackground}
            >
              Process {selectedArray.length}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Remove text/watermark modal */}
      <Modal
        isOpen={removeWatermarkOpen}
        onClose={() => setRemoveWatermarkOpen(false)}
        title="Remove text, watermarks & logos"
        size="md"
      >
        <div className="flex flex-col gap-3 text-sm text-gray-200">
          <div className="rounded-md border border-gray-700 bg-gray-900/60 px-3 py-2 text-xs text-gray-400">
            Detects the selected region types with Grounding DINO and reconstructs the background with LaMa
            inpainting. Fully local — the models download once.
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-gray-300">What to remove</span>
            <div className="flex flex-wrap gap-3">
              <label className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  className="accent-cyan-500"
                  checked={removeWatermarkOpts.text}
                  onChange={e => setRemoveWatermarkOpts(o => ({ ...o, text: e.target.checked }))}
                />
                Text
              </label>
              <label className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  className="accent-cyan-500"
                  checked={removeWatermarkOpts.watermark}
                  onChange={e => setRemoveWatermarkOpts(o => ({ ...o, watermark: e.target.checked }))}
                />
                Watermarks
              </label>
              <label className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  className="accent-cyan-500"
                  checked={removeWatermarkOpts.logo}
                  onChange={e => setRemoveWatermarkOpts(o => ({ ...o, logo: e.target.checked }))}
                />
                Logos
              </label>
            </div>
          </div>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Extra phrases to detect (optional)</span>
            <input
              type="text"
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              placeholder="e.g. timestamp, signature, QR code"
              value={removeWatermarkOpts.prompt}
              onChange={e => setRemoveWatermarkOpts(o => ({ ...o, prompt: e.target.value }))}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Detection model</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={removeWatermarkOpts.model}
              onChange={e =>
                setRemoveWatermarkOpts(o => ({ ...o, model: e.target.value as RemoveWatermarkOptions['model'] }))
              }
            >
              <option value="grounding-dino-tiny">Grounding DINO tiny (fast, ~0.7 GB)</option>
              <option value="grounding-dino-base">Grounding DINO base (more accurate, ~0.9 GB)</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">
              Detection threshold ({removeWatermarkOpts.threshold.toFixed(2)})
            </span>
            <input
              type="range"
              min={0.1}
              max={0.6}
              step={0.05}
              value={removeWatermarkOpts.threshold}
              onChange={e => setRemoveWatermarkOpts(o => ({ ...o, threshold: parseFloat(e.target.value) }))}
              className="accent-cyan-400"
            />
            <span className="text-xs text-gray-500">
              Lower catches fainter/smaller marks but risks erasing wanted detail; higher is more conservative.
            </span>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Region padding ({removeWatermarkOpts.dilate}px)</span>
            <input
              type="range"
              min={0}
              max={32}
              step={1}
              value={removeWatermarkOpts.dilate}
              onChange={e => setRemoveWatermarkOpts(o => ({ ...o, dilate: parseInt(e.target.value) }))}
              className="accent-cyan-400"
            />
            <span className="text-xs text-gray-500">Grows each detected region before inpainting to catch soft edges/shadows.</span>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Save as</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={removeWatermarkOpts.outputMode}
              onChange={e =>
                setRemoveWatermarkOpts(o => ({
                  ...o,
                  outputMode: e.target.value as RemoveWatermarkOptions['outputMode'],
                }))
              }
            >
              <option value="sidecar">Sidecar (&lt;name&gt;.clean.png next to original)</option>
              <option value="replace">Replace original (keeps format)</option>
            </select>
          </label>
          <div className="mt-2 flex justify-end gap-2">
            <Button
              className="rounded-md border border-gray-600 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-800"
              onClick={() => setRemoveWatermarkOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="rounded-md bg-cyan-700 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-60"
              onClick={runRemoveWatermark}
              disabled={!removeWatermarkOpts.text && !removeWatermarkOpts.watermark && !removeWatermarkOpts.logo}
            >
              Process {selectedArray.length}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Upscale modal */}
      <Modal isOpen={upscaleOpen} onClose={() => setUpscaleOpen(false)} title="Upscale" size="md">
        <div className="flex flex-col gap-3 text-sm text-gray-200">
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Model</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={upscaleOpts.model}
              onChange={e => setUpscaleOpts(o => ({ ...o, model: e.target.value as UpscaleOptions['model'] }))}
            >
              <option value="x2">Real-ESRGAN ×2</option>
              <option value="x4">Real-ESRGAN ×4</option>
              <option value="x4plus">Real-ESRGAN ×4 plus</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Cap longest side at (px, 0 = no cap)</span>
            <input
              type="number"
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={upscaleOpts.maxSide}
              onChange={e => setUpscaleOpts(o => ({ ...o, maxSide: parseInt(e.target.value) || 0 }))}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Save as</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={upscaleOpts.outputMode}
              onChange={e => setUpscaleOpts(o => ({ ...o, outputMode: e.target.value as UpscaleOptions['outputMode'] }))}
            >
              <option value="sidecar">Sidecar (&lt;name&gt;.upscaled.png next to original)</option>
              <option value="replace">Replace original (forces PNG extension)</option>
            </select>
          </label>
          <p className="rounded-md bg-amber-950/40 px-2 py-1.5 text-xs text-amber-100">
            Runs on GPU. First use will download the model (~67 MB).
          </p>
          <div className="mt-2 flex justify-end gap-2">
            <Button
              className="rounded-md border border-gray-600 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-800"
              onClick={() => setUpscaleOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="rounded-md bg-cyan-700 px-3 py-1.5 text-sm font-medium text-white"
              onClick={runUpscale}
            >
              Upscale {selectedArray.length}
            </Button>
          </div>
        </div>
      </Modal>

      {/* Auto-Crop modal */}
      <Modal isOpen={autoCropOpen} onClose={() => setAutoCropOpen(false)} title="Auto-Crop" size="md">
        <div className="flex flex-col gap-3 text-sm text-gray-200">
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Body part to crop around</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={autoCropOpts.target}
              onChange={e => setAutoCropOpts(o => ({ ...o, target: e.target.value as AutoCropOptions['target'] }))}
            >
              <option value="face">Face (default)</option>
              <option value="upper_body">Upper body (head + torso)</option>
              <option value="torso">Torso (no head)</option>
              <option value="person">Person (full body)</option>
            </select>
          </label>
          {autoCropOpts.target === 'torso' && (
            <label className="flex flex-col gap-1">
              <span className="text-gray-300">Torso tightness</span>
              <select
                className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                value={autoCropOpts.torsoTightness}
                onChange={e =>
                  setAutoCropOpts(o => ({
                    ...o,
                    torsoTightness: e.target.value as AutoCropOptions['torsoTightness'],
                  }))
                }
              >
                <option value="tight">Tight (shoulders + upper chest)</option>
                <option value="medium">Medium (chest down to mid-torso)</option>
                <option value="wide">Wide (down to hip area)</option>
              </select>
            </label>
          )}
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">
              Padding around detection ({Math.round(autoCropOpts.padding * 100)}%)
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={autoCropOpts.padding}
              onChange={e => setAutoCropOpts(o => ({ ...o, padding: parseFloat(e.target.value) }))}
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Output size</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={autoCropOpts.outputSizeMode}
              onChange={e =>
                setAutoCropOpts(o => ({ ...o, outputSizeMode: e.target.value as AutoCropOptions['outputSizeMode'] }))
              }
            >
              <option value="none">Square crop, native size (no resize)</option>
              <option value="square">Square at N px</option>
              <option value="exact">Exact W × H (rectangular)</option>
            </select>
          </label>
          {autoCropOpts.outputSizeMode === 'square' && (
            <label className="flex flex-col gap-1">
              <span className="text-gray-300">Square size (px)</span>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  className="w-full rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                  value={autoCropOpts.squareSize}
                  onChange={e => setAutoCropOpts(o => ({ ...o, squareSize: parseInt(e.target.value) || 0 }))}
                />
                <div className="flex shrink-0 gap-1">
                  {[256, 512, 768, 1024].map(p => (
                    <Button
                      key={p}
                      className="rounded border border-gray-600 px-2 py-1 text-xs hover:bg-gray-800"
                      onClick={() => setAutoCropOpts(o => ({ ...o, squareSize: p }))}
                    >
                      {p}
                    </Button>
                  ))}
                </div>
              </div>
            </label>
          )}
          {autoCropOpts.outputSizeMode === 'exact' && (
            <>
              <div className="grid grid-cols-2 gap-3">
                <label className="flex flex-col gap-1">
                  <span className="text-gray-300">Width</span>
                  <input
                    type="number"
                    className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                    value={autoCropOpts.exactWidth}
                    onChange={e => setAutoCropOpts(o => ({ ...o, exactWidth: parseInt(e.target.value) || 0 }))}
                  />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-gray-300">Height</span>
                  <input
                    type="number"
                    className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
                    value={autoCropOpts.exactHeight}
                    onChange={e => setAutoCropOpts(o => ({ ...o, exactHeight: parseInt(e.target.value) || 0 }))}
                  />
                </label>
              </div>
            </>
          )}
          <label className="flex flex-col gap-1">
            <span className="text-gray-300">Save as</span>
            <select
              className="rounded-md border border-gray-700 bg-gray-950 px-2 py-1.5"
              value={autoCropOpts.outputMode}
              onChange={e =>
                setAutoCropOpts(o => ({ ...o, outputMode: e.target.value as AutoCropOptions['outputMode'] }))
              }
            >
              <option value="sidecar">
                Sidecar (&lt;name&gt;.crop.&lt;ext&gt; — auto-incremented on re-runs)
              </option>
              <option value="replace">Replace original</option>
            </select>
          </label>
          <p className="rounded-md bg-amber-950/40 px-2 py-1.5 text-xs text-amber-100">
            Uses the largest detected {autoCropOpts.target.replace('_', ' ')} per image. Images with no detection are
            skipped. In sidecar mode, re-running auto-crop never overwrites — successive crops are saved as
            &lt;name&gt;.crop.png, &lt;name&gt;.crop2.png, &lt;name&gt;.crop3.png… First face-target use will download a
            ~280 MB detection model.
          </p>
          <div className="mt-2 flex justify-end gap-2">
            <Button
              className="rounded-md border border-gray-600 px-3 py-1.5 text-sm text-gray-200 hover:bg-gray-800"
              onClick={() => setAutoCropOpen(false)}
            >
              Cancel
            </Button>
            <Button
              className="rounded-md bg-cyan-700 px-3 py-1.5 text-sm font-medium text-white"
              onClick={runAutoCrop}
            >
              Auto-Crop {selectedArray.length}
            </Button>
          </div>
        </div>
      </Modal>
    </>
  );
}
