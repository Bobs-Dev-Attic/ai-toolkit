import { useMemo, useState, useRef, useCallback, useEffect, createContext, useContext } from 'react';
import { Virtuoso, VirtuosoHandle } from 'react-virtuoso';
import useSampleImages from '@/hooks/useSampleImages';
import SampleImageCard from './SampleImageCard';
import { Job } from '@prisma/client';
import { JobConfig } from '@/types';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { Button } from '@headlessui/react';
import { FaDownload, FaFolderOpen, FaSortAmountDown, FaSortAmountUp } from 'react-icons/fa';
import { apiClient } from '@/utils/api';
import classNames from 'classnames';
import { FaCaretDown, FaCaretUp } from 'react-icons/fa';
import SampleImageViewer from './SampleImageViewer';

export type SampleSortOrder = 'newest' | 'oldest';

interface SampleSortContextValue {
  sortOrder: SampleSortOrder;
  setSortOrder: React.Dispatch<React.SetStateAction<SampleSortOrder>>;
}

// Shared so the Sort By control (rendered in the page tab row via `menuItem`) and the
// gallery (rendered in the main content area) can stay in sync.
export const SampleSortContext = createContext<SampleSortContextValue | null>(null);

interface SampleImagesMenuProps {
  job?: Job | null;
}

export const SampleImagesMenu = ({ job }: SampleImagesMenuProps) => {
  const [isZipping, setIsZipping] = useState(false);
  const [isOpeningFolder, setIsOpeningFolder] = useState(false);
  const sortCtx = useContext(SampleSortContext);
  // Only offer Download / Sort once there is at least one sample to act on.
  const { sampleImages } = useSampleImages(job?.id ?? '', 5000);
  const hasSamples = sampleImages.length > 0;

  const openSamplesFolder = async () => {
    if (!job || isOpeningFolder) return;
    setIsOpeningFolder(true);
    try {
      await apiClient.post(`/api/jobs/${job.id}/open-folder`);
    } catch (err) {
      console.error('Error opening samples folder:', err);
    } finally {
      setIsOpeningFolder(false);
    }
  };

  const downloadZip = async () => {
    if (isZipping) return;
    setIsZipping(true);

    try {
      const res = await apiClient.post('/api/zip', {
        zipTarget: 'samples',
        jobName: job?.name,
      });

      const zipPath = res.data.zipPath; // e.g. /mnt/Train2/out/ui/.../samples.zip
      if (!zipPath) throw new Error('No zipPath in response');

      const downloadPath = `/api/files/${encodeURIComponent(zipPath)}`;
      const a = document.createElement('a');
      a.href = downloadPath;
      // optional: suggest filename (browser may ignore if server sets Content-Disposition)
      a.download = 'samples.zip';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (err) {
      console.error('Error downloading zip:', err);
    } finally {
      setIsZipping(false);
    }
  };
  return (
    <>
      <Button
        onClick={openSamplesFolder}
        disabled={isOpeningFolder}
        title="Open the samples folder in your file explorer"
        className={classNames(
          `flex-1 sm:flex-initial justify-center px-2 sm:px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center`,
          {
            'opacity-50 cursor-not-allowed': isOpeningFolder,
          },
        )}
      >
        {isOpeningFolder ? (
          <LuLoader className="animate-spin inline-block sm:mr-2" />
        ) : (
          <FaFolderOpen className="inline-block sm:mr-2" />
        )}
        <span className="hidden sm:inline">Open Folder</span>
      </Button>

      {hasSamples && sortCtx && (
        <label className="flex items-center gap-2 px-2 sm:px-4 h-8 text-sm">
          {sortCtx.sortOrder === 'newest' ? (
            <FaSortAmountDown className="text-gray-400" />
          ) : (
            <FaSortAmountUp className="text-gray-400" />
          )}
          <span className="hidden sm:inline">Sort by</span>
          <select
            value={sortCtx.sortOrder}
            onChange={e => sortCtx.setSortOrder(e.target.value as SampleSortOrder)}
            className="h-6 rounded border border-gray-600 bg-gray-700 text-gray-100 px-1 text-sm focus:outline-none"
          >
            <option value="newest">Created: Newer to Older</option>
            <option value="oldest">Created: Older to Newer</option>
          </select>
        </label>
      )}

      {hasSamples && (
        <Button
          onClick={downloadZip}
          className={classNames(
            `flex-1 sm:flex-initial justify-center px-2 sm:px-4 py-1 h-8 hover:bg-gray-200 dark:hover:bg-gray-700 flex items-center`,
            {
              'opacity-50 cursor-not-allowed': isZipping,
            },
          )}
        >
          {isZipping ? (
            <LuLoader className="animate-spin inline-block sm:mr-2" />
          ) : (
            <FaDownload className="inline-block sm:mr-2" />
          )}
          <span className="hidden sm:inline">{isZipping ? 'Preparing' : 'Download'}</span>
        </Button>
      )}
    </>
  );
};

interface SampleImagesProps {
  job: Job;
}

export default function SampleImages({ job }: SampleImagesProps) {
  const { sampleImages, status, refreshSampleImages } = useSampleImages(job.id, 5000);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);
  const [scrollParent, setScrollParent] = useState<HTMLDivElement | null>(null);
  const scrollParentCallback = useCallback((el: HTMLDivElement | null) => setScrollParent(el), []);
  const virtuosoRef = useRef<VirtuosoHandle>(null);
  // Sort order is controlled from the Sort By dropdown in the page tab row (see SampleImagesMenu).
  const sortCtx = useContext(SampleSortContext);
  const sortOrder = sortCtx?.sortOrder ?? 'newest';

  const numSamples = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      const sampleConfig = jobConfig.config.process[0].sample;
      const numPrompts = sampleConfig.prompts ? sampleConfig.prompts.length : 0;
      const numSamples = sampleConfig.samples.length;
      return Math.max(numPrompts, numSamples, 1);
    }
    return 10;
  }, [job]);

  // Group samples into rows of `numSamples` for the virtualized list — one row per sample iteration.
  // The underlying list is chronological (filenames are timestamp-prefixed), so grouping first
  // keeps each iteration's images together in prompt order.
  const rows = useMemo(() => {
    const out: string[][] = [];
    for (let i = 0; i < sampleImages.length; i += numSamples) {
      out.push(sampleImages.slice(i, i + numSamples));
    }
    return out;
  }, [sampleImages, numSamples]);

  // For "Newer to Older" we reverse the iteration order while preserving the
  // left-to-right prompt order within each row.
  const displayRows = useMemo(
    () => (sortOrder === 'newest' ? [...rows].reverse() : rows),
    [rows, sortOrder],
  );

  // When the sort order flips, jump to the end that holds the newest samples.
  useEffect(() => {
    if (displayRows.length === 0) return;
    if (sortOrder === 'newest') {
      virtuosoRef.current?.scrollToIndex({ index: 0, align: 'start' });
    } else {
      virtuosoRef.current?.scrollToIndex({ index: 'LAST', align: 'end' });
    }
  }, [sortOrder]);

  const scrollToBottom = () => {
    virtuosoRef.current?.scrollToIndex({ index: 'LAST', align: 'end' });
  };

  const scrollToTop = () => {
    virtuosoRef.current?.scrollToIndex({ index: 0, align: 'start' });
  };

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (sampleImages.length > 0) return null;

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Samples';
      subtitle = 'Please wait while we fetch your samples...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Samples';
      subtitle = 'There was a problem fetching the samples.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && sampleImages.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Samples Found';
      subtitle = 'No samples have been generated yet';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, sampleImages.length]);

  // Inline style instead of Tailwind grid-cols-N classes — Tailwind only ships grid-cols-1..12,
  // so class-based columns silently break for larger sample counts.
  const gridCols = Math.max(numSamples, 3);

  const sampleConfig = useMemo(() => {
    if (job?.job_config) {
      const jobConfig = JSON.parse(job.job_config) as JobConfig;
      return jobConfig.config.process[0].sample;
    }
    return null;
  }, [job]);

  return (
    <div ref={scrollParentCallback} className="absolute top-[80px] left-0 right-0 bottom-0 overflow-y-auto">
      <div className="pb-4">
        {PageInfoContent}
        {sampleImages && displayRows.length > 0 && scrollParent && (
          <Virtuoso
            ref={virtuosoRef}
            customScrollParent={scrollParent}
            totalCount={displayRows.length}
            initialTopMostItemIndex={sortOrder === 'newest' ? 0 : displayRows.length - 1}
            followOutput={sortOrder === 'newest' ? false : 'auto'}
            increaseViewportBy={400}
            computeItemKey={index => displayRows[index]?.[0] ?? index}
            itemContent={index => {
              const row = displayRows[index];
              if (!row) return null;

              // Only pad the final row when numSamples < MIN_COLS and the row is short.
              const MIN_COLS = 3;
              const shouldPad = numSamples < MIN_COLS && row.length < MIN_COLS;
              const padsNeeded = shouldPad ? MIN_COLS - row.length : 0;

              return (
                // pb-1 recreates the vertical gap between rows that the original single CSS grid provided via `gap-1`.
                <div className="grid gap-1 pb-1" style={{ gridTemplateColumns: `repeat(${gridCols}, minmax(0, 1fr))` }}>
                  {row.map(sample => (
                    <SampleImageCard
                      key={sample}
                      imageUrl={sample}
                      numSamples={numSamples}
                      sampleImages={sampleImages}
                      alt="Sample Image"
                      onClick={() => setSelectedSamplePath(sample)}
                      observerRoot={scrollParent}
                    />
                  ))}
                  {Array.from({ length: padsNeeded }).map((_, i) => (
                    <div key={`pad-${index}-${i}`} className="invisible" />
                  ))}
                </div>
              );
            }}
          />
        )}
      </div>
      <SampleImageViewer
        imgPath={selectedSamplePath}
        numSamples={numSamples}
        sampleImages={sampleImages}
        onChange={setPath => setSelectedSamplePath(setPath)}
        sampleConfig={sampleConfig}
        refreshSampleImages={refreshSampleImages}
      />
      <div
        className="hidden md:flex fixed top-20 mt-4 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToTop}
        title="Scroll to Top"
      >
        <FaCaretUp className="text-gray-500 dark:text-gray-400" />
      </div>
      <div
        className="hidden md:flex fixed bottom-5 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToBottom}
        title="Scroll to Bottom"
      >
        <FaCaretDown className="text-gray-500 dark:text-gray-400" />
      </div>
    </div>
  );
}
