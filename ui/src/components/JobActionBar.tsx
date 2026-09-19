import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { Eye, Trash2, Pen, Play, Pause, Cog, X, Copy, Save, OctagonX, Image, FolderInput } from 'lucide-react';
import { LuLoader } from 'react-icons/lu';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { Job } from '@prisma/client';
import {
  startJob,
  stopJob,
  deleteJob,
  getAvaliableJobActions,
  getJobConfig,
  markJobAsStopped,
  saveJobNow,
  sampleJobNow,
  copyFinalCheckpoint,
} from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { redirect } from 'next/navigation';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import StopJobModal from '@/components/StopJobModal';
import FolderBrowserModal from '@/components/FolderBrowserModal';
import PreflightModal from '@/components/PreflightModal';
import { setNestedValue } from '@/utils/hooks';
import { apiClient } from '@/utils/api';
import { JobConfig } from '@/types';
import { FindingFix } from '@/utils/preflight';

interface JobActionBarProps {
  job: Job;
  onRefresh?: () => void;
  afterDelete?: () => void;
  hideView?: boolean;
  className?: string;
  autoStartQueue?: boolean;
  // Where the gear menu opens; defaults to below. Use 'top end' when the bar is
  // pinned to the bottom of the screen so the menu opens upward into view.
  menuAnchor?: 'bottom' | 'top end';
}

type PendingAction = 'start' | 'remove' | 'stop' | 'delete' | 'save' | 'sample' | 'markStopped';

// If the job never reports a state change (e.g. the request silently failed
// server-side), unlock the bar after this long so it can't stay stuck.
const PENDING_TIMEOUT_MS = 30_000;

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  autoStartQueue = false,
  menuAnchor = 'bottom',
}: JobActionBarProps) {
  const { canStart, canStop, canDelete, canEdit, canRemoveFromQueue } = getAvaliableJobActions(job);
  const [stopOpen, setStopOpen] = useState(false);
  const [copyDestOpen, setCopyDestOpen] = useState(false);
  // Restart pre-flight: seeded from the job's saved config when the user hits
  // Play on a training job. restartDirty tracks whether the user applied any
  // strategy change, so we only re-persist the config when it actually changed.
  const [preflightOpen, setPreflightOpen] = useState(false);
  const [restartConfig, setRestartConfig] = useState<JobConfig | null>(null);
  const [restartDirty, setRestartDirty] = useState(false);

  if (!afterDelete) afterDelete = onRefresh;

  // The action currently in flight. While set, every other action is disabled
  // and the pending one shows a spinner. It is cleared once the job comes back
  // in a different state (status / stop flag), on error, or after a timeout.
  const [pending, setPending] = useState<PendingAction | null>(null);
  // Job state key captured when the pending action was fired. null means the
  // action does not change job state and clears as soon as the request resolves.
  const pendingStateKeyRef = useRef<string | null>(null);
  const jobStateKey = `${job.status}|${job.stop ? 1 : 0}`;

  const clearPending = () => {
    pendingStateKeyRef.current = null;
    setPending(null);
  };

  useEffect(() => {
    if (pending && pendingStateKeyRef.current !== null && pendingStateKeyRef.current !== jobStateKey) {
      clearPending();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobStateKey, pending]);

  useEffect(() => {
    if (!pending) return;
    const timer = setTimeout(clearPending, PENDING_TIMEOUT_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending]);

  const runAction = async (action: PendingAction, fn: () => Promise<void>, waitForJobUpdate = true) => {
    if (pending) return;
    pendingStateKeyRef.current = waitForJobUpdate ? jobStateKey : null;
    setPending(action);
    try {
      await fn();
      if (!waitForJobUpdate) clearPending();
    } catch (e) {
      console.error(`Error running job action "${action}":`, e);
      clearPending();
    }
  };

  const isBusy = pending !== null;

  // Start (or resume) the job, tracked by the busy system. When `config` is
  // provided, persist it to the existing job first (POST /api/jobs with the job
  // id updates in place) so the chosen strategy takes effect on this run;
  // startJob then reads it back.
  const runStart = async (config: JobConfig | null) => {
    await runAction('start', async () => {
      if (config) {
        await apiClient.post('/api/jobs', {
          id: job.id,
          name: job.name,
          gpu_ids: job.gpu_ids,
          job_config: config,
        });
      }
      await startJob(job.id);
      if (autoStartQueue) {
        await startQueue(job.gpu_ids);
      }
      if (onRefresh) onRefresh();
    });
  };

  const handleStartClick = () => {
    if (!canStart || isBusy) return;
    // Training jobs get the pre-flight/strategy review; other job types (e.g.
    // captioning) have nothing to review, so start them directly.
    if (job.job_type === 'train') {
      try {
        setRestartConfig(getJobConfig(job));
        setRestartDirty(false);
        setPreflightOpen(true);
      } catch {
        // If the config can't be parsed, fall back to starting directly.
        void runStart(null);
      }
    } else {
      void runStart(null);
    }
  };

  const iconSizeClass = 'w-5 h-5 sm:w-6 sm:h-6';
  const actionButtonClass = 'ml-1 sm:ml-2 opacity-100 disabled:opacity-40 disabled:cursor-not-allowed';
  const menuItemClass = 'px-4 py-1 rounded flex items-center gap-2';
  const menuItemEnabledClass = `${menuItemClass} cursor-pointer hover:bg-gray-800`;
  const menuItemDisabledClass = `${menuItemClass} opacity-40 cursor-not-allowed`;
  const spinner = <LuLoader className={`${iconSizeClass} animate-spin`} />;
  const menuSpinner = <LuLoader className="w-4 h-4 animate-spin" />;
  const menuPending = pending === 'save' || pending === 'sample' || pending === 'markStopped';

  return (
    <div className={`flex items-center flex-shrink-0 ${className ?? ''}`}>
      {canStart && (
        <Button disabled={isBusy} onClick={handleStartClick} className={actionButtonClass}>
          {pending === 'start' ? spinner : <Play className={iconSizeClass} />}
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          disabled={isBusy}
          onClick={() => {
            if (!canRemoveFromQueue) return;
            runAction('remove', async () => {
              await markJobAsStopped(job.id);
              if (onRefresh) onRefresh();
            });
          }}
          className={actionButtonClass}
        >
          {pending === 'remove' ? spinner : <X className={iconSizeClass} />}
        </Button>
      )}
      {canStop && (
        <Button
          disabled={isBusy}
          onClick={() => {
            if (!canStop || isBusy) return;
            // The fork uses a richer StopJobModal (with stop options) instead of a
            // plain confirm; open it here.
            setStopOpen(true);
          }}
          className={actionButtonClass}
        >
          {pending === 'stop' ? spinner : <Pause className={iconSizeClass} />}
        </Button>
      )}
      <StopJobModal
        open={stopOpen}
        job={job}
        onClose={() => {
          setStopOpen(false);
          if (onRefresh) onRefresh();
        }}
        onComplete={() => {
          if (onRefresh) onRefresh();
        }}
      />
      {restartConfig && (
        <PreflightModal
          open={preflightOpen}
          jobConfig={restartConfig}
          confirmVerb="Start"
          onConfirm={() => {
            setPreflightOpen(false);
            // Only re-persist if the user applied a strategy/fix; otherwise start as-is.
            void runStart(restartDirty ? restartConfig : null);
          }}
          onCancel={() => setPreflightOpen(false)}
          onApplyFixes={(fixes: FindingFix[]) => {
            setRestartConfig(prev => {
              if (!prev) return prev;
              let updated = prev;
              for (const fix of fixes) {
                updated = setNestedValue(updated, fix.value, fix.path);
              }
              return updated;
            });
            setRestartDirty(true);
          }}
        />
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className="ml-1 sm:ml-2 text-gray-200 hover:text-gray-100 inline-block">
          <Eye className={iconSizeClass} />
        </Link>
      )}
      {job.job_type === 'caption' && canEdit && (
        <div
          className="ml-1 sm:ml-2 hover:text-gray-100 inline-block cursor-pointer"
          onClick={() =>
            openCaptionDatasetModal(
              job.job_ref || '',
              () => {
                if (onRefresh) onRefresh();
              },
              { jobId: job.id },
            )
          }
        >
          <Pen className={iconSizeClass} />
        </div>
      )}
      {job.job_type === 'train' && canEdit && (
        <Link href={`/jobs/new?id=${job.id}`} className="ml-1 sm:ml-2 hover:text-gray-100 inline-block">
          <Pen className={iconSizeClass} />
        </Link>
      )}
      <Button
        disabled={isBusy}
        onClick={() => {
          if (isBusy) return;
          let message = `Are you sure you want to delete the job "${job.name}"? This will also permanently remove it from your disk.`;
          if (job.status === 'running') {
            message += ' WARNING: The job is currently running. You should stop it first if you can.';
          }
          openConfirm({
            title: 'Delete Job',
            message: message,
            type: 'warning',
            confirmText: 'Delete',
            onConfirm: () =>
              runAction('delete', async () => {
                if (job.status === 'running') {
                  try {
                    await stopJob(job.id);
                  } catch (e) {
                    console.error('Error stopping job before deleting:', e);
                  }
                }
                await deleteJob(job.id);
                if (afterDelete) afterDelete();
              }),
          });
        }}
        className={actionButtonClass}
      >
        {pending === 'delete' ? spinner : <Trash2 className={iconSizeClass} />}
      </Button>
      <div className="border-r border-1 border-gray-700 ml-1 sm:ml-2 inline"></div>
      <Menu>
        <MenuButton className={'ml-1 sm:ml-2'}>{menuPending ? spinner : <Cog className={iconSizeClass} />}</MenuButton>
        <MenuItems
          anchor={{ to: menuAnchor, gap: 16 }}
          className="bg-gray-900 border border-gray-700 rounded shadow-lg w-60 px-2 py-2 z-50"
        >
          {job.job_type === 'train' && (
            <MenuItem>
              <Link
                href={`/jobs/new?cloneId=${job.id}`}
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
              >
                <Copy className="w-4 h-4" />
                Clone Job
              </Link>
            </MenuItem>
          )}
          {job.job_type === 'train' && (
            <MenuItem>
              <div
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
                onClick={() => setCopyDestOpen(true)}
              >
                <FolderInput className="w-4 h-4" />
                Copy Final Checkpoint…
              </div>
            </MenuItem>
          )}
          {job.job_type === 'train' && canStop && (
            <MenuItem disabled={isBusy}>
              <div
                className={isBusy ? menuItemDisabledClass : menuItemEnabledClass}
                onClick={() => {
                  if (isBusy) return;
                  // does not change job state, so release as soon as the request resolves
                  runAction(
                    'save',
                    async () => {
                      await saveJobNow(job.id);
                      if (onRefresh) onRefresh();
                    },
                    false,
                  );
                }}
              >
                {pending === 'save' ? menuSpinner : <Save className="w-4 h-4" />}
                Save Next Step
              </div>
            </MenuItem>
          )}
          {job.job_type === 'train' && canStop && (
            <MenuItem disabled={isBusy}>
              <div
                className={isBusy ? menuItemDisabledClass : menuItemEnabledClass}
                onClick={() => {
                  if (isBusy) return;
                  // does not change job state, so release as soon as the request resolves
                  runAction(
                    'sample',
                    async () => {
                      await sampleJobNow(job.id);
                      if (onRefresh) onRefresh();
                    },
                    false,
                  );
                }}
              >
                {pending === 'sample' ? menuSpinner : <Image className="w-4 h-4" />}
                Sample Next Step
              </div>
            </MenuItem>
          )}
          <MenuItem disabled={isBusy}>
            <div
              className={isBusy ? menuItemDisabledClass : menuItemEnabledClass}
              onClick={() => {
                if (isBusy) return;
                let message = `Are you sure you want to mark this job as stopped? This will set the job status to 'stopped' if the status is hung. Only do this if you are 100% sure the job is stopped. This will NOT stop the job.`;
                openConfirm({
                  title: 'Mark Job as Stopped',
                  message: message,
                  type: 'warning',
                  confirmText: 'Mark as Stopped',
                  onConfirm: () =>
                    runAction('markStopped', async () => {
                      await markJobAsStopped(job.id);
                      onRefresh && onRefresh();
                    }),
                });
              }}
            >
              {pending === 'markStopped' ? menuSpinner : <OctagonX className="w-4 h-4" />}
              Mark as Stopped
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
      <FolderBrowserModal
        open={copyDestOpen}
        onClose={() => setCopyDestOpen(false)}
        onSelect={async folder => {
          setCopyDestOpen(false);
          try {
            const res = await copyFinalCheckpoint(job.id, folder);
            openConfirm({
              title: 'Checkpoint Copied',
              message: `Copied the latest checkpoint to:\n${res.to}`,
              type: 'info',
              confirmText: 'OK',
            });
          } catch (e: any) {
            openConfirm({
              title: 'Copy Failed',
              message: e?.response?.data?.error || 'Failed to copy the checkpoint.',
              type: 'warning',
              confirmText: 'OK',
            });
          }
        }}
      />
    </div>
  );
}
