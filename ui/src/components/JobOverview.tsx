import { Job } from '@prisma/client';
import useGPUInfo from '@/hooks/useGPUInfo';
import useCPUInfo from '@/hooks/useCPUInfo';
import GPUWidget from '@/components/GPUWidget';
import CPUWidget from '@/components/CPUWidget';
import FilesWidget from '@/components/FilesWidget';
import JobLogPanel from '@/components/JobLogPanel';
import { getTotalSteps } from '@/utils/jobs';
import { Cpu, HardDrive, Info, Gauge } from 'lucide-react';
import { useMemo } from 'react';

interface JobOverviewProps {
  job: Job;
}

// speed_string is written as "X.XX iter/sec" or "X.XX sec/iter" (see
// DiffusionTrainer.handle_timing_print_hook). Parse it back to seconds/iter.
function parseSecPerIter(speedString: string): number | null {
  if (!speedString) return null;
  const m = speedString.match(/([\d.]+)\s*(iter\/sec|sec\/iter)/);
  if (!m) return null;
  const val = parseFloat(m[1]);
  if (!Number.isFinite(val) || val <= 0) return null;
  return m[2] === 'iter/sec' ? 1 / val : val;
}

// Compact duration like "1h 04m 12s" / "3m 20s" / "45s".
function formatDuration(totalSec: number): string {
  if (!Number.isFinite(totalSec) || totalSec < 0) return '—';
  const s = Math.floor(totalSec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m ${String(ss).padStart(2, '0')}s`;
  if (m > 0) return `${m}m ${String(ss).padStart(2, '0')}s`;
  return `${ss}s`;
}

export default function JobOverview({ job }: JobOverviewProps) {
  const gpuIds = useMemo(() => {
    if (job.gpu_ids === 'mps') {
      return [0]; // For MPS, we can just return a single GPU ID since it's virtualized
    }
    return job.gpu_ids.split(',').map(id => parseInt(id));
  }, [job.gpu_ids]);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(gpuIds, 5000);
  const { cpuInfo, isCPUInfoLoaded } = useCPUInfo(5000);
  const totalSteps = getTotalSteps(job);
  const progress = (job.step / totalSteps) * 100;
  const isStopping = job.stop && job.status === 'running';

  // Elapsed / ETA estimated from the current iteration speed (no persisted start
  // time exists). Both are approximate and assume steady speed.
  const timing = useMemo(() => {
    const secPerIter = parseSecPerIter(job.speed_string);
    if (secPerIter == null || totalSteps <= 0) return null;
    const remaining = Math.max(0, totalSteps - job.step);
    return {
      elapsed: job.step * secPerIter,
      eta: remaining * secPerIter,
    };
  }, [job.speed_string, job.step, totalSteps]);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'running':
        return 'bg-emerald-500/10 text-emerald-500';
      case 'stopping':
        return 'bg-amber-500/10 text-amber-500';
      case 'stopped':
        return 'bg-gray-500/10 text-gray-400';
      case 'completed':
        return 'bg-blue-500/10 text-blue-500';
      case 'error':
        return 'bg-rose-500/10 text-rose-500';
      default:
        return 'bg-gray-500/10 text-gray-400';
    }
  };

  const jobType = job?.job_type || 'unknown';

  let status = job.status;
  if (isStopping) {
    status = 'stopping';
  }

  return (
    <div className="grid grid-cols-1 gap-4 md:gap-6 md:grid-cols-3">
      {/* Job Information Panel */}
      <div className="md:col-span-2 bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
        <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
          <h2 className="text-gray-100">
            <Info className="w-5 h-5 mr-2 -mt-1 text-amber-600 dark:text-amber-400 inline-block" /> {job.info}
          </h2>
          <span className={`px-3 py-1 rounded-full text-sm ${getStatusColor(job.status)}`}>{job.status}</span>
        </div>

        <div className="p-4 space-y-6 flex flex-col flex-grow">
          {/* Progress Bar */}
          {totalSteps > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm gap-3 flex-wrap">
                <div className="flex items-center gap-3">
                  <span className="text-gray-400">Progress</span>
                  {timing && (
                    <span className="text-xs text-gray-500 tabular-nums" title="Estimated from current speed">
                      Elapsed ~{formatDuration(timing.elapsed)}
                      <span className="mx-1 text-gray-700">•</span>
                      ETA ~{formatDuration(timing.eta)}
                    </span>
                  )}
                </div>
                <span className="text-gray-200">
                  Step {job.step} of {totalSteps} ({progress.toFixed(1)}%)
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2">
                <div className="h-2 rounded-full bg-blue-500 transition-all" style={{ width: `${progress}%` }} />
              </div>
            </div>
          )}

          {/* Job Info Grid */}
          <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
            <div className="flex items-center space-x-4">
              <HardDrive className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <div>
                <p className="text-xs text-gray-400">Job Name</p>
                <p className="text-sm font-medium text-gray-200">{job.name}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <Cpu className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              <div>
                <p className="text-xs text-gray-400">Assigned GPUs</p>
                <p className="text-sm font-medium text-gray-200">GPUs: {job.gpu_ids}</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <Gauge className="w-5 h-5 text-green-600 dark:text-green-400" />
              <div>
                <p className="text-xs text-gray-400">Speed</p>
                <p className="text-sm font-medium text-gray-200">{job.speed_string == '' ? '?' : job.speed_string}</p>
              </div>
            </div>
          </div>

          {/* Log panel: raw / table views, copy, verbose, filter, export */}
          <JobLogPanel jobID={job.id} />
        </div>
      </div>

      {/* GPU Widget Panel */}
      <div className="md:col-span-1">
        <div>{isCPUInfoLoaded && cpuInfo && <CPUWidget cpu={cpuInfo} />}</div>
        <div className="mt-4">{isGPUInfoLoaded && gpuList.length > 0 && <GPUWidget gpu={gpuList[0]} />}</div>
        {jobType === 'train' && (
          <div className="mt-4">
            <FilesWidget jobID={job.id} jobName={job.name} />
          </div>
        )}
      </div>
    </div>
  );
}
