import { JobConfig } from '@/types';
import { Finding } from './preflight';

// ---------------------------------------------------------------------------
// Runtime post-mortem: read a finished/running job's own telemetry
// (system_stats.jsonl + loss_log.db) and surface what actually happened —
// speed, utilization, loss trend — plus honest, numbers-cited suggestions.
//
// This is descriptive, not predictive: it reports measured values and applies
// a few deterministic rules over them. It does NOT learn or infer quality —
// whether the output resembles the subject is a human judgement captured
// separately by the run rating.
// ---------------------------------------------------------------------------

export interface RunPoint {
  t?: number;
  elapsed?: number;
  step?: number;
  cpu_percent?: number;
  ram_used_mb?: number;
  ram_total_mb?: number;
  ram_percent?: number;
  proc_ram_mb?: number;
  vram_used_mb?: number;
  vram_total_mb?: number;
  vram_percent?: number;
  gpu_percent?: number;
  disk_percent?: number;
}

export interface RunSummary {
  sampleCount: number;
  elapsedSec: number | null;
  stepsCovered: number | null;
  secPerStep: number | null;
  gpuMeanPct: number | null;
  gpuP90Pct: number | null;
  vramPeakPct: number | null;
  vramPeakGB: number | null;
  vramTotalGB: number | null;
  ramPeakPct: number | null;
  procRamPeakGB: number | null;
  cpuMeanPct: number | null;
  lossFirst: number | null;
  lossLast: number | null;
  lossDeltaPct: number | null;
}

export interface RunAnalysis {
  summary: RunSummary;
  findings: Finding[];
}

function nums(arr: (number | null | undefined)[]): number[] {
  return arr.filter((x): x is number => typeof x === 'number' && !Number.isNaN(x));
}
function mean(xs: number[]): number | null {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}
function percentile(xs: number[], p: number): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
  return s[idx];
}
function max(xs: number[]): number | null {
  return xs.length ? Math.max(...xs) : null;
}

/**
 * Analyze a run from its telemetry.
 * @param points system_stats.jsonl rows (any subset of fields present)
 * @param loss   loss values in step order (may be empty)
 * @param job    the job config (for offloading/arch context)
 */
export function analyzeRun(points: RunPoint[], loss: number[], job: JobConfig | null): RunAnalysis {
  const process = job?.config?.process?.[0];
  const model = process?.model;
  const offloading = !!model?.layer_offloading;
  const lowVram = !!model?.low_vram;
  const offloadPct = model?.layer_offloading_transformer_percent ?? null;
  const arch = model?.arch || 'model';

  const elapsedVals = nums(points.map(p => p.elapsed));
  const stepVals = nums(points.map(p => p.step));
  const elapsedSec = elapsedVals.length ? (max(elapsedVals)! - Math.min(...elapsedVals)) : null;
  const stepsCovered = stepVals.length ? (max(stepVals)! - Math.min(...stepVals)) : null;
  const secPerStep = elapsedSec != null && stepsCovered && stepsCovered > 0 ? elapsedSec / stepsCovered : null;

  const gpuVals = nums(points.map(p => p.gpu_percent));
  const vramPctVals = nums(points.map(p => p.vram_percent));
  const vramGbVals = nums(points.map(p => (p.vram_used_mb != null ? p.vram_used_mb / 1024 : undefined)));
  const vramTotVals = nums(points.map(p => (p.vram_total_mb != null ? p.vram_total_mb / 1024 : undefined)));
  const ramPctVals = nums(points.map(p => p.ram_percent));
  const procRamVals = nums(points.map(p => (p.proc_ram_mb != null ? p.proc_ram_mb / 1024 : undefined)));
  const cpuVals = nums(points.map(p => p.cpu_percent));

  const lossFirst = loss.length ? mean(loss.slice(0, Math.max(1, Math.floor(loss.length * 0.1)))) : null;
  const lossLast = loss.length ? mean(loss.slice(-Math.max(1, Math.floor(loss.length * 0.1)))) : null;
  const lossDeltaPct = lossFirst != null && lossLast != null && lossFirst !== 0 ? ((lossLast - lossFirst) / lossFirst) * 100 : null;

  const summary: RunSummary = {
    sampleCount: points.length,
    elapsedSec,
    stepsCovered,
    secPerStep,
    gpuMeanPct: mean(gpuVals),
    gpuP90Pct: percentile(gpuVals, 90),
    vramPeakPct: max(vramPctVals),
    vramPeakGB: max(vramGbVals),
    vramTotalGB: max(vramTotVals),
    ramPeakPct: max(ramPctVals),
    procRamPeakGB: max(procRamVals),
    cpuMeanPct: mean(cpuVals),
    lossFirst,
    lossLast,
    lossDeltaPct,
  };

  const findings: Finding[] = [];

  if (points.length < 5) {
    findings.push({
      id: 'run-nodata',
      level: 'info',
      title: 'Not enough telemetry yet',
      detail: 'Fewer than 5 system-stats samples were recorded, so a meaningful post-mortem is not possible. This is normal for a run that just started or was stopped early.',
    });
    return { summary, findings };
  }

  // ---- VRAM headroom while offloading = wasted speed --------------------
  // The most reliable "you're leaving speed on the table" signal: offloading
  // is on (moves weights to CPU each step, slow) yet peak VRAM stayed low.
  if (offloading && summary.vramPeakPct != null && summary.vramPeakPct < 65) {
    const suggestPct = offloadPct != null ? Math.max(0, Math.round((offloadPct - 0.3) * 100) / 100) : 0.2;
    findings.push({
      id: 'run-vram-headroom',
      level: 'warning',
      title: 'GPU memory underused while offloading is on',
      detail:
        `Peak VRAM was ${summary.vramPeakPct.toFixed(0)}%` +
        (summary.vramPeakGB != null && summary.vramTotalGB != null ? ` (~${summary.vramPeakGB.toFixed(1)} / ${summary.vramTotalGB.toFixed(0)} GB)` : '') +
        `, but layer offloading is enabled — which moves weights to CPU each step and is a major speed cost` +
        (summary.secPerStep != null ? ` (this run averaged ${summary.secPerStep.toFixed(1)} s/step)` : '') +
        `. You likely have room to offload less and train faster.`,
      setting: 'model.layer_offloading_transformer_percent',
      current: offloadPct != null ? String(offloadPct) : 'on',
      recommended: `try ${suggestPct} (or disable offloading)`,
    });
  }

  // ---- Low GPU utilization (supporting, caveated) ----------------------
  // gpu_percent is an instantaneous sample and often reads 0 between steps, so
  // this is phrased as "worth checking", not a hard verdict.
  if (summary.gpuMeanPct != null && summary.gpuMeanPct < 25 && summary.sampleCount >= 20) {
    findings.push({
      id: 'run-gpu-idle',
      level: 'info',
      title: 'GPU utilization sampled low',
      detail:
        `Mean GPU utilization across ${summary.sampleCount} samples was ${summary.gpuMeanPct.toFixed(0)}% (90th pct ${summary.gpuP90Pct?.toFixed(0) ?? '?'}%). ` +
        `Note this metric is sampled instantaneously and often catches idle gaps between steps, so treat it as a hint, not proof. ` +
        `Combined with a slow step time it usually points to an I/O or offloading bottleneck rather than the GPU being the limit.`,
    });
  }

  // ---- RAM near the ceiling --------------------------------------------
  if (summary.ramPeakPct != null && summary.ramPeakPct >= 92) {
    findings.push({
      id: 'run-ram-tight',
      level: 'warning',
      title: 'System RAM ran near its limit',
      detail:
        `Peak system RAM hit ${summary.ramPeakPct.toFixed(0)}%` +
        (summary.procRamPeakGB != null ? ` (training process ~${summary.procRamPeakGB.toFixed(1)} GB)` : '') +
        `. That is close enough to the ceiling to risk OS paging (slowdowns) or an out-of-memory kill on a spike. ` +
        `Enabling quantization or closing other apps gives headroom.`,
      setting: 'model.quantize / quantize_te',
      current: `${summary.ramPeakPct.toFixed(0)}% peak`,
      recommended: 'keep peak < 90%',
    });
  }

  // ---- Step speed note --------------------------------------------------
  if (summary.secPerStep != null && summary.secPerStep >= 8) {
    findings.push({
      id: 'run-slow-step',
      level: 'info',
      title: 'Slow training steps',
      detail:
        `Average ${summary.secPerStep.toFixed(1)} s/step` +
        (summary.stepsCovered != null && summary.elapsedSec != null
          ? ` (${summary.stepsCovered} steps in ${(summary.elapsedSec / 60).toFixed(0)} min)`
          : '') +
        `. For ${arch} this usually reflects offloading/low-VRAM overhead. If VRAM has headroom (see above), reducing offloading is the biggest lever.`,
    });
  }

  // ---- Loss trend (reported honestly) ----------------------------------
  if (summary.lossFirst != null && summary.lossLast != null && loss.length >= 20) {
    const dp = summary.lossDeltaPct ?? 0;
    if (dp > -3) {
      findings.push({
        id: 'run-loss-flat',
        level: 'info',
        title: 'Loss is roughly flat (this is normal)',
        detail:
          `Mean loss went ${summary.lossFirst.toFixed(4)} → ${summary.lossLast.toFixed(4)} (${dp >= 0 ? '+' : ''}${dp.toFixed(1)}%). ` +
          `In diffusion/flow-matching training, per-step loss is dominated by which timestep is sampled, so a flat curve is expected and does NOT mean training failed. ` +
          `Judge results from the sample images (and rate the run below), not from loss.`,
      });
    } else {
      findings.push({
        id: 'run-loss-down',
        level: 'ok',
        title: 'Loss trended down',
        detail: `Mean loss went ${summary.lossFirst.toFixed(4)} → ${summary.lossLast.toFixed(4)} (${dp.toFixed(1)}%). Still, judge final quality from the samples, not loss alone.`,
      });
    }
  }

  const order: Record<string, number> = { error: 0, warning: 1, info: 2, ok: 3 };
  findings.sort((a, b) => order[a.level] - order[b.level]);
  return { summary, findings };
}
