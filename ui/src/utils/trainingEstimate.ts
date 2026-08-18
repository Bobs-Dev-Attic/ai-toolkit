// Per-step training-time model for the pre-flight goal tabs.
//
// The model is `secPerStep = base(arch) * stepMultiplier(profile)`. The
// multiplier captures the RELATIVE cost of resolution / offloading / precision /
// low-VRAM / TE-cache / batch; `base` is the per-arch anchor for one step at
// 512px, batch 1, quantized-and-resident, on THIS machine.
//
// `base` starts from a hardcoded default (calibrated to a 32 GB Blackwell card)
// but is overridden by `calibrateBaseByArch()` whenever the machine has real
// past-run telemetry: each past run's observed s/it is divided by its own
// multiplier to recover the machine's true base, and the median is used. That
// turns "rough estimate" into a measured one for arches the user has trained.

export interface StepFactors {
  resolutionPx: number; // longest training edge (max across datasets)
  batchSize: number;
  gradientAccumulation: number;
  quantize: boolean; // transformer 8-bit (true) vs bf16 (false)
  layerOffloading: boolean;
  lowVram: boolean;
  cacheTextEmbeddings: boolean;
}

export interface EstimateInput extends StepFactors {
  arch: string;
  steps: number;
}

export interface Estimate {
  secPerStep: number;
  totalSeconds: number;
}

const SETUP_SECONDS = 120; // one-off model load + quantization pass

// Family key so arch variants (krea2 / krea2:turbo …) share one calibration.
export function archFamily(arch: string): string {
  const a = (arch || '').toLowerCase();
  if (a.includes('krea2')) return 'krea2';
  if (a.includes('minimax')) return 'minimax';
  if (a.includes('qwen')) return 'qwen';
  if (a.includes('flux2') || a.includes('klein')) return 'flux2';
  if (a.includes('flux')) return 'flux';
  if (a.includes('wan')) return 'wan';
  return a || 'other';
}

// Hardcoded fallback base (s/step at 512px, quantized-resident, batch 1).
function defaultBase(arch: string): number {
  switch (archFamily(arch)) {
    case 'krea2':
      return 1.05; // calibrated: fail-safe@512 -> ~2.05 s/it (observed 2.08)
    case 'minimax':
      return 1.2;
    case 'qwen':
      return 1.0;
    case 'flux2':
      return 0.85;
    case 'flux':
      return 0.8;
    case 'wan':
      return 1.2;
    default:
      return 0.9;
  }
}

// Product of every factor except the per-arch base. Compute scales ~ with token
// count (pixel area) at 512 reference; the rest are measured penalties.
export function stepMultiplier(f: StepFactors): number {
  const resFactor = Math.pow(Math.max(f.resolutionPx, 256) / 512, 2);
  const offloadFactor = f.layerOffloading ? 1.6 : 1.0; // PCIe streaming penalty
  const lowVramFactor = f.lowVram ? 1.1 : 1.0;
  const precisionFactor = f.quantize ? 1.0 : 1.15; // bf16 moves ~2x weight data
  const teFactor = f.cacheTextEmbeddings ? 1.0 : 1.15; // text encoder runs each step if uncached
  const workFactor = Math.max(1, f.batchSize) * Math.max(1, f.gradientAccumulation);
  return resFactor * offloadFactor * lowVramFactor * precisionFactor * teFactor * workFactor;
}

export function estimateTraining(inp: EstimateInput, calibratedBase?: number | null): Estimate {
  const base = calibratedBase && calibratedBase > 0 ? calibratedBase : defaultBase(inp.arch);
  const secPerStep = base * stepMultiplier(inp);
  return {
    secPerStep,
    totalSeconds: SETUP_SECONDS + secPerStep * Math.max(0, inp.steps),
  };
}

// ---- Calibration from past-run telemetry --------------------------------

export interface CalibrationSample extends StepFactors {
  arch: string;
  observedSecPerStep: number; // parsed from a past run's speed_string
}

export interface ArchCalibration {
  base: number; // median implied base for this arch family
  samples: number; // how many runs contributed
}

// speed_string is "X.XX iter/sec" or "X.XX sec/iter"
// (DiffusionTrainer.handle_timing_print_hook). Parse to seconds/iter.
export function parseSpeedString(speedString: string | null | undefined): number | null {
  if (!speedString) return null;
  const m = speedString.match(/([\d.]+)\s*(iter\/sec|sec\/iter)/);
  if (!m) return null;
  const val = parseFloat(m[1]);
  if (!Number.isFinite(val) || val <= 0) return null;
  return m[2] === 'iter/sec' ? 1 / val : val;
}

// Recover each machine's true per-arch base from real runs: base = observed /
// multiplier. Median over same-family runs, ignoring implausible values.
export function calibrateBaseByArch(samples: CalibrationSample[]): Record<string, ArchCalibration> {
  const byFamily: Record<string, number[]> = {};
  for (const s of samples) {
    const mult = stepMultiplier(s);
    if (!(mult > 0) || !(s.observedSecPerStep > 0)) continue;
    const implied = s.observedSecPerStep / mult;
    if (!(implied >= 0.02) || implied > 60) continue; // reject garbage/outliers
    (byFamily[archFamily(s.arch)] ??= []).push(implied);
  }
  const out: Record<string, ArchCalibration> = {};
  for (const [family, arr] of Object.entries(byFamily)) {
    arr.sort((a, b) => a - b);
    const mid = Math.floor(arr.length / 2);
    const median = arr.length % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
    out[family] = { base: median, samples: arr.length };
  }
  return out;
}

// "3h 12m" / "45m" / "30s"
export function formatDuration(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m`;
  return `${s}s`;
}
