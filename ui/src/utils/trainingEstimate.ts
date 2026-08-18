// Rough per-step training-time model for the pre-flight goal tabs.
//
// IMPORTANT: absolute numbers are approximate (±30% or worse across GPUs). The
// model is calibrated against one real data point on a 32 GB Blackwell-class
// card — Krea 2 @ 512px, 8-bit + layer offloading + Low VRAM, uncached text
// embeddings measured ~2.08 s/it. The RELATIVE ordering between Speed / Quality
// / Fail-safe is far more reliable than the absolute totals, which is the point:
// the tabs let the user compare profiles, not trust a number to the minute.

export interface EstimateInput {
  arch: string;
  resolutionPx: number; // longest training edge (max across datasets)
  batchSize: number;
  gradientAccumulation: number;
  quantize: boolean; // transformer 8-bit (true) vs bf16 (false)
  layerOffloading: boolean;
  lowVram: boolean;
  cacheTextEmbeddings: boolean;
  steps: number;
}

export interface Estimate {
  secPerStep: number;
  totalSeconds: number;
}

// Baseline seconds/step for a quantized, VRAM-resident transformer at 512px,
// batch 1, on the reference GPU. Per-arch; large-model default otherwise.
function baseSecPerStep(arch: string): number {
  const a = (arch || '').toLowerCase();
  if (a.includes('krea2')) return 1.05; // calibrated: fail-safe@512 -> ~2.05 s/it (observed 2.08)
  if (a.includes('minimax')) return 1.2;
  if (a.includes('qwen')) return 1.0;
  if (a.includes('flux2') || a.includes('klein')) return 0.85;
  if (a.includes('flux')) return 0.8;
  if (a.includes('wan')) return 1.2;
  return 0.9;
}

const SETUP_SECONDS = 120; // one-off model load + quantization pass

export function estimateTraining(inp: EstimateInput): Estimate {
  const base = baseSecPerStep(inp.arch);
  // Compute scales roughly with token count (~pixel area) at 512 reference.
  const resFactor = Math.pow(Math.max(inp.resolutionPx, 256) / 512, 2);
  const offloadFactor = inp.layerOffloading ? 1.6 : 1.0; // PCIe streaming penalty
  const lowVramFactor = inp.lowVram ? 1.1 : 1.0;
  const precisionFactor = inp.quantize ? 1.0 : 1.15; // bf16 moves ~2x weight data
  const teFactor = inp.cacheTextEmbeddings ? 1.0 : 1.15; // text encoder runs each step if uncached
  const workFactor = Math.max(1, inp.batchSize) * Math.max(1, inp.gradientAccumulation);

  const secPerStep = base * resFactor * offloadFactor * lowVramFactor * precisionFactor * teFactor * workFactor;
  return {
    secPerStep,
    totalSeconds: SETUP_SECONDS + secPerStep * Math.max(0, inp.steps),
  };
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
