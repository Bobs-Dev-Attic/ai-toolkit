import { JobConfig } from '@/types';

// ---------------------------------------------------------------------------
// Hardware snapshot (assembled by the modal from /api/gpu, /api/cpu, /api/disk)
// ---------------------------------------------------------------------------
export interface PreflightHardware {
  gpus: { name: string; memTotalMB: number; memFreeMB: number }[];
  cpuCores: number;
  ramTotalMB: number;
  ramFreeMB: number;
  diskFreeGB: number | null;
  diskTotalGB: number | null;
}

export type FindingLevel = 'error' | 'warning' | 'info' | 'ok';

export interface Finding {
  id: string;
  level: FindingLevel;
  title: string;
  detail: string;
  setting?: string;
  current?: string;
  recommended?: string;
}

// ---------------------------------------------------------------------------
// Rough per-arch weight sizes in GB at bf16 (full precision). These are
// deliberately approximate — they exist to catch "won't fit" situations, not to
// be exact. transformer = denoiser; te = text encoder(s).
// ---------------------------------------------------------------------------
interface ArchSize {
  transformerGB: number;
  teGB: number;
  label: string;
}

function archSize(arch: string): ArchSize {
  const a = (arch || '').toLowerCase();
  // order matters: match most specific first
  if (a.includes('flux2') || a.includes('klein')) return { transformerGB: 18, teGB: 16, label: 'Flux.2 Klein 9B' };
  if (a.includes('wan22_14b') || a.includes('wan21_14b') || a.includes('wan2_14b'))
    return { transformerGB: 28, teGB: 11, label: 'Wan 14B (dual expert)' };
  if (a.includes('wan22_5b') || a.includes('wan22_ti2v') || a.includes('5b'))
    return { transformerGB: 10, teGB: 11, label: 'Wan 5B' };
  if (a.includes('qwen_image') || a.includes('qwen-image')) return { transformerGB: 40, teGB: 16, label: 'Qwen-Image' };
  if (a.includes('flux')) return { transformerGB: 24, teGB: 10, label: 'Flux.1' };
  if (a.includes('sd3') || a.includes('sd35')) return { transformerGB: 16, teGB: 10, label: 'SD3' };
  if (a.includes('sdxl')) return { transformerGB: 5, teGB: 1.5, label: 'SDXL' };
  if (a.includes('sd1') || a.includes('sd15')) return { transformerGB: 2, teGB: 0.5, label: 'SD1.x' };
  return { transformerGB: 12, teGB: 6, label: arch || 'unknown model' };
}

// Fraction of full-precision size a quantization type keeps.
function quantFactor(qtype: string | undefined, quantize: boolean | undefined): number {
  if (!quantize) return 1;
  const q = (qtype || '').toLowerCase();
  if (q.includes('uint4') || q.includes('int4') || q.includes('4bit')) return 0.3;
  if (q.includes('float8') || q.includes('int8') || q.includes('8bit')) return 0.55;
  return 0.55; // assume 8-bit-ish
}

function fmtGB(gb: number): string {
  if (gb >= 100) return `${gb.toFixed(0)} GB`;
  return `${gb.toFixed(1)} GB`;
}

// ---------------------------------------------------------------------------
// Main analysis. Returns findings ordered error -> warning -> info -> ok.
// ---------------------------------------------------------------------------
export function analyzePreflight(job: JobConfig, hw: PreflightHardware): Finding[] {
  const findings: Finding[] = [];
  const process = job?.config?.process?.[0];
  if (!process) {
    return [{ id: 'noconfig', level: 'error', title: 'Invalid config', detail: 'Could not read the job configuration.' }];
  }

  const model = process.model;
  const train = process.train;
  const sample = process.sample;
  const datasets = process.datasets ?? [];

  const size = archSize(model?.arch);
  const tFactor = quantFactor(model?.qtype, model?.quantize);
  const teFactor = quantFactor(model?.qtype_te, model?.quantize_te);

  const transformerEff = size.transformerGB * tFactor;
  const teEff = size.teGB * teFactor;
  const weightsEff = transformerEff + teEff;
  const weightsFull = size.transformerGB + size.teGB;

  const gpu = hw.gpus?.[0];
  const vramGB = gpu ? gpu.memTotalMB / 1024 : 0;
  const ramGB = hw.ramTotalMB / 1024;
  const offloading = !!model?.layer_offloading;
  const lowVram = !!model?.low_vram;

  // ---- GPU present? ----
  if (process.device?.startsWith('cuda') && (!hw.gpus || hw.gpus.length === 0)) {
    findings.push({
      id: 'nogpu',
      level: 'error',
      title: 'No NVIDIA GPU detected',
      detail:
        'The job targets CUDA but no GPU was found via nvidia-smi. Training will fail. Check drivers, or switch device if you meant to use another accelerator.',
      setting: 'device',
      current: process.device,
    });
  }

  // ---- System RAM vs weights (the classic silent-crash-on-load case) ----
  // With offloading/low_vram, weights are held in CPU RAM. Unquantized loads
  // also spike RAM to full precision transiently.
  if (ramGB > 0) {
    const ramBudget = ramGB * 0.85; // leave headroom for OS + framework
    if (weightsEff > ramBudget) {
      const wantQuant = !model?.quantize || !model?.quantize_te;
      findings.push({
        id: 'ram-weights',
        level: 'error',
        title: 'Model likely will not fit in system RAM',
        detail:
          `${size.label} needs ~${fmtGB(weightsEff)} of weights (transformer ~${fmtGB(transformerEff)} + text encoder ~${fmtGB(teEff)}), ` +
          `but this machine has ${fmtGB(ramGB)} RAM. With layer offloading the weights live in system RAM, so loading will likely be killed by the OS (a silent crash with no traceback). ` +
          (wantQuant
            ? 'Enable quantization on the transformer and text encoder to roughly halve (8-bit) or quarter (4-bit) this.'
            : 'Consider a smaller model or more RAM.'),
        setting: 'quantize / quantize_te',
        current: `quantize=${!!model?.quantize}, quantize_te=${!!model?.quantize_te}`,
        recommended: wantQuant ? 'both on (float8 / uint4)' : 'add RAM',
      });
    } else if (weightsFull > ramBudget && (!model?.quantize || !model?.quantize_te)) {
      // Fits once quantized, but the unquantized *load transient* may not.
      findings.push({
        id: 'ram-transient',
        level: 'warning',
        title: 'Tight system RAM during model load',
        detail:
          `Loading ${size.label} unquantized momentarily needs ~${fmtGB(weightsFull)} before quantization frees memory, close to your ${fmtGB(ramGB)} of RAM. ` +
          'If loading crashes silently, enable quantization on both the transformer and text encoder, and close other memory-heavy apps.',
        setting: 'quantize / quantize_te',
        current: `quantize=${!!model?.quantize}, quantize_te=${!!model?.quantize_te}`,
        recommended: 'both on',
      });
    }
  }

  // ---- VRAM vs model ----
  if (vramGB > 0) {
    // With offloading, only a fraction of the transformer sits on the GPU at
    // once; without it (and not low_vram), roughly the whole transformer must fit
    // alongside activations.
    const offloadPct = model?.layer_offloading_transformer_percent ?? 0.5;
    const residentGB = offloading ? transformerEff * (1 - offloadPct) : transformerEff;
    const activationsGB = estimateActivationsGB(train, sample, datasets);
    const vramNeed = residentGB + activationsGB + (model?.quantize_te ? teEff : offloading ? 0 : teEff) * 0.0 + 1.5;

    const vramBudget = vramGB * 0.92;
    if (vramNeed > vramBudget && !offloading && !lowVram) {
      findings.push({
        id: 'vram-fit',
        level: 'warning',
        title: 'Model may not fit in VRAM',
        detail:
          `Estimated peak VRAM ~${fmtGB(vramNeed)} vs ${fmtGB(vramGB)} on ${gpu?.name ?? 'your GPU'}. ` +
          'Enable Low VRAM and/or layer offloading, enable quantization, or lower the sample resolution.',
        setting: 'low_vram / layer_offloading',
        current: `low_vram=${lowVram}, layer_offloading=${offloading}`,
        recommended: 'enable at least one',
      });
    } else if (vramNeed > vramBudget) {
      findings.push({
        id: 'vram-tight',
        level: 'info',
        title: 'VRAM will be tight',
        detail:
          `Estimated peak VRAM ~${fmtGB(vramNeed)} vs ${fmtGB(vramGB)}. Offloading/low-VRAM is on, which should help, but an OOM at the first sample or a large batch is possible.`,
      });
    }
  }

  // ---- Quantization suggestion when nothing is on but the model is large ----
  if ((size.transformerGB >= 12) && !model?.quantize && vramGB > 0 && vramGB < size.transformerGB * 1.3) {
    findings.push({
      id: 'quant-suggest',
      level: 'warning',
      title: 'Quantization is off for a large model',
      detail:
        `${size.label} is large relative to your ${fmtGB(vramGB)} of VRAM and quantization is disabled. 8-bit (float8) usually trains fine and roughly halves memory; 4-bit (uint4) with an accuracy-recovery adapter halves it again.`,
      setting: 'model.quantize',
      current: 'false',
      recommended: 'true (qfloat8)',
    });
  }

  // ---- I2V sampling without a first-frame image ----
  const archLower = (model?.arch || '').toLowerCase();
  const isI2V = archLower.includes('i2v');
  const samplingOn = !train?.disable_sampling;
  if (isI2V && samplingOn) {
    const samples = sample?.samples ?? [];
    const missing = samples.filter(s => !s.ctrl_img).length;
    const hasPromptsOnly = (sample?.prompts?.length ?? 0) > 0;
    if (missing > 0 || hasPromptsOnly) {
      findings.push({
        id: 'i2v-samples',
        level: 'warning',
        title: 'I2V samples have no control image',
        detail:
          `${missing || sample?.prompts?.length} sample prompt(s) have no first-frame (control) image. Image-to-video sampling needs one, so those samples will be skipped. ` +
          'Add a control image to each sample, or disable sampling to avoid the wasted setup.',
        setting: 'sample.samples[].ctrl_img / train.disable_sampling',
        current: 'no ctrl_img',
        recommended: 'add ctrl_img or disable sampling',
      });
    }
  }

  // ---- Disk space ----
  if (hw.diskFreeGB != null) {
    // LoRA saves are small; full/diffusers saves are large. Estimate keep-count × per-save.
    const isLora = process.network?.type === 'lora';
    const perSaveGB = isLora ? 0.6 : Math.max(2, size.transformerGB * tFactor);
    const keep = (train as any)?.max_step_saves_to_keep ?? process.save?.max_step_saves_to_keep ?? 4;
    const cacheGB = datasets.some(d => d.cache_latents_to_disk) ? 2 : 0;
    const needGB = perSaveGB * (keep + 1) + cacheGB + 1;
    if (hw.diskFreeGB < needGB) {
      findings.push({
        id: 'disk-low',
        level: hw.diskFreeGB < needGB * 0.5 ? 'error' : 'warning',
        title: 'Low free disk space',
        detail:
          `~${fmtGB(needGB)} may be needed for checkpoints${cacheGB ? ' + latent cache' : ''}, but only ${fmtGB(hw.diskFreeGB)} is free on the training drive. ` +
          'Free up space or lower "Max step saves to keep".',
        setting: 'save.max_step_saves_to_keep',
        current: String(keep),
        recommended: hw.diskFreeGB < needGB * 0.5 ? 'free disk space' : `lower to ${Math.max(1, Math.floor(hw.diskFreeGB / perSaveGB) - 1)}`,
      });
    }
  }

  // ---- Dataset sanity ----
  if (datasets.length === 0) {
    findings.push({
      id: 'no-dataset',
      level: 'error',
      title: 'No dataset configured',
      detail: 'The job has no dataset folder. Add at least one dataset before training.',
    });
  }

  // ---- Steps / save cadence sanity ----
  const steps = train?.steps ?? 0;
  const saveEvery = process.save?.save_every ?? 0;
  if (steps > 0 && saveEvery > 0 && saveEvery > steps) {
    findings.push({
      id: 'save-never',
      level: 'warning',
      title: 'Checkpoints never save',
      detail: `"Save every" (${saveEvery}) is greater than total steps (${steps}); no intermediate checkpoint will be written.`,
      setting: 'save.save_every',
      current: String(saveEvery),
      recommended: `≤ ${Math.max(1, Math.floor(steps / 4))}`,
    });
  }

  // ---- CPU cores (minor) ----
  if (hw.cpuCores && hw.cpuCores <= 4) {
    findings.push({
      id: 'cpu-cores',
      level: 'info',
      title: 'Few CPU cores',
      detail: `${hw.cpuCores} cores detected. Data loading / latent caching may bottleneck; consider caching latents to disk.`,
    });
  }

  // sort: error, warning, info, ok
  const order: Record<FindingLevel, number> = { error: 0, warning: 1, info: 2, ok: 3 };
  findings.sort((a, b) => order[a.level] - order[b.level]);
  return findings;
}

// Very rough activation/working-set estimate driven by sample resolution and
// video frame count (video sampling dominates VRAM).
function estimateActivationsGB(train: any, sample: any, datasets: any[]): number {
  const w = sample?.width ?? 1024;
  const h = sample?.height ?? 1024;
  const frames = sample?.num_frames ?? 1;
  const px = (w * h) / (1024 * 1024); // megapixels
  const base = 1.5 * px; // ~1.5GB per MP as a coarse anchor
  const video = frames > 1 ? base * Math.min(frames, 81) * 0.15 : 0;
  const batch = train?.batch_size ?? 1;
  return (base + video) * Math.max(1, batch);
}
