import { JobConfig } from '@/types';
import { Finding, PreflightHardware, archSize } from './preflight';

// ---------------------------------------------------------------------------
// Stage A: rules-based training-config review.
//
// This is deliberately deterministic and explainable — every finding cites the
// numbers it is based on, so the advice can be checked rather than trusted. It
// encodes widely-used conventions (LoRA learning rates, steps-per-image ranges,
// bucket divisibility), NOT learned or model-derived judgements. Conventions
// are guidance, not laws; findings are phrased accordingly and none block a run.
//
// It is separate from preflight.ts on purpose: preflight answers "will this run
// on this machine?", this answers "are these training settings sane?". Both
// emit the same Finding shape and render in the same modal.
// ---------------------------------------------------------------------------

// Latent bucket divisibility per architecture, mirroring each model's
// get_bucket_divisibility() on the Python side. Resolutions not divisible by
// this get rounded down at bucketing time, so the trained resolution differs
// from what was requested.
function bucketDivisibility(arch: string): number {
  const a = (arch || '').toLowerCase();
  if (a.includes('krea2')) return 16; // vae 8 * patch 2
  if (a.includes('qwen_image') || a.includes('qwen-image')) return 32; // vae 16 * patch 2
  if (a.includes('wan')) return 16;
  if (a.includes('flux')) return 16;
  if (a.includes('sd3') || a.includes('sd35')) return 16;
  if (a.includes('sdxl')) return 8;
  if (a.includes('sd1') || a.includes('sd15')) return 8;
  return 8; // safe default; most VAEs are 8x
}

// Rough weight class of the denoiser, used to scale LR expectations. Big
// transformers generally want gentler LoRA LRs than small ones.
function transformerIsLarge(arch: string): boolean {
  const a = (arch || '').toLowerCase();
  return (
    a.includes('krea2') ||
    a.includes('qwen') ||
    a.includes('flux2') ||
    a.includes('klein') ||
    a.includes('wan22_14b') ||
    a.includes('wan21_14b') ||
    a.includes('hidream')
  );
}

function fmtLr(lr: number): string {
  return lr.toExponential(1).replace('e', 'e');
}

/**
 * Review the training-quality aspects of a job config.
 *
 * @param job          the full job config
 * @param imageCount   total real images across the job's datasets (before
 *                     num_repeats), or null if it could not be counted. When
 *                     null, the steps-per-image check is skipped rather than
 *                     guessed.
 * @param hardware     detected machine (RAM/VRAM), or null. When provided, adds
 *                     hardware-aware findings that cross-reference the config
 *                     against what this machine can afford.
 */
export function reviewTrainingConfig(
  job: JobConfig,
  imageCount: number | null,
  hardware: PreflightHardware | null = null,
): Finding[] {
  const findings: Finding[] = [];
  const process = job?.config?.process?.[0];
  if (!process) return findings;

  const train = process.train;
  const model = process.model;
  const network = process.network;
  const datasets = process.datasets ?? [];
  const arch = model?.arch || '';
  const isLora = network?.type === 'lora' || network?.type === 'lokr';

  // ---- Steps per image -------------------------------------------------
  // The single most useful signal for likeness/style LoRA. Too few steps per
  // image underfits; too many overfits (memorises the set, poor prompt
  // adherence). Ranges are conventions for LoRA, widened for full finetunes.
  const steps = train?.steps ?? 0;
  if (imageCount != null && imageCount > 0 && steps > 0) {
    // imageCount is the summed real image count across datasets; apply the max
    // num_repeats as an upper-bound multiplier when set.
    const maxRepeats = Math.max(1, ...datasets.map(d => d.num_repeats ?? 1));
    const exposures = imageCount * maxRepeats;
    const perImage = steps / exposures;

    const lo = isLora ? 40 : 20;
    const hi = isLora ? 200 : 120;
    const base =
      `${steps} steps ÷ ${exposures} image exposure(s)` +
      (maxRepeats > 1 ? ` (${imageCount} images × ${maxRepeats} repeats)` : ` (${imageCount} images)`) +
      ` ≈ ${perImage.toFixed(0)} steps/image.`;

    if (perImage < lo) {
      findings.push({
        id: 'steps-low',
        level: 'info',
        title: 'Few steps per image',
        detail:
          `${base} That is on the low side for a ${isLora ? 'LoRA' : 'finetune'}; likeness/style may not fully converge. ` +
          `Typical range is ~${lo}–${hi} steps/image. Consider more steps or fewer images if results look weak.`,
        setting: 'train.steps',
        current: String(steps),
        recommended: `~${lo * exposures}–${hi * exposures}`,
      });
    } else if (perImage > hi) {
      findings.push({
        id: 'steps-high',
        level: 'warning',
        title: 'Many steps per image — overfitting risk',
        detail:
          `${base} That is high for a ${isLora ? 'LoRA' : 'finetune'} (typical ~${lo}–${hi} steps/image), which risks overfitting: ` +
          `the model may memorise the training images and lose prompt flexibility. Watch the samples for rigidity, or reduce steps.`,
        setting: 'train.steps',
        current: String(steps),
        recommended: `~${lo * exposures}–${hi * exposures}`,
      });
    }
  }

  // ---- Learning rate ---------------------------------------------------
  const lr = train?.lr;
  if (typeof lr === 'number' && lr > 0) {
    if (isLora) {
      if (lr > 5e-4) {
        findings.push({
          id: 'lr-high',
          level: 'warning',
          title: 'Learning rate is high for a LoRA',
          detail:
            `LR ${fmtLr(lr)} is above the usual LoRA range (~1e-4). High LRs can train fast but often "fry" the adapter — ` +
            `colour shifts, artefacts, poor generalisation. If samples degrade early, lower it.`,
          setting: 'train.lr',
          current: fmtLr(lr),
          recommended: '1e-4',
          fix: [{ path: 'config.process[0].train.lr', value: 1e-4 }],
        });
      } else if (lr < 2e-5) {
        findings.push({
          id: 'lr-low',
          level: 'info',
          title: 'Learning rate is low',
          detail: `LR ${fmtLr(lr)} is below the usual LoRA range (~1e-4); training may be slow to converge. This is safe, just slower.`,
          setting: 'train.lr',
          current: fmtLr(lr),
          recommended: '1e-4',
          fix: [{ path: 'config.process[0].train.lr', value: 1e-4 }],
        });
      } else if (lr >= 1e-4 && transformerIsLarge(arch)) {
        findings.push({
          id: 'lr-large-model',
          level: 'info',
          title: 'Consider a gentler LR for a large model',
          detail:
            `LR ${fmtLr(lr)} is a common LoRA default, but large transformers (${arch}) often train more cleanly at ~5e-5, ` +
            `trading a little speed for stability. Optional — 1e-4 frequently works too.`,
          setting: 'train.lr',
          current: fmtLr(lr),
          recommended: '5e-5',
          fix: [{ path: 'config.process[0].train.lr', value: 5e-5 }],
        });
      }
    } else {
      // full / non-LoRA finetune expects much smaller LRs
      if (lr >= 1e-4) {
        findings.push({
          id: 'lr-full-high',
          level: 'warning',
          title: 'Learning rate is high for a full finetune',
          detail:
            `LR ${fmtLr(lr)} is LoRA-scale, but this job is not a LoRA. Full/other finetunes usually need ~1e-5 or lower; ` +
            `this high can destabilise the base weights.`,
          setting: 'train.lr',
          current: fmtLr(lr),
          recommended: '1e-5',
          fix: [{ path: 'config.process[0].train.lr', value: 1e-5 }],
        });
      }
    }
  }

  // ---- Alpha vs rank ---------------------------------------------------
  // The LoRA scale applied at inference is alpha/rank. When they differ, the
  // effective LR is scaled, which is easy to overlook.
  if (isLora && network) {
    const rank = network.linear;
    const alpha = network.linear_alpha;
    if (rank > 0 && alpha > 0 && alpha !== rank) {
      const scale = alpha / rank;
      findings.push({
        id: 'alpha-rank',
        level: 'info',
        title: 'Alpha differs from rank',
        detail:
          `linear_alpha ${alpha} / linear rank ${rank} gives a LoRA scale of ${scale.toFixed(2)}×, which effectively ` +
          `${scale < 1 ? 'reduces' : 'increases'} the learning rate by that factor. Common setups use alpha = rank (scale 1.0). ` +
          `If you intended full-strength training, set alpha = rank.`,
        setting: 'network.linear_alpha',
        current: String(alpha),
        recommended: `= rank (${rank})`,
        fix: [{ path: 'config.process[0].network.linear_alpha', value: rank }],
      });
    }
  }

  // ---- Resolution divisibility ----------------------------------------
  const div = bucketDivisibility(arch);
  const badRes = new Set<number>();
  for (const d of datasets) {
    for (const r of d.resolution ?? []) {
      if (r % div !== 0) badRes.add(r);
    }
  }
  if (badRes.size > 0) {
    const examples = [...badRes]
      .slice(0, 3)
      .map(r => `${r}→${Math.floor(r / div) * div}`)
      .join(', ');
    // Fix: round each affected dataset's resolution array down to the nearest
    // multiple of div (deduped, preserving order).
    const resFix = datasets
      .map((d, i) => ({ i, res: d.resolution ?? [] }))
      .filter(({ res }) => res.some(r => r % div !== 0))
      .map(({ i, res }) => ({
        path: `config.process[0].datasets[${i}].resolution`,
        value: [...new Set(res.map(r => Math.floor(r / div) * div))],
      }));
    findings.push({
      id: 'res-divis',
      level: 'warning',
      title: 'Resolution not divisible by bucket size',
      detail:
        `${arch} buckets to multiples of ${div}px, but ${badRes.size} configured resolution(s) are not multiples: ${examples}. ` +
        `Those images train at the rounded-down size, so you lose a little detail and the effective resolution differs from what you set.`,
      setting: 'datasets[].resolution',
      current: [...badRes].join(', '),
      recommended: `multiples of ${div}`,
      fix: resFix,
    });
  }

  // ---- Trigger-word tokenization (heuristic) --------------------------
  // A genuine heuristic, clearly labelled: we cannot tokenize client-side, so
  // this flags patterns that commonly tokenize into several weak sub-tokens.
  const trigger = process.trigger_word;
  if (trigger && trigger.trim()) {
    const t = trigger.trim();
    const hyphenated = /[-_\s]/.test(t);
    if (hyphenated) {
      findings.push({
        id: 'trigger-tokens',
        level: 'info',
        title: 'Trigger word may tokenize into several tokens (heuristic)',
        detail:
          `"${t}" contains hyphens/spaces, so most text encoders split it into multiple sub-tokens. Likeness then has to bind ` +
          `across several weak tokens instead of one, which can dilute it. A single uncommon token (e.g. a short made-up word) ` +
          `often binds a subject more strongly. This is a rule-of-thumb, not a measurement.`,
        setting: 'trigger_word',
        current: t,
        recommended: 'a single uncommon token',
      });
    }
  }

  // ---- Caption dropout vs trigger -------------------------------------
  // If the default caption IS the trigger and dropout is high, the trigger is
  // omitted from that fraction of steps, weakening the association.
  const highDropoutIdx = datasets.findIndex(
    d => (d.caption_dropout_rate ?? 0) >= 0.2 && (d.default_caption ?? '').trim() === (trigger ?? '').trim() && !!trigger,
  );
  if (highDropoutIdx >= 0) {
    const rate = datasets[highDropoutIdx].caption_dropout_rate;
    findings.push({
      id: 'caption-dropout',
      level: 'info',
      title: 'High caption dropout with a trigger word',
      detail:
        `caption_dropout_rate ${rate} means the caption (your trigger "${trigger}") is dropped on ~${Math.round(rate * 100)}% of steps. ` +
        `Some dropout aids generalisation, but a high rate on a likeness job can weaken how strongly the trigger binds the subject.`,
      setting: 'datasets[].caption_dropout_rate',
      current: String(rate),
      recommended: '≤ 0.1 for likeness',
      fix: [{ path: `config.process[0].datasets[${highDropoutIdx}].caption_dropout_rate`, value: 0.1 }],
    });
  }

  // ---- Hardware-aware: quality headroom -------------------------------
  // preflight.ts flags when settings WON'T fit. This is the opposite
  // direction: when the machine has enough headroom to RELAX a
  // memory-saving setting for better quality. Only fires with hardware.
  if (hardware) {
    const ramGB = hardware.ramTotalMB / 1024;
    const vramGB = hardware.gpus?.[0] ? hardware.gpus[0].memTotalMB / 1024 : 0;
    const size = archSize(arch);
    const bf16Weights = size.transformerGB + size.teGB; // full-precision load footprint
    const offloading = !!model?.layer_offloading;
    const large = transformerIsLarge(arch);

    // Large model, transformer quantized, but RAM is ample and offloading is on
    // (so the transformer lives in CPU RAM, not VRAM). With enough RAM the
    // quality cost of quantizing the transformer is optional.
    if (large && model?.quantize && offloading && ramGB > 0) {
      const ramHeadroom = ramGB * 0.85;
      if (bf16Weights < ramHeadroom) {
        findings.push({
          id: 'hw-quant-headroom',
          level: 'info',
          title: 'RAM headroom — bf16 transformer is an option',
          detail:
            `This machine has ~${ramGB.toFixed(0)} GB RAM, comfortably more than ${size.label}'s ~${bf16Weights.toFixed(0)} GB of weights, ` +
            `and layer offloading is on (so the transformer sits in CPU RAM, not VRAM). ` +
            `Transformer quantization is therefore trading quality for memory you may not need — try quantize off (bf16) for higher-fidelity training. ` +
            `Keep text-encoder quantization if VRAM (~${vramGB.toFixed(0)} GB) is tight during the encode pass.`,
          setting: 'model.quantize',
          current: `true (${model?.qtype ?? 'qfloat8'})`,
          recommended: 'try false (bf16) with offloading',
          fix: [{ path: 'config.process[0].model.quantize', value: false }],
        });
      }
    }

    // NOTE: deliberately NOT suggesting "disable offloading to go faster" from a
    // static VRAM estimate. Real telemetry (the Melissa krea2 run) peaked at 98%
    // VRAM *with* offloading already on, so a naive "the transformer fits, drop
    // offloading" rule would cause an OOM. Whether offloading can be relaxed is
    // decided from measured peak VRAM in the run post-mortem (runAnalysis.ts),
    // not guessed here.
    void vramGB;
  }

  return findings;
}
