import { JobConfig } from '@/types';
import { Finding, FindingOption, PreflightHardware, archSize, isQwenImage21 } from './preflight';

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
  if (a.includes('minimax_h3') || a.includes('minimax')) return 32; // vae 16 * patch 2
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
  // ~7B single-stream DiT: not in the 12-20B class this LR nudge is written for,
  // and the Qwen-Image 2.1 guide gives no LR advice, so don't invent one.
  if (isQwenImage21(a)) return false;
  return (
    a.includes('krea2') ||
    a.includes('minimax') ||
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

    // ---- Trigger missing from sample prompts -------------------------
    // The subject is bound to the trigger token during training, but the
    // trainer injects it into a sample prompt ONLY where a [trigger]/[name]
    // placeholder exists (inject_trigger_into_prompt add_if_not_present=False).
    // A prompt with neither the placeholder nor the literal trigger samples a
    // generic subject — so likeness "doesn't work" even when the LoRA is fine.
    // This is the single most common reason trained samples look nothing like
    // the dataset, so it is a warning, not an info.
    const samplingOn = !train?.disable_sampling;
    const promptStrings: string[] = [
      ...((process.sample?.samples ?? []).map(s => s?.prompt ?? '')),
      ...((process.sample?.prompts ?? []) as string[]),
    ].filter(p => typeof p === 'string' && p.trim() !== '');
    if (samplingOn && promptStrings.length > 0) {
      const tl = t.toLowerCase();
      const carriesTrigger = (p: string) => {
        const pl = p.toLowerCase();
        return pl.includes(tl) || pl.includes('[trigger]') || pl.includes('[name]');
      };
      const withTrigger = promptStrings.filter(carriesTrigger).length;
      if (withTrigger === 0) {
        findings.push({
          id: 'trigger-missing-in-samples',
          level: 'warning',
          title: 'Sample prompts never use the trigger word',
          detail:
            `The trigger "${t}" binds the subject during training, but none of your ${promptStrings.length} sample prompt(s) contain it or a [trigger] placeholder. ` +
            `The trainer does not add the trigger automatically, so every sample renders a generic subject and will not resemble the dataset — ` +
            `even when the LoRA trained correctly. Add "${t}" (or [trigger]) to each sample prompt.`,
          setting: 'sample.samples[].prompt',
          current: `0 of ${promptStrings.length} prompts include the trigger`,
          recommended: 'add the trigger / [trigger] to each prompt',
        });
      }
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

  // ---- Caption dropout vs cached text embeddings ----------------------
  // Caching text embeddings encodes each caption once and reuses it, so
  // caption_dropout_rate (and token dropout / shuffle) never takes effect —
  // the dataloader explicitly skips dropout when cache_text_embeddings is on.
  // The two settings silently cancel, which surprises people who set both.
  if (train?.cache_text_embeddings) {
    const dropIdx = datasets.findIndex(d => (d.caption_dropout_rate ?? 0) > 0);
    if (dropIdx >= 0) {
      const rate = datasets[dropIdx].caption_dropout_rate;
      findings.push({
        id: 'dropout-vs-cache-te',
        level: 'warning',
        title: 'Caption dropout is disabled by cached text embeddings',
        detail:
          `cache_text_embeddings is on, which encodes each caption once and reuses it — so caption_dropout_rate (${rate}) has no effect; the trainer skips dropout (and token dropout / shuffle) whenever embeddings are cached. ` +
          `Turn Cache Text Embeddings OFF to get real caption dropout, or set caption_dropout_rate to 0 so the config reflects what actually happens.`,
        setting: 'train.cache_text_embeddings / datasets[].caption_dropout_rate',
        current: `cache_text_embeddings=true, caption_dropout_rate=${rate}`,
        recommended: 'cache off (for dropout), or dropout 0',
      });
    }
  }

  // ---- Krea 2 model-specific recipe (Krea / RunComfy guidance) --------
  // Krea 2 trains with a flow-matching schedule and a model-specific time
  // distribution, so Linear is the correct timestep type. The Turbo variants
  // are distilled few-step students: they need the assistant training adapter
  // to expose a usable training signal, and must be validated in their native
  // ~8-step / guidance-1 regime rather than the Raw preview recipe.
  {
    const a = arch.toLowerCase();
    if (a.includes('krea2')) {
      const isTurbo = a.includes('turbo');
      const ts = (train?.timestep_type || '').toLowerCase();

      if (ts && ts !== 'linear') {
        findings.push({
          id: 'krea2-timestep-linear',
          level: 'warning',
          title: 'Krea 2 should use Linear timesteps',
          detail:
            `timestep_type is "${train?.timestep_type}", but Krea 2 uses a flow-matching schedule where Linear is the correct setting (with the FlowMatch scheduler). ` +
            `Weighted/Sigmoid recipes copied from FLUX, video, or older diffusion setups push learning toward the wrong noise regions — the symptom is weak concept pickup or a LoRA that only works at extreme weight.`,
          setting: 'train.timestep_type',
          current: String(train?.timestep_type ?? ''),
          recommended: 'linear',
          fix: [{ path: 'config.process[0].train.timestep_type', value: 'linear' }],
        });
      }

      if (isTurbo) {
        const adapter = model?.assistant_lora_path;
        if (!adapter || String(adapter).trim() === '') {
          findings.push({
            id: 'krea2-turbo-adapter',
            level: 'warning',
            title: 'Krea 2 Turbo needs its training adapter',
            detail:
              `This is a Krea 2 Turbo (distilled) run, but no assistant training adapter is set. Turbo has a compressed few-step trajectory that ordinary fine-tuning erases quickly; the adapter temporarily de-distills the student so a LoRA can train. ` +
              `Without it, 8-step previews degrade while high-step previews look deceptively better — that is damaged distillation, not a good LoRA.`,
            setting: 'model.assistant_lora_path',
            current: 'unset',
            recommended: 'ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors',
            fix: [
              {
                path: 'config.process[0].model.assistant_lora_path',
                value: 'ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors',
              },
            ],
          });
        }

        const ss = process.sample?.sample_steps;
        const gs = process.sample?.guidance_scale;
        const samplingOn = !train?.disable_sampling;
        if (samplingOn && ((typeof ss === 'number' && ss > 12) || (typeof gs === 'number' && gs > 2))) {
          findings.push({
            id: 'krea2-turbo-sampling',
            level: 'warning',
            title: 'Validate Krea 2 Turbo at ~8 steps, guidance 1',
            detail:
              `Turbo is a few-step model, but previews are set to ${ss ?? '?'} steps / guidance ${gs ?? '?'}. Raw-style 25–30 steps and guidance 4 evaluate a different regime — they over-steer the distilled trajectory, exaggerate artifacts, and hide early drift. ` +
              `Sample at ~8 steps and guidance 1 so each checkpoint is judged at its deployment settings.`,
            setting: 'sample.sample_steps / sample.guidance_scale',
            current: `${ss ?? '?'} steps, guidance ${gs ?? '?'}`,
            recommended: '~8 steps, guidance 1',
            fix: [
              { path: 'config.process[0].sample.sample_steps', value: 8 },
              { path: 'config.process[0].sample.guidance_scale', value: 1 },
            ],
          });
        }
      }
    }
  }

  // ---- MiniMax H3 model-specific recipe (RunComfy / model-card guidance) ----
  // H3 is a CFG-distilled video+audio model with a strict temporal grid and
  // pre-quantized weights. The costly mistakes are: validating at CFG > 1 (the
  // distilled trajectory over-saturates), feeding clips off the 17n+5 grid (they
  // get silently trimmed), and re-quantizing weights that already ship quantized.
  {
    const a = arch.toLowerCase();
    if (a.includes('minimax_h3') || a.includes('minimax')) {
      const samplingOn = !train?.disable_sampling;
      const gs = process.sample?.guidance_scale;

      // Guidance MUST be 1.0 — the model is CFG-distilled.
      if (samplingOn && typeof gs === 'number' && gs !== 1) {
        findings.push({
          id: 'minimax_h3-guidance-one',
          level: 'warning',
          title: 'MiniMax H3 validates at Guidance 1.0 (CFG = 1)',
          detail:
            `Preview guidance_scale is ${gs}, but MiniMax H3 is CFG-distilled — its sampler runs without classifier-free guidance. ` +
            `Validating at 3.5/7.0 evaluates a regime the model never uses: the previews come out over-saturated and artifact-heavy and misrepresent the LoRA's true state, hiding real progress or drift.`,
          setting: 'sample.guidance_scale',
          current: String(gs),
          recommended: '1',
          fix: [{ path: 'config.process[0].sample.guidance_scale', value: 1 }],
        });
      }

      // Flow-matching schedule is required.
      const ns = (train?.noise_scheduler || '').toLowerCase();
      if (ns && ns !== 'flowmatch') {
        findings.push({
          id: 'minimax_h3-flowmatch',
          level: 'warning',
          title: 'MiniMax H3 needs the FlowMatch scheduler',
          detail:
            `noise_scheduler is "${train?.noise_scheduler}", but H3 is a flow-matching model and trains with the FlowMatch scheduler (which applies the model's own timestep/sigma shift). Other schedulers sample the wrong noise regime and the LoRA will barely learn.`,
          setting: 'train.noise_scheduler',
          current: String(train?.noise_scheduler ?? ''),
          recommended: 'flowmatch',
          fix: [{ path: 'config.process[0].train.noise_scheduler', value: 'flowmatch' }],
        });
      }

      // Weights ship pre-quantized (int8-ConvRot DiT + nvfp4 TE) — re-quantizing
      // is wasted work at best and can corrupt the packed scales.
      if (model?.quantize || model?.quantize_te) {
        findings.push({
          id: 'minimax_h3-no-requantize',
          level: 'warning',
          title: 'MiniMax H3 weights are already quantized',
          detail:
            `quantize=${!!model?.quantize} / quantize_te=${!!model?.quantize_te}, but the Comfy-Org H3 checkpoint already ships as int8-ConvRot (DiT) and nvfp4-AWQ (Qwen3-VL text encoder). Turning quantization on runs a redundant pass over pre-packed weights — leave it off and manage VRAM with Low VRAM mode and layer offloading instead.`,
          setting: 'model.quantize / model.quantize_te',
          current: `quantize=${!!model?.quantize}, quantize_te=${!!model?.quantize_te}`,
          recommended: 'both off',
          fix: [
            { path: 'config.process[0].model.quantize', value: false },
            { path: 'config.process[0].model.quantize_te', value: false },
          ],
        });
      }

      // Temporal grid: video clips must be 17n+5 (5, 22, 39, 56, …) or a single
      // still. Off-grid counts are trimmed DOWN at load, wasting decode and
      // training on fewer frames than configured. auto_frame_count snaps for you.
      const validFrame = (n: number) => n === 1 || (n >= 5 && (n - 5) % 17 === 0);
      const offGrid = datasets.filter(
        d => !d.auto_frame_count && typeof d.num_frames === 'number' && d.num_frames > 1 && !validFrame(d.num_frames),
      );
      if (offGrid.length > 0) {
        const examples = offGrid.map(d => d.num_frames).slice(0, 4).join(', ');
        findings.push({
          id: 'minimax_h3-frame-grid',
          level: 'warning',
          title: 'MiniMax H3 clips must be on the 17n+5 frame grid',
          detail:
            `${offGrid.length} dataset(s) set num_frames to ${examples}, which is not on H3's temporal grid (5, 22, 39, 56, 73, 90, 107 …). The VAE trims off-grid clips DOWN to the nearest valid count, so you decode and train on fewer frames than you asked for. Use a valid count (39 ≈ 1.63s @ 24fps is the character sweet spot), or turn on Auto Frame Count to snap automatically.`,
          setting: 'datasets[].num_frames',
          current: examples,
          recommended: '5, 22, 39, 56, … (or enable auto_frame_count)',
        });
      }

      // Rank baseline for character likeness. Higher ranks tend to absorb
      // lighting/background into the identity.
      if (isLora && typeof network?.linear === 'number' && network.linear > 16) {
        findings.push({
          id: 'minimax_h3-rank-baseline',
          level: 'info',
          title: 'MiniMax H3 character LoRAs favour Rank 16',
          detail:
            `Network rank is ${network.linear}. For H3 character likeness the RunComfy baseline is Linear Rank 16 / Alpha 16 (Alpha 8 also works). Jumping to 32+ early tends to bake lighting and background into the character rather than the face/body identity. Raise rank only if 16 underfits.`,
          setting: 'network.linear / network.linear_alpha',
          current: String(network.linear),
          recommended: '16 / 16',
        });
      }

      // Audio supervision reminder — only when it's actually on.
      const audioOn = datasets.some(d => d.do_audio);
      if (audioOn) {
        findings.push({
          id: 'minimax_h3-audio-on',
          level: 'info',
          title: 'Audio supervision is on',
          detail:
            `At least one dataset has Do Audio enabled, so H3 will jointly learn audio with appearance. That is correct for voice/lipsync, but it costs extra compute and needs clean, synchronized audio clips. For a pure physical-appearance character LoRA, turn Do Audio OFF so audio gradients don't distort the visual identity.`,
          setting: 'datasets[].do_audio',
          current: 'on',
          recommended: 'off for pure-appearance characters',
        });
      }
    }
  }

  // ---- Qwen-Image 2.1 model-specific recipe (RunComfy guide) ----------
  // One checkpoint covers text-to-image, reference editing and native RGBA. The
  // costly mistakes: leaving the shipped convrot8 quantization (another qtype
  // re-quantizes layer by layer), mixing RGBA with alpha_mask (the dataloader
  // silently drops the alpha, so transparency is never learned), mismatched
  // reference counts inside a batch, and sampling an edit LoRA with no reference
  // (that previews text-to-image, not the edit it was trained for).
  if (isQwenImage21(arch)) {
    const samplingOn = !train?.disable_sampling;
    const modelKwargs: any = (model as any)?.model_kwargs ?? {};
    const rgba = !!modelKwargs.rgba;
    const ds = datasets as any[];

    // Matched quantization: both switches on, both formats convrot8.
    const quantMatched =
      !!model?.quantize && !!model?.quantize_te && model?.qtype === 'convrot8' && model?.qtype_te === 'convrot8';
    if (!quantMatched) {
      findings.push({
        id: 'qwen21-matched-quant',
        level: 'warning',
        title: 'Keep Qwen-Image 2.1 on its shipped convrot8 quantization',
        detail:
          `quantize=${!!model?.quantize} (${model?.qtype ?? 'unset'}), quantize_te=${!!model?.quantize_te} (${model?.qtype_te ?? 'unset'}). ` +
          `The Comfy-Org Qwen-Image 2.1 checkpoint ships int8-convrot for both the transformer and the text encoder, and the tested defaults keep both switches on at convrot8 so those weights load unchanged. ` +
          `Another format re-quantizes layer by layer (slower to load, different numerics from the shipped weights), and turning quantization off departs from the tested recipe. Establish a working baseline on the matched settings first.`,
        setting: 'model.quantize / quantize_te / qtype / qtype_te',
        current: `quantize=${!!model?.quantize}/${model?.qtype ?? '-'}, quantize_te=${!!model?.quantize_te}/${model?.qtype_te ?? '-'}`,
        recommended: 'both on, convrot8',
        fix: [
          { path: 'config.process[0].model.quantize', value: true },
          { path: 'config.process[0].model.quantize_te', value: true },
          { path: 'config.process[0].model.qtype', value: 'convrot8' },
          { path: 'config.process[0].model.qtype_te', value: 'convrot8' },
        ],
      });
    }

    // Flow-matching scheduler, sampler and the `shift` timestep type.
    const schedFix: { path: string; value: unknown }[] = [];
    const schedBits: string[] = [];
    const lc = (v: unknown) => String(v ?? '').toLowerCase();
    if (lc(train?.noise_scheduler) !== 'flowmatch') {
      schedFix.push({ path: 'config.process[0].train.noise_scheduler', value: 'flowmatch' });
      schedBits.push(`noise_scheduler=${train?.noise_scheduler ?? 'unset'}`);
    }
    if (lc(process.sample?.sampler) !== 'flowmatch') {
      schedFix.push({ path: 'config.process[0].sample.sampler', value: 'flowmatch' });
      schedBits.push(`sampler=${process.sample?.sampler ?? 'unset'}`);
    }
    if (lc(train?.timestep_type) !== 'shift') {
      schedFix.push({ path: 'config.process[0].train.timestep_type', value: 'shift' });
      schedBits.push(`timestep_type=${train?.timestep_type ?? 'unset'}`);
    }
    if (schedFix.length > 0) {
      findings.push({
        id: 'qwen21-flowmatch-shift',
        level: 'warning',
        title: 'Qwen-Image 2.1 trains with FlowMatch and Shift timesteps',
        detail:
          `${schedBits.join(', ')}. Qwen-Image 2.1 is a flow-matching model: the recipe uses the FlowMatch scheduler and sampler with the Shift timestep type. ` +
          `Weighted/Sigmoid defaults carried over from other models bias learning toward the wrong noise regions, which shows up as weak concept pickup.`,
        setting: 'train.noise_scheduler / sample.sampler / train.timestep_type',
        current: schedBits.join(', '),
        recommended: 'flowmatch / flowmatch / shift',
        fix: schedFix,
      });
    }

    // Preview guidance: start at 3.0 and compare lower before blaming the LoRA.
    const gs = process.sample?.guidance_scale;
    if (samplingOn && typeof gs === 'number' && gs > 4.5) {
      findings.push({
        id: 'qwen21-guidance',
        level: 'info',
        title: 'Start Qwen-Image 2.1 previews near guidance 3',
        detail:
          `Preview guidance_scale is ${gs}. The training UI default for Qwen-Image 2.1 is 3.0; at ${gs} previews tend to look harsh or over-sharpened, which is easy to mistake for an over-trained LoRA. ` +
          `Keep guidance, prompts and references fixed while comparing checkpoints, and try a modestly lower value before adding steps.`,
        setting: 'sample.guidance_scale',
        current: String(gs),
        recommended: '3.0',
        fix: [{ path: 'config.process[0].sample.guidance_scale', value: 3 }],
      });
    }

    // RGBA vs alpha_mask: both read the alpha channel, with different meanings.
    // The dataloader gives alpha_mask precedence, so the transparency is dropped.
    const alphaMaskIdx = ds.map((d, i) => (d?.alpha_mask ? i : -1)).filter(i => i >= 0);
    if (rgba && alphaMaskIdx.length > 0) {
      findings.push({
        id: 'qwen21-rgba-alpha-mask',
        level: 'warning',
        title: 'RGBA and alpha_mask cannot be combined',
        detail:
          `Transparency (RGBA) is on, but ${alphaMaskIdx.length} dataset(s) also set alpha_mask. RGBA treats alpha as image content the model learns to generate; alpha_mask uses it to weight the loss instead. ` +
          `With both set, alpha_mask wins and the alpha is not learned as content, so the LoRA will not produce transparency. Turn alpha_mask off for those datasets (masked training is not a way to learn transparent output), or turn RGBA off.`,
        setting: 'model.model_kwargs.rgba / datasets[].alpha_mask',
        current: `rgba=true, alpha_mask on dataset(s) ${alphaMaskIdx.map(i => i + 1).join(', ')}`,
        recommended: 'one or the other',
      });
    }

    // RGBA reminders: only when it is on.
    if (rgba) {
      findings.push({
        id: 'qwen21-rgba-prompt',
        level: 'info',
        title: 'Transparency (RGBA) is on — state it in captions and previews',
        detail:
          `Alpha is kept for dataset images and references, encoded through all four VAE channels, and samples are saved as PNG. The checkbox preserves alpha; it does not remove backgrounds, so targets must carry real transparency (not a baked-in checkerboard). ` +
          `Use the official format in captions and sample prompts, for example: "This is an RGBA image with transparency. <subject>. The image has alpha channel and the background is transparent." ` +
          `Do not label opaque targets as transparent. Toggling RGBA re-caches latents, so decide before the run.`,
        setting: 'model.model_kwargs.rgba',
        current: 'on',
        recommended: 'real alpha targets + RGBA wording in captions',
      });
    }

    // Editing: reference streams must line up between datasets and previews.
    const refCount = (d: any) =>
      [d?.control_path_1, d?.control_path_2, d?.control_path_3].filter(p => p && String(p).trim() !== '').length;
    const counts = ds.map(refCount);
    const trainsEdit = counts.some(c => c > 0);
    if (new Set(counts).size > 1) {
      findings.push({
        id: 'qwen21-reference-counts',
        level: 'warning',
        title: 'Datasets use different numbers of reference images',
        detail:
          `Reference streams per dataset: ${counts.map((c, i) => `#${i + 1}=${c}`).join(', ')}. Training needs matching reference-token counts within a batch, so keep the number of references consistent across examples rather than mixing text-to-image data with paired edit data, or datasets with incomplete pairs. ` +
          `Each reference stream should carry a distinct input, not duplicated placeholders.`,
        setting: 'datasets[].control_path_1..3',
        current: counts.map(String).join(' / '),
        recommended: 'same count in every dataset',
      });
    }
    if (samplingOn) {
      const sampleItems = ((process.sample as any)?.samples ?? []) as any[];
      const hasRef = (s: any) =>
        [s?.ctrl_img, s?.ctrl_img_1, s?.ctrl_img_2, s?.ctrl_img_3].some(p => p && String(p).trim() !== '');
      const withoutRef = sampleItems.filter(s => !hasRef(s)).length;
      if (trainsEdit && sampleItems.length > 0 && withoutRef > 0) {
        findings.push({
          id: 'qwen21-edit-samples-need-reference',
          level: 'warning',
          title: 'Edit LoRA previews need held-out reference images',
          detail:
            `Datasets train with reference images, but ${withoutRef} of ${sampleItems.length} sample prompt(s) have none. An edit LoRA sampled without a reference is tested as plain text-to-image, not the edit it learned. ` +
            `Give each sample a held-out reference (one not in the training set) with the same roles and order as training. Sample references are configured separately from dataset references.`,
          setting: 'sample.samples[].ctrl_img_1..3',
          current: `${withoutRef} sample(s) without a reference`,
          recommended: 'held-out reference on every sample',
        });
      } else if (!trainsEdit && sampleItems.some(hasRef)) {
        findings.push({
          id: 'qwen21-samples-extra-reference',
          level: 'info',
          title: 'Previews use references the datasets do not',
          detail:
            `Some sample prompts include reference images, but no dataset has control paths, so this run trains text-to-image. Adding references only to previews does not change the training task. ` +
            `Leave sample references empty for text-to-image, or add paired controls to the datasets to train editing.`,
          setting: 'sample.samples[].ctrl_img_1..3 / datasets[].control_path_1..3',
          current: 'refs in samples, none in datasets',
          recommended: 'make them match',
        });
      }
    }
  }

  // ---- Krea2: conv layers aren't part of the recipe -------------------
  // Krea 2's SingleStreamDiT LoRA recipe trains linear layers only; the UI
  // disables the conv section for every krea2 variant. A raw config with a
  // positive network.conv builds Conv2d adapters (config_modules reads
  // conv=None as "off"), adding params outside the recipe. Setting conv and
  // conv_alpha to null disables them cleanly.
  if (isLora && arch.toLowerCase().includes('krea2') && typeof network?.conv === 'number' && network.conv > 0) {
    findings.push({
      id: 'krea2-conv-unused',
      level: 'warning',
      title: 'Conv layers are outside the Krea 2 recipe',
      detail:
        `network.conv is ${network.conv}, so the trainer builds Conv2d LoRA adapters. Krea 2's recipe trains linear layers only ` +
        `(the UI hides the conv section for krea2), so these are extra parameters trained off-recipe. Set conv / conv_alpha to null to disable them unless you specifically want conv adapters.`,
      setting: 'network.conv / network.conv_alpha',
      current: `conv=${network.conv}, conv_alpha=${network.conv_alpha ?? network.conv}`,
      recommended: 'null (linear only)',
      fix: [
        { path: 'config.process[0].network.conv', value: null },
        { path: 'config.process[0].network.conv_alpha', value: null },
      ],
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

    // Quantization, layer offloading and Low VRAM mode are NOT independent
    // yes/no knobs — they trade VRAM, speed and fidelity against each other, and
    // the "right" combination depends on what the user is optimising for.
    // Emitting them as separate findings produced advice that looked
    // self-contradictory (one card "quantize on", another "quantize off").
    // Instead, collapse the decision into ONE finding that offers mutually
    // exclusive, intent-labelled options; the user picks the profile that
    // matches their goal and gets a self-consistent set of settings.
    // Qwen-Image 2.1: the weights are pre-quantized, so the goal is not "which
    // quantization" (it stays matched at convrot8 in every profile) but how the
    // memory-saving switches and the text-embedding cache are set. archSize is the
    // resident footprint, so weights = transformer + text encoder with no scaling.
    if (isQwenImage21(arch) && vramGB > 0 && ramGB > 0) {
      const weights = bf16Weights;
      const fitsResident = weights + 4 < vramGB * 0.92; // weights + ~4GB activations
      const matched = [
        { path: 'config.process[0].model.quantize', value: true },
        { path: 'config.process[0].model.quantize_te', value: true },
        { path: 'config.process[0].model.qtype', value: 'convrot8' },
        { path: 'config.process[0].model.qtype_te', value: 'convrot8' },
      ];
      const resident = [
        { path: 'config.process[0].model.low_vram', value: false },
        { path: 'config.process[0].model.layer_offloading', value: false },
      ];
      const noFit = `The resident weights (~${weights.toFixed(0)} GB) plus activations may exceed your ${vramGB.toFixed(0)} GB VRAM — risk of OOM. Prefer Fail-safe.`;

      const options: FindingOption[] = [
        {
          id: 'qwen21-speed',
          profile: 'speed',
          label: 'Fastest throughput',
          detail:
            `Matched convrot8 weights resident on the GPU (Low VRAM and layer offloading off), text embeddings cached so the ~${size.teGB.toFixed(0)} GB text encoder is unloaded after caching. ` +
            `Nothing streams over PCIe, so this is the highest throughput. The guide suggests keeping Low VRAM on for a first run — switch to this once a first sample has rendered cleanly.`,
          recommended: fitsResident,
          note: fitsResident ? undefined : noFit,
          fix: [...matched, ...resident, { path: 'config.process[0].train.cache_text_embeddings', value: true }],
        },
        {
          id: 'qwen21-quality',
          profile: 'quality',
          label: 'Best training signal',
          detail:
            `Same matched convrot8 weights (they are the checkpoint's native format, so there is no higher-fidelity load to switch to), resident on the GPU, with text-embedding caching OFF. ` +
            `Captions are re-encoded each step, so caption dropout and token shuffle actually apply (cached embeddings skip them) — better prompt adherence at guidance, at the cost of keeping the text encoder resident and slower steps.`,
          recommended: false,
          note: fitsResident ? undefined : noFit,
          fix: [...matched, ...resident, { path: 'config.process[0].train.cache_text_embeddings', value: false }],
        },
        {
          id: 'qwen21-safe',
          profile: 'safe',
          label: 'Lowest VRAM',
          detail:
            `Matched convrot8 weights with Low VRAM (which also tiles the VAE decode for samples) and layer offloading both on, text embeddings cached. ` +
            `The most resistant to out-of-memory on ${vramGB.toFixed(0)} GB and the guide's recommended setup for an initial run — the slowest steps, and the fallback if either faster profile OOMs. Offloading is a capacity tool, not a speed-up.`,
          recommended: !fitsResident,
          fix: [
            ...matched,
            { path: 'config.process[0].model.low_vram', value: true },
            { path: 'config.process[0].model.layer_offloading', value: true },
            { path: 'config.process[0].train.cache_text_embeddings', value: true },
          ],
        },
      ];

      findings.push({
        id: 'hw-memory-strategy',
        level: 'info',
        title: 'Configuration goal: Speed, Quality, or Fail-safe',
        detail:
          `Qwen-Image 2.1 ships pre-quantized (~${weights.toFixed(0)} GB resident), so quantization stays matched at convrot8 in every profile; the goal changes Low VRAM, layer offloading and text-embedding caching on your ${vramGB.toFixed(0)} GB VRAM / ${ramGB.toFixed(0)} GB RAM. ` +
          `Pick the goal that matches this run; each tab shows exactly which of your current settings it would change.`,
        setting: 'model.low_vram / model.layer_offloading / train.cache_text_embeddings',
        options,
      });
    }

    if (large && vramGB > 0 && ramGB > 0 && !isQwenImage21(arch)) {
      const halfTransformer = size.transformerGB * 0.5; // ~qfloat8 resident footprint
      const vramBudget = vramGB * 0.92;
      const ramBudget = ramGB * 0.85;
      // Speed: quantized weights must fit in VRAM alongside ~a few GB of activations.
      const speedFits = halfTransformer + 4 < vramBudget;
      // Quality: full bf16 weights must fit in the RAM they're offloaded to.
      const qualityFits = bf16Weights < ramBudget;

      // Text embeddings can be cached (encode once, unload the text encoder) in
      // every profile: it frees VRAM AND removes per-step text-encoder work, and
      // the captions are static so there's no fidelity cost. It's the single
      // biggest speed lever, so each goal bundle includes it.
      const cacheTE = { path: 'config.process[0].train.cache_text_embeddings', value: true };

      // All three goals are always offered as tabs; a `note` marks one that may
      // not fit this machine (the UI warns and blocks its apply) rather than
      // hiding it, so the speed/quality/fail-safe comparison is always complete.
      const options: FindingOption[] = [
        {
          id: 'strategy-speed',
          profile: 'speed',
          label: 'Fastest throughput',
          detail:
            `${model?.qtype ?? 'qfloat8'} transformer resident in VRAM, layer offloading and Low VRAM off, text embeddings cached. ` +
            `The quantized weights (~${halfTransformer.toFixed(0)} GB) fit your ${vramGB.toFixed(0)} GB, so nothing streams over PCIe and the text encoder is unloaded after caching — the highest throughput. Minor quality cost from 8-bit weights.`,
          recommended: speedFits,
          note: speedFits ? undefined : `The 8-bit transformer (~${halfTransformer.toFixed(0)} GB) plus activations may exceed your ${vramGB.toFixed(0)} GB VRAM — risk of OOM. Prefer Fail-safe.`,
          fix: [
            { path: 'config.process[0].model.quantize', value: true },
            { path: 'config.process[0].model.layer_offloading', value: false },
            { path: 'config.process[0].model.low_vram', value: false },
            cacheTE,
          ],
        },
        {
          id: 'strategy-quality',
          profile: 'quality',
          label: 'Highest fidelity',
          detail:
            `Full-precision bf16 transformer (no quantization) parked in your ${ramGB.toFixed(0)} GB RAM and streamed to the GPU via layer offloading, text embeddings cached. ` +
            `${size.label}'s ~${bf16Weights.toFixed(0)} GB of weights fit that RAM, giving the best likeness — at the cost of some speed lost to RAM↔GPU transfer.`,
          recommended: false,
          note: qualityFits ? undefined : `${size.label}'s ~${bf16Weights.toFixed(0)} GB bf16 weights may not fit your ${ramGB.toFixed(0)} GB RAM — loading could be killed. Prefer Speed or Fail-safe.`,
          fix: [
            { path: 'config.process[0].model.quantize', value: false },
            { path: 'config.process[0].model.layer_offloading', value: true },
            { path: 'config.process[0].model.low_vram', value: false },
            cacheTE,
          ],
        },
        {
          id: 'strategy-safe',
          profile: 'safe',
          label: 'Lowest VRAM',
          detail:
            `8-bit transformer and text encoder, with layer offloading and Low VRAM mode both on, and text embeddings cached. ` +
            `The smallest VRAM footprint and the most resistant to out-of-memory crashes on ${vramGB.toFixed(0)} GB — the slowest steps, but the safe fallback if either faster profile OOMs.`,
          recommended: !speedFits,
          fix: [
            { path: 'config.process[0].model.quantize', value: true },
            { path: 'config.process[0].model.quantize_te', value: true },
            { path: 'config.process[0].model.layer_offloading', value: true },
            { path: 'config.process[0].model.low_vram', value: true },
            cacheTE,
          ],
        },
      ];

      findings.push({
        id: 'hw-memory-strategy',
        level: 'info',
        title: 'Configuration goal: Speed, Quality, or Fail-safe',
        detail:
          `Quantization, layer offloading, Low VRAM mode and text-embedding caching only make sense as a set — together they trade VRAM, speed and fidelity on ${size.label} ` +
          `(~${bf16Weights.toFixed(0)} GB at bf16) given your ${vramGB.toFixed(0)} GB VRAM / ${ramGB.toFixed(0)} GB RAM. ` +
          `Pick the goal that matches this run; each tab shows exactly which of your current settings it would change.`,
        setting: 'model.quantize / model.layer_offloading / model.low_vram / train.cache_text_embeddings',
        options,
      });
    }

    // Latent cache location. cache_latents_to_disk writes latents to disk to
    // save RAM; with the cache held in RAM instead, training skips the per-epoch
    // disk round-trip and data loading is faster. The latent cache is small
    // relative to the weights, so only suggest the switch when RAM clearly has
    // room for BOTH the (offloaded) weights and the cache, with slack.
    const diskCached = datasets
      .map((d, i) => ({ d, i }))
      .filter(({ d }) => d.cache_latents_to_disk);
    if (diskCached.length > 0 && ramGB > 0) {
      const ramBudget = ramGB * 0.85;
      // Coarse latent size: a 1-megapixel frame is ~0.5 MB at bf16 (16-channel,
      // 8x VAE latent). Scale by resolution and video frame count.
      const maxMPFrames = Math.max(
        ...diskCached.map(({ d }) => {
          const res = Math.max(1, ...(d.resolution ?? [1024]));
          const mp = (res * res) / (1024 * 1024);
          return mp * Math.max(1, d.num_frames ?? 1);
        }),
      );
      const estCacheGB =
        imageCount != null && imageCount > 0 ? (imageCount * maxMPFrames * 0.5) / 1024 : null;
      // Require headroom for weights + cache (+ slack). When the cache size is
      // unknown, demand generous slack rather than guess.
      const roomForCache =
        estCacheGB != null ? bf16Weights + estCacheGB + 4 < ramBudget : bf16Weights + 8 < ramBudget;
      if (roomForCache) {
        const sizeClause =
          estCacheGB != null
            ? `The latent cache is small (~${estCacheGB < 1 ? '<1' : estCacheGB.toFixed(1)} GB for ${imageCount} image(s)), `
            : 'The latent cache is typically small, ';
        findings.push({
          id: 'hw-latent-cache-ram',
          level: 'info',
          title: 'RAM headroom — cache latents in RAM for faster loading',
          detail:
            `${diskCached.length} dataset(s) set cache_latents_to_disk, which writes latents to disk to save memory. ` +
            `${sizeClause}and this machine has ~${ramGB.toFixed(0)} GB RAM, comfortably more than ${size.label}'s ~${bf16Weights.toFixed(0)} GB of weights. ` +
            `Turning disk caching off keeps latents in RAM and skips the per-epoch disk round-trip, speeding up data loading. ` +
            `Leave it on for very large datasets or when running other memory-heavy apps alongside training.`,
          setting: 'datasets[].cache_latents_to_disk',
          current: 'true',
          recommended: 'false (cache in RAM)',
          fix: diskCached.map(({ i }) => ({
            path: `config.process[0].datasets[${i}].cache_latents_to_disk`,
            value: false,
          })),
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
