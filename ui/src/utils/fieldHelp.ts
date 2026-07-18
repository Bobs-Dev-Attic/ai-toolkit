/**
 * Short hover tooltip text for form fields on the New / Edit Training Job page.
 *
 * Keyed by a normalized form of the label (lowercased, alphanumerics + spaces).
 * If you add a new field whose label exactly matches an entry here, the
 * tooltip will appear automatically — no per-input wiring required.
 *
 * Keep entries concise (one or two short sentences). The CircleHelp icon
 * still opens the full ConfigDoc modal for fields that have one in src/docs.tsx.
 */

const RAW: Record<string, string> = {
  // ───── Job card ─────
  'training name':
    'Unique name for this run. Used as the folder name under your training folder and as the saved model filename. Alphanumerics, underscores, and dashes only — no spaces.',
  'gpu id':
    'GPU index to run training on. Only one GPU per job from the UI; start multiple jobs in parallel for multi-GPU. On Mac this auto-routes to MPS.',
  'trigger word':
    'Optional. A token that activates your concept at inference. Auto-prepended to captions that don\'t already contain it. Use [trigger] inside a caption to control where it appears.',

  // ───── Model card ─────
  'model architecture':
    'Which base model family this LoRA targets. Changing this swaps in arch-specific defaults and may reset architecture-dependent fields.',
  'model path':
    'HuggingFace repo id (e.g. black-forest-labs/FLUX.1-dev) or an absolute local path. Gated models require a HuggingFace token in Settings.',
  'name or path':
    'HuggingFace repo id or absolute local path to the base model.',
  'name_or_path':
    'HuggingFace repo id or absolute local path to the base model.',

  // ───── Quantization ─────
  'quantization': 'Reduce model weight precision to fit in less VRAM. Stronger quantization saves memory but can soften fine details.',
  'quantize compile': 'Combines the Quantization controls (which precision to load the model in) with the Compile controls (whether to run torch.compile on the model for faster steps).',
  'transformer':
    'Quantization precision for the diffusion transformer weights — the biggest single VRAM saving. qfloat8 is the standard balanced choice; uint3/uint4 + an Accuracy Recovery Adapter goes much smaller at the cost of some speed and quality.',
  'transformer quantization':
    'Quantize the diffusion transformer weights. The biggest single VRAM saving for most models. Pick a heavier dtype (qfloat8) when you have headroom.',
  'text encoder':
    'Quantization precision for the text encoder. Smaller than the transformer but still worth quantizing when memory is tight; qfloat8 is almost free in quality. Use bf16 to leave it un-quantized.',
  'text encoder quantization':
    'Quantize the text encoder weights. Smaller than the transformer but worth doing when memory is tight.',
  'compile options':
    'torch.compile turns the model into a fused, optimized graph at startup. The first step is slow (compilation pass), subsequent steps are noticeably faster. Disable if you hit a compile error.',
  'compile model':
    'Enable torch.compile on the transformer. Faster per-step training after the one-time compile, at the cost of a ~1-2 minute warmup on job start.',
  'low vram':
    'Aggressive memory-saving path: weights stream to GPU on demand. Slower per step, but lets bigger models fit.',
  'low vram mode':
    'Aggressive memory-saving path: weights stream to GPU on demand. Slower per step, but lets bigger models fit.',
  'layer offloading':
    'Move a fraction of model layers to CPU RAM between forward passes. Trades training speed for VRAM headroom.',
  'transformer offload':
    'Fraction (0.0–1.0) of transformer layers offloaded to CPU. Higher = less VRAM, slower steps. Start at 0.5 if 24 GB is tight.',
  'text encoder offload':
    'Fraction (0.0–1.0) of text encoder layers offloaded to CPU. Usually safe at 1.0 since the text encoder is only used during caching.',

  // ───── Multistage ─────
  'multistage': 'Train multiple training stages back-to-back with different settings. Common for Wan 2.2 high/low-noise expert stages.',
  'switch boundary every':
    'How many steps between stage switches when training a multistage model. Smaller values cycle stages more often.',

  // ───── Target / Network ─────
  'target type':
    'What kind of weights you are training. lora trains a low-rank adapter (typical); lokr/loha are alternative low-rank decompositions; full retrains the entire model (huge VRAM, rarely used).',
  'data type':
    'Compute precision for training. bfloat16 is the modern default (numerically stable + memory-friendly). float16 is faster on older cards but can NaN on some setups. float32 is precise but uses 2x the VRAM.',
  'lora weight':
    'Multiplier applied to the LoRA layers during training. 1.0 means the LoRA is fully active; lower values dampen its influence so the base model dominates more.',
  'caption dropout rate':
    'Probability per step that the caption is replaced with an empty string. Forces the model to learn from images alone, improving robustness and preventing the LoRA from over-relying on trigger words. 0.05–0.1 is a healthy default; 0 disables.',
  'network type':
    'lora is the standard rank-adapter. lokr / loha are alternative low-rank decompositions that can be more parameter-efficient.',
  'linear rank':
    'Rank of the low-rank decomposition. Higher rank = more capacity to capture detail and more VRAM. 16–32 is a strong default for character/style LoRAs.',
  'linear alpha':
    'Scaling factor for the LoRA update at inference. A common convention is alpha = rank; lower alpha = subtler effect at the trained scale.',
  'network rank':
    'Rank of the LoRA. Higher rank = more capacity for detail at the cost of size and VRAM. 16 is a safe default.',
  'network alpha':
    'Alpha = scale of the LoRA. Typically set equal to rank; lower values dampen the trained effect.',
  'network multiplier':
    'Multiplier applied to the LoRA at inference time (a.k.a. LoRA scale or strength). 1.0 is trained strength; 0 disables the LoRA.',
  'lora scale':
    'Multiplier applied to the LoRA at inference (network_multiplier). 1.0 is trained strength; <1 softens the effect; >1 over-applies and often produces artifacts.',
  'dropout': 'Random drop fraction applied to the LoRA update during training. Light regularization (0.05–0.1) helps prevent overfitting.',
  'network dropout': 'Random drop fraction applied to the LoRA update during training. Light regularization (0.05–0.1) helps prevent overfitting.',

  // ───── Training card ─────
  'batch size':
    'Number of samples processed per forward pass. Bigger batches stabilize gradients but use more VRAM. Keep at 1 and use gradient accumulation if memory is tight.',
  'gradient accumulation':
    'Accumulate gradients across N forward passes before updating weights. Effective batch size = batch_size × gradient_accumulation. Use to simulate a bigger batch without using more VRAM.',
  'gradient checkpointing':
    'Recompute activations during backprop instead of storing them. Large VRAM savings at the cost of ~25% slower steps. Almost always on for LoRA training.',
  'steps': 'Total optimizer updates. 1500–4000 is typical for a character/style LoRA depending on dataset size.',
  'total steps': 'Total optimizer updates over the run.',
  'learning rate':
    'Step size of the optimizer. Lower = slower, more stable; higher = faster, riskier. 1e-4 is a strong default for AdamW8bit LoRA training.',
  'lr': 'Step size of the optimizer. Lower is slower & more stable; higher is faster & riskier. Try 1e-4 for AdamW LoRA.',
  'optimizer':
    'Algorithm that turns gradients into weight updates. adamw8bit is the standard memory-friendly choice; prodigy auto-tunes the LR.',
  'lr scheduler':
    'How the learning rate changes over training. constant is robust; cosine gently decays; linear decays linearly.',
  'noise scheduler':
    'How noise is added during training. flowmatch matches modern flow-matching models; ddpm is classic diffusion.',
  'timestep bias':
    'Skew which timesteps get sampled more often during training. Positive values bias toward higher-noise (harder, structural) timesteps; negative toward lower-noise (details). 0 = uniform.',
  'loss type':
    'Loss function used to compare the model output against the noise target. mse is the standard mean-squared error; huber is more robust to outliers; flow_match is required for flow-matching models like FLUX.',
  'ema': 'Exponential moving average of the network weights. Often produces smoother, slightly higher-quality saved checkpoints. Adds a small VRAM cost (copy of the LoRA weights) but no per-step slowdown.',
  'ema exponential moving average':
    'Exponential moving average of the network weights. Often produces smoother, slightly higher-quality saved checkpoints. Adds a small VRAM cost (copy of the LoRA weights) but no per-step slowdown.',
  'ema rate': 'How quickly the EMA forgets old weights. 0.99 is typical; closer to 1 = slower decay = more stable but less responsive.',
  'ema decay': 'How quickly the EMA forgets old weights. 0.99 is typical; closer to 1 = slower decay = more stable but less responsive.',
  'use ema': 'Track an exponential moving average of the LoRA weights and save it alongside the regular checkpoint. Often produces slightly smoother results.',
  'unload text encoder':
    'Drop the text encoder from VRAM after caching text embeddings. Frees several GB but only safe when text embeddings are pre-cached.',
  'cache text embeddings':
    'Pre-compute and cache caption embeddings to disk. One-time cost; lets you unload the text encoder during training.',
  'diff output preservation':
    'Penalize divergence from the base model output on a held-out prompt to preserve general knowledge. Helps avoid catastrophic forgetting.',
  'blank prompt preservation':
    'Penalize changes to outputs for empty/blank prompts. Keeps the base model\'s general behavior intact while training the concept.',

  // ───── Save card ─────
  'save every':
    'Steps between intermediate checkpoint saves. 250–500 is common. Lower values use more disk but give finer recovery points.',
  'max step saves to keep':
    'Rolling window of saved checkpoints — older intermediate saves are pruned. Set high if disk is plentiful.',
  'save dtype':
    'Precision used when writing the saved LoRA. float16 / bfloat16 are standard and cut file size in half versus float32.',

  // ───── Sample card ─────
  'sample every':
    'Steps between sample generations. Sampling is slow — every 250–500 steps is a reasonable cadence to monitor progress.',
  'sampler':
    'Sampling algorithm used to generate sample images. flowmatch for flow-matching models; euler for diffusion baselines.',
  'sample steps':
    'Denoising steps per sample image. 20–30 for fast sampling; 40+ for higher-fidelity previews.',
  'sample width':
    'Width in pixels for sample images. Pick a multiple of the model\'s patch size (usually 64). Bigger = slower + more VRAM.',
  'sample height':
    'Height in pixels for sample images. Pick a multiple of the model\'s patch size (usually 64).',
  'width':
    'Width override for this sample, in pixels. Leave blank to inherit the job-level width.',
  'height':
    'Height override for this sample, in pixels. Leave blank to inherit the job-level height.',
  'guidance scale':
    'Classifier-free guidance scale. Higher = closer to prompt but can over-saturate; ~3.5 for FLUX, 4–7 for SDXL.',
  'cfg':
    'Classifier-free guidance scale. Higher = stricter prompt adherence; lower = more creative.',
  'seed':
    'Random seed for sampling. Same seed + prompt + model = identical sample. Leave blank to inherit the job seed (or job\'s walk-seed +1 per prompt).',
  'walk seed':
    'Increment the seed by 1 for each prompt instead of reusing the same seed. Useful for getting variety across the sample grid.',
  'skip first sample':
    'Don\'t generate a sample at step 0. Saves time when you only care about later checkpoints.',
  'force first sample':
    'Generate a sample at step 0 regardless of the skip setting. Useful as a baseline comparison.',
  'prompt':
    'Caption used to generate the sample image. Include your trigger word here if you want to see how the trained concept renders.',
  'sample prompts':
    'Prompts generated at each sample interval. Mix in/out the trigger word to gauge concept strength vs. drift.',

  // ───── Dataset card ─────
  'target dataset':
    'Pick a folder from your datasets directory. Images and matching .txt caption files in that folder become the training set.',
  'dataset path':
    'Absolute path to a folder containing images + matching .txt caption files. Use a folder from /datasets or any local path.',
  'caption ext':
    'File extension expected for caption sidecars. ".txt" is standard.',
  'caption extension': 'File extension expected for caption sidecars. ".txt" is standard.',
  'control path':
    'Folder of control images (one per matching base image). For ControlNet / IP-Adapter / Kontext-style training.',
  'control paths': 'Folder(s) of control images aligned by filename to the base images.',
  'num repeats':
    'How many times each image is sampled per epoch. Use to up-weight smaller datasets when mixing multiple datasets together.',
  'cache latents to disk':
    'Pre-compute VAE latents and reuse them every epoch. Huge speed win after the first epoch; uses extra disk.',
  'resolution':
    'Training resolution(s). Mixing multiple resolutions ([512, 768, 1024]) gives the LoRA robustness across aspect ratios at the cost of slower epochs.',
  'flip horizontal':
    'Random horizontal flip augmentation. Disable for asymmetric subjects (text, faces with specific features, branded gear).',
  'flip vertical':
    'Random vertical flip augmentation. Rarely useful — most subjects have a clear "up".',
  'flip x': 'Random horizontal flip augmentation. Disable for asymmetric subjects.',
  'flip y': 'Random vertical flip augmentation. Rarely useful.',
  'do i2v':
    'Image-to-video training mode. The first frame of each clip is supplied as a conditioning image.',
  'num frames':
    'Number of frames per video clip used in training. More frames = better temporal coherence but much more VRAM.',
  'auto frame count':
    'Pick the frame count automatically from each clip\'s length, clipped to the model\'s max. Useful when clips vary.',
  'frame rate':
    'Frames per second for generated samples. Affects how motion is sampled — keep it close to your source footage FPS.',
  'frames':
    'Number of frames per sample. More frames cost VRAM and time proportionally.',
  'do audio': 'Train an audio model branch. Only meaningful for audio-capable architectures.',
  'audio normalize': 'Loudness-normalize audio clips before training.',
  'audio preserve pitch': 'Preserve pitch when resampling/normalizing audio.',

  // ───── Advanced (collapsed) ─────
  'unet trainable': 'Train the diffusion transformer weights directly (full fine-tune). Almost never used with LoRA training.',
  'text encoder trainable': 'Train the text encoder weights. Rarely used; can hurt prompt understanding if overtrained.',
  'cache latents': 'Cache VAE latents in memory. Speeds up epochs at the cost of RAM.',
  'cache text embeddings to disk': 'Cache caption embeddings to disk so the text encoder can be unloaded between epochs.',
  'first sample': 'Sample once at step 0 to capture a baseline before any training has happened.',
  'random crop': 'Random crop augmentation instead of center crop. Adds variety; mildly hurts framing fidelity.',
  'token shuffling': 'Shuffle the order of tokens in captions during training. Helps the model not over-rely on token order.',
  'shuffle tokens': 'Shuffle the order of tokens in captions during training. Helps the model not over-rely on token order.',
  'noise offset': 'Add a small bias to the noise level. Helps with deeper blacks / brighter whites. Try 0.05–0.1.',
  'min snr gamma': 'Min-SNR-γ loss reweighting. Helps stabilize training. 5 is the canonical value.',
  'snr gamma': 'Min-SNR-γ loss reweighting. Helps stabilize training. 5 is the canonical value.',
  'timestep type': 'Distribution used to sample timesteps. flowmatch_sigmoid for FLUX; uniform for classic diffusion.',
};

const KEY_SYNONYMS: Record<string, string> = {
  // docKey → label normalization for known fields without an exact label match
  'config.name': 'training name',
  'gpuids': 'gpu id',
  'config.process[0].trigger_word': 'trigger word',
  'config.process[0].model.name_or_path': 'name or path',
  'datasets.flip': 'flip horizontal',
  'datasets.num_frames': 'num frames',
  'datasets.do_i2v': 'do i2v',
  'datasets.do_audio': 'do audio',
  'datasets.audio_normalize': 'audio normalize',
  'datasets.audio_preserve_pitch': 'audio preserve pitch',
  'datasets.auto_frame_count': 'auto frame count',
  'datasets.control_path': 'control path',
  'datasets.multi_control_paths': 'control paths',
  'dataset.num_repeats': 'num repeats',
  'model.layer_offloading': 'layer offloading',
  'model.multistage': 'multistage',
  'train.unload_text_encoder': 'unload text encoder',
  'train.cache_text_embeddings': 'cache text embeddings',
  'train.switch_boundary_every': 'switch boundary every',
  'train.force_first_sample': 'force first sample',
  'train.diff_output_preservation': 'diff output preservation',
  'train.blank_prompt_preservation': 'blank prompt preservation',
  'train.audio_loss_multiplier': 'audio loss multiplier',
};

function normalize(s: unknown): string {
  if (typeof s !== 'string') return '';
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * Look up a tooltip for a given label and optional doc key. Returns `null`
 * when neither matches, so the caller can fall back to nothing. The label
 * type is widened to `unknown` because some inputs pass JSX/numbers.
 */
export function getFieldTooltip(label?: unknown, docKey?: string | null): string | null {
  if (typeof label === 'string' && label.length > 0) {
    const direct = RAW[normalize(label)];
    if (direct) return direct;
  }
  if (typeof docKey === 'string' && docKey.length > 0) {
    const viaKey = KEY_SYNONYMS[docKey];
    if (viaKey && RAW[viaKey]) return RAW[viaKey];
    // Try a tail match (last two dotted segments) — covers many "model.foo" / "train.foo" keys.
    const parts = docKey.split('.');
    if (parts.length >= 2) {
      const tail = parts.slice(-2).join('.');
      if (KEY_SYNONYMS[tail] && RAW[KEY_SYNONYMS[tail]]) return RAW[KEY_SYNONYMS[tail]];
    }
  }
  return null;
}
