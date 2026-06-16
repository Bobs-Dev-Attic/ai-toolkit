# Proposed Training Optimizations

Status: **DESIGN, NOT IMPLEMENTED.**

Each item below is genuinely missing from upstream (verified via grep on
`main` at the time this fork was created) and would either improve
throughput, quality, or both. All are behind opt-in config flags so the
existing training path stays bit-identical when the flag is off.

## What's already in upstream (don't duplicate)

For reference — these were candidates I considered but the toolkit
**already has**:

- `torch.compile` on the backbone (`compile: true` in model config, see
  recent upstream commit "More work on compiling models")
- Min-SNR and SNR-gamma loss weighting (`min_snr_gamma` /
  `snr_gamma` in `train` config; logic in
  `jobs/process/BaseSDTrainProcess.py` around `apply_snr_weight()`)
- AdEMAMix-8bit, Lion-8bit, Prodigy / Prodigy-8bit, dadaptation
  optimizers
- Flow-matching + linear-timesteps + mean-flow schedulers
- torchao uint3 quantization with accuracy-recovery adapters (Qwen-Image)
- qfloat8 text encoders, NF4 base via bitsandbytes
- EMA, gradient checkpointing, latent + text-embedding caching
- Multi-resolution bucketed batching
- DoRA (`toolkit/network/DoRA.py`), LoRAFormer, iLoRA variants

That's a lot. The proposals below fill specific gaps.

---

## Tier 1 — high ROI, low risk

### 1. LoRA+ (separate learning rates for A and B)

**What:** Hao et al. 2024 — A is updated with the base LR, B with a much
larger LR (commonly 16×). Convergence speedup of ~30–50% on many LoRA
tasks for essentially zero code complexity.

**Why it's safe to add:** Pure split of the optimizer's parameter groups.
When the multiplier is 1.0 the behavior is identical to current. No loss
or scheduler changes.

**Where:**

1. `toolkit/config_modules.py` → add `loraplus_lr_ratio: float | None`
   to the `TrainConfig` model (default `None`).
2. Find where the trainable LoRA params are gathered for the optimizer.
   `toolkit/network/lora.py` (or `LoRAModule.prepare_optimizer_params`
   if it exists) is the most likely site. Look for any method that
   returns `params` to `optimizer.py`.
3. When `loraplus_lr_ratio` is set:
   ```python
   # split by parameter name: ".lora_B" vs ".lora_A" / ".lora_down" vs ".lora_up"
   a_params, b_params = [], []
   for name, p in lora_module.named_parameters():
       if not p.requires_grad: continue
       if 'lora_B' in name or 'lora_up' in name:
           b_params.append(p)
       else:
           a_params.append(p)
   return [
       {'params': a_params, 'lr': base_lr},
       {'params': b_params, 'lr': base_lr * loraplus_lr_ratio},
   ]
   ```
4. Document in `config/examples/*.yaml`:
   ```yaml
   train:
     loraplus_lr_ratio: 16   # try 8-32; null/omit to disable
   ```

**Validation plan:**
1. Train same config with `loraplus_lr_ratio: null` and `: 16` side by
   side on the same dataset.
2. Compare loss curves and sample-image quality at matched-step.
3. Expected: faster convergence; quality comparable or better.

**Risk:** Low. Reversible by setting the flag to null.

**Effort:** 1–2 hours including testing.

---

### 2. PiSSA initialization for LoRA

**What:** *Meng et al. 2024* — Initialize the A matrix from the top-K
right singular vectors of the base weight matrix instead of Gaussian-zero.
The "important" subspace of the base weight is what LoRA adapts; PiSSA
starts in that subspace.

**Why it's safe:** Only changes the **initial values** of A and B (and
subtracts the residual from the base weight to keep the network output
identical at step 0). Training proceeds with the same optimizer/loss.

**Where:** In `toolkit/network/lora.py` (or wherever `LoRAModule.__init__`
constructs the down/up matrices). Add an `init_method: "default" | "pissa"`
option. PiSSA pseudo-code:

```python
# W is the base weight; rank is LoRA rank
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
A_init = (Vh[:rank] * S[:rank].sqrt().unsqueeze(-1))  # rank × in
B_init = (U[:, :rank] * S[:rank].sqrt().unsqueeze(0)) # out × rank
# Subtract the residual from the base weight so output stays identical:
W_residual = W - B_init @ A_init
self.base_layer.weight.data = W_residual
self.lora_A.weight.data = A_init
self.lora_B.weight.data = B_init
```

**Validation plan:** Same as LoRA+ — A/B against current default init.

**Risk:** Slightly higher than LoRA+ because the base weight is modified
(then "fixed" by subtracting the residual). Numerical-precision drift
possible at FP16. Save the original base weight before modification so a
restore is trivial.

**Effort:** ~3–4 hours including SVD memory-management for big matrices.

---

### 3. Liger Kernels for the text encoders

**What:** Meta's fused Triton kernels for RMSNorm / SwiGLU / RoPE
(`pip install liger-kernel`). Drop-in patch for Qwen-family models;
typically 20%+ throughput on those layers.

**Why it's safe:** Numerically equivalent (the kernels fuse the same
ops). Only patches text-encoder modules during training; sampling /
inference unaffected.

**Where:**

1. `toolkit/models/loaders/umt5.py` or wherever Qwen-Image's text
   encoder is loaded.
2. After loading:
   ```python
   try:
       from liger_kernel.transformers import apply_liger_kernel_to_qwen2
       apply_liger_kernel_to_qwen2(model=text_encoder)
   except ImportError:
       pass  # liger optional
   ```
3. Surface via `train.use_liger_kernels: true` in config.

**Validation plan:** Time `train_unet_step` over 100 steps with and
without liger active. Should see TE-side speedup proportional to TE
share of step time.

**Risk:** Low. Liger has been around since mid-2024 and is widely used.
Only affects models with Llama-family TEs (Qwen). No-op for FLUX's T5.

**Effort:** 1 hour for Qwen-Image; can extend to other TEs incrementally.

---

## Tier 2 — quality wins, moderate effort

### 4. REPA loss (representation alignment)

**What:** *Yu et al. 2024* — an auxiliary loss aligning intermediate DiT
features to features from a frozen DINOv2 encoder. Substantial
fine-tuning fidelity improvement (paper shows ~5× faster convergence to
matched FID). Small VRAM cost (one DINOv2 forward per batch) and one
extra MLP head.

**Where:**

1. Add config keys: `train.repa.enabled: bool`, `train.repa.weight: 0.5`,
   `train.repa.dinov2_size: "small"|"base"|"large"`,
   `train.repa.dit_layer: int` (which DiT block to extract features from,
   usually mid-network).
2. In `BaseSDTrainProcess.py` setup, lazily load
   `facebook/dinov2-small` (or chosen size) and a `Linear(dit_dim,
   dinov2_dim)` projection head.
3. In the training step, capture the activation from `dit_layer` via a
   forward hook, project it, and add
   `mse(projection, dinov2(input_image))` to the loss with the
   configured weight.
4. Persist the projection head with the LoRA checkpoint (small, ~1 MB).

**Validation plan:**
1. Train two LoRAs side-by-side: one with REPA, one without.
2. Compare validation-set FID (or CLIP-score) at matched steps and
   final-step image quality.
3. Watch VRAM: should add ~1–2 GB at 1024×1024.

**Risk:** Medium. The projection head is per-checkpoint, so a config
mismatch when loading old checkpoints needs handling. DINOv2 must
match the training resolution stride; document that REPA requires
multiples-of-14 image sizes (DINOv2 patch size).

**Effort:** 1–2 days including validation.

---

### 5. Native FP8 mixed-precision training on Blackwell

**What:** `torchao.float8` training APIs swap matmuls in `nn.Linear`
layers to FP8 (E4M3 for fwd, E5M2 for bwd) with automatic scaling.
On Blackwell tensor cores this can deliver up to **2× throughput** on
attention/MLP and lower memory pressure. Already a dependency
(`torchao==0.10.0`).

**Where:**

1. After model load, before the trainer wraps things:
   ```python
   from torchao.float8 import convert_to_float8_training, Float8LinearConfig
   if train_config.fp8_training:
       config = Float8LinearConfig.from_recipe_name("rowwise")  # safer than tensorwise
       convert_to_float8_training(model, config=config)
   ```
2. Skip already-quantized models — incompatible with `quantize: true` /
   torchao uint3.
3. Tightly scope: only swap the DiT transformer blocks; **not** the
   norm / embedding / VAE / TE.

**Validation plan:**
1. Train both with and without FP8 for 500 steps; loss curves should
   align within noise.
2. Profile step time; expect 30–60% speedup on transformer-heavy
   workloads, less on data-bound runs.
3. Sample images at same seed/step — should be visually
   indistinguishable.

**Risk:** Medium-high. FP8 training stability still has edge cases.
Default off; mark experimental.

**Effort:** 2–3 days including stability tuning.

---

## Tier 3 — UX / observability

### 6. Real-time loss + sample preview in the UI

**What:** The toolkit already writes TensorBoard event files and sample
images during training (`output/<job_name>/samples/`). The UI doesn't
surface these. Add:

1. **Loss chart** on the job-detail page, parsing the most recent TF
   events file from `output/<job>/runs/`. Refresh every 5–10s.
2. **Sample grid** showing the most recent N sample images, click to
   enlarge.

**Where:**

1. Job detail page (find existing in `ui/src/app/jobs/`).
2. New API route `/api/jobs/[jobID]/metrics` reading the event file and
   returning the scalar history (use `node-tfevents` or just a tiny
   protobuf parser — events have stable schema).
3. New API route `/api/jobs/[jobID]/samples` listing recent sample
   images and serving via existing `/api/img/`.

**Risk:** None — read-only additive UI.

**Effort:** 1–2 days.

---

### 7. Validation set + best-checkpoint selection

**What:** Currently saves every `N` steps and keeps the last `K`. The
*best* checkpoint isn't necessarily the most recent one. Add:

1. `datasets[*].validation_split: 0.1` (or path to a separate validation
   folder).
2. Periodic forward pass on the validation set; track validation loss.
3. Persist the best-loss checkpoint as `best.safetensors` in addition to
   step-numbered saves.

**Where:** `jobs/process/BaseSDTrainProcess.py` save logic and dataset
loader.

**Risk:** Low. Pure addition; existing fixed-step saves still happen.

**Effort:** ~1 day.

---

## Recommended order

If picking up this work later, do them in this order — each builds on
infrastructure you'll already have:

1. **LoRA+** — 1 line in the config, biggest convergence win for cost
2. **PiSSA init** — same area of code as LoRA+
3. **UI loss/sample preview** — improves the *next* item by giving you
   a way to visually compare runs
4. **Validation set + best checkpoint** — once you can see the curve,
   you'll want to pick the best step
5. **Liger Kernels (Qwen)** — throughput, Qwen-specific
6. **REPA loss** — quality; depends on having validation infrastructure
7. **FP8 training** — last because it's the most experimental

## What I considered but rejected

- **Schedule-Free AdamW**: promising but newer than I'd want for a default;
  user already has Prodigy which serves a similar "no LR schedule"
  niche.
- **CAME optimizer**: marginal benefit over AdEMAMix in our setting.
- **Adam-mini**: memory savings overlap with what 8-bit Adam already
  provides.
- **Q-GaLore**: a full-finetune-style alternative; the toolkit is
  LoRA-first, so this is a different product, not an optimization.
- **DMD / Hyper / PCM distillation**: a separate training mode entirely,
  not an incremental optimization.
