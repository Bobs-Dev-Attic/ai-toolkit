import os
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager

from diffusers import (
    Krea2Pipeline,
    Krea2Transformer2DModel,
    AutoencoderKLQwenImage,
)
from transformers import (
    Qwen3VLModel,
    Qwen2Tokenizer,
)

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

# Matches krea/Krea-2-Raw scheduler/scheduler_config.json. Note max_image_seq_len
# is 6400 here (Qwen-Image uses 8192) and shift_terminal is null, so dynamic
# shifting is driven entirely by calculate_shift on the image sequence length.
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 6400,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Krea 2 packs latents into patch_size-square patches before the transformer.
# The pipeline stores this on the instance rather than in the transformer config.
PATCH_SIZE = 2


class Krea2Model(BaseModel):
    arch = "krea2"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["Krea2Transformer2DModel"]

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        # vae_scale_factor (8) * patch_size (2). Krea2Pipeline.check_inputs
        # rejects any height/width not divisible by this.
        return 8 * PATCH_SIZE

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Krea 2 model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path

        if base_model_path.endswith(".safetensors"):
            # a single file has no tokenizer/vae/te alongside it, pull those from the hub
            base_model_path = "krea/Krea-2-Raw"

        self.print_and_status_update("Loading transformer")

        if model_path.endswith(".safetensors"):
            transformer = Krea2Transformer2DModel.from_single_file(
                model_path,
                config="krea/Krea-2-Raw",
                subfolder="transformer",
                torch_dtype=dtype,
            )
            transformer.to(dtype)
        else:
            transformer_path = model_path
            transformer_subfolder = "transformer"
            if os.path.exists(transformer_path):
                transformer_subfolder = None
                transformer_path = os.path.join(transformer_path, "transformer")
                # if the text encoder is here too, this is a full checkpoint, use it as the base
                te_folder_path = os.path.join(model_path, "text_encoder")
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = Krea2Transformer2DModel.from_pretrained(
                transformer_path, subfolder=transformer_subfolder, torch_dtype=dtype
            )

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        self.print_and_status_update("Loading Text Encoder")
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer"
        )
        text_encoder = Qwen3VLModel.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
        )

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKLQwenImage.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype
        )

        self.noise_scheduler = Krea2Model.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: Krea2Pipeline = Krea2Pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
        )
        # for quantization, it works best to attach these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        if not self.low_vram:
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = Krea2Model.get_train_scheduler()

        pipeline: Krea2Pipeline = Krea2Pipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: Krea2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)

        if gen_config.ctrl_img is not None:
            raise NotImplementedError(
                "Control image generation is not supported in Krea 2 model"
            )

        flush_between_steps = self.model_config.low_vram

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if flush_between_steps:
                flush()
            return {"latents": callback_kwargs["latents"]}

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            prompt_embeds_mask=conditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.bool
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds,
            negative_prompt_embeds_mask=unconditional_embeds.attention_mask.to(
                self.device_torch, dtype=torch.bool
            ),
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)
        batch_size, num_channels_latents, height, width = latent_model_input.shape

        p = PATCH_SIZE

        # pack image tokens into patch_size-square patches, matching Krea2Pipeline._pack_latents
        latent_model_input = latent_model_input.view(
            batch_size, num_channels_latents, height // p, p, width // p, p
        )
        latent_model_input = latent_model_input.permute(0, 2, 4, 1, 3, 5)
        latent_model_input = latent_model_input.reshape(
            batch_size, (height // p) * (width // p), num_channels_latents * (p * p)
        )

        enc_hs = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        # Krea 2 uses a boolean mask; it is also consumed as rotary position input below.
        encoder_attention_mask = text_embeddings.attention_mask.to(
            self.device_torch, dtype=torch.bool
        )

        # Rotary coordinates for the joint text+image sequence. Text tokens sit at
        # the origin, image tokens carry their (0, h, w) latent-grid coordinates.
        grid_height, grid_width = height // p, width // p
        position_ids = self.pipeline.prepare_position_ids(
            enc_hs.shape[1], grid_height, grid_width, self.device_torch
        )

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(
                self.device_torch, self.torch_dtype
            ).detach(),
            encoder_hidden_states=enc_hs.detach(),
            timestep=(timestep / 1000).detach(),
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask.detach(),
            return_dict=False,
            **kwargs,
        )[0]

        # unpack back to latent grid
        noise_pred = noise_pred.view(
            batch_size, height // p, width // p, num_channels_latents, p, p
        )
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels_latents, height, width)
        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        # returns (batch, seq_len, num_text_layers, dim) -- Krea 2 stacks several
        # text encoder hidden layers per token, so these embeds are 4D, not 3D.
        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt,
            device=self.device_torch,
            num_images_per_prompt=1,
        )
        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_embeds_mask
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        # only save the transformer
        transformer: Krea2Transformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "krea2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        # refiner_blocks are the text fusion stage; only the main image blocks
        # are targeted for block-wise offloading / LoRA layer selection.
        return ["transformer_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)

        # AutoencoderKLQwenImage is a video vae, add a frame count dim
        images = images.unsqueeze(2)
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std
        latents = latents.to(device, dtype=dtype)

        latents = latents.squeeze(2)  # remove the frame count dimension

        return latents

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)

        # add frame count dim for the video vae
        latents = latents.unsqueeze(2)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        images = self.vae.decode(latents).sample

        images = images.squeeze(2)  # remove the frame count dimension

        return images.to(device, dtype=dtype)
